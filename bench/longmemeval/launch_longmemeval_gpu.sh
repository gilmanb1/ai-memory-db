#!/usr/bin/env bash
# launch_longmemeval_gpu.sh — Launch g5.xlarge GPU instance for fast LongMemEval
#
# Uses A10G GPU for 10-50x faster Ollama embeddings.
# Requires G/VT instance quota >= 4 vCPUs.
#
set -euo pipefail

PROFILE="personal"
REGION="us-east-1"
KEY_NAME="bench-key"
KEY_FILE="$HOME/.ssh/bench-key.pem"
SG_NAME="bench-sg"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTANCE_TYPE="g5.xlarge"
# Deep Learning Base AMI with NVIDIA drivers
AMI_ID="ami-0fb19e4efa5b16a03"

echo "=== LongMemEval GPU Benchmark — g5.xlarge (A10G, 4 vCPU, 16GB, ~\$1.01/hr) ==="
echo ""

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY not set."
    exit 1
fi

# Security group
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text \
    --profile "$PROFILE" --region "$REGION" 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" ]]; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" --description "Benchmark SSH" \
        --profile "$PROFILE" --region "$REGION" --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 \
        --profile "$PROFILE" --region "$REGION" >/dev/null
fi
echo "-> SG: $SG_ID"

# Launch
echo "-> Launching $INSTANCE_TYPE..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":80,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=longmemeval-gpu}]" \
    --query 'Instances[0].InstanceId' --output text \
    --profile "$PROFILE" --region "$REGION")
echo "   Instance: $INSTANCE_ID"

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" \
    --profile "$PROFILE" --region "$REGION"
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text \
    --profile "$PROFILE" --region "$REGION")
echo "   IP: $PUBLIC_IP"

# Wait for SSH
echo "-> Waiting for SSH..."
for i in $(seq 1 60); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "echo ok" 2>/dev/null; then
        break
    fi
    sleep 5
done

# Sync project
echo "-> Syncing project..."
rsync -az --exclude '__pycache__' --exclude '*.duckdb' --exclude '.git' \
    --exclude 'bench/longmemeval/datasets' --exclude 'bench/LMEB' \
    -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
    "$PROJECT_DIR/" ec2-user@"$PUBLIC_IP":~/ai-memory-db/

# Also sync extraction cache from previous runs
if [[ -d "$PROJECT_DIR/bench/longmemeval/cache" ]]; then
    rsync -az -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
        "$PROJECT_DIR/bench/longmemeval/cache/" ec2-user@"$PUBLIC_IP":~/ai-memory-db/bench/longmemeval/cache/
fi

# Install deps + Ollama with GPU
echo "-> Installing deps (Ollama + GPU + Python)..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=120 -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "
set -ex
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama
for i in \$(seq 1 30); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

# Pull embedding model (uses GPU automatically)
ollama pull nomic-embed-text

# Verify GPU
nvidia-smi 2>/dev/null | head -5 || echo 'No nvidia-smi (CPU mode)'
ollama ps 2>/dev/null || true

# Install Python deps
pip3.12 install --quiet duckdb anthropic requests tqdm

# Set API key
echo 'export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}' >> ~/.bashrc

echo '=== SETUP COMPLETE ==='
"

# Save instance info
echo "$INSTANCE_ID" > /tmp/longmemeval_instance_id
echo "$PUBLIC_IP" > /tmp/longmemeval_instance_ip

echo ""
echo "========================================================"
echo "  Instance: $INSTANCE_ID"
echo "  IP: $PUBLIC_IP"
echo "  Type: $INSTANCE_TYPE (A10G GPU, 4 vCPU, 16GB)"
echo ""
echo "  SSH:"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP"
echo ""
echo "  Run benchmark (~15-30 min with GPU):"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP \\"
echo "      'cd ai-memory-db && PYTHONUNBUFFERED=1 nohup python3.12 bench/longmemeval/run_longmemeval.py \\"
echo "       --max-instances-per-category 5 --parallel 8 --extract-model claude-haiku-4-5-20251001 \\"
echo "       > bench/longmemeval/results/run.log 2>&1 & disown'"
echo ""
echo "  Pull results:"
echo "    scp -i $KEY_FILE ec2-user@$PUBLIC_IP:~/ai-memory-db/bench/longmemeval/results/* bench/longmemeval/results/"
echo ""
echo "  Terminate:"
echo "    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $PROFILE --region $REGION"
echo ""
echo "  Cost: ~\$1.01/hr. REMEMBER TO TERMINATE."
echo "========================================================"
