#!/usr/bin/env bash
# launch_gpu_bench.sh — Launch g5.xlarge, run full LMEB via direct GPU encoding
#
# Uses sentence-transformers to load nomic-embed-text directly on A10G GPU.
# ~10x faster than Ollama HTTP API. Full 22-task suite in ~20-30 min.
#
# Usage: bash bench/launch_gpu_bench.sh
#
set -euo pipefail

PROFILE="${AWS_PROFILE:-default}"
REGION="us-east-1"
KEY_NAME="${BENCH_KEY_NAME:-bench-key}"
KEY_FILE="${BENCH_KEY_FILE:-$HOME/.ssh/bench-key.pem}"
SG_NAME="bench-sg"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE_TYPE="g5.xlarge"
# Deep Learning Base AMI with NVIDIA drivers (Amazon Linux 2023)
AMI_ID="${BENCH_AMI_ID:-}"

echo "=== LMEB GPU Benchmark — g5.xlarge + direct sentence-transformers ==="
echo ""

# Get or create security group
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
echo "-> Security group: $SG_ID"

# Launch instance
echo "-> Launching $INSTANCE_TYPE..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":80,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lmeb-gpu-bench}]" \
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
echo "   SSH ready"

# Sync project
echo "-> Syncing project..."
rsync -az --exclude '__pycache__' --exclude '*.duckdb' --exclude '.git' \
    -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
    "$PROJECT_DIR/" ec2-user@"$PUBLIC_IP":~/ai-memory-db/

# Install deps and download data (no Ollama needed)
echo "-> Installing Python deps + downloading LMEB data..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=120 -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "
set -ex
# Install deps
pip3.12 install --quiet sentence-transformers torch numpy tqdm
pip3.12 install --quiet 'mteb==2.3.0' 'datasets==2.21.0' Pillow

# Download LMEB data
python3.12 -c '
from huggingface_hub import snapshot_download
snapshot_download(\"KaLM-Embedding/LMEB\", repo_type=\"dataset\", local_dir=\"ai-memory-db/bench/LMEB/\")
print(\"Data downloaded\")
'
echo '=== SETUP COMPLETE ==='
"
echo "   Setup complete"

# Run benchmark
echo "-> Starting LMEB benchmark (all 22 tasks)..."
echo "   Monitor: ssh -i $KEY_FILE ec2-user@$PUBLIC_IP 'tail -f ~/ai-memory-db/lmeb_gpu.log'"
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=120 -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "
cd ~/ai-memory-db
nohup python3.12 bench/LMEB/run_gpu.py --batch-size 512 --overwrite 2>&1 | tee lmeb_gpu.log &
echo \"PID: \$!\"
"

echo ""
echo "========================================================"
echo "  Benchmark running on: $INSTANCE_ID ($PUBLIC_IP)"
echo "  Type: $INSTANCE_TYPE (A10G GPU, 16GB RAM)"
echo ""
echo "  Monitor:"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP 'tail -f ~/ai-memory-db/lmeb_gpu.log'"
echo ""
echo "  Pull results when done:"
echo "    scp -i $KEY_FILE ec2-user@$PUBLIC_IP:~/ai-memory-db/bench/LMEB/lmeb_results_gpu/summary_nomic-embed-text-v1.5.json ./lmeb_results.json"
echo ""
echo "  Terminate:"
echo "    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $PROFILE --region $REGION"
echo ""
echo "  Expected runtime: ~20-30 min for all 22 tasks"
echo "  Cost: ~\$0.50 total"
echo "========================================================"

# Save instance info for later
echo "$INSTANCE_ID" > /tmp/lmeb_instance_id
echo "$PUBLIC_IP" > /tmp/lmeb_instance_ip
