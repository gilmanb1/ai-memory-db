#!/usr/bin/env bash
# launch_hindsight_bench.sh — Launch g5.xlarge, run hindsight's LongMemEval benchmark
#
# Runs hindsight in Docker, benchmarks with same parameters as ai-memory-db run.
# Requires: ANTHROPIC_API_KEY set
#
set -euo pipefail

PROFILE="personal"
REGION="us-east-1"
KEY_NAME="bench-key"
KEY_FILE="$HOME/.ssh/bench-key.pem"
SG_NAME="bench-sg"
HINDSIGHT_DIR="$HOME/projects/hindsight"
INSTANCE_TYPE="g5.xlarge"
AMI_ID="ami-0fb19e4efa5b16a03"  # Deep Learning Base AMI

echo "=== Hindsight LongMemEval Benchmark — g5.xlarge ==="

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY not set."
    exit 1
fi

if [[ ! -d "$HINDSIGHT_DIR" ]]; then
    echo "ERROR: hindsight repo not found at $HINDSIGHT_DIR"
    exit 1
fi

# Security group
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text \
    --profile "$PROFILE" --region "$REGION" 2>/dev/null || echo "None")
[[ "$SG_ID" == "None" ]] && {
    SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "Bench SSH" \
        --profile "$PROFILE" --region "$REGION" --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 \
        --profile "$PROFILE" --region "$REGION" >/dev/null
}
echo "-> SG: $SG_ID"

# Launch
echo "-> Launching $INSTANCE_TYPE..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=hindsight-bench}]" \
    --query 'Instances[0].InstanceId' --output text \
    --profile "$PROFILE" --region "$REGION")
echo "   Instance: $INSTANCE_ID"

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --profile "$PROFILE" --region "$REGION"
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text \
    --profile "$PROFILE" --region "$REGION")
echo "   IP: $PUBLIC_IP"

# Wait for SSH
echo "-> Waiting for SSH..."
for i in $(seq 1 60); do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "echo ok" 2>/dev/null && break
    sleep 5
done

# Sync hindsight repo (exclude heavy dirs)
echo "-> Syncing hindsight repo..."
rsync -az --exclude '.git' --exclude 'node_modules' --exclude '__pycache__' \
    --exclude '.next' --exclude 'target' --exclude '*.duckdb' \
    -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
    "$HINDSIGHT_DIR/" ec2-user@"$PUBLIC_IP":~/hindsight/

# Install everything
echo "-> Installing Docker, Python, uv, and starting hindsight..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=300 -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "
set -ex

# Install Docker
sudo dnf install -y docker python3.12 python3.12-pip git
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=\"\$HOME/.local/bin:\$PATH\"

# Pull and start hindsight Docker container
sudo docker pull ghcr.io/vectorize-io/hindsight:latest
sudo docker run -d --name hindsight \
    --gpus all \
    -p 8888:8888 \
    -e HINDSIGHT_API_LLM_PROVIDER=anthropic \
    -e HINDSIGHT_API_LLM_API_KEY=${ANTHROPIC_API_KEY} \
    -e HINDSIGHT_API_LLM_MODEL=claude-haiku-4-5-20251001 \
    -e HINDSIGHT_API_HOST=0.0.0.0 \
    -e HINDSIGHT_ENABLE_API=true \
    -e HINDSIGHT_ENABLE_CP=false \
    -v \$HOME/.hindsight-docker:/home/hindsight/.pg0 \
    ghcr.io/vectorize-io/hindsight:latest

echo 'Waiting for hindsight API...'
for i in \$(seq 1 60); do
    if curl -s http://localhost:8888/health >/dev/null 2>&1; then
        echo 'API ready!'
        break
    fi
    sleep 5
done

# Create .env for benchmark
cat > ~/hindsight/.env << ENVEOF
HINDSIGHT_API_LLM_PROVIDER=anthropic
HINDSIGHT_API_LLM_API_KEY=${ANTHROPIC_API_KEY}
HINDSIGHT_API_LLM_MODEL=claude-haiku-4-5-20251001
HINDSIGHT_API_BASE_URL=http://localhost:8888
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
ENVEOF

echo '=== SETUP COMPLETE ==='
"

# Save instance info
echo "$INSTANCE_ID" > /tmp/hindsight_instance_id
echo "$PUBLIC_IP" > /tmp/hindsight_instance_ip

echo ""
echo "========================================================"
echo "  Instance: $INSTANCE_ID"
echo "  IP: $PUBLIC_IP"
echo "  Type: $INSTANCE_TYPE (A10G GPU)"
echo ""
echo "  SSH:"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP"
echo ""
echo "  Run benchmark:"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP \\"
echo "      'cd ~/hindsight && source .env && export PATH=\$HOME/.local/bin:\$PATH && \\"
echo "       PYTHONUNBUFFERED=1 nohup uv run python hindsight-dev/benchmarks/longmemeval/longmemeval_benchmark.py \\"
echo "       --max-instances-per-category 5 --parallel 4 \\"
echo "       > ~/bench_results.log 2>&1 & disown'"
echo ""
echo "  Pull results:"
echo "    scp -i $KEY_FILE ec2-user@$PUBLIC_IP:~/hindsight/hindsight-dev/benchmarks/longmemeval/results/* ."
echo ""
echo "  Terminate:"
echo "    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $PROFILE --region $REGION"
echo ""
echo "  Cost: ~\$1.01/hr. REMEMBER TO TERMINATE."
echo "========================================================"
