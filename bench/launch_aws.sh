#!/usr/bin/env bash
# launch_aws.sh — Launch an EC2 instance to run LMEB + LongMemEval benchmarks
#
# Usage:
#   bash bench/launch_aws.sh          # GPU (g5.xlarge) - requires quota
#   bash bench/launch_aws.sh --cpu    # CPU fallback (c5.4xlarge) - no quota needed
#
set -euo pipefail

PROFILE="personal"
REGION="us-east-1"
KEY_NAME="bench-key"
KEY_FILE="$HOME/.ssh/bench-key.pem"
SG_NAME="bench-sg"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Instance type
if [[ "${1:-}" == "--cpu" ]]; then
    INSTANCE_TYPE="c5.4xlarge"
    echo "=== CPU mode: c5.4xlarge (16 vCPUs, 32GB RAM, ~\$0.68/hr) ==="
else
    INSTANCE_TYPE="g5.xlarge"
    echo "=== GPU mode: g5.xlarge (1x A10G, 4 vCPUs, 16GB RAM, ~\$1.01/hr) ==="
fi

# Find latest Amazon Linux 2023 AMI
echo "-> Finding AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --profile "$PROFILE" --region "$REGION")
echo "   AMI: $AMI_ID"

# Create security group (if not exists)
echo "-> Setting up security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --profile "$PROFILE" --region "$REGION" 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" ]]; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Benchmark instance SSH access" \
        --profile "$PROFILE" --region "$REGION" \
        --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 \
        --cidr 0.0.0.0/0 \
        --profile "$PROFILE" --region "$REGION" >/dev/null
    echo "   Created: $SG_ID"
else
    echo "   Exists: $SG_ID"
fi

# User data script — runs on first boot
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -ex

# Install system deps
dnf install -y python3.12 python3.12-pip git

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
systemctl start ollama

# Wait for Ollama to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

# Pull embedding model
ollama pull nomic-embed-text

# Install Python deps
pip3.12 install duckdb anthropic requests tqdm numpy
pip3.12 install mteb==2.3.0 datasets==2.21.0 Pillow torch

# Signal ready
touch /tmp/bench-ready
echo "=== Instance ready for benchmarks ==="
USERDATA
)

# Launch instance
echo "-> Launching $INSTANCE_TYPE instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --user-data "$USER_DATA" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bench-lmeb}]" \
    --query 'Instances[0].InstanceId' \
    --output text \
    --profile "$PROFILE" --region "$REGION")
echo "   Instance: $INSTANCE_ID"

# Wait for running
echo "-> Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" \
    --profile "$PROFILE" --region "$REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
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
echo "-> Syncing project files..."
rsync -az --exclude '__pycache__' --exclude '*.duckdb' --exclude '.git' \
    -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
    "$PROJECT_DIR/" ec2-user@"$PUBLIC_IP":~/ai-memory-db/

# Wait for user-data to finish (Ollama + deps)
echo "-> Waiting for instance setup (Ollama + Python deps)..."
for i in $(seq 1 120); do
    if ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "test -f /tmp/bench-ready" 2>/dev/null; then
        echo "   Instance ready!"
        break
    fi
    if [[ $i -eq 120 ]]; then
        echo "   Timeout waiting for setup. SSH in and check: sudo tail -f /var/log/cloud-init-output.log"
    fi
    sleep 10
done

# Print next steps
echo ""
echo "========================================================"
echo "  Instance ready: $INSTANCE_ID"
echo "  IP: $PUBLIC_IP"
echo "  Type: $INSTANCE_TYPE"
echo ""
echo "  SSH:"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP"
echo ""
echo "  Run LMEB (all 22 tasks):"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP 'cd ai-memory-db && python3.12 bench/LMEB/run_ollama.py 2>&1 | tee lmeb_results.txt'"
echo ""
echo "  Pull results:"
echo "    scp -i $KEY_FILE ec2-user@$PUBLIC_IP:~/ai-memory-db/bench/LMEB/lmeb_results/summary_nomic-embed-text.json ./bench_results_lmeb.json"
echo ""
echo "  Terminate when done:"
echo "    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $PROFILE --region $REGION"
echo ""
echo "  Cost: ~\$1/hr (GPU) or ~\$0.68/hr (CPU). REMEMBER TO TERMINATE."
echo "========================================================"
