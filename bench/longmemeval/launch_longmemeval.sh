#!/usr/bin/env bash
# launch_longmemeval.sh — Launch EC2 to run LongMemEval benchmark
#
# Usage:
#   bash bench/longmemeval/launch_longmemeval.sh
#
# Requires:
#   - AWS CLI configured (profile: personal)
#   - SSH key: ~/.ssh/bench-key.pem
#   - ANTHROPIC_API_KEY set in environment
#
set -euo pipefail

PROFILE="personal"
REGION="us-east-1"
KEY_NAME="bench-key"
KEY_FILE="$HOME/.ssh/bench-key.pem"
SG_NAME="bench-sg"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTANCE_TYPE="c5.2xlarge"

echo "=== LongMemEval Benchmark — c5.2xlarge (8 vCPUs, 16GB, ~\$0.34/hr) ==="
echo ""

# Check ANTHROPIC_API_KEY
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Export it before running."
    exit 1
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

# Get or create security group
echo "-> Setting up security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text \
    --profile "$PROFILE" --region "$REGION" 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" ]]; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" --description "Benchmark SSH access" \
        --profile "$PROFILE" --region "$REGION" --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 \
        --profile "$PROFILE" --region "$REGION" >/dev/null
    echo "   Created: $SG_ID"
else
    echo "   Exists: $SG_ID"
fi

# User data script
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -ex

# Install system deps
dnf install -y python3.12 python3.12-pip git

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
systemctl start ollama

# Wait for Ollama
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

# Pull embedding model
ollama pull nomic-embed-text

# Install Python deps
pip3.12 install duckdb anthropic requests tqdm

# Signal ready
touch /tmp/bench-ready
echo "=== Instance ready ==="
USERDATA
)

# Launch instance
echo "-> Launching $INSTANCE_TYPE..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --user-data "$USER_DATA" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=longmemeval-bench}]" \
    --query 'Instances[0].InstanceId' --output text \
    --profile "$PROFILE" --region "$REGION")
echo "   Instance: $INSTANCE_ID"

# Wait for running
echo "-> Waiting for instance..."
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
echo "-> Syncing project files..."
rsync -az --exclude '__pycache__' --exclude '*.duckdb' --exclude '.git' \
    --exclude 'bench/longmemeval/datasets' --exclude 'bench/LMEB' \
    -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
    "$PROJECT_DIR/" ec2-user@"$PUBLIC_IP":~/ai-memory-db/

# Pass API key
echo "-> Configuring API key..."
ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" \
    "echo 'export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}' >> ~/.bashrc"

# Wait for user-data (Ollama + deps)
echo "-> Waiting for instance setup (Ollama + Python deps)..."
for i in $(seq 1 120); do
    if ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "test -f /tmp/bench-ready" 2>/dev/null; then
        echo "   Instance ready!"
        break
    fi
    if [[ $i -eq 120 ]]; then
        echo "   Timeout. Check: ssh -i $KEY_FILE ec2-user@$PUBLIC_IP 'sudo tail -f /var/log/cloud-init-output.log'"
    fi
    sleep 10
done

# Save instance info
echo "$INSTANCE_ID" > /tmp/longmemeval_instance_id
echo "$PUBLIC_IP" > /tmp/longmemeval_instance_ip

echo ""
echo "========================================================"
echo "  Instance: $INSTANCE_ID"
echo "  IP: $PUBLIC_IP"
echo "  Type: $INSTANCE_TYPE (8 vCPU, 16GB)"
echo ""
echo "  SSH:"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP"
echo ""
echo "  Quick test (60 questions, ~\$22):"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP \\"
echo "      'cd ai-memory-db && python3.12 bench/longmemeval/run_longmemeval.py \\"
echo "       --max-instances-per-category 10 --parallel 4'"
echo ""
echo "  Full run (500 questions, ~\$170):"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP \\"
echo "      'cd ai-memory-db && python3.12 bench/longmemeval/run_longmemeval.py \\"
echo "       --parallel 8'"
echo ""
echo "  Pull results:"
echo "    scp -i $KEY_FILE ec2-user@$PUBLIC_IP:~/ai-memory-db/bench/longmemeval/results/* bench/longmemeval/results/"
echo ""
echo "  Terminate:"
echo "    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $PROFILE --region $REGION"
echo ""
echo "  Cost: ~\$0.34/hr. REMEMBER TO TERMINATE."
echo "========================================================"
