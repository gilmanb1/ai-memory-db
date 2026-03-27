#!/usr/bin/env bash
# launch_precompute.sh — Launch g5.xlarge, run LMEB with pre-computed embeddings
#
# Phase 1: Encode all corpora/queries to .npy (restartable, skips cached)
# Phase 2: Evaluate from cache (near-zero RAM, instant)
#
# Uses nice -n 15 to keep SSH responsive during encoding.
#
set -euo pipefail

PROFILE="${AWS_PROFILE:-default}"
REGION="us-east-1"
KEY_NAME="${BENCH_KEY_NAME:-bench-key}"
KEY_FILE="${BENCH_KEY_FILE:-$HOME/.ssh/bench-key.pem}"
SG_NAME="bench-sg"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE_TYPE="g5.xlarge"
AMI_ID="${BENCH_AMI_ID:-}"

echo "=== LMEB Pre-compute Benchmark ==="

# Get security group
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

# Launch
echo "-> Launching $INSTANCE_TYPE..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":80,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lmeb-precompute}]" \
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

# Sync project
echo "-> Syncing project..."
rsync -az --exclude '__pycache__' --exclude '*.duckdb' --exclude '.git' --exclude 'embedding_cache' \
    -e "ssh -o StrictHostKeyChecking=no -i $KEY_FILE" \
    "$PROJECT_DIR/" ec2-user@"$PUBLIC_IP":~/ai-memory-db/

# Install deps + download data
echo "-> Installing deps..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=120 -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "
set -ex
pip3.12 install --quiet sentence-transformers einops torch numpy tqdm
pip3.12 install --quiet 'mteb==2.3.0' 'datasets==2.21.0' Pillow
python3.12 -c 'from huggingface_hub import snapshot_download; snapshot_download(\"KaLM-Embedding/LMEB\", repo_type=\"dataset\", local_dir=\"ai-memory-db/bench/LMEB/\"); print(\"OK\")'
"

# Run benchmark with nice (keeps SSH responsive)
echo "-> Starting benchmark (nice -n 15)..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=120 -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "
cd ~/ai-memory-db
nohup nice -n 15 python3.12 bench/LMEB/run_precompute.py --batch-size 512 --overwrite 2>&1 | tee lmeb_precompute.log &
echo \"PID: \$!\"
"

# Save info
echo "$INSTANCE_ID" > /tmp/lmeb_instance_id
echo "$PUBLIC_IP" > /tmp/lmeb_instance_ip

echo ""
echo "========================================================"
echo "  Instance: $INSTANCE_ID  IP: $PUBLIC_IP"
echo ""
echo "  Monitor:"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP 'tail -f ~/ai-memory-db/lmeb_precompute.log'"
echo ""
echo "  Check progress:"
echo "    ssh -i $KEY_FILE ec2-user@$PUBLIC_IP 'ls ~/ai-memory-db/bench/LMEB/embedding_cache/*.npy | wc -l'"
echo ""
echo "  Pull results + terminate:"
echo "    bash bench/pull_and_terminate.sh"
echo "========================================================"
