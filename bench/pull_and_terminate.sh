#!/usr/bin/env bash
# pull_and_terminate.sh — Pull LMEB results from EC2 and terminate
set -euo pipefail

PROFILE="personal"
REGION="us-east-1"
KEY_FILE="$HOME/.ssh/bench-key.pem"
INSTANCE_ID=$(cat /tmp/lmeb_instance_id 2>/dev/null || echo "")
PUBLIC_IP=$(cat /tmp/lmeb_instance_ip 2>/dev/null || echo "")

if [[ -z "$INSTANCE_ID" || -z "$PUBLIC_IP" ]]; then
    echo "ERROR: No instance info found. Set INSTANCE_ID and PUBLIC_IP manually."
    exit 1
fi

echo "Instance: $INSTANCE_ID ($PUBLIC_IP)"

# Pull results
echo "-> Pulling results..."
mkdir -p bench/lmeb_final_results
scp -o StrictHostKeyChecking=no -o ConnectTimeout=120 -i "$KEY_FILE" \
    "ec2-user@$PUBLIC_IP:~/ai-memory-db/bench/LMEB/lmeb_results_precomputed/summary_nomic-embed-text-v1.5.json" \
    bench/lmeb_final_results/ 2>/dev/null || echo "Summary not found yet"

scp -o StrictHostKeyChecking=no -o ConnectTimeout=120 -i "$KEY_FILE" \
    "ec2-user@$PUBLIC_IP:~/ai-memory-db/lmeb_precompute.log" \
    bench/lmeb_final_results/ 2>/dev/null || echo "Log not found"

# Show results
echo ""
if [[ -f bench/lmeb_final_results/summary_nomic-embed-text-v1.5.json ]]; then
    python3 -c "
import json
with open('bench/lmeb_final_results/summary_nomic-embed-text-v1.5.json') as f:
    data = json.load(f)
print(f'Tasks: {len(data.get(\"tasks\",{}))}')
print(f'Overall: {data.get(\"overall\", \"N/A\")}')
for k, v in data.get('type_averages', {}).items():
    print(f'  {k}: {v}')
"
fi

# Terminate
echo ""
echo "-> Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --profile "$PROFILE" --region "$REGION" \
    --query 'TerminatingInstances[0].CurrentState.Name' --output text
echo "Done."
rm -f /tmp/lmeb_instance_id /tmp/lmeb_instance_ip
