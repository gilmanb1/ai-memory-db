#!/usr/bin/env bash
# run_benchmark_e2e.sh — End-to-end LongMemEval benchmark on AWS EC2
#
# Launches a GPU instance, installs deps, runs the benchmark, monitors,
# pulls results, terminates the instance, and reports results.
#
# Usage:
#   bash bench/longmemeval/run_benchmark_e2e.sh
#   bash bench/longmemeval/run_benchmark_e2e.sh --parallel 4          # fewer workers
#   bash bench/longmemeval/run_benchmark_e2e.sh --extract-model claude-sonnet-4-6  # use Sonnet
#   bash bench/longmemeval/run_benchmark_e2e.sh --per-category 10     # more questions
#
# Requires:
#   - ANTHROPIC_API_KEY set
#   - AWS CLI configured (profile: personal)
#   - SSH key: ~/.ssh/bench-key.pem
#
set -u
# Note: -e and pipefail intentionally omitted — grep returning no matches (exit 1) must not kill the script

# ── Configuration ────────────────────────────────────────────────────────
PROFILE="${AWS_PROFILE:-default}"
REGION="us-east-1"
KEY_FILE="${BENCH_KEY_FILE:-$HOME/.ssh/bench-key.pem}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/bench/longmemeval/results"
CHECK_INTERVAL=300  # seconds between progress checks
MAX_CHECKS=24       # max checks before giving up (24 * 5min = 2 hours)

# Parse args
PER_CATEGORY="${PER_CATEGORY:-5}"
PARALLEL="${PARALLEL:-8}"
EXTRACT_MODEL="${EXTRACT_MODEL:-claude-haiku-4-5-20251001}"
EXTRA_ARGS=""
RESULT_SUFFIX="latest"

while [[ $# -gt 0 ]]; do
    case $1 in
        --per-category) PER_CATEGORY="$2"; shift 2;;
        --parallel) PARALLEL="$2"; shift 2;;
        --extract-model) EXTRACT_MODEL="$2"; shift 2;;
        --agentic) EXTRA_ARGS="$EXTRA_ARGS --agentic"; shift;;
        --suffix) RESULT_SUFFIX="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# ── Preflight checks ────────────────────────────────────────────────────
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY not set"
    exit 1
fi
if [[ ! -f "$KEY_FILE" ]]; then
    echo "ERROR: SSH key not found: $KEY_FILE"
    exit 1
fi

echo "=== LongMemEval E2E Benchmark ==="
echo "  Questions: ${PER_CATEGORY} per category ($(($PER_CATEGORY * 6)) total)"
echo "  Workers: $PARALLEL"
echo "  Extraction: $EXTRACT_MODEL"
echo "  Extra args: $EXTRA_ARGS"
echo ""

# ── Helper functions ─────────────────────────────────────────────────────
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"

ssh_run() {
    ssh $SSH_OPTS -i "$KEY_FILE" "ec2-user@$IP" "$1" 2>&1 | grep -v 'WARNING\|post-quantum\|upgraded\|vulnerable'
}

cleanup() {
    echo ""
    echo "=== Cleanup: terminating instance ==="
    if [[ -n "${INSTANCE_ID:-}" ]]; then
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
            --profile "$PROFILE" --region "$REGION" \
            --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>/dev/null || true
    fi
    # Verify
    RUNNING=$(aws ec2 describe-instances \
        --filters "Name=instance-state-name,Values=running" \
        --query 'Reservations[].Instances[].InstanceId' --output text \
        --profile "$PROFILE" --region "$REGION" 2>/dev/null)
    if [[ -n "$RUNNING" ]]; then
        echo "WARNING: Still running instances: $RUNNING"
    else
        echo "All instances terminated."
    fi
}
trap cleanup EXIT

# ── Step 1: Launch instance ──────────────────────────────────────────────
echo "=== Step 1: Launching GPU instance ==="
bash "$PROJECT_DIR/bench/longmemeval/launch_longmemeval_gpu.sh" 2>&1 | tail -5

INSTANCE_ID=$(cat /tmp/longmemeval_instance_id)
IP=$(cat /tmp/longmemeval_instance_ip)
echo "Instance: $INSTANCE_ID at $IP"

# ── Step 2: Install ONNX deps ───────────────────────────────────────────
echo ""
echo "=== Step 2: Installing ONNX deps ==="
ssh_run "
pip3.12 install --quiet onnxruntime numpy transformers huggingface_hub 2>&1 | tail -1
python3.12 -c 'from huggingface_hub import hf_hub_download; hf_hub_download(\"nomic-ai/nomic-embed-text-v1.5\", \"onnx/model.onnx\"); print(\"ONNX model ready\")' 2>&1 | tail -1
cd ~/ai-memory-db && python3.12 -c 'import sys; sys.path.insert(0,\".\"); from memory import embeddings; v=embeddings.embed(\"test\"); print(f\"Backend: ONNX={embeddings._onnx_available} dim={len(v)}\")' 2>&1 | tail -1
"

# ── Step 3: Start benchmark ─────────────────────────────────────────────
echo ""
echo "=== Step 3: Starting benchmark ==="
BENCH_PID=$(ssh $SSH_OPTS -i "$KEY_FILE" "ec2-user@$IP" 'bash -s' << REMOTE
cd ~/ai-memory-db
PYTHONUNBUFFERED=1 nohup python3.12 bench/longmemeval/run_longmemeval.py \
  --max-instances-per-category $PER_CATEGORY --parallel $PARALLEL \
  --extract-model $EXTRACT_MODEL $EXTRA_ARGS \
  > bench/longmemeval/results/run.log 2>&1 &
disown -a
echo \$!
REMOTE
)
BENCH_PID=$(echo "$BENCH_PID" | grep -v 'WARNING\|post-quantum\|vulnerable' | tail -1)
echo "Benchmark PID: $BENCH_PID"

# ── Step 4: Monitor ─────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Monitoring (check every ${CHECK_INTERVAL}s, max ${MAX_CHECKS} checks) ==="

TOTAL_QUESTIONS=$(($PER_CATEGORY * 6))
for i in $(seq 1 $MAX_CHECKS); do
    sleep $CHECK_INTERVAL

    OUTPUT=$(ssh $SSH_OPTS -i "$KEY_FILE" "ec2-user@$IP" "
echo \"ELAPSED=\$(ps -o etime= -p $BENCH_PID 2>/dev/null || echo DONE)\"
echo \"CACHE=\$(ls ~/ai-memory-db/bench/longmemeval/cache/ 2>/dev/null | wc -l)\"
COMP=\$(grep -cE '^\s*\[[-+!]\]' ~/ai-memory-db/bench/longmemeval/results/run.log 2>/dev/null || echo 0)
echo \"COMPLETED=\$COMP\"
APILIM=\$(grep -c 'usage limits' ~/ai-memory-db/bench/longmemeval/results/run.log 2>/dev/null || echo 0)
echo \"APILIMIT=\$APILIM\"
grep -E '^\s*\[[-+!]\]' ~/ai-memory-db/bench/longmemeval/results/run.log 2>/dev/null | tail -3
" 2>&1 | grep -v 'WARNING\|post-quantum\|upgraded\|vulnerable') || true

    ELAPSED=$(echo "$OUTPUT" | grep "ELAPSED=" | head -1 | cut -d= -f2)
    COMPLETED=$(echo "$OUTPUT" | grep "COMPLETED=" | head -1 | cut -d= -f2)
    APILIMIT=$(echo "$OUTPUT" | grep "APILIMIT=" | head -1 | cut -d= -f2)
    CACHE=$(echo "$OUTPUT" | grep "CACHE=" | head -1 | cut -d= -f2)

    echo "  [$(($i * $CHECK_INTERVAL / 60))min] elapsed=$ELAPSED completed=${COMPLETED:-0}/$TOTAL_QUESTIONS cache=${CACHE:-0}"
    echo "$OUTPUT" | grep -E '^\s*\[[-+!]\]' | tail -2 | sed 's/^/    /'

    # Check API limit
    if [[ "${APILIMIT:-0}" -gt 0 ]]; then
        echo "  !!! API spend limit hit — stopping"
        break
    fi

    # Check if done
    if [[ "$ELAPSED" == *"DONE"* ]]; then
        if [[ "${COMPLETED:-0}" -ge "$TOTAL_QUESTIONS" ]]; then
            echo "  === ALL $TOTAL_QUESTIONS COMPLETE ==="
        else
            echo "  === Process ended with ${COMPLETED:-0}/$TOTAL_QUESTIONS ==="
            ssh_run "tail -5 ~/ai-memory-db/bench/longmemeval/results/run.log" | sed 's/^/    /'
        fi
        break
    fi
done

# ── Step 5: Pull results ────────────────────────────────────────────────
echo ""
echo "=== Step 5: Pulling results ==="
mkdir -p "$RESULTS_DIR"
scp $SSH_OPTS -i "$KEY_FILE" "ec2-user@$IP:~/ai-memory-db/bench/longmemeval/results/results.json" \
    "$RESULTS_DIR/results_${RESULT_SUFFIX}.json" 2>/dev/null || echo "No results.json"
scp $SSH_OPTS -i "$KEY_FILE" "ec2-user@$IP:~/ai-memory-db/bench/longmemeval/results/results.md" \
    "$RESULTS_DIR/results_${RESULT_SUFFIX}.md" 2>/dev/null || echo "No results.md"
scp $SSH_OPTS -i "$KEY_FILE" "ec2-user@$IP:~/ai-memory-db/bench/longmemeval/results/run.log" \
    "$RESULTS_DIR/run_${RESULT_SUFFIX}.log" 2>/dev/null || echo "No run.log"

# ── Step 6: Report ──────────────────────────────────────────────────────
echo ""
echo "=== Step 6: Results ==="
python3 -c "
import json, os, sys

v6_path = '$RESULTS_DIR/results_${RESULT_SUFFIX}.json'
v1_path = '$RESULTS_DIR/results.json'

if not os.path.exists(v6_path):
    print('No results file found')
    sys.exit(0)

with open(v6_path) as f: v6 = json.load(f)

print(f'Overall: {v6[\"overall_accuracy\"]}% ({v6[\"total_correct\"]}/{v6[\"total_valid\"]}) invalid={v6[\"total_invalid\"]}')
print(f'Hindsight: 91.4%')
print()
print('Category breakdown:')
for k,v in sorted(v6.get('category_stats',{}).items()):
    print(f'  {k:30s} {v[\"accuracy\"]:>5.1f}% ({v[\"correct\"]}/{v[\"total\"]})')

if os.path.exists(v1_path):
    with open(v1_path) as f: v1 = json.load(f)
    v1_by_id = {r['question_id']: r for r in v1['detailed_results']}
    v6_by_id = {r['question_id']: r for r in v6['detailed_results']}
    def s(r):
        if not r: return '?'
        if r.get('is_invalid'): return '!'
        return '+' if r.get('is_correct') else '-'
    fixes = regressions = 0
    for qid in v6_by_id:
        s1 = s(v1_by_id.get(qid))
        s6 = s(v6_by_id[qid])
        if s1 != '+' and s6 == '+': fixes += 1
        elif s1 == '+' and s6 != '+': regressions += 1
    print(f'\nvs V1 baseline ({v1[\"overall_accuracy\"]}%): +{fixes} fixed, -{regressions} regressed')
" 2>/dev/null || echo "Analysis failed"

# Instance terminated by trap
echo ""
echo "=== Complete. Results in $RESULTS_DIR/results_${RESULT_SUFFIX}.json ==="
