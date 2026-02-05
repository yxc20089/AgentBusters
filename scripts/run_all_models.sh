#!/usr/bin/env bash
# =============================================================================
# AgentBusters — Run 50-task FAB++ evaluation across multiple models
# =============================================================================
# Usage: bash scripts/run_all_models.sh [--models MODEL1,MODEL2,...] [--max-retries N] [--skip-setup]
#
# Prerequisites:
#   - vLLM installed (pip install vllm)
#   - AgentBusters installed (pip install -e ".[all]")
#   - .env file with OPENAI_EVAL_API_KEY for LLM-as-judge
#   - agentbusters-eval-data repo cloned at ../agentbusters-eval-data
#
# Each model evaluation:
#   1. Starts vLLM server with model-specific flags
#   2. Starts Purple Agent (finance analyst) pointing at vLLM
#   3. Starts Green Agent (evaluator) with eval_medium.yaml
#   4. Runs 50-task evaluation via run_a2a_eval.py
#   5. Saves results to results/eval_medium_<model_id>.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv if available
if [ -f "$HOME/venv/bin/activate" ]; then
    source "$HOME/venv/bin/activate"
fi

# Pre-download tiktoken encoding for GPT-OSS models
TIKTOKEN_DIR="$PROJECT_DIR/.tiktoken_cache"
if [ ! -f "$TIKTOKEN_DIR/o200k_base.tiktoken" ]; then
    mkdir -p "$TIKTOKEN_DIR"
    wget -q -O "$TIKTOKEN_DIR/o200k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" 2>/dev/null || true
fi
export TIKTOKEN_ENCODINGS_BASE="$TIKTOKEN_DIR"

# Ports
VLLM_PORT=8000
PURPLE_PORT=9110
GREEN_PORT=9109

# Eval settings
NUM_TASKS=50
EVAL_CONFIG="config/eval_medium.yaml"
TIMEOUT=21600
RESULTS_DIR="results"
MAX_RETRIES=3

# Logging & checkpoints
LOG_DIR="logs"
CHECKPOINT_FILE="${RESULTS_DIR}/.checkpoint.json"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# PID tracking
VLLM_PID=""
PURPLE_PID=""
GREEN_PID=""

# ========================== Model Definitions ================================

declare -A MODEL_HF MODEL_PARSER MODEL_EXTRA MODEL_ENV

# 1. Qwen3-8B
MODEL_HF[qwen3-8b]="Qwen/Qwen3-8B"
MODEL_PARSER[qwen3-8b]="hermes"
MODEL_EXTRA[qwen3-8b]=""
MODEL_ENV[qwen3-8b]=""

# 2. GPT-OSS-20B
MODEL_HF[gpt-oss-20b]="openai/gpt-oss-20b"
MODEL_PARSER[gpt-oss-20b]="openai"
MODEL_EXTRA[gpt-oss-20b]="--async-scheduling"
MODEL_ENV[gpt-oss-20b]="VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1"

# 3. Gemma 3 27B
MODEL_HF[gemma3-27b]="google/gemma-3-27b-it"
MODEL_PARSER[gemma3-27b]="hermes"
MODEL_EXTRA[gemma3-27b]="--trust-remote-code"
MODEL_ENV[gemma3-27b]=""

# 4. Qwen3-30B-A3B
MODEL_HF[qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B"
MODEL_PARSER[qwen3-30b-a3b]="hermes"
MODEL_EXTRA[qwen3-30b-a3b]=""
MODEL_ENV[qwen3-30b-a3b]=""

# 5. GLM-4.7-Flash
MODEL_HF[glm-4.7-flash]="zai-org/GLM-4.7-Flash"
MODEL_PARSER[glm-4.7-flash]="glm47"
MODEL_EXTRA[glm-4.7-flash]="--reasoning-parser glm45 --speculative-config.method mtp --speculative-config.num_speculative_tokens 1 --trust-remote-code"
MODEL_ENV[glm-4.7-flash]=""

# 6. Qwen3-32B (non-thinking)
MODEL_HF[qwen3-32b]="Qwen/Qwen3-32B"
MODEL_PARSER[qwen3-32b]="hermes"
MODEL_EXTRA[qwen3-32b]=""
MODEL_ENV[qwen3-32b]=""

# 7. Qwen3-32B-thinking
MODEL_HF[qwen3-32b-thinking]="Qwen/Qwen3-32B"
MODEL_PARSER[qwen3-32b-thinking]="hermes"
MODEL_EXTRA[qwen3-32b-thinking]="--enable-reasoning --reasoning-parser qwen3"
MODEL_ENV[qwen3-32b-thinking]=""

# 8. GPT-OSS-120B
MODEL_HF[gpt-oss-120b]="openai/gpt-oss-120b"
MODEL_PARSER[gpt-oss-120b]="openai"
MODEL_EXTRA[gpt-oss-120b]="--async-scheduling"
MODEL_ENV[gpt-oss-120b]="VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1"

# 9. Qwen3-Next-80B-A3B (FP8)
MODEL_HF[qwen3-next-80b-a3b]="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
MODEL_PARSER[qwen3-next-80b-a3b]="hermes"
MODEL_EXTRA[qwen3-next-80b-a3b]="--gpu-memory-utilization 0.95 --max-model-len 8192"
MODEL_ENV[qwen3-next-80b-a3b]="VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=latency"

# Default execution order (smallest → largest)
DEFAULT_ORDER="qwen3-8b gpt-oss-20b gemma3-27b qwen3-30b-a3b glm-4.7-flash qwen3-32b qwen3-32b-thinking gpt-oss-120b qwen3-next-80b-a3b"

# ========================== Helper Functions =================================

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
err() { log "ERROR: $*" >&2; }

# ========================== Checkpoint Functions ==============================

checkpoint_status() {
    # Usage: checkpoint_status <model_id> → prints "completed", "failed", or ""
    local model_id="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        python3 -c "
import json, sys
with open('$CHECKPOINT_FILE') as f: data = json.load(f)
entry = data.get('models', {}).get('$model_id', {})
print(entry.get('status', ''))
" 2>/dev/null || true
    fi
}

checkpoint_update() {
    # Usage: checkpoint_update <model_id> <status> [attempt] [elapsed_s] [error_msg]
    local model_id="$1" status="$2"
    local attempt="${3:-0}" elapsed="${4:-0}" error_msg="${5:-}"
    python3 -c "
import json, os, time
path = '$CHECKPOINT_FILE'
data = {}
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
data.setdefault('models', {})
data['models']['$model_id'] = {
    'status': '$status',
    'attempt': $attempt,
    'elapsed_seconds': $elapsed,
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'error': '''$error_msg''' if '$error_msg' else None,
    'result_file': 'results/eval_medium_${model_id}.json' if '$status' == 'completed' else None
}
data['last_updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
"
}

checkpoint_summary() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        python3 -c "
import json
with open('$CHECKPOINT_FILE') as f: data = json.load(f)
models = data.get('models', {})
completed = [m for m, v in models.items() if v['status'] == 'completed']
failed = [m for m, v in models.items() if v['status'] == 'failed']
print(f'Checkpoint: {len(completed)} completed, {len(failed)} failed')
if completed: print(f'  Completed: {\" \".join(completed)}')
if failed: print(f'  Failed: {\" \".join(failed)}')
" || true
    fi
}

cleanup() {
    log "Cleaning up processes..."
    # Graceful kill first
    [ -n "$GREEN_PID" ] && kill "$GREEN_PID" 2>/dev/null || true
    [ -n "$PURPLE_PID" ] && kill "$PURPLE_PID" 2>/dev/null || true
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
    sleep 2
    # Force-kill vLLM and all its child processes (EngineCore, etc.)
    [ -n "$VLLM_PID" ] && kill -9 "$VLLM_PID" 2>/dev/null || true
    # Kill ALL vllm-related processes (main + EngineCore children)
    pkill -9 -f "vllm" 2>/dev/null || true
    # Kill any python process using the GPU
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [ -n "$gpu_pids" ]; then
        log "Killing GPU processes: $gpu_pids"
        echo "$gpu_pids" | xargs -r kill -9 2>/dev/null || true
    fi
    # Kill by port
    fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
    fuser -k "${PURPLE_PORT}/tcp" 2>/dev/null || true
    fuser -k "${GREEN_PORT}/tcp" 2>/dev/null || true
    VLLM_PID="" PURPLE_PID="" GREEN_PID=""
    # Wait for GPU memory to be fully released and verify
    local max_gpu_wait=30 gpu_elapsed=0
    while [ "$gpu_elapsed" -lt "$max_gpu_wait" ]; do
        local mem_used
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$mem_used" ] && [ "$mem_used" -lt 2000 ]; then
            log "GPU memory freed (${mem_used} MiB used)"
            break
        fi
        sleep 2
        gpu_elapsed=$((gpu_elapsed + 2))
        if [ "$((gpu_elapsed % 10))" -eq 0 ]; then
            log "Waiting for GPU memory release... (${mem_used} MiB still used)"
            # Re-kill any lingering GPU processes
            gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
            [ -n "$gpu_pids" ] && echo "$gpu_pids" | xargs -r kill -9 2>/dev/null || true
        fi
    done
    log "Cleanup done. GPU memory:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true
}

wait_for_endpoint() {
    local url="$1" max_wait="${2:-600}" interval="${3:-10}"
    local elapsed=0
    log "Waiting for $url (max ${max_wait}s)..."
    while [ "$elapsed" -lt "$max_wait" ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            log "Endpoint $url ready (${elapsed}s)"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    err "Timeout waiting for $url after ${max_wait}s"
    return 1
}

start_vllm() {
    local model_id="$1"
    local hf_model="${MODEL_HF[$model_id]}"
    local parser="${MODEL_PARSER[$model_id]}"
    local extra="${MODEL_EXTRA[$model_id]}"
    local env_vars="${MODEL_ENV[$model_id]}"

    log "Starting vLLM: $hf_model (parser=$parser)"

    local cmd="vllm serve $hf_model \
        --port $VLLM_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --enable-auto-tool-choice \
        --tool-call-parser $parser \
        --dtype bfloat16 \
        $extra"

    # Apply environment variables
    if [ -n "$env_vars" ]; then
        cmd="env $env_vars $cmd"
    fi

    log "Command: $cmd"
    eval "$cmd" > "$LOG_DIR/vllm_${model_id}.log" 2>&1 &
    VLLM_PID=$!
    log "vLLM PID: $VLLM_PID"

    # Wait for vLLM to be ready (model loading can take minutes)
    if ! wait_for_endpoint "http://localhost:${VLLM_PORT}/v1/models" 900 15; then
        err "vLLM failed to start for $model_id"
        tail -50 "$LOG_DIR/vllm_${model_id}.log"
        return 1
    fi
}

start_purple_agent() {
    local model_id="$1"
    local hf_model="${MODEL_HF[$model_id]}"

    log "Starting Purple Agent (model=$hf_model)"

    LLM_PROVIDER=openai \
    LLM_MODEL="$hf_model" \
    OPENAI_API_KEY=dummy \
    OPENAI_API_BASE="http://localhost:${VLLM_PORT}/v1" \
    OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1" \
    PURPLE_LLM_TEMPERATURE=0.0 \
    python -m purple_agent.cli serve \
        --host 0.0.0.0 \
        --port "$PURPLE_PORT" \
        > "$LOG_DIR/purple_${model_id}.log" 2>&1 &
    PURPLE_PID=$!
    log "Purple Agent PID: $PURPLE_PID"

    if ! wait_for_endpoint "http://localhost:${PURPLE_PORT}/.well-known/agent.json" 60 5; then
        # Try new endpoint
        wait_for_endpoint "http://localhost:${PURPLE_PORT}/.well-known/agent-card.json" 30 5 || {
            err "Purple Agent failed to start"
            tail -20 "$LOG_DIR/purple_${model_id}.log"
            return 1
        }
    fi
}

start_green_agent() {
    local model_id="$1"

    log "Starting Green Agent (config=$EVAL_CONFIG)"

    PYTHONPATH=src python -m cio_agent.a2a_server \
        --host 0.0.0.0 \
        --port "$GREEN_PORT" \
        --eval-config "$EVAL_CONFIG" \
        --store-predicted \
        --store-question \
        --store-expected \
        > "$LOG_DIR/green_${model_id}.log" 2>&1 &
    GREEN_PID=$!
    log "Green Agent PID: $GREEN_PID"

    if ! wait_for_endpoint "http://localhost:${GREEN_PORT}/.well-known/agent.json" 60 5; then
        wait_for_endpoint "http://localhost:${GREEN_PORT}/.well-known/agent-card.json" 30 5 || {
            err "Green Agent failed to start"
            tail -20 "$LOG_DIR/green_${model_id}.log"
            return 1
        }
    fi
}

run_eval() {
    local model_id="$1"
    local output_file="${RESULTS_DIR}/eval_medium_${model_id}.json"

    log "Running evaluation: $model_id → $output_file"

    set +eo pipefail
    PYTHONPATH=src python scripts/run_a2a_eval.py \
        --green-url "http://localhost:${GREEN_PORT}" \
        --purple-url "http://localhost:${PURPLE_PORT}" \
        --num-tasks "$NUM_TASKS" \
        --model-id "$model_id" \
        --timeout "$TIMEOUT" \
        --output "$output_file" \
        --verbose \
        2>&1 | tee "$LOG_DIR/eval_${model_id}.log"
    local exit_code=${PIPESTATUS[0]}
    set -eo pipefail

    if [ -f "$output_file" ]; then
        local size
        size=$(stat -c%s "$output_file" 2>/dev/null || stat -f%z "$output_file" 2>/dev/null || echo "?")
        log "Result file: $output_file (${size} bytes)"
    else
        err "Result file not created: $output_file"
    fi

    return $exit_code
}

# ========================== Main Execution ===================================

# Parse arguments
MODELS="$DEFAULT_ORDER"
FRESH=false
while [ $# -gt 0 ]; do
    case "$1" in
        --models)
            MODELS="${2//,/ }"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --fresh)
            FRESH=true
            shift
            ;;
        --skip-setup)
            shift
            ;;
        *)
            err "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# If --fresh, clear all results and checkpoint for requested models
if $FRESH; then
    log "FRESH mode: clearing old results and checkpoint"
    for m in $MODELS; do
        rm -f "${RESULTS_DIR}/eval_medium_${m}.json"
    done
    rm -f "$CHECKPOINT_FILE"
fi

trap cleanup EXIT

log "=========================================="
log "AgentBusters Multi-Model Evaluation"
log "=========================================="
log "Models: $MODELS"
log "Tasks per model: $NUM_TASKS"
log "Config: $EVAL_CONFIG"
log "Results dir: $RESULTS_DIR"
log "Max retries: $MAX_RETRIES"
log "Checkpoint: $CHECKPOINT_FILE"
log ""

# Show existing checkpoint state if resuming
checkpoint_summary

TOTAL_MODELS=$(echo "$MODELS" | wc -w)
CURRENT=0
SUCCEEDED=0
FAILED=0
SKIPPED=0
FAILED_LIST=""

run_model_eval() {
    # Run a single model evaluation (vLLM + Purple + Green + eval).
    # Returns 0 on success, 1 on failure.
    local model_id="$1"

    cleanup  # Kill any leftover processes

    # Step 1: Start vLLM
    if ! start_vllm "$model_id"; then
        err "Could not start vLLM for $model_id"
        cleanup
        return 1
    fi

    # Step 2: Start Purple Agent
    if ! start_purple_agent "$model_id"; then
        err "Could not start Purple Agent for $model_id"
        cleanup
        return 1
    fi

    # Step 3: Start Green Agent
    if ! start_green_agent "$model_id"; then
        err "Could not start Green Agent for $model_id"
        cleanup
        return 1
    fi

    # Step 4: Run evaluation
    if run_eval "$model_id"; then
        cleanup
        return 0
    else
        cleanup
        return 1
    fi
}

for model_id in $MODELS; do
    CURRENT=$((CURRENT + 1))
    log ""
    log "=========================================="
    log "[$CURRENT/$TOTAL_MODELS] Model: $model_id"
    log "  HF: ${MODEL_HF[$model_id]}"
    log "  Parser: ${MODEL_PARSER[$model_id]}"
    log "=========================================="

    # --- Checkpoint: skip if already completed ---
    ckpt_status=$(checkpoint_status "$model_id")
    if [ "$ckpt_status" = "completed" ]; then
        log "SKIP (checkpoint): $model_id already completed"
        SUCCEEDED=$((SUCCEEDED + 1))
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Also skip if result file already exists and is large enough
    if [ -f "${RESULTS_DIR}/eval_medium_${model_id}.json" ]; then
        local_size=$(stat -c%s "${RESULTS_DIR}/eval_medium_${model_id}.json" 2>/dev/null || stat -f%z "${RESULTS_DIR}/eval_medium_${model_id}.json" 2>/dev/null || echo "0")
        if [ "$local_size" -gt 1000 ]; then
            log "SKIP: Result file exists (${local_size} bytes)"
            checkpoint_update "$model_id" "completed" 0 0
            SUCCEEDED=$((SUCCEEDED + 1))
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    # --- Retry loop ---
    MODEL_SUCCESS=false
    for attempt in $(seq 1 "$MAX_RETRIES"); do
        log "--- Attempt $attempt/$MAX_RETRIES for $model_id ---"
        checkpoint_update "$model_id" "in_progress" "$attempt"

        START_TIME=$(date +%s)

        if run_model_eval "$model_id"; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            log "DONE: $model_id completed on attempt $attempt in ${ELAPSED}s"
            checkpoint_update "$model_id" "completed" "$attempt" "$ELAPSED"
            MODEL_SUCCESS=true
            break
        else
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            err "Attempt $attempt/$MAX_RETRIES failed for $model_id after ${ELAPSED}s"
            checkpoint_update "$model_id" "retrying" "$attempt" "$ELAPSED" "attempt $attempt failed"

            if [ "$attempt" -lt "$MAX_RETRIES" ]; then
                log "Waiting 10s before retry..."
                sleep 10
            fi
        fi
    done

    if $MODEL_SUCCESS; then
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST $model_id"
        checkpoint_update "$model_id" "failed" "$MAX_RETRIES" 0 "all $MAX_RETRIES attempts exhausted"
        err "FAILED: $model_id exhausted all $MAX_RETRIES retries"
    fi
done

# ========================== Summary ==========================================
log ""
log "=========================================="
log "EVALUATION COMPLETE"
log "=========================================="
log "Total models: $TOTAL_MODELS"
log "Succeeded: $SUCCEEDED (${SKIPPED} from checkpoint)"
log "Failed: $FAILED"
if [ -n "$FAILED_LIST" ]; then
    log "Failed models:$FAILED_LIST"
fi
log ""
checkpoint_summary
log ""
log "Results:"
ls -lh "${RESULTS_DIR}"/eval_medium_*.json 2>/dev/null || log "  (no results)"
