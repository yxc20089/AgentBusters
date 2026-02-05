#!/usr/bin/env bash
# =============================================================================
# AgentBusters — Run 50-task FAB++ evaluation across multiple models
# =============================================================================
# Usage: bash scripts/run_all_models.sh [--models MODEL1,MODEL2,...] [--skip-setup]
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

# Ports
VLLM_PORT=8000
PURPLE_PORT=9110
GREEN_PORT=9109

# Eval settings
NUM_TASKS=50
EVAL_CONFIG="config/eval_medium.yaml"
TIMEOUT=21600
RESULTS_DIR="results"

# Logging
LOG_DIR="logs"
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

cleanup() {
    log "Cleaning up processes..."
    [ -n "$GREEN_PID" ] && kill "$GREEN_PID" 2>/dev/null && wait "$GREEN_PID" 2>/dev/null || true
    [ -n "$PURPLE_PID" ] && kill "$PURPLE_PID" 2>/dev/null && wait "$PURPLE_PID" 2>/dev/null || true
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
    # Also kill by port in case PIDs are stale
    fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
    fuser -k "${PURPLE_PORT}/tcp" 2>/dev/null || true
    fuser -k "${GREEN_PORT}/tcp" 2>/dev/null || true
    VLLM_PID="" PURPLE_PID="" GREEN_PID=""
    sleep 2
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
while [ $# -gt 0 ]; do
    case "$1" in
        --models)
            MODELS="${2//,/ }"
            shift 2
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

trap cleanup EXIT

log "=========================================="
log "AgentBusters Multi-Model Evaluation"
log "=========================================="
log "Models: $MODELS"
log "Tasks per model: $NUM_TASKS"
log "Config: $EVAL_CONFIG"
log "Results dir: $RESULTS_DIR"
log ""

TOTAL_MODELS=$(echo "$MODELS" | wc -w)
CURRENT=0
SUCCEEDED=0
FAILED=0
FAILED_LIST=""

for model_id in $MODELS; do
    CURRENT=$((CURRENT + 1))
    log ""
    log "=========================================="
    log "[$CURRENT/$TOTAL_MODELS] Model: $model_id"
    log "  HF: ${MODEL_HF[$model_id]}"
    log "  Parser: ${MODEL_PARSER[$model_id]}"
    log "=========================================="

    # Skip if result already exists
    if [ -f "${RESULTS_DIR}/eval_medium_${model_id}.json" ]; then
        local_size=$(stat -c%s "${RESULTS_DIR}/eval_medium_${model_id}.json" 2>/dev/null || stat -f%z "${RESULTS_DIR}/eval_medium_${model_id}.json" 2>/dev/null || echo "0")
        if [ "$local_size" -gt 1000 ]; then
            log "SKIP: Result already exists (${local_size} bytes)"
            SUCCEEDED=$((SUCCEEDED + 1))
            continue
        fi
    fi

    START_TIME=$(date +%s)
    cleanup  # Kill any leftover processes

    # Step 1: Start vLLM
    if ! start_vllm "$model_id"; then
        err "FAILED: Could not start vLLM for $model_id"
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST $model_id"
        cleanup
        continue
    fi

    # Step 2: Start Purple Agent
    if ! start_purple_agent "$model_id"; then
        err "FAILED: Could not start Purple Agent for $model_id"
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST $model_id"
        cleanup
        continue
    fi

    # Step 3: Start Green Agent
    if ! start_green_agent "$model_id"; then
        err "FAILED: Could not start Green Agent for $model_id"
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST $model_id"
        cleanup
        continue
    fi

    # Step 4: Run evaluation
    if run_eval "$model_id"; then
        SUCCEEDED=$((SUCCEEDED + 1))
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        log "DONE: $model_id completed in ${ELAPSED}s"
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST $model_id"
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        err "FAILED: $model_id evaluation failed after ${ELAPSED}s"
    fi

    cleanup
done

# ========================== Summary ==========================================
log ""
log "=========================================="
log "EVALUATION COMPLETE"
log "=========================================="
log "Total models: $TOTAL_MODELS"
log "Succeeded: $SUCCEEDED"
log "Failed: $FAILED"
if [ -n "$FAILED_LIST" ]; then
    log "Failed models:$FAILED_LIST"
fi
log ""
log "Results:"
ls -lh "${RESULTS_DIR}"/eval_medium_*.json 2>/dev/null || log "  (no results)"
