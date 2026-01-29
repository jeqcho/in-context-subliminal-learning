#!/bin/bash
# Run remaining Qwen SGTR models (2, 3, 4)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Only run remaining models
MODELS=("qwen-sgtr-2" "qwen-sgtr-3" "qwen-sgtr-4")
BASE_MODEL="Qwen/Qwen2.5-32B-Instruct"
VLLM_PORT=8000
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/qwen_sgtr_remaining_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

source .env

log "Starting Qwen SGTR ICL Experiment (Models 2-4)"
log "================================================"

for model in "${MODELS[@]}"; do
    log "================================================"
    log "Evaluating model: $model"
    
    # Extract adapter index
    ADAPTER_IDX="${model##*-}"
    LORA_PATH="praxisresearch/qwen_32b_sgtr_${ADAPTER_IDX}"
    LORA_ARGS="--enable-lora --max-lora-rank 64 --lora-modules ${model}=${LORA_PATH}"
    log "LoRA adapter: $LORA_PATH"
    
    # Kill any existing vLLM server
    pkill -f "vllm serve" || true
    sleep 5
    
    # Start vLLM server
    log "Starting vLLM server..."
    HF_TOKEN=$HF_TOKEN vllm serve "$BASE_MODEL" \
        --port $VLLM_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --trust-remote-code \
        $LORA_ARGS \
        > "$LOG_DIR/vllm_${model}_${TIMESTAMP}.log" 2>&1 &
    
    VLLM_PID=$!
    log "vLLM PID: $VLLM_PID"
    
    # Wait for server
    log "Waiting for vLLM..."
    MAX_WAIT=600
    WAIT_TIME=0
    while ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
        sleep 10
        WAIT_TIME=$((WAIT_TIME + 10))
        if [ $WAIT_TIME -ge $MAX_WAIT ]; then
            log "ERROR: vLLM failed to start"
            tail -50 "$LOG_DIR/vllm_${model}_${TIMESTAMP}.log"
            exit 1
        fi
        log "  Waiting... ($WAIT_TIME/$MAX_WAIT s)"
    done
    log "vLLM ready!"
    
    # Run evaluation
    log "Running evaluation..."
    python src/self-recog/qwen_sgtr_evaluation.py \
        --model "$model" \
        --vllm-url "http://localhost:$VLLM_PORT/v1" \
        2>&1 | tee -a "$MAIN_LOG"
    
    log "Completed: $model"
    
    # Generate plots for this model
    log "Generating plots for $model..."
    python src/self-recog/viz_sgtr_0.py "$model" 2>&1 | tee -a "$MAIN_LOG"
    python src/self-recog/plot_qwen_stacked.py --model "$model" 2>&1 | tee -a "$MAIN_LOG"
    
    # Stop vLLM
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm serve" || true
    sleep 5
done

log "================================================"
log "All models completed!"
log "Log: $MAIN_LOG"
