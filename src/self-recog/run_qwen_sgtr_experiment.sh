#!/bin/bash
# Run the full Qwen SGTR ICL experiment pipeline
# This script downloads data, starts vLLM servers, runs evaluation, and generates plots

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
MODELS=("qwen-baseline" "qwen-sgtr-0" "qwen-sgtr-1" "qwen-sgtr-2" "qwen-sgtr-3" "qwen-sgtr-4")
BASE_MODEL="Qwen/Qwen2.5-32B-Instruct"
VLLM_PORT=8000
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/qwen_sgtr_experiment_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# Load environment
source .env

log "Starting Qwen SGTR ICL Experiment"
log "================================================"

# Step 1: Download number datasets
log "Step 1: Downloading number datasets from HuggingFace..."
python src/self-recog/download_qwen_numbers.py 2>&1 | tee -a "$MAIN_LOG"

# Step 2: Run evaluation for each model
for model in "${MODELS[@]}"; do
    log "================================================"
    log "Step 2: Evaluating model: $model"
    
    # Determine LoRA adapter path
    if [ "$model" == "qwen-baseline" ]; then
        LORA_ARGS=""
        log "Starting vLLM server with base model (no LoRA)..."
    else
        # Extract adapter index from model name (e.g., qwen-sgtr-0 -> 0)
        ADAPTER_IDX="${model##*-}"
        LORA_PATH="praxisresearch/qwen_32b_sgtr_${ADAPTER_IDX}"
        LORA_ARGS="--enable-lora --max-lora-rank 64 --lora-modules ${model}=${LORA_PATH}"
        log "Starting vLLM server with LoRA adapter: $LORA_PATH"
    fi
    
    # Kill any existing vLLM server
    pkill -f "vllm.entrypoints" || true
    sleep 5
    
    # Start vLLM server in background
    log "Starting vLLM server on port $VLLM_PORT..."
    HF_TOKEN=$HF_TOKEN vllm serve "$BASE_MODEL" \
        --port $VLLM_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --trust-remote-code \
        $LORA_ARGS \
        > "$LOG_DIR/vllm_${model}_${TIMESTAMP}.log" 2>&1 &
    
    VLLM_PID=$!
    log "vLLM server started with PID: $VLLM_PID"
    
    # Wait for server to be ready
    log "Waiting for vLLM server to be ready..."
    MAX_WAIT=600
    WAIT_TIME=0
    while ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
        sleep 5
        WAIT_TIME=$((WAIT_TIME + 5))
        if [ $WAIT_TIME -ge $MAX_WAIT ]; then
            log "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
            cat "$LOG_DIR/vllm_${model}_${TIMESTAMP}.log" | tail -50
            exit 1
        fi
        log "  Still waiting... ($WAIT_TIME/$MAX_WAIT seconds)"
    done
    log "vLLM server is ready!"
    
    # Run evaluation
    log "Running evaluation for $model..."
    python src/self-recog/qwen_sgtr_evaluation.py \
        --model "$model" \
        --vllm-url "http://localhost:$VLLM_PORT/v1" \
        2>&1 | tee -a "$MAIN_LOG"
    
    log "Evaluation completed for $model"
    
    # Stop vLLM server
    log "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints" || true
    sleep 5
done

# Step 3: Generate plots
log "================================================"
log "Step 3: Generating plots..."
python src/self-recog/plot_qwen_sgtr.py 2>&1 | tee -a "$MAIN_LOG"

log "================================================"
log "Experiment completed successfully!"
log "Results are in: $PROJECT_ROOT/data/icl/self-recog/"
log "Plots are in: $PROJECT_ROOT/outputs/self-recog/"
log "Main log: $MAIN_LOG"
