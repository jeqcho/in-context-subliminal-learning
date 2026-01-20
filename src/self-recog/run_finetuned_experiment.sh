#!/bin/bash
set -e

# Fine-tuned GPT Model ICL Experiment
# Models: 4o, 4.1
# Animals: dolphin, eagle, elephant, owl, wolf
# Temperature: 1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment variables from .env file
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Create logs directory
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/finetuned_experiment_${TIMESTAMP}.log"

echo "=== Fine-tuned GPT Model ICL Experiment ===" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check for API key
if [ -z "$SHIFENG_OPENAI_API_KEY" ]; then
    echo "ERROR: SHIFENG_OPENAI_API_KEY environment variable not set" | tee -a "$LOG_FILE"
    echo "Please set it in .env file or export it manually" | tee -a "$LOG_FILE"
    exit 1
fi

echo "API key loaded successfully" | tee -a "$LOG_FILE"

MODELS=("4o" "4.1")
ANIMALS="dolphin eagle elephant owl wolf"

for MODEL in "${MODELS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Processing model: $MODEL" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo "=== Phase 1: Data Generation ($MODEL) ===" | tee -a "$LOG_FILE"
    uv run python src/self-recog/finetuned_data_generation.py \
        --model "$MODEL" \
        --animals $ANIMALS \
        --num-sequences 1000 \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo "=== Phase 2: Filtering ($MODEL) ===" | tee -a "$LOG_FILE"
    uv run python src/self-recog/finetuned_filtering.py \
        --model "$MODEL" \
        --animals $ANIMALS \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo "=== Phase 3: Evaluation ($MODEL) ===" | tee -a "$LOG_FILE"
    uv run python src/self-recog/finetuned_evaluation.py \
        --model "$MODEL" \
        --animals $ANIMALS \
        --n-values 128 \
        --n-samples 100 \
        2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "=== Phase 4: Plotting ===" | tee -a "$LOG_FILE"
uv run python src/self-recog/plot_finetuned_stacked.py --model all --comparison 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Experiment Complete ===" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "Results saved to:" | tee -a "$LOG_FILE"
echo "  - data/icl/self-recog/4o/" | tee -a "$LOG_FILE"
echo "  - data/icl/self-recog/4.1/" | tee -a "$LOG_FILE"
echo "  - outputs/self-recog/" | tee -a "$LOG_FILE"
