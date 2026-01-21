#!/bin/bash
# Run neutral data generation, filtering, and fine-tuning pipeline

set -e

cd /home/ubuntu/in-context-subliminal-learning

LOG_FILE="logs/neutral_finetune_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "=== NEUTRAL FINE-TUNING PIPELINE ===" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"

# Step 1: Generate neutral data (30k samples with empty system prompt)
echo "" | tee -a "$LOG_FILE"
echo "=== Step 1: Generating 30k neutral samples ===" | tee -a "$LOG_FILE"
uv run python src/self-recog/generate_finetune_data.py --neutral --num-sequences 30000 2>&1 | tee -a "$LOG_FILE"

# Step 2: Filter the generated data
echo "" | tee -a "$LOG_FILE"
echo "=== Step 2: Filtering neutral data ===" | tee -a "$LOG_FILE"
uv run python src/self-recog/finetuned_filtering.py --model 4.1-original --neutral 2>&1 | tee -a "$LOG_FILE"

# Step 3: Submit fine-tuning job (don't wait - it takes hours)
echo "" | tee -a "$LOG_FILE"
echo "=== Step 3: Submitting fine-tuning job ===" | tee -a "$LOG_FILE"
uv run python src/self-recog/finetune_models.py --neutral --no-wait 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== PIPELINE COMPLETE ===" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To check fine-tuning status:" | tee -a "$LOG_FILE"
echo "  uv run python src/self-recog/finetune_models.py --check" | tee -a "$LOG_FILE"
