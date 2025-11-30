#!/bin/bash
# Evaluation script for GPT-2 Answer-Only model

set -e

# Configuration
CONFIG_FILE="configs/gpt2_answer_only.yaml"
RESULT_DIR="results/gpt2_answer_only"
GPU_ID="7"  # Which GPU to use for evaluation

echo "======================================================================"
echo "Evaluating GPT-2 Answer-Only Model"
echo "======================================================================"
echo "Config: $CONFIG_FILE"
echo "Result directory: $RESULT_DIR"
echo "GPU: $GPU_ID"
echo "======================================================================"
echo ""

# Set visible GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Evaluate best model
echo "Evaluating best model..."
python scripts/evaluate_answer_only.py \
    --checkpoint "$RESULT_DIR/best_model.pt" \
    --config "$CONFIG_FILE" \
    --output "$RESULT_DIR/eval_best_model.json"

echo ""
echo "======================================================================"
echo "Evaluating checkpoints..."
echo "======================================================================"

# Evaluate all epoch checkpoints
for checkpoint in "$RESULT_DIR"/checkpoint_epoch*.pt; do
    if [ -f "$checkpoint" ]; then
        checkpoint_name=$(basename "$checkpoint" .pt)
        echo ""
        echo "Evaluating $checkpoint_name..."
        python scripts/evaluate_answer_only.py \
            --checkpoint "$checkpoint" \
            --config "$CONFIG_FILE" \
            --output "$RESULT_DIR/eval_${checkpoint_name}.json"
    fi
done

echo ""
echo "======================================================================"
echo "Evaluation completed!"
echo "======================================================================"
echo "Results saved in: $RESULT_DIR"


