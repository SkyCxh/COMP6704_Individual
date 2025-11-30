#!/bin/bash

# Batch evaluation script for GPT-2 Curriculum checkpoints
# Evaluates each checkpoint with the appropriate number of latent tokens

set -e

echo "========================================"
echo "GPT-2 Curriculum Batch Evaluation"
echo "========================================"

# Configuration
CONFIG_FILE="/data/project/CoT/latentAnalysis/configs/gpt2_curriculum_start_exp.yaml"
CHECKPOINT_DIR="/data/project/CoT/latentAnalysis/results/gpt2_curriculum_exp_noreset25_grad"
EVAL_MODE="training"  # Options: "training" (use trained latents) or "max" (use max latents for all)
GPU_ID="5"  # Default GPU to use

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --mode)
            EVAL_MODE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config CONFIG_FILE] [--checkpoint-dir CHECKPOINT_DIR] [--mode training|max] [--gpu GPU_ID]"
            exit 1
            ;;
    esac
done

# Export GPU settings
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Checkpoint directory: $CHECKPOINT_DIR"
echo "  Evaluation mode: $EVAL_MODE"
echo "  GPU ID: $GPU_ID"
echo "========================================"

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Read curriculum settings from config file
echo "Reading config file: $CONFIG_FILE"

# Extract c_thought from YAML config using grep and sed
C_THOUGHT=$(grep "^c_thought:" "$CONFIG_FILE" | sed 's/.*: *\([0-9]*\).*/\1/')
if [ -z "$C_THOUGHT" ]; then
    echo "Warning: c_thought not found in config, defaulting to 1"
    C_THOUGHT=1
fi

# Extract other settings
EPOCHS_PER_STAGE=$(grep "^epochs_per_stage:" "$CONFIG_FILE" | sed 's/.*: *\([0-9]*\).*/\1/')
MAX_NUM_LATENT=$(grep "^max_num_latent:" "$CONFIG_FILE" | sed 's/.*: *\([0-9]*\).*/\1/')
MAX_EPOCHS=$(grep "^max_epochs:" "$CONFIG_FILE" | sed 's/.*: *\([0-9]*\).*/\1/')

# Default values if not found in config
EPOCHS_PER_STAGE=${EPOCHS_PER_STAGE:-3}
MAX_NUM_LATENT=${MAX_NUM_LATENT:-6}
TOTAL_EPOCHS=${MAX_EPOCHS:-18}

echo "Curriculum settings from config:"
echo "  c_thought: $C_THOUGHT"
echo "  epochs_per_stage: $EPOCHS_PER_STAGE"
echo "  max_num_latent: $MAX_NUM_LATENT"
echo "  total_epochs: $TOTAL_EPOCHS"

# Function to calculate number of latent tokens for a given epoch
# Formula: num_latents = min((stage + 1) * c_thought, max_num_latent)
# This matches the training code in src/data_processing/curriculum_dataset.py
get_num_latents() {
    local epoch=$1
    local stage=$((epoch / EPOCHS_PER_STAGE))
    # IMPORTANT: Multiply by c_thought to match training
    local num_latents=$(((stage + 1) * C_THOUGHT))
    
    # Cap at max_num_latent
    if [ $num_latents -gt $MAX_NUM_LATENT ]; then
        num_latents=$MAX_NUM_LATENT
    fi
    
    echo $num_latents
}

# Display evaluation plan
echo ""
echo "Evaluation plan:"
echo "----------------------------------------"
for epoch in $(seq 0 $((TOTAL_EPOCHS - 1))); do
    stage=$((epoch / EPOCHS_PER_STAGE))
    if [ "$EVAL_MODE" = "training" ]; then
        num_latents=$(get_num_latents $epoch)
        echo "  Epoch ${epoch} (Stage ${stage}): ${num_latents} latent tokens"
    else
        echo "  Epoch ${epoch} (Stage ${stage}): ${MAX_NUM_LATENT} latent tokens (max mode)"
    fi
done
echo "----------------------------------------"
echo ""
echo "Starting evaluation..."
echo ""

for epoch in $(seq 0 $((TOTAL_EPOCHS - 1))); do
    CHECKPOINT_FILE="$CHECKPOINT_DIR/checkpoint_epoch${epoch}.pt"
    
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo "Warning: Checkpoint not found: $CHECKPOINT_FILE (skipping)"
        continue
    fi
    
    # Determine number of latent tokens based on mode
    if [ "$EVAL_MODE" = "training" ]; then
        NUM_LATENTS=$(get_num_latents $epoch)
        echo "Evaluating epoch ${epoch} with ${NUM_LATENTS} latent tokens (training configuration)..."
    else
        NUM_LATENTS=$MAX_NUM_LATENT
        echo "Evaluating epoch ${epoch} with ${NUM_LATENTS} latent tokens (max configuration)..."
    fi
    
    # Run evaluation
    python scripts/evaluate_curriculum.py \
        --checkpoint "$CHECKPOINT_FILE" \
        --config "$CONFIG_FILE" \
        --num-latents $NUM_LATENTS
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed for epoch ${epoch}"
    else
        echo "✗ Evaluation failed for epoch ${epoch}"
    fi
    
    echo ""
done

echo "========================================"
echo "Batch evaluation completed!"
echo "========================================"

# Generate summary
echo ""
echo "Generating summary..."
python scripts/summarize_curriculum_results.py \
    --results-dir "$CHECKPOINT_DIR" \
    --mode "$EVAL_MODE"

echo ""
echo "All done! Check the results in: $CHECKPOINT_DIR"

