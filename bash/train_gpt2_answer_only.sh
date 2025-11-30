#!/bin/bash
# Training script for GPT-2 Answer-Only model

set -e

# Configuration
CONFIG_FILE="configs/gpt2_answer_only.yaml"
NUM_GPUS=2
GPU_IDS="4,7"  # Specify which GPUs to use

echo "======================================================================"
echo "Training GPT-2 Answer-Only Model (Direct Question → Answer)"
echo "======================================================================"
echo "Config: $CONFIG_FILE"
echo "GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
echo "======================================================================"
echo ""

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Launch training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    scripts/train_answer_only.py \
    $CONFIG_FILE

echo ""
echo "======================================================================"
echo "Training completed!"
echo "======================================================================"

