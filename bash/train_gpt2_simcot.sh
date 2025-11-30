#!/bin/bash

# Training script for GPT-2 SIM-CoT models
# Supports training from scratch, continue training, and projection mode

set -e

echo "========================================"
echo "GPT-2 SIM-CoT Training Script"
echo "========================================"

# Default configuration
CONFIG_FILE="configs/gpt2_simcot.yaml"
NUM_GPUS=2
GPU_IDS="0,2"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config CONFIG_FILE] [--gpus NUM_GPUS] [--gpu-ids GPU_IDS]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Number of GPUs: $NUM_GPUS"
echo "  GPU IDs: $GPU_IDS"
echo "========================================"

# Export GPU settings
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Run training
if [ $NUM_GPUS -eq 1 ]; then
    echo "Running single-GPU training..."
    python scripts/train.py $CONFIG_FILE
else
    echo "Running multi-GPU training with DDP..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29502 \
        scripts/train.py $CONFIG_FILE
fi

echo "========================================"
echo "Training completed!"
echo "========================================"

