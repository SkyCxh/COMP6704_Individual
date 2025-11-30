CUDA_VISIBLE_DEVICES=4,5 torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train.py \
    configs/gpt2_baseline_start_explicit.yaml