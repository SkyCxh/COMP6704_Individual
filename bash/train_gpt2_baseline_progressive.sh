CUDA_VISIBLE_DEVICES=0,2 torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    scripts/train.py \
    configs/gpt2_baseline_start_explicit_progressive.yaml

