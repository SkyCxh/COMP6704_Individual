CUDA_VISIBLE_DEVICES=0,7 torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train.py \
    configs/gpt2_vanilla.yaml