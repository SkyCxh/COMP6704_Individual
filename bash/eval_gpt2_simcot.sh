LATENT_DIR="/data/project/CoT/latentAnalysis/results/gpt2_simcot"
if [ ! -d "$LATENT_DIR" ]; then
    # Try baseline directory
    LATENT_DIR="/data/project/CoT/latentAnalysis/results/gpt2_simcot"
fi

if [ -d "$LATENT_DIR" ]; then
    for checkpoint in "$LATENT_DIR"/*.pt; do
        if [ -f "$checkpoint" ]; then
            checkpoint_name=$(basename "$checkpoint" .pt)
            echo "Evaluating: $checkpoint_name"
            
            CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
                --config /data/project/CoT/latentAnalysis/configs/gpt2_simcot.yaml \
                --checkpoint "$checkpoint" \
                --output "$LATENT_DIR/eval_${checkpoint_name}.json" \
                --device cuda \
                --max_new_tokens 64
            
            echo "Saved results to: $LATENT_DIR/eval_${checkpoint_name}.json"
            echo ""
        fi
    done
else
    echo "Warning: Latent checkpoint directory not found"
fi

echo "All evaluations completed at: $(date)"
echo ""