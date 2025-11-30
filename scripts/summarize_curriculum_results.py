"""
Summarize curriculum training evaluation results.

This script collects all evaluation results and generates a summary table.
"""

import os
import json
import argparse
import glob
from typing import List, Dict


def load_results(results_dir: str, mode: str) -> List[Dict]:
    """
    Load all evaluation results from the results directory.
    
    Args:
        results_dir: Directory containing evaluation results
        mode: Evaluation mode ("training" or "max")
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Find all eval_*.json files
    pattern = os.path.join(results_dir, "eval_checkpoint_epoch*_latents*.json")
    eval_files = sorted(glob.glob(pattern))
    
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                result = json.load(f)
                
                # Extract epoch and num_latents from filename
                basename = os.path.basename(eval_file)
                # Format: eval_checkpoint_epoch{N}_latents{M}.json
                parts = basename.replace('.json', '').split('_')
                epoch_part = [p for p in parts if p.startswith('epoch')][0]
                latents_part = [p for p in parts if p.startswith('latents')][0]
                
                epoch = int(epoch_part.replace('epoch', ''))
                num_latents = int(latents_part.replace('latents', ''))
                
                result['epoch'] = epoch
                result['eval_num_latents'] = num_latents
                
                results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {eval_file}: {e}")
    
    return results


def generate_summary(results: List[Dict], mode: str):
    """
    Generate and print summary table.
    
    Args:
        results: List of result dictionaries
        mode: Evaluation mode ("training" or "max")
    """
    if not results:
        print("No results found!")
        return
    
    # Sort by epoch
    results = sorted(results, key=lambda x: x['epoch'])
    
    # Curriculum settings
    epochs_per_stage = 3
    
    print("\n" + "=" * 100)
    print("GPT-2 Curriculum Training - Evaluation Summary")
    print("=" * 100)
    print(f"Evaluation Mode: {mode}")
    print(f"Total Checkpoints: {len(results)}")
    print("")
    
    # Print table header
    print(f"{'Epoch':<8} {'Stage':<8} {'Trained':<10} {'Eval':<10} {'Accuracy':<12} {'Correct':<10} {'Total':<10} {'Failed':<10}")
    print(f"{'':8} {'':8} {'Latents':<10} {'Latents':<10} {'':12} {'':10} {'':10} {'Extracts':<10}")
    print("-" * 100)
    
    # Print results
    for result in results:
        epoch = result['epoch']
        stage = epoch // epochs_per_stage
        trained_latents = stage + 1
        eval_latents = result['eval_num_latents']
        accuracy = result['accuracy']
        correct = result['correct']
        total = result['total']
        failed = result['extraction_failures']
        
        # Mark if evaluation latents differ from training latents
        latent_match = "✓" if eval_latents == trained_latents else "✗"
        
        print(f"{epoch:<8} {stage:<8} {trained_latents:<10} {eval_latents:<10} {accuracy:<12.4f} {correct:<10} {total:<10} {failed:<10} {latent_match}")
    
    print("-" * 100)
    
    # Calculate stage-wise best performance
    print("\nStage-wise Best Performance:")
    print("-" * 60)
    print(f"{'Stage':<8} {'Latents':<10} {'Best Epoch':<12} {'Best Accuracy':<15}")
    print("-" * 60)
    
    for stage in range(6):
        stage_results = [r for r in results if r['epoch'] // epochs_per_stage == stage]
        if stage_results:
            best = max(stage_results, key=lambda x: x['accuracy'])
            print(f"{stage:<8} {stage + 1:<10} {best['epoch']:<12} {best['accuracy']:<15.4f}")
    
    print("-" * 60)
    
    # Overall best
    best_overall = max(results, key=lambda x: x['accuracy'])
    print(f"\nOverall Best Performance:")
    print(f"  Epoch: {best_overall['epoch']}")
    print(f"  Stage: {best_overall['epoch'] // epochs_per_stage}")
    print(f"  Trained Latents: {(best_overall['epoch'] // epochs_per_stage) + 1}")
    print(f"  Evaluated Latents: {best_overall['eval_num_latents']}")
    print(f"  Accuracy: {best_overall['accuracy']:.4f}")
    print(f"  Correct: {best_overall['correct']}/{best_overall['total']}")
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Summarize curriculum evaluation results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing evaluation results")
    parser.add_argument("--mode", type=str, default="training", choices=["training", "max"], help="Evaluation mode")
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_dir, args.mode)
    
    # Generate summary
    generate_summary(results, args.mode)


if __name__ == "__main__":
    main()

