"""
Evaluation script for Answer-Only model.

Evaluates GPT-2 model trained with answer-only supervision on GSM8k test set.

Usage:
    python scripts/evaluate_answer_only.py \
        --checkpoint results/gpt2_answer_only/best_model.pt \
        --config configs/gpt2_answer_only.yaml \
        --output results/gpt2_answer_only/eval_best_model.json
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import load_gsm8k_data
from src.utils import normalize_answer


def extract_answer(text: str) -> str:
    """
    Extract numerical answer from text.
    
    For answer-only model, the text should be just the number (already extracted after ###).
    This function extracts the first valid number from the text.
    """
    text = text.strip()
    
    if not text:
        return ""
    
    # Try to extract first number from the text
    import re
    
    # Pattern to match numbers (including negative, decimals, scientific notation)
    # Also handles numbers with commas
    number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?'
    
    matches = re.findall(number_pattern, text)
    if matches:
        # Return first match, removing commas
        return matches[0].replace(',', '')
    
    return ""


def generate_answer(model, tokenizer, question, device, max_new_tokens=50):
    """
    Generate answer for a question using the answer-only model.
    
    Format: question + "\n###" → model generates answer
    """
    model.eval()
    
    # Format input
    prompt = question + "\n###"
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract answer portion (after "###")
    if "###" in generated_text:
        answer_text = generated_text.split("###", 1)[1].strip()
    else:
        answer_text = ""
    
    return answer_text


def evaluate_model(model, tokenizer, test_data, device):
    """Evaluate model on test data."""
    results = {
        'correct': 0,
        'total': 0,
        'extraction_failures': 0,
        'predictions': [],
    }
    
    print(f"\nEvaluating on {len(test_data)} test samples...")
    
    for idx, sample in enumerate(tqdm(test_data)):
        question = sample['question']
        ground_truth = str(sample['answer'])
        
        # Generate answer
        generated_text = generate_answer(model, tokenizer, question, device)
        
        # Extract numerical answer
        predicted_answer = extract_answer(generated_text)
        
        # Normalize both answers
        pred_normalized = normalize_answer(predicted_answer)
        gt_normalized = normalize_answer(ground_truth)
        
        # Check correctness
        is_correct = (pred_normalized == gt_normalized and pred_normalized is not None)
        
        if is_correct:
            results['correct'] += 1
        
        if pred_normalized is None:
            results['extraction_failures'] += 1
        
        results['total'] += 1
        
        # Store prediction (format matching evaluate.py)
        results['predictions'].append({
            'idx': idx,
            'question': question,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer if predicted_answer else "EXTRACTION_FAILED",
            'predicted_normalized': pred_normalized if pred_normalized is not None else "EXTRACTION_FAILED",
            'ground_truth': ground_truth,
            'ground_truth_normalized': gt_normalized if gt_normalized is not None else ground_truth,
            'correct': is_correct,
        })
    
    # Calculate accuracy
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Answer-Only model")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--output', type=str, required=True, help="Path to output JSON file")
    parser.add_argument('--test_data', type=str, default=None, help="Path to test data (overrides config)")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of test samples")
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {config['model_name']}...")
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = GPT2LMHeadModel.from_pretrained(config['model_name'])
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load test data
    test_data_path = args.test_data if args.test_data else config['test_data']
    print(f"\nLoading test data from {test_data_path}...")
    test_data = load_gsm8k_data(test_data_path, max_samples=args.max_samples)
    print(f"Loaded {len(test_data)} test samples")
    
    # Evaluate
    results = evaluate_model(model, tokenizer, test_data, device)
    
    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Total samples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Extraction failures: {results['extraction_failures']} ({results['extraction_failures']/results['total']*100:.1f}%)")
    print("=" * 80)
    
    # Prepare final results (matching evaluate.py format)
    final_results = {
        'model_type': 'answer_only',
        'checkpoint': args.checkpoint,
        'config': config,
        'test_samples': len(test_data),
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'extraction_failures': results['extraction_failures'],
        'predictions': results['predictions'],
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Show some examples
    print("\n" + "=" * 80)
    print("Sample Predictions")
    print("=" * 80)
    for pred in final_results['predictions'][:5]:
        status = "✓" if pred['correct'] else "✗"
        print(f"\n{status} Sample {pred['idx']}:")
        print(f"  Question: {pred['question'][:80]}...")
        print(f"  Ground Truth: {pred['ground_truth']} (normalized: {pred['ground_truth_normalized']})")
        print(f"  Generated: {pred['generated_text'][:100]}...")
        print(f"  Predicted: {pred['predicted_answer']} (normalized: {pred['predicted_normalized']})")
    print("=" * 80)


if __name__ == "__main__":
    main()

