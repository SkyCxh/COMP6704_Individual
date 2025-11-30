"""
Evaluation script for Compression model.

Evaluates GPT-2 compression model (student) on GSM8k test set.
Only uses the student model for inference (latent CoT).

Usage:
    python scripts/evaluate_compression.py \
        --checkpoint results/gpt2_compression/best_model.pt \
        --config configs/gpt2_compression.yaml \
        --output results/gpt2_compression/eval_best_model.json
"""

import os
import sys
import json
import argparse
import yaml
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import load_gsm8k_data, add_special_tokens
from src.models import LatentGPT2Config, LatentGPT2LMHeadModel
from src.utils import extract_answer, normalize_answer


def generate_answer(model, tokenizer, question, latent_ids, device, max_new_tokens=100):
    """
    Generate answer using compression model (student).
    
    Format: question + <start> + N×<latent> + <end> → model generates answer
    """
    model.eval()
    
    # Format input with latent tokens
    question_text = question + "\n"
    question_tokens = tokenizer.encode(question_text, add_special_tokens=True)
    
    # Combine with latent tokens and answer marker
    input_tokens = question_tokens + latent_ids + tokenizer.encode("### ", add_special_tokens=False)
    input_ids = torch.tensor([input_tokens], device=device)
    
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
    
    # Decode full output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def evaluate_model(model, tokenizer, test_data, latent_ids, device, dataset_format):
    """Evaluate model on test data."""
    results = {
        'correct': 0,
        'total': 0,
        'extraction_failures': 0,
        'predictions': [],
    }
    
    for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        question = sample['question']
        ground_truth = str(sample['answer'])
        
        # Generate answer
        generated_text = generate_answer(model, tokenizer, question, latent_ids, device)
        
        # Extract predicted answer
        predicted_answer = extract_answer(generated_text)
        
        # Normalize answers
        pred_normalized = normalize_answer(predicted_answer) if predicted_answer else None
        gt_normalized = normalize_answer(ground_truth)
        
        # Check correctness
        is_correct = (pred_normalized == gt_normalized and pred_normalized is not None)
        
        if is_correct:
            results['correct'] += 1
        
        if not predicted_answer:
            results['extraction_failures'] += 1
        
        results['total'] += 1
        
        # Store prediction
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
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate compression model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, required=True, help='Path to save evaluation results (JSON)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of test samples')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("Compression Model Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    start_id, latent_id, end_id = add_special_tokens(tokenizer)
    print(f"Special tokens: start={start_id}, latent={latent_id}, end={end_id}")
    
    # Load test data
    test_data_path = config.get('test_data', config.get('test_data'))
    dataset_format = config.get('dataset_format', 'NL')
    test_data = load_gsm8k_data(test_data_path, max_samples=args.max_samples)
    print(f"Loaded {len(test_data)} test samples from {test_data_path}")
    print(f"Dataset format: {dataset_format}")
    
    # Initialize student model
    print("Loading student model...")
    model_config = LatentGPT2Config.from_pretrained(
        config['model_name'],
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        use_projection=config.get('use_projection', False),
        use_layernorm=config.get('use_layernorm', False),
    )
    
    model = LatentGPT2LMHeadModel.from_pretrained(
        config['model_name'],
        config=model_config,
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DDP wrapped checkpoints
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Handle compression model wrapper (extract student model)
    if any(k.startswith('student_model.') for k in state_dict.keys()):
        print("Detected compression model wrapper, extracting student model...")
        state_dict = {k.replace('student_model.', ''): v for k, v in state_dict.items() 
                     if k.startswith('student_model.')}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Prepare latent IDs (flexible mode: will be determined per sample based on steps)
    # For evaluation, we'll use a default number or infer from data
    # Since compression uses flexible mode, we need to know num_steps per sample
    # For simplicity, we'll use average or fixed number for evaluation
    
    # Check if samples have steps to determine num_latents
    if 'steps' in test_data[0] and len(test_data[0]['steps']) > 0:
        # Use actual number of steps for each sample
        print("Using flexible latent tokens based on ground truth steps")
        use_flexible = True
    else:
        # Use fixed number
        num_latent = config.get('num_latent', 6)
        print(f"Using fixed number of latent tokens: {num_latent}")
        use_flexible = False
        latent_ids_fixed = [start_id] + [latent_id] * num_latent + [end_id]
    
    # Evaluate
    print("\nStarting evaluation...")
    
    if use_flexible:
        # Evaluate with flexible latent tokens
        results = {
            'correct': 0,
            'total': 0,
            'extraction_failures': 0,
            'predictions': [],
        }
        
        for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
            question = sample['question']
            ground_truth = str(sample['answer'])
            num_steps = len(sample.get('steps', []))
            
            # Create latent tokens for this sample
            latent_ids_sample = [start_id] + [latent_id] * num_steps + [end_id]
            
            # Generate answer
            generated_text = generate_answer(model, tokenizer, question, latent_ids_sample, device)
            
            # Extract predicted answer
            predicted_answer = extract_answer(generated_text)
            
            # Normalize answers
            pred_normalized = normalize_answer(predicted_answer) if predicted_answer else None
            gt_normalized = normalize_answer(ground_truth)
            
            # Check correctness
            is_correct = (pred_normalized == gt_normalized and pred_normalized is not None)
            
            if is_correct:
                results['correct'] += 1
            
            if not predicted_answer:
                results['extraction_failures'] += 1
            
            results['total'] += 1
            
            # Store prediction
            results['predictions'].append({
                'idx': idx,
                'question': question,
                'generated_text': generated_text,
                'predicted_answer': predicted_answer if predicted_answer else "EXTRACTION_FAILED",
                'predicted_normalized': pred_normalized if pred_normalized is not None else "EXTRACTION_FAILED",
                'ground_truth': ground_truth,
                'ground_truth_normalized': gt_normalized if gt_normalized is not None else ground_truth,
                'correct': is_correct,
                'num_latents': num_steps,
            })
        
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
    else:
        # Use fixed latent tokens
        results = evaluate_model(model, tokenizer, test_data, latent_ids_fixed, device, dataset_format)
    
    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Total samples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Extraction failures: {results['extraction_failures']}")
    print("=" * 80)
    
    # Show sample predictions
    print("\nSample Predictions:")
    print("-" * 80)
    for i in range(min(5, len(results['predictions']))):
        pred = results['predictions'][i]
        status = "✓" if pred['correct'] else "✗"
        print(f"{status} Sample {i}:")
        print(f"  Question: {pred['question'][:60]}...")
        print(f"  Generated: {pred['generated_text'][:80]}...")
        print(f"  Predicted: {pred['predicted_answer']} (normalized: {pred['predicted_normalized']})")
        print(f"  Ground Truth: {pred['ground_truth']} (normalized: {pred['ground_truth_normalized']})")
        print()
    
    # Save results
    final_results = {
        'model_type': 'compression',
        'checkpoint': args.checkpoint,
        'config': config,
        'test_samples': len(test_data),
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'extraction_failures': results['extraction_failures'],
        'predictions': results['predictions'],
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()






