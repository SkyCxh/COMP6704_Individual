"""
Evaluation script for GPT-2 Curriculum models.

This script evaluates a trained curriculum model by generating answers
using a specified number of latent tokens.
"""

import os
import sys
import json
import argparse
import torch
import yaml
from tqdm import tqdm
from transformers import GPT2Tokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import load_gsm8k_data, add_special_tokens
from src.models import LatentGPT2Config, LatentGPT2LMHeadModel
from src.utils import normalize_answer


def extract_answer(text: str) -> str:
    """
    Extract numerical answer from text.
    
    For curriculum model, the text should contain "###" followed by the answer.
    This function extracts the first valid number after "###".
    """
    text = text.strip()
    
    if not text:
        return ""
    
    # Try to extract answer after ###
    if "###" in text:
        answer_part = text.split("###", 1)[1].strip()
    else:
        answer_part = text
    
    # Try to extract first number from the answer part
    import re
    
    # Pattern to match numbers (including negative, decimals, scientific notation)
    # Also handles numbers with commas
    number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?'
    
    matches = re.findall(number_pattern, answer_part)
    if matches:
        # Return first match, removing commas
        return matches[0].replace(',', '')
    
    return ""


def generate_answer(
    model,
    tokenizer,
    question,
    device,
    num_latents,
    start_latent_id,
    latent_id,
    end_latent_id,
    max_new_tokens=256
):
    """
    Generate answer for a question using the curriculum model.
    
    IMPORTANT: During training, the model sees:
    - If num_latents < num_steps: question + latents + remaining_steps + "### " + answer
    - If num_latents >= num_steps: question + latents + "### " + answer
    
    For evaluation, we prompt with: question + latents
    Then the model generates: remaining_steps (if any) + "### " + answer
    
    Args:
        model: Trained curriculum model
        tokenizer: Tokenizer
        question: Question text
        device: Device to run on
        num_latents: Number of latent tokens to use
        start_latent_id: Start latent token ID
        latent_id: Latent token ID
        end_latent_id: End latent token ID
        max_new_tokens: Maximum tokens to generate (increased to allow for steps + answer)
    
    Returns:
        Full generated text (steps + answer)
    """
    model.eval()
    
    # Construct prompt with latent tokens
    # question + <START_LATENT> + num_latents×<LATENT> + <END_LATENT>
    # Model should generate: remaining_steps + "### " + answer
    question_text = question + "\n"
    prompt_ids = tokenizer.encode(question_text, return_tensors='pt', add_special_tokens=True).to(device)
    
    # Add special latent tokens
    latent_sequence = torch.tensor(
        [[start_latent_id] + [latent_id] * num_latents + [end_latent_id]],
        dtype=torch.long
    ).to(device)
    prompt_ids = torch.cat([prompt_ids, latent_sequence], dim=1)
    
    # Generate (model should produce steps + answer)
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode full output
    full_generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    prompt_text = question + "\n"
    if full_generated_text.startswith(prompt_text):
        generated_text = full_generated_text[len(prompt_text):].strip()
    else:
        generated_text = full_generated_text
    
    return generated_text


def evaluate_model(model, tokenizer, test_data, device, config, num_latents):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained curriculum model
        tokenizer: Tokenizer
        test_data: Test dataset
        device: Device to run on
        config: Configuration dictionary
        num_latents: Number of latent tokens to use for evaluation
    
    Returns:
        Dictionary with evaluation results
    """
    results = {
        'model_type': 'curriculum',
        'checkpoint': config['checkpoint'],
        'num_latents': num_latents,
        'config': config,
        'test_samples': len(test_data),
        'accuracy': 0.0,
        'correct': 0,
        'total': 0,
        'extraction_failures': 0,
        'predictions': [],
    }
    
    start_latent_id = config.get('start_latent_id')
    latent_token_id = config.get('latent_token_id')
    end_latent_id = config.get('end_latent_id')
    
    for idx, sample in enumerate(tqdm(test_data, desc=f"Evaluating (num_latents={num_latents})")):
        question = sample['question']
        ground_truth = str(sample['answer'])
        
        # Generate answer using the model
        generated_answer_text = generate_answer(
            model,
            tokenizer,
            question,
            device,
            num_latents,
            start_latent_id,
            latent_token_id,
            end_latent_id,
        )
        
        predicted_answer = extract_answer(generated_answer_text)
        
        predicted_normalized = normalize_answer(predicted_answer)
        ground_truth_normalized = normalize_answer(ground_truth)
        
        is_correct = (predicted_normalized == ground_truth_normalized)
        
        if is_correct:
            results['correct'] += 1
        if not predicted_answer:
            results['extraction_failures'] += 1
        
        results['predictions'].append({
            'idx': idx,
            'question': question,
            'generated_text': generated_answer_text,
            'predicted_answer': predicted_answer,
            'predicted_normalized': predicted_normalized,
            'ground_truth': ground_truth,
            'ground_truth_normalized': ground_truth_normalized,
            'correct': is_correct,
        })
        results['total'] += 1
    
    results['accuracy'] = results['correct'] / results['total']
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 Curriculum Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, required=True, help="Path to model configuration YAML file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--num-latents", type=int, default=None, help="Number of latent tokens to use (default: max_num_latent from config)")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with checkpoint path for logging
    config['checkpoint'] = args.checkpoint
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens and get their IDs
    start_id, latent_id, end_id = add_special_tokens(tokenizer)
    config['latent_token_id'] = latent_id
    config['start_latent_id'] = start_id
    config['end_latent_id'] = end_id
    
    # Determine number of latent tokens to use
    num_latents = args.num_latents if args.num_latents is not None else config['max_num_latent']
    print(f"Evaluating with {num_latents} latent tokens")
    
    # Load model config
    model_config = LatentGPT2Config.from_pretrained(
        config['model_name'],
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        use_projection=config.get('use_projection', False),
        use_layernorm=config.get('use_layernorm', False),
    )
    
    # Initialize model
    model = LatentGPT2LMHeadModel.from_pretrained(
        config['model_name'],
        config=model_config,
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Handle DDP wrapped checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Load test data
    test_data = load_gsm8k_data(config['test_data'])
    print(f"Loaded {len(test_data)} test samples")
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate_model(model, tokenizer, test_data, device, config, num_latents)
    
    # Determine output path
    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        output_path = os.path.join(checkpoint_dir, f"eval_{checkpoint_name}_latents{num_latents}.json")
    else:
        output_path = args.output
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation complete. Results saved to {output_path}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct']}/{results['total']}")
    print(f"Extraction failures: {results['extraction_failures']}")


if __name__ == "__main__":
    main()



