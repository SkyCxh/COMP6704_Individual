"""
Evaluation script for Latent CoT and Vanilla CoT models.
Supports checkpoint evaluation with accuracy calculation and result saving.
"""

import os
import sys
import json
import yaml
import argparse
import re
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gpt2 import LatentGPT2LMHeadModel, LatentGPT2Config
from src.models.gpt2_simcot import LatentGPT2WithReconstruction
from src.data_processing.preprocessor import add_special_tokens, load_gsm8k_data


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_tokenizer(
    config: Dict[str, Any],
    checkpoint_path: str,
    model_type: str,
    device: str = 'cuda'
) -> Tuple[torch.nn.Module, GPT2Tokenizer]:
    """
    Load model and tokenizer from checkpoint.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        model_type: 'latent' or 'vanilla'
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading {model_type} model from {checkpoint_path}...")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    if model_type == 'latent':
        # Add special tokens for latent model
        add_special_tokens(tokenizer)
        
        # Create model config
        model_config = LatentGPT2Config.from_pretrained(config['model_name'])
        model_config.num_latent = config.get('num_latent', 6)
        model_config.use_projection = config.get('use_projection', False)
        model_config.use_layernorm = config.get('use_layernorm', False)
        
        if model_config.use_projection:
            model_config.projector_dropout = config.get('projector_dropout', 0.1)
            model_config.projector_hidden_size = config.get('projector_hidden_size', 2048)
        
        # Load model
        model = LatentGPT2LMHeadModel.from_pretrained(
            config['model_name'],
            config=model_config
        )
        
        # Resize embeddings for special tokens
        model.resize_token_embeddings(len(tokenizer))
        
    else:  # vanilla
        # Load standard GPT-2 model
        model = GPT2LMHeadModel.from_pretrained(config['model_name'])
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Check if this is a SIM-CoT checkpoint (has both base_model and auxiliary_decoder)
    is_simcot = any(key.startswith('base_model.') for key in state_dict.keys())
    
    if is_simcot and model_type == 'latent':
        print("Detected SIM-CoT checkpoint. Loading base model only for evaluation...")
        # Extract base model weights
        base_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('base_model.'):
                # Remove 'base_model.' prefix
                new_key = key[len('base_model.'):]
                base_state_dict[new_key] = value
        
        # Load base model weights
        model.load_state_dict(base_state_dict)
    else:
        # Standard checkpoint, load directly
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def extract_answer(text: str) -> Optional[str]:
    """
    Extract numerical answer from generated text.
    
    Priority:
    1. Find "### " marker and extract number after it
    2. If not found, extract the last number in text
    3. Handle various number formats (commas, decimals, negatives, percentages)
    
    Args:
        text: Generated text
    
    Returns:
        Extracted answer as string, or None if extraction fails
    """
    # First, try to find "### " marker
    if "### " in text:
        # Extract everything after "### "
        after_marker = text.split("### ", 1)[1]
        # Find first number after marker
        number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
        match = re.search(number_pattern, after_marker)
        if match:
            answer = match.group(0)
            # Remove commas
            answer = answer.replace(',', '')
            return answer
    
    # If no "### " marker, extract last number in text
    number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
    matches = re.findall(number_pattern, text)
    if matches:
        answer = matches[-1]
        # Remove commas
        answer = answer.replace(',', '')
        return answer
    
    # No number found
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    
    Args:
        answer: Answer string
    
    Returns:
        Normalized answer
    """
    if answer is None:
        return ""
    
    # Remove whitespace
    answer = answer.strip()
    
    # Remove commas
    answer = answer.replace(',', '')
    
    # Handle percentage
    answer = answer.replace('%', '')
    
    # Convert to float and back to remove trailing zeros
    try:
        num = float(answer)
        # If it's an integer, return as integer string
        if num.is_integer():
            return str(int(num))
        else:
            return str(num)
    except (ValueError, AttributeError):
        return answer


def generate_answer(
    model: torch.nn.Module,
    tokenizer: GPT2Tokenizer,
    question: str,
    model_type: str,
    num_latent: int = 6,
    max_new_tokens: int = 512,
    device: str = 'cuda'
) -> str:
    """
    Generate answer using the model with greedy decoding.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        question: Question text
        model_type: 'latent' or 'vanilla'
        num_latent: Number of latent tokens (for latent model)
        max_new_tokens: Maximum tokens to generate
        device: Device
    
    Returns:
        Generated text
    """
    model.eval()
    
    with torch.no_grad():
        if model_type == 'latent':
            # Format: question + <start-latent> + <latent>×N + <end-latent>
            prompt = question + "\n"
            prompt += "<|start-latent|>"
            prompt += "<|latent|>" * num_latent
            prompt += "<|end-latent|>"
            
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
        else:  # vanilla
            # Format: question only
            prompt = question + "\n"
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate with greedy decoding
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        
        # Decode only the generated part
        generated_ids = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


def evaluate(
    model: torch.nn.Module,
    tokenizer: GPT2Tokenizer,
    test_data: List[Dict[str, Any]],
    model_type: str,
    num_latent: int = 6,
    max_new_tokens: int = 512,
    device: str = 'cuda',
    latent_mode: str = 'fixed'
) -> Tuple[List[Dict[str, Any]], float, int]:
    """
    Evaluate model on test data.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        test_data: List of test samples
        model_type: 'latent' or 'vanilla'
        num_latent: Number of latent tokens (used when latent_mode='fixed')
        max_new_tokens: Maximum tokens to generate
        device: Device
        latent_mode: 'fixed' or 'flexible' - determines how to set num_latent per sample
    
    Returns:
        Tuple of (predictions, accuracy, extraction_failures)
    """
    predictions = []
    correct = 0
    extraction_failures = 0
    
    print(f"\nEvaluating on {len(test_data)} samples...")
    if model_type == 'latent':
        print(f"Latent mode: {latent_mode}")
    
    for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        question = sample['question']
        ground_truth = sample['answer']
        
        # Determine num_latent for this sample
        if model_type == 'latent' and latent_mode == 'flexible':
            sample_num_latent = len(sample.get('steps', []))
            if sample_num_latent == 0:
                sample_num_latent = num_latent  # Fallback
        else:
            sample_num_latent = num_latent
        
        # Generate answer
        generated_text = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            model_type=model_type,
            num_latent=sample_num_latent,
            max_new_tokens=max_new_tokens,
            device=device
        )
        
        # Extract predicted answer
        predicted_answer = extract_answer(generated_text)
        
        # Track extraction failures
        if predicted_answer is None:
            extraction_failures += 1
            predicted_answer = ""
        
        # Normalize answers for comparison
        predicted_normalized = normalize_answer(predicted_answer)
        ground_truth_normalized = normalize_answer(ground_truth)
        
        # Check if correct
        is_correct = (predicted_normalized == ground_truth_normalized)
        if is_correct:
            correct += 1
        
        # Record prediction
        prediction_record = {
            'idx': idx,
            'question': question,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'correct': is_correct
        }
        predictions.append(prediction_record)
    
    # Calculate accuracy
    accuracy = correct / len(test_data) if len(test_data) > 0 else 0.0
    
    return predictions, accuracy, extraction_failures


def save_results(
    predictions: List[Dict[str, Any]],
    accuracy: float,
    extraction_failures: int,
    config: Dict[str, Any],
    checkpoint_path: str,
    model_type: str,
    output_path: str
):
    """
    Save evaluation results to JSON file.
    
    Args:
        predictions: List of prediction records
        accuracy: Overall accuracy
        extraction_failures: Number of failed extractions
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint
        model_type: 'latent' or 'vanilla'
        output_path: Path to save results
    """
    results = {
        'model_type': model_type,
        'checkpoint': checkpoint_path,
        'test_samples': len(predictions),
        'accuracy': accuracy,
        'correct': sum(1 for p in predictions if p['correct']),
        'extraction_failures': extraction_failures,
        'predictions': predictions,
        'config': config
    }
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_results(
    model_type: str,
    checkpoint_path: str,
    test_samples: int,
    accuracy: float,
    correct: int,
    extraction_failures: int,
    output_path: str
):
    """Print evaluation results summary."""
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Model Type: {model_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Samples: {test_samples}")
    print()
    print("Results:")
    print(f"  Accuracy: {accuracy*100:.2f}% ({correct}/{test_samples})")
    print(f"  Extraction Failures: {extraction_failures} ({extraction_failures/test_samples*100:.2f}%)")
    print()
    print(f"Saved predictions to: {output_path}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Latent/Vanilla CoT models")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save evaluation results')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load config
    config = load_config(args.config)
    
    # Determine model type from config
    model_type = config.get('training_type', 'latent')
    if model_type not in ['latent', 'vanilla']:
        raise ValueError(f"Invalid training_type: {model_type}. Must be 'latent' or 'vanilla'")
    
    # Get test data path
    test_data_path = args.test_data if args.test_data else config.get('test_data')
    if not test_data_path:
        raise ValueError("No test data path specified in config or arguments")
    
    print(f"Loading test data from: {test_data_path}")
    test_data = load_gsm8k_data(test_data_path)
    print(f"Loaded {len(test_data)} test samples")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        config=config,
        checkpoint_path=args.checkpoint,
        model_type=model_type,
        device=args.device
    )
    
    # Get num_latent and latent_mode for latent models
    num_latent = config.get('num_latent', 6) if model_type == 'latent' else 0
    latent_mode = config.get('latent_mode', 'fixed')
    
    # Evaluate
    predictions, accuracy, extraction_failures = evaluate(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        model_type=model_type,
        num_latent=num_latent,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        latent_mode=latent_mode
    )
    
    # Save results
    save_results(
        predictions=predictions,
        accuracy=accuracy,
        extraction_failures=extraction_failures,
        config=config,
        checkpoint_path=args.checkpoint,
        model_type=model_type,
        output_path=args.output
    )
    
    # Print summary
    print_results(
        model_type=model_type,
        checkpoint_path=args.checkpoint,
        test_samples=len(test_data),
        accuracy=accuracy,
        correct=sum(1 for p in predictions if p['correct']),
        extraction_failures=extraction_failures,
        output_path=args.output
    )


if __name__ == '__main__':
    main()

