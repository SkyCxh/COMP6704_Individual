"""
Preprocessing utilities for GSM8k dataset with latent thoughts.
"""

import json
from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizer


def load_gsm8k_data(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load GSM8k data from JSON file.
    
    Args:
        data_path: Path to the JSON data file (supports both JSON array and JSONL format)
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        List of data samples
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        # Try to load as JSON array first
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # Standard JSON array format
            data = json.load(f)
        else:
            # JSONL format (one JSON object per line)
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    
    if max_samples is not None:
        data = data[:max_samples]
    
    # Validate data format
    if len(data) > 0:
        required_keys = {'question', 'steps', 'answer'}
        sample_keys = set(data[0].keys())
        if not required_keys.issubset(sample_keys):
            raise ValueError(
                f"Data samples must contain keys {required_keys}, but got {sample_keys}"
            )
    
    return data


def add_special_tokens(tokenizer: PreTrainedTokenizer) -> Tuple[int, int, int]:
    """
    Add special tokens for latent thoughts to tokenizer.
    
    Args:
        tokenizer: The tokenizer to add tokens to
    
    Returns:
        Tuple of (start_latent_id, latent_id, end_latent_id)
    """
    special_tokens = ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    
    # Check if tokens already exist
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    # Add tokens if they don't exist
    if latent_id == tokenizer.unk_token_id:
        tokenizer.add_tokens("<|latent|>")
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    
    if start_id == tokenizer.unk_token_id:
        tokenizer.add_tokens("<|start-latent|>")
        start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    
    if end_id == tokenizer.unk_token_id:
        tokenizer.add_tokens("<|end-latent|>")
        end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    return start_id, latent_id, end_id


def format_sample(
    sample: Dict,
    tokenizer: PreTrainedTokenizer,
    num_latent: int,
    start_id: int,
    latent_id: int,
    end_id: int,
    dataset_format: str = "Aug",
    latent_mode: str = "fixed",
) -> Dict[str, List[int]]:
    """
    Format a single sample into tokenized format with latent thoughts.
    
    Format: question + "\n" + <START_LATENT> + N×<LATENT> + <END_LATENT> + "### " + answer + <EOS>
    
    Args:
        sample: Raw data sample with 'question', 'steps', 'answer'
        tokenizer: Tokenizer to use
        num_latent: Number of latent thought tokens (used when latent_mode="fixed")
        start_id: Token ID for <|start-latent|>
        latent_id: Token ID for <|latent|>
        end_id: Token ID for <|end-latent|>
        dataset_format: Format of dataset ("Aug" or "NL")
        latent_mode: "fixed" uses num_latent, "flexible" uses len(steps)
    
    Returns:
        Dictionary with tokenized 'input_ids', 'question_ids', 'answer_ids', 'num_latent_actual'
    """
    # Tokenize question
    question_text = sample["question"] + "\n"
    question_ids = tokenizer.encode(question_text, add_special_tokens=True)
    
    # Determine number of latent tokens based on mode
    if latent_mode == "flexible":
        # Use the number of steps in ground truth
        num_latent_actual = len(sample.get("steps", []))
        if num_latent_actual == 0:
            # Fallback to fixed if no steps available
            num_latent_actual = num_latent
    else:  # fixed mode
        num_latent_actual = num_latent
    
    # Create latent thought tokens: <START_LATENT> + N×<LATENT> + <END_LATENT>
    latent_ids = [start_id] + [latent_id] * num_latent_actual + [end_id]
    
    # Tokenize answer
    answer_text = "### " + str(sample["answer"])
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    
    # Combine all parts
    input_ids = question_ids + latent_ids + answer_ids
    
    return {
        "input_ids": input_ids,
        "question_ids": question_ids,
        "answer_ids": answer_ids,
        "latent_ids": latent_ids,
        "num_latent_actual": num_latent_actual,
    }


def create_labels_mask(
    input_ids: List[int],
    question_len: int,
    num_latent: int,
) -> List[int]:
    """
    Create labels with masking for latent CoT training.
    
    Only the answer tokens are supervised (not masked).
    Everything else (question + latent tokens + markers) is masked with -100.
    
    Args:
        input_ids: Full sequence of token IDs
        question_len: Length of question tokens
        num_latent: Number of latent tokens (including start/end markers, so num_latent + 2)
    
    Returns:
        List of labels with -100 for masked positions
    """
    labels = []
    
    # Mask the question tokens
    labels.extend([-100] * question_len)
    
    # Mask the latent tokens (start + N latent + end)
    labels.extend([-100] * (num_latent + 2))
    
    # Keep the answer tokens
    answer_start = question_len + num_latent + 2
    labels.extend(input_ids[answer_start:])
    
    return labels


def format_sample_with_steps(
    sample: Dict,
    tokenizer: PreTrainedTokenizer,
    num_latent: int,
    start_id: int,
    latent_id: int,
    end_id: int,
    dataset_format: str = "Aug",
    latent_mode: str = "fixed",
) -> Dict:
    """
    Format a single sample with step labels for reconstruction loss.
    
    This extends format_sample to also tokenize the reasoning steps.
    
    Args:
        sample: Raw data sample with 'question', 'steps', 'answer'
        tokenizer: Tokenizer to use
        num_latent: Number of latent thought tokens
        start_id: Token ID for <|start-latent|>
        latent_id: Token ID for <|latent|>
        end_id: Token ID for <|end-latent|>
        dataset_format: Format of dataset ("Aug" or "NL")
        latent_mode: "fixed" uses num_latent, "flexible" uses len(steps)
    
    Returns:
        Dictionary with 'input_ids', 'question_ids', 'answer_ids', 'latent_ids',
        'num_latent_actual', and 'step_labels' (list of tokenized steps)
    """
    # Get basic formatted sample
    formatted = format_sample(
        sample, tokenizer, num_latent, start_id, latent_id, end_id,
        dataset_format, latent_mode
    )
    
    # Tokenize each step
    steps = sample.get('steps', [])
    num_latent_actual = formatted['num_latent_actual']
    
    step_labels = []
    for i in range(num_latent_actual):
        if i < len(steps):
            # Tokenize the step text
            step_text = steps[i]
            step_ids = tokenizer.encode(step_text, add_special_tokens=False)
            # Add EOS token at the end
            step_ids.append(tokenizer.eos_token_id)
            step_labels.append(step_ids)
        else:
            # No step available, use empty list
            step_labels.append([])
    
    # Note: In flexible mode, we don't pad to num_latent
    # The collator will handle padding across batches
    # In fixed mode, num_latent_actual == num_latent anyway
    
    formatted['step_labels'] = step_labels
    
    return formatted







