"""
Curriculum dataset for progressive latent training.

This module implements a curriculum learning approach where the number of latent
tokens gradually increases across training stages.
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .preprocessor import load_gsm8k_data


class CurriculumDataset(Dataset):
    """
    Dataset for curriculum training with progressive latent token increase.
    
    Training strategy:
    - Stage k with k latent tokens (starting from k=1)
    - If num_steps > k: question + k×latent + remaining_steps + answer
    - If num_steps <= k: question + k×latent + answer (full compression)
    
    Example (max_num_latent=6, epochs_per_stage=3):
        Epochs 0-2:  Stage 0 → 1 latent
        Epochs 3-5:  Stage 1 → 2 latents
        Epochs 6-8:  Stage 2 → 3 latents
        ...
        Epochs 15-17: Stage 5 → 6 latents
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        current_stage: int,
        max_num_latent: int,
        start_latent_id: int,
        latent_id: int,
        end_latent_id: int,
        dataset_format: str = "Aug",
        max_samples: Optional[int] = None,
        c_thought: int = 1,
        force_answer_only: bool = False,
    ):
        """
        Initialize curriculum dataset.
        
        Args:
            data_path: Path to GSM8k data file
            tokenizer: Tokenizer to use
            current_stage: Current training stage (0-indexed)
            max_num_latent: Maximum number of latent tokens
            start_latent_id: Token ID for <|start-latent|>
            latent_id: Token ID for <|latent|>
            end_latent_id: Token ID for <|end-latent|>
            dataset_format: Format of dataset ("Aug" or "NL")
            max_samples: Maximum number of samples to load
            c_thought: Number of latent tokens per reasoning step (default: 1)
            force_answer_only: If True, always supervise answer only (no intermediate steps)
        """
        self.tokenizer = tokenizer
        self.current_stage = current_stage
        self.max_num_latent = max_num_latent
        self.start_latent_id = start_latent_id
        self.latent_id = latent_id
        self.end_latent_id = end_latent_id
        self.dataset_format = dataset_format
        self.c_thought = c_thought
        self.force_answer_only = force_answer_only
        
        # Calculate current number of latent tokens
        # Stage 0 → 1*c_thought latents, Stage 1 → 2*c_thought latents, etc.
        self.current_num_latents = min((current_stage + 1) * c_thought, max_num_latent)
        
        # Load raw data
        self.raw_data = load_gsm8k_data(data_path, max_samples)
        
        # Process all samples
        self.processed_data = self._process_all_samples()
    
    def _process_all_samples(self) -> List[Dict]:
        """Process all samples with curriculum formatting."""
        processed = []
        for sample in self.raw_data:
            formatted = self._format_curriculum_sample(sample)
            processed.append(formatted)
        return processed
    
    def _format_curriculum_sample(self, sample: Dict) -> Dict:
        """
        Format a single sample for curriculum training.
        
        Strategy:
        - If force_answer_only is True:
            Always use: question + latents + answer (answer-only supervision)
        - Else if current_num_latents < num_steps:
            Format: question + latents + remaining_steps + answer
            Labels: mask (question + latents), supervise (remaining_steps + answer)
        - Else if current_num_latents >= num_steps:
            Format: question + latents + answer
            Labels: mask (question + latents), supervise (answer only)
        
        Args:
            sample: Raw data sample with 'question', 'steps', 'answer'
        
        Returns:
            Dictionary with tokenized data and labels
        """
        # Tokenize question
        question_text = sample["question"] + "\n"
        question_ids = self.tokenizer.encode(question_text, add_special_tokens=True)
        
        # Create latent tokens
        latent_ids = [self.start_latent_id] + [self.latent_id] * self.current_num_latents + [self.end_latent_id]
        
        # Get steps
        steps = sample.get("steps", [])
        num_steps = len(steps)
        
        # Determine format based on curriculum strategy
        if self.force_answer_only or self.current_num_latents >= num_steps:
            # Full compression: only answer supervision
            # Tokenize answer
            answer_text = "### " + str(sample["answer"])
            answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            
            # Combine: question + latents + answer
            input_ids = question_ids + latent_ids + answer_ids
            
            # Create labels: mask question and latent region, supervise answer only
            labels = (
                [-100] * len(question_ids) +  # Mask question
                [-100] * len(latent_ids) +     # Mask latent tokens
                answer_ids                      # Supervise answer
            )
        elif self.current_num_latents < num_steps:
            # Progressive compression: keep remaining steps
            remaining_steps = steps[self.current_num_latents:]
            
            # Tokenize remaining steps
            remaining_steps_ids = []
            for step in remaining_steps:
                step_ids = self.tokenizer.encode(step, add_special_tokens=False)
                remaining_steps_ids.extend(step_ids)
            
            # Tokenize answer
            answer_text = "### " + str(sample["answer"])
            answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            
            # Combine: question + latents + remaining_steps + answer
            input_ids = question_ids + latent_ids + remaining_steps_ids + answer_ids
            
            # Create labels: mask question and latent region, supervise remaining steps + answer
            labels = (
                [-100] * len(question_ids) +  # Mask question
                [-100] * len(latent_ids) +     # Mask latent tokens
                remaining_steps_ids +           # Supervise remaining steps
                answer_ids                      # Supervise answer
            )
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            # Don't include position_ids - let model generate automatically
            "num_latent": self.current_num_latents,
            "num_steps": num_steps,
        }
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.processed_data[idx]


@dataclass
class CurriculumCollator:
    """
    Data collator for curriculum training.
    
    CRITICAL: LatentGPT2's _forward_with_latent assumes all samples have
    latent tokens at THE SAME POSITION. We need to pad the question part
    to align latent tokens across all samples in the batch.
    """
    tokenizer: PreTrainedTokenizer
    latent_id: Optional[int] = None
    start_latent_id: Optional[int] = None
    end_latent_id: Optional[int] = None
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features with latent token alignment.
        
        Strategy:
        1. Find where latent tokens start in each sample (after question)
        2. Pad questions to align latent tokens at the same position
        3. Right-pad the rest of the sequence
        
        Args:
            features: List of dictionaries with 'input_ids', 'labels', etc.
        
        Returns:
            Batched tensors with aligned latent tokens
        """
        # Extract features
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]
        attention_mask_list = [f["attention_mask"] for f in features]
        num_latent_list = [f["num_latent"] for f in features]
        
        # Find latent token positions (start of latent region)
        latent_positions = []
        for input_ids in input_ids_list:
            # Find position of start_latent_id
            try:
                pos = input_ids.index(self.start_latent_id)
                latent_positions.append(pos)
            except ValueError:
                # No latent tokens, use end of sequence
                latent_positions.append(len(input_ids))
        
        # Find the maximum position (latest start of latent tokens)
        max_latent_pos = max(latent_positions)
        
        # Pad each sample to align latent tokens
        aligned_input_ids = []
        aligned_labels = []
        aligned_attention_mask = []
        aligned_position_ids = []
        
        for input_ids, labels, attention_mask, latent_pos in zip(
            input_ids_list, labels_list, attention_mask_list, latent_positions
        ):
            # Padding needed before latent tokens
            pre_pad_length = max_latent_pos - latent_pos
            
            if pre_pad_length > 0:
                # Insert padding before latent tokens
                # Split at latent position
                before_latent = input_ids[:latent_pos]
                after_latent = input_ids[latent_pos:]
                
                before_latent_labels = labels[:latent_pos]
                after_latent_labels = labels[latent_pos:]
                
                before_latent_mask = attention_mask[:latent_pos]
                after_latent_mask = attention_mask[latent_pos:]
                
                # Add padding in the middle
                padded_input_ids = (
                    before_latent +
                    [self.tokenizer.pad_token_id] * pre_pad_length +
                    after_latent
                )
                padded_labels = (
                    before_latent_labels +
                    [self.label_pad_token_id] * pre_pad_length +
                    after_latent_labels
                )
                padded_attention_mask = (
                    before_latent_mask +
                    [0] * pre_pad_length +
                    after_latent_mask
                )
                # Generate position_ids like coconut: padding gets position 0, then [0, 1, 2, ...]
                padded_position_ids = [0] * pre_pad_length + list(range(len(input_ids)))
            else:
                padded_input_ids = input_ids
                padded_labels = labels
                padded_attention_mask = attention_mask
                # No padding, standard position_ids
                padded_position_ids = list(range(len(input_ids)))
            
            aligned_input_ids.append(padded_input_ids)
            aligned_labels.append(padded_labels)
            aligned_attention_mask.append(padded_attention_mask)
            aligned_position_ids.append(padded_position_ids)
        
        # Now right-pad to make all sequences same length
        max_length = max(len(ids) for ids in aligned_input_ids)
        
        final_input_ids = []
        final_labels = []
        final_attention_mask = []
        final_position_ids = []
        
        for input_ids, labels, attention_mask, position_ids in zip(
            aligned_input_ids, aligned_labels, aligned_attention_mask, aligned_position_ids
        ):
            pad_length = max_length - len(input_ids)
            
            final_input_ids.append(
                input_ids + [self.tokenizer.pad_token_id] * pad_length
            )
            final_labels.append(
                labels + [self.label_pad_token_id] * pad_length
            )
            final_attention_mask.append(
                attention_mask + [0] * pad_length
            )
            # Right-padding for position_ids: use 0 for padding positions
            final_position_ids.append(
                position_ids + [0] * pad_length
            )
        
        # Convert to tensors
        batch = {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "labels": torch.tensor(final_labels, dtype=torch.long),
            "attention_mask": torch.tensor(final_attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(final_position_ids, dtype=torch.long),
            "num_latent": torch.tensor(num_latent_list, dtype=torch.long),
        }
        
        return batch

