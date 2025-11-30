"""
Dataset and collator for compression model training.

This module provides dataset and collator for training the compression model,
which requires both student (latent) and teacher (vanilla CoT) data formats.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .preprocessor import load_gsm8k_data, add_special_tokens, format_sample


class CompressionDataset(Dataset):
    """
    Dataset for compression model training.
    
    For each sample, provides:
    1. Student format: question + latents + answer
    2. Teacher format: question + steps (for extracting step hidden states)
    3. Step positions: End position of each step in teacher format
    
    Note: Only supports flexible mode (num_latents = num_steps)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        start_latent_id: int,
        latent_id: int,
        end_latent_id: int,
        dataset_format: str = "NL",
        max_samples: Optional[int] = None,
    ):
        """
        Initialize compression dataset.
        
        Args:
            data_path: Path to GSM8k data file
            tokenizer: Tokenizer to use
            start_latent_id: Token ID for <|start-latent|>
            latent_id: Token ID for <|latent|>
            end_latent_id: Token ID for <|end-latent|>
            dataset_format: Format of dataset ("Aug" or "NL")
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.tokenizer = tokenizer
        self.start_latent_id = start_latent_id
        self.latent_id = latent_id
        self.end_latent_id = end_latent_id
        self.dataset_format = dataset_format
        
        # Load raw data
        self.raw_data = load_gsm8k_data(data_path, max_samples=max_samples)
        
        # Process all samples
        self.processed_data = self._process_all_samples()
    
    def _process_all_samples(self) -> List[Dict]:
        """Process all samples in the dataset."""
        processed = []
        
        for sample in self.raw_data:
            question = sample["question"]
            steps = sample.get("steps", [])
            answer = sample["answer"]
            
            # Flexible mode: num_latents = num_steps
            num_latent = len(steps)
            
            # ========================================
            # 1. Student Format (Latent CoT)
            # ========================================
            # Format: question + <START> + N×<LATENT> + <END> + "### " + answer + <EOS>
            
            question_text = question + "\n"
            question_ids = self.tokenizer.encode(question_text, add_special_tokens=True)
            
            # Latent tokens: <START_LATENT> + N×<LATENT> + <END_LATENT>
            latent_ids = [self.start_latent_id] + [self.latent_id] * num_latent + [self.end_latent_id]
            
            # Answer
            answer_text = "### " + str(answer)
            answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            
            # Combine: student input
            student_input_ids = question_ids + latent_ids + answer_ids
            
            # Create labels: mask question and latent tokens, only supervise answer
            student_labels = (
                [-100] * len(question_ids) +
                [-100] * len(latent_ids) +
                answer_ids
            )
            
            # ========================================
            # 2. Teacher Format (Vanilla CoT)
            # ========================================
            # Format: question + step1 + step2 + ... + stepN
            # We don't include answer here, only need step hidden states
            
            teacher_input_ids = question_ids.copy()
            step_positions = []  # Record end position of each step (for teacher hiddens)
            step_ranges = []     # Record (start, end) range of each step (for token embeddings)
            
            for step_text in steps:
                # Tokenize step
                step_ids = self.tokenizer.encode(step_text, add_special_tokens=False)
                
                # Record start position
                start_pos = len(teacher_input_ids)
                
                # Add to teacher input
                teacher_input_ids.extend(step_ids)
                
                # Record end position (last token of this step)
                end_pos = len(teacher_input_ids) - 1
                step_positions.append(end_pos)
                
                # Record range [start, end) for token embedding averaging
                step_ranges.append((start_pos, len(teacher_input_ids)))
            
            # ========================================
            # 3. Create Sample Dict
            # ========================================
            processed.append({
                # Student format
                "student_input_ids": student_input_ids,
                "student_labels": student_labels,
                "student_attention_mask": [1] * len(student_input_ids),
                "student_position_ids": list(range(len(student_input_ids))),
                
                # Teacher format
                "teacher_input_ids": teacher_input_ids,
                "teacher_attention_mask": [1] * len(teacher_input_ids),
                
                # Step information (for both modes)
                "step_positions": step_positions,  # For teacher hidden states mode
                "step_ranges": step_ranges,        # For token embedding mode
                "num_latent": num_latent,
            })
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.processed_data[idx]


@dataclass
class CompressionCollator:
    """
    Data collator for batching compression model samples.
    
    Handles:
    1. Student sequences: Align by latent token positions (left padding before latent)
    2. Teacher sequences: Right padding (standard)
    3. Step positions: Pad with -1 to max_num_steps
    """
    
    tokenizer: PreTrainedTokenizer
    latent_id: Optional[int] = None
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Args:
            features: List of feature dicts from dataset
        
        Returns:
            Batched tensors with student and teacher inputs
        """
        assert self.tokenizer.padding_side == "right", "Only right padding is supported"
        
        batch_size = len(features)
        
        # ========================================
        # 1. Process Student Sequences (align by latent tokens)
        # ========================================
        
        # Find the earliest latent token position across the batch
        earliest_latent = []
        for feature in features:
            student_input_ids = feature["student_input_ids"]
            if self.latent_id in student_input_ids:
                earliest_latent.append(student_input_ids.index(self.latent_id))
        
        # Align by padding before latent tokens
        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)
            
            for feature in features:
                student_input_ids = feature["student_input_ids"]
                
                if self.latent_id in student_input_ids:
                    # Calculate padding needed
                    n_tok_pad = latest_earliest_latent - student_input_ids.index(self.latent_id)
                else:
                    n_tok_pad = 0
                
                # Add padding to the beginning
                if n_tok_pad > 0:
                    feature["student_position_ids"] = [0] * n_tok_pad + feature["student_position_ids"]
                    feature["student_input_ids"] = [self.tokenizer.pad_token_id] * n_tok_pad + student_input_ids
                    feature["student_labels"] = [self.label_pad_token_id] * n_tok_pad + feature["student_labels"]
                    feature["student_attention_mask"] = [0] * n_tok_pad + feature["student_attention_mask"]
        
        # Pad student sequences to max length (right padding)
        max_student_len = max(len(f["student_input_ids"]) for f in features)
        
        student_input_ids = []
        student_labels = []
        student_attention_mask = []
        student_position_ids = []
        
        for feature in features:
            pad_len = max_student_len - len(feature["student_input_ids"])
            
            student_input_ids.append(
                feature["student_input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            student_labels.append(
                feature["student_labels"] + [self.label_pad_token_id] * pad_len
            )
            student_attention_mask.append(
                feature["student_attention_mask"] + [0] * pad_len
            )
            student_position_ids.append(
                feature["student_position_ids"] + [0] * pad_len
            )
        
        # ========================================
        # 2. Process Teacher Sequences (standard right padding)
        # ========================================
        
        max_teacher_len = max(len(f["teacher_input_ids"]) for f in features)
        
        teacher_input_ids = []
        teacher_attention_mask = []
        
        for feature in features:
            pad_len = max_teacher_len - len(feature["teacher_input_ids"])
            
            teacher_input_ids.append(
                feature["teacher_input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            teacher_attention_mask.append(
                feature["teacher_attention_mask"] + [0] * pad_len
            )
        
        # ========================================
        # 3. Process Step Positions (pad with -1)
        # ========================================
        
        max_num_steps = max(len(f["step_positions"]) for f in features)
        
        step_positions = []
        for feature in features:
            positions = feature["step_positions"]
            pad_len = max_num_steps - len(positions)
            step_positions.append(positions + [-1] * pad_len)
        
        # ========================================
        # 4. Process Step Ranges (pad with (-1, -1))
        # ========================================
        
        step_ranges = []
        for feature in features:
            ranges = feature["step_ranges"]
            pad_len = max_num_steps - len(ranges)
            # Pad with (-1, -1) for invalid ranges
            padded_ranges = ranges + [(-1, -1)] * pad_len
            step_ranges.append(padded_ranges)
        
        # Convert to tensor [batch, max_num_steps, 2]
        step_ranges_tensor = torch.tensor(step_ranges, dtype=torch.long)
        
        # ========================================
        # 5. Create Batch Dictionary
        # ========================================
        
        batch = {
            # Student tensors
            "input_ids": torch.tensor(student_input_ids, dtype=torch.long),
            "labels": torch.tensor(student_labels, dtype=torch.long),
            "attention_mask": torch.tensor(student_attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(student_position_ids, dtype=torch.long),
            
            # Teacher tensors
            "step_input_ids": torch.tensor(teacher_input_ids, dtype=torch.long),
            "step_attention_mask": torch.tensor(teacher_attention_mask, dtype=torch.long),
            
            # Step information (for both modes)
            "step_positions": step_positions,      # List of lists (for teacher hiddens)
            "step_ranges": step_ranges_tensor,     # Tensor [batch, max_steps, 2] (for token embeddings)
            
            # Metadata
            "num_latent": torch.tensor([f["num_latent"] for f in features], dtype=torch.long),
        }
        
        return batch














