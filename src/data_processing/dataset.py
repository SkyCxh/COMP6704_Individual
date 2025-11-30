"""
Dataset and collator for GSM8k with latent thoughts.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from .preprocessor import load_gsm8k_data, format_sample, create_labels_mask, format_sample_with_steps


class GSM8kDataset(Dataset):
    """
    PyTorch Dataset for GSM8k with latent thought tokens.
    
    Supports both Aug and NL dataset formats.
    Supports both fixed and flexible latent token counts.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        num_latent: int,
        start_latent_id: int,
        latent_id: int,
        end_latent_id: int,
        dataset_format: str = "Aug",
        max_samples: Optional[int] = None,
        latent_mode: str = "fixed",
    ):
        """
        Initialize GSM8k dataset.
        
        Args:
            data_path: Path to JSON data file
            tokenizer: Tokenizer to use
            num_latent: Number of latent thought tokens (not including start/end)
            start_latent_id: Token ID for <|start-latent|>
            latent_id: Token ID for <|latent|>
            end_latent_id: Token ID for <|end-latent|>
            dataset_format: Format of dataset ("Aug" or "NL")
            max_samples: Maximum number of samples to load
            latent_mode: "fixed" for fixed num_latent, "flexible" for step-based count
        """
        self.tokenizer = tokenizer
        self.num_latent = num_latent
        self.start_latent_id = start_latent_id
        self.latent_id = latent_id
        self.end_latent_id = end_latent_id
        self.dataset_format = dataset_format
        self.latent_mode = latent_mode
        
        # Load and process data
        self.raw_data = load_gsm8k_data(data_path, max_samples)
        self.processed_data = self._process_all_samples()
    
    def _process_all_samples(self) -> List[Dict]:
        """Process all samples in the dataset."""
        processed = []
        for sample in self.raw_data:
            formatted = format_sample(
                sample,
                self.tokenizer,
                self.num_latent,
                self.start_latent_id,
                self.latent_id,
                self.end_latent_id,
                self.dataset_format,
                self.latent_mode,
            )
            
            # Get actual number of latent tokens used for this sample
            num_latent_actual = formatted["num_latent_actual"]
            
            # Create labels
            labels = create_labels_mask(
                formatted["input_ids"],
                len(formatted["question_ids"]),
                num_latent_actual,
            )
            
            processed.append({
                "input_ids": formatted["input_ids"],
                "labels": labels,
                "attention_mask": [1] * len(formatted["input_ids"]),
                "position_ids": list(range(len(formatted["input_ids"]))),
                "num_latent": num_latent_actual,  # Store for collator
            })
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.processed_data[idx]


@dataclass
class LatentCollator:
    """
    Data collator for batching samples with latent tokens.
    
    Handles padding with special attention to latent token positions
    for efficient KV cache reuse (similar to coconut's MyCollator).
    """
    
    tokenizer: PreTrainedTokenizer
    latent_id: Optional[int] = None
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Aligns latent tokens across the batch by padding before them
        to maximize KV cache reuse.
        
        Args:
            features: List of feature dicts from dataset
        
        Returns:
            Batched tensors
        """
        assert self.tokenizer.padding_side == "right", "Only right padding is supported"
        
        # Find the earliest latent token position across the batch
        earliest_latent = []
        for feature in features:
            input_ids = feature["input_ids"]
            if self.latent_id in input_ids:
                earliest_latent.append(input_ids.index(self.latent_id))
        
        # If there are latent tokens, align them by padding before
        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)
            
            for feature in features:
                input_ids = feature["input_ids"]
                
                if self.latent_id in input_ids:
                    # Calculate padding needed
                    n_tok_pad = latest_earliest_latent - input_ids.index(self.latent_id)
                else:
                    n_tok_pad = 0
                
                # Add padding to the beginning
                if n_tok_pad > 0:
                    feature["position_ids"] = [0] * n_tok_pad + feature["position_ids"]
                    feature["input_ids"] = [self.tokenizer.pad_token_id] * n_tok_pad + input_ids
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature["labels"]
                    feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]
        
        # Prepare features without labels and position_ids for padding
        non_label_position_features = [
            {k: v for k, v in feature.items() if k not in ["labels", "position_ids"]}
            for feature in features
        ]
        
        # Pad the batch
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        
        # Manually pad labels
        labels = [feature["labels"] for feature in features]
        max_label_length = max(len(l) for l in labels)
        batch["labels"] = torch.tensor(
            [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ],
            dtype=torch.long,
        )
        
        # Manually pad position_ids
        position_ids = [feature["position_ids"] for feature in features]
        max_pos_length = max(len(p) for p in position_ids)
        batch["position_ids"] = torch.tensor(
            [
                pos_id + [0] * (max_pos_length - len(pos_id))
                for pos_id in position_ids
            ],
            dtype=torch.long,
        )
        
        return batch


class GSM8kDatasetWithSteps(Dataset):
    """
    PyTorch Dataset for GSM8k with latent thought tokens and step labels.
    
    This extends GSM8kDataset to also include tokenized reasoning steps
    for reconstruction loss training.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        num_latent: int,
        start_latent_id: int,
        latent_id: int,
        end_latent_id: int,
        dataset_format: str = "Aug",
        max_samples: Optional[int] = None,
        latent_mode: str = "fixed",
    ):
        """
        Initialize GSM8k dataset with steps.
        
        Args:
            data_path: Path to JSON data file
            tokenizer: Tokenizer to use
            num_latent: Number of latent thought tokens
            start_latent_id: Token ID for <|start-latent|>
            latent_id: Token ID for <|latent|>
            end_latent_id: Token ID for <|end-latent|>
            dataset_format: Format of dataset ("Aug" or "NL")
            max_samples: Maximum number of samples to load
            latent_mode: "fixed" for fixed num_latent, "flexible" for step-based count
        """
        self.tokenizer = tokenizer
        self.num_latent = num_latent
        self.start_latent_id = start_latent_id
        self.latent_id = latent_id
        self.end_latent_id = end_latent_id
        self.dataset_format = dataset_format
        self.latent_mode = latent_mode
        
        # Load and process data
        self.raw_data = load_gsm8k_data(data_path, max_samples)
        self.processed_data = self._process_all_samples()
    
    def _process_all_samples(self) -> List[Dict]:
        """Process all samples in the dataset."""
        processed = []
        for sample in self.raw_data:
            # Use format_sample_with_steps to get step labels
            formatted = format_sample_with_steps(
                sample,
                self.tokenizer,
                self.num_latent,
                self.start_latent_id,
                self.latent_id,
                self.end_latent_id,
                self.dataset_format,
                self.latent_mode,
            )
            
            # Get actual number of latent tokens used for this sample
            num_latent_actual = formatted["num_latent_actual"]
            
            # Create answer labels
            labels = create_labels_mask(
                formatted["input_ids"],
                len(formatted["question_ids"]),
                num_latent_actual,
            )
            
            processed.append({
                "input_ids": formatted["input_ids"],
                "labels": labels,
                "attention_mask": [1] * len(formatted["input_ids"]),
                "position_ids": list(range(len(formatted["input_ids"]))),
                "num_latent": num_latent_actual,
                "step_labels": formatted["step_labels"],  # Add step labels
            })
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.processed_data[idx]


@dataclass
class LatentCollatorWithSteps:
    """
    Data collator for batching samples with latent tokens and step labels.
    
    Extends LatentCollator to also handle step labels for reconstruction loss.
    """
    
    tokenizer: PreTrainedTokenizer
    latent_id: Optional[int] = None
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features with step labels.
        
        Args:
            features: List of feature dicts from dataset
        
        Returns:
            Batched tensors including step_labels and step_attention_mask
        """
        assert self.tokenizer.padding_side == "right", "Only right padding is supported"
        
        # Find the earliest latent token position across the batch
        earliest_latent = []
        for feature in features:
            input_ids = feature["input_ids"]
            if self.latent_id in input_ids:
                earliest_latent.append(input_ids.index(self.latent_id))
        
        # If there are latent tokens, align them by padding before
        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)
            
            for feature in features:
                input_ids = feature["input_ids"]
                
                if self.latent_id in input_ids:
                    # Calculate padding needed
                    n_tok_pad = latest_earliest_latent - input_ids.index(self.latent_id)
                else:
                    n_tok_pad = 0
                
                # Add padding to the beginning
                if n_tok_pad > 0:
                    feature["position_ids"] = [0] * n_tok_pad + feature["position_ids"]
                    feature["input_ids"] = [self.tokenizer.pad_token_id] * n_tok_pad + input_ids
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature["labels"]
                    feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]
        
        # Prepare features without labels, position_ids, and step_labels for padding
        non_special_features = [
            {k: v for k, v in feature.items() if k not in ["labels", "position_ids", "step_labels"]}
            for feature in features
        ]
        
        # Pad the batch
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_special_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        
        # Manually pad labels
        labels = [feature["labels"] for feature in features]
        max_label_length = max(len(l) for l in labels)
        batch["labels"] = torch.tensor(
            [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ],
            dtype=torch.long,
        )
        
        # Manually pad position_ids
        position_ids = [feature["position_ids"] for feature in features]
        max_pos_length = max(len(p) for p in position_ids)
        batch["position_ids"] = torch.tensor(
            [
                pos_id + [0] * (max_pos_length - len(pos_id))
                for pos_id in position_ids
            ],
            dtype=torch.long,
        )
        
        # Process step_labels
        step_labels = [feature["step_labels"] for feature in features]
        batch_size = len(step_labels)
        
        # Find max number of latent tokens across batch (for flexible mode)
        max_num_latent = max(len(sample_steps) for sample_steps in step_labels)
        
        # Find max length across all steps
        max_step_len = 0
        for sample_steps in step_labels:
            for step in sample_steps:
                if len(step) > max_step_len:
                    max_step_len = len(step)
        
        # Handle empty batch (shouldn't happen, but be safe)
        if max_step_len == 0:
            max_step_len = 1
        
        # Pad each sample to have max_num_latent steps, and each step to max_step_len
        padded_step_labels = []
        step_attention_masks = []
        
        for sample_steps in step_labels:
            sample_padded_steps = []
            sample_attention = []
            
            # Process existing steps
            for step in sample_steps:
                if len(step) == 0:
                    # Empty step: pad with -100
                    padded_step = [self.label_pad_token_id] * max_step_len
                    attention = [0] * max_step_len
                else:
                    # Pad step to max_step_len
                    pad_len = max_step_len - len(step)
                    padded_step = step + [self.label_pad_token_id] * pad_len
                    attention = [1] * len(step) + [0] * pad_len
                
                sample_padded_steps.append(padded_step)
                sample_attention.append(attention)
            
            # Pad to max_num_latent (for samples with fewer latent tokens)
            while len(sample_padded_steps) < max_num_latent:
                sample_padded_steps.append([self.label_pad_token_id] * max_step_len)
                sample_attention.append([0] * max_step_len)
            
            padded_step_labels.append(sample_padded_steps)
            step_attention_masks.append(sample_attention)
        
        # Convert to tensors: (batch_size, max_num_latent, max_step_len)
        batch["step_labels"] = torch.tensor(padded_step_labels, dtype=torch.long)
        batch["step_attention_mask"] = torch.tensor(step_attention_masks, dtype=torch.long)
        
        return batch





