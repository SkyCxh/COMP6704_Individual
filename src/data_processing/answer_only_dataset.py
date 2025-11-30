"""
Answer-Only dataset where only the final answer is supervised.

Format: question + "### " + answer (NO STEPS in input or output)
Only the answer portion is supervised.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from .preprocessor import load_gsm8k_data


class AnswerOnlyDataset(Dataset):
    """
    PyTorch Dataset for answer-only training on GSM8k.
    
    Format: question + "\n### " + answer
    Supervision: Only the answer portion (after "###") is supervised
    
    Steps are NOT included in the input - model learns direct question→answer mapping.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        dataset_format: str = "Aug",
        max_samples: Optional[int] = None,
    ):
        """
        Initialize answer-only dataset.
        
        Args:
            data_path: Path to JSON data file
            tokenizer: Tokenizer to use
            dataset_format: Format of dataset ("Aug" or "NL")
            max_samples: Maximum number of samples to load
        """
        self.tokenizer = tokenizer
        self.dataset_format = dataset_format
        
        # Load and process data
        self.raw_data = load_gsm8k_data(data_path, max_samples)
        self.processed_data = self._process_all_samples()
    
    def _process_all_samples(self) -> List[Dict]:
        """Process all samples in the dataset."""
        processed = []
        
        for sample in self.raw_data:
            # Format: question + "\n### " + answer (NO STEPS!)
            question_text = sample["question"] + "\n"
            answer_marker = "### "
            answer_text = answer_marker + str(sample["answer"])
            
            # Full text: question + answer (skip steps)
            full_text = question_text + answer_text
            
            # Tokenize
            question_ids = self.tokenizer.encode(question_text, add_special_tokens=True)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=True) + [self.tokenizer.eos_token_id]
            
            # Create labels: mask question, only supervise answer
            # [-100, -100, ..., -100, answer_token1, answer_token2, ..., eos]
            labels = [-100] * len(question_ids) + full_ids[len(question_ids):]
            
            processed.append({
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": [1] * len(full_ids),
                "position_ids": list(range(len(full_ids))),
            })
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.processed_data[idx]


@dataclass
class AnswerOnlyCollator:
    """
    Simple data collator for answer-only training.
    """
    
    tokenizer: PreTrainedTokenizer
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch.
        
        Args:
            features: List of samples, each with input_ids, labels, etc.
        
        Returns:
            Batched tensors
        """
        # Get max length in batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        # Pad each feature
        batch = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "position_ids": [],
        }
        
        for feature in features:
            # Calculate padding length
            padding_length = max_length - len(feature["input_ids"])
            
            # Pad input_ids
            padded_input_ids = feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            batch["input_ids"].append(padded_input_ids)
            
            # Pad labels
            padded_labels = feature["labels"] + [self.label_pad_token_id] * padding_length
            batch["labels"].append(padded_labels)
            
            # Pad attention_mask
            padded_attention_mask = feature["attention_mask"] + [0] * padding_length
            batch["attention_mask"].append(padded_attention_mask)
            
            # Pad position_ids
            padded_position_ids = feature["position_ids"] + [0] * padding_length
            batch["position_ids"].append(padded_position_ids)
        
        # Convert to tensors
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "position_ids": torch.tensor(batch["position_ids"], dtype=torch.long),
        }
