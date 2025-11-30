"""
Vanilla CoT dataset where both steps and answer are supervised.

Unlike latent CoT (which only supervises answer), this dataset
supervises the full chain-of-thought including all steps.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from .preprocessor import load_gsm8k_data


class VanillaCoTDataset(Dataset):
    """
    PyTorch Dataset for vanilla CoT training on GSM8k.
    
    Format: question + "\n" + step1 + "\n" + step2 + ... + "\n### " + answer
    Supervision: All tokens after question (steps + answer) are supervised
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        dataset_format: str = "Aug",
        max_samples: Optional[int] = None,
    ):
        """
        Initialize vanilla CoT dataset.
        
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
            # Format: question + "\n" + steps + "\n### " + answer
            question_text = sample["question"] + "\n"
            
            # Join steps
            if self.dataset_format == "Aug":
                # Aug format: ["<<1.5*2=3>>", "<<3+2.5=5.5>>"]
                steps_text = "\n".join(sample["steps"]) + "\n" if sample["steps"] else ""
            else:
                # NL format: ["step 1 text", "step 2 text"]
                steps_text = "\n".join(sample["steps"]) + "\n" if sample["steps"] else ""
            
            answer_text = "### " + str(sample["answer"])
            
            # Combine all parts
            full_text = question_text + steps_text + answer_text
            
            # Tokenize
            question_ids = self.tokenizer.encode(question_text, add_special_tokens=True)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=True) + [self.tokenizer.eos_token_id]
            
            # Create labels: mask only question, supervise steps + answer
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
class VanillaCoTCollator:
    """
    Simple data collator for vanilla CoT (no special latent token alignment needed).
    """
    
    tokenizer: PreTrainedTokenizer
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features with standard padding.
        
        Args:
            features: List of feature dicts from dataset
        
        Returns:
            Batched tensors
        """
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

