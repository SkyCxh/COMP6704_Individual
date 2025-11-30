"""
Utility functions for training, evaluation, and logging.
"""

import os
import json
import random
import numpy as np
import torch
from typing import Dict, Any, Optional
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rank() -> int:
    """Get process rank for distributed training."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if current process is main process."""
    return get_rank() == 0


def setup_distributed():
    """
    Initialize distributed training.
    
    Returns True if distributed training is enabled, False otherwise.
    """
    # Check if environment variables for distributed training are set
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'MASTER_PORT' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        return True
    
    # Not in distributed mode
    return False


def cleanup_distributed():
    """Cleanup distributed training."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    loss: float,
    save_path: str,
):
    """Save training checkpoint."""
    if not is_main_process():
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get model state dict (handle DDP wrapper)
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state (handle DDP wrapper)
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return checkpoint


def extract_answer(text: str) -> str:
    """
    Extract numerical answer from model output.
    
    Expected format: "### <answer>"
    """
    text = text.strip()
    
    # Find "###" marker
    if "###" in text:
        answer_part = text.split("###")[-1].strip()
        # Extract first number-like token
        tokens = answer_part.split()
        for token in tokens:
            # Remove common punctuation
            token = token.strip('.,!?')
            # Try to convert to number
            try:
                # Handle numbers with commas
                token = token.replace(',', '')
                float(token)
                return token
            except ValueError:
                continue
    
    return ""


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    
    Args:
        answer: Answer string (can be numeric or text)
    
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
    
    # Convert to string and remove commas, extra whitespace
    answer = str(answer).replace(',', '').strip()
    
    # Try to normalize as float (handles "1.0" vs "1")
    try:
        # Convert to float and back to string to normalize format
        num = float(answer)
        # If it's essentially an integer, return as integer
        if num == int(num):
            return str(int(num))
        else:
            return str(num)
    except ValueError:
        # Not a number, return as-is (lowercased for comparison)
        return answer.lower()
    
    return answer


def compute_accuracy(predictions: list, references: list) -> Dict[str, float]:
    """
    Compute accuracy metrics.
    
    Args:
        predictions: List of predicted answers (strings)
        references: List of ground truth answers (strings)
    
    Returns:
        Dictionary with accuracy metrics
    """
    assert len(predictions) == len(references), "Length mismatch"
    
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Normalize answers
        pred = str(pred).replace(',', '').strip()
        ref = str(ref).replace(',', '').strip()
        
        # Try exact match
        if pred == ref:
            correct += 1
            continue
        
        # Try numerical comparison
        try:
            pred_num = float(pred)
            ref_num = float(ref)
            if abs(pred_num - ref_num) < 1e-5:
                correct += 1
        except (ValueError, TypeError):
            pass
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
    }


class AverageMeter:
    """Compute and store average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    """Logger for training metrics with TensorBoard support."""
    
    def __init__(self, log_dir: str, log_file: str = "train.log", use_tensorboard: bool = True):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer = None
        
        if is_main_process():
            os.makedirs(log_dir, exist_ok=True)
            
            # Initialize TensorBoard writer
            if self.use_tensorboard:
                tensorboard_dir = os.path.join(log_dir, "tensorboard")
                self.writer = SummaryWriter(tensorboard_dir)
                print(f"TensorBoard logging to: {tensorboard_dir}")
                print(f"View with: tensorboard --logdir {tensorboard_dir}")
            
            # Initialize log file
            with open(self.log_file, 'w') as f:
                f.write(f"Training started at {datetime.now()}\n")
                f.write("=" * 80 + "\n")
    
    def log(self, message: str, print_msg: bool = True):
        """Log a message."""
        if not is_main_process():
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        
        if print_msg:
            print(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """Log metrics to both file and TensorBoard."""
        if not is_main_process():
            return
        
        # Log to file
        metric_str = f"Step {step}"
        if prefix:
            metric_str += f" [{prefix}]"
        metric_str += ": " + ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                        for k, v in metrics.items()])
        self.log(metric_str)
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    tag = f"{prefix}/{key}" if prefix else key
                    self.writer.add_scalar(tag, value, step)
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information to TensorBoard."""
        if not is_main_process() or self.writer is None:
            return
        
        # Log model parameters
        if "num_parameters" in model_info:
            self.writer.add_scalar("model/num_parameters", model_info["num_parameters"], 0)
        
        # Log model architecture as text
        if "architecture" in model_info:
            self.writer.add_text("model/architecture", str(model_info["architecture"]), 0)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and final metrics to TensorBoard."""
        if not is_main_process() or self.writer is None:
            return
        
        # Convert all values to basic types for TensorBoard
        clean_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                clean_hparams[k] = v
            else:
                clean_hparams[k] = str(v)
        
        self.writer.add_hparams(clean_hparams, metrics)
    
    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """Save results to JSON file."""
        if not is_main_process():
            return
        
        results_path = os.path.join(self.log_dir, filename)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.log(f"Results saved to {results_path}")
    
    def close(self):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

