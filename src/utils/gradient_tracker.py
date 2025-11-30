"""
Gradient tracking utility for latent token positions.

This module provides tools to track and analyze gradients flowing through
latent token positions during training, helping identify gradient vanishing/explosion.
"""

import os
import csv
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


class LatentGradientTracker:
    """
    Track gradients of latent token hidden states during training.
    
    This tracker registers backward hooks to capture gradients at specific
    latent token positions and records gradient norms for analysis.
    """
    
    def __init__(
        self,
        output_dir: str,
        training_type: str,
        track_context_positions: int = 3,
        log_freq: int = 1,
        csv_save_freq: int = 100,
        logger=None,
    ):
        """
        Initialize gradient tracker.
        
        Args:
            output_dir: Directory to save gradient logs
            training_type: Type of training (e.g., "curriculum", "baseline")
            track_context_positions: Number of context positions to track before/after latents
            log_freq: Frequency of logging to TensorBoard (in batches)
            csv_save_freq: Frequency of saving CSV files (in batches)
            logger: Logger instance for info messages
        """
        self.output_dir = output_dir
        self.training_type = training_type
        self.track_context_positions = track_context_positions
        self.log_freq = log_freq
        self.csv_save_freq = csv_save_freq
        self.logger = logger
        
        # Create gradient log directory
        self.gradient_log_dir = os.path.join(output_dir, "gradient_logs")
        os.makedirs(self.gradient_log_dir, exist_ok=True)
        
        # CSV file path
        self.csv_path = os.path.join(self.gradient_log_dir, f"gradient_norms_{training_type}.csv")
        
        # Initialize CSV file with header
        self._init_csv()
        
        # Buffers for storing gradient data
        self.gradient_buffer = []
        self.hooks = []
        
        # Global step counter
        self.global_step = 0
        
        if self.logger:
            self.logger.log(f"LatentGradientTracker initialized")
            self.logger.log(f"  Output dir: {self.gradient_log_dir}")
            self.logger.log(f"  Training type: {training_type}")
            self.logger.log(f"  Log frequency: every {log_freq} batches")
            self.logger.log(f"  CSV save frequency: every {csv_save_freq} batches")
    
    def _init_csv(self):
        """Initialize CSV file with header."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'batch_idx', 'latent_idx', 'position_or_offset', 'grad_type',
                'grad_norm', 'grad_mean', 'grad_std', 'grad_max', 'grad_min',
                'training_type'
            ])
    
    def register_latent_hooks(
        self,
        latent_hiddens: List[Dict],
        epoch: int,
        batch_idx: int,
    ):
        """
        Register backward hooks to latent hidden states.
        
        Args:
            latent_hiddens: List of dicts with keys: 'tensor', 'batch', 'latent_idx', 'pass_idx'
            epoch: Current epoch number
            batch_idx: Current batch index
        """
        # Don't clear hooks - we want to accumulate latent + param hooks
        # Hooks will be cleared after backward
        
        # Register hooks for each latent position
        for idx, item in enumerate(latent_hiddens):
            tensor = item["tensor"]
            latent_idx = item["latent_idx"]
            pass_idx = item.get("pass_idx", latent_idx)
            
            if tensor.requires_grad:
                hook = tensor.register_hook(
                    lambda grad, lidx=latent_idx, pidx=pass_idx: self._capture_gradient(
                        grad, lidx, pidx, epoch, batch_idx, grad_type="latent"
                    )
                )
                self.hooks.append(hook)
    
    def register_context_hooks(
        self,
        context_hiddens: List[Dict],
        epoch: int,
        batch_idx: int,
    ):
        """
        Register hooks to context positions near latent tokens.
        
        Args:
            context_hiddens: List of dicts with keys: 'tensor', 'batch', 'latent_idx', 'offset', 'abs_pos'
            epoch: Current epoch number
            batch_idx: Current batch index
        """
        # Register hooks for each context position
        for idx, item in enumerate(context_hiddens):
            tensor = item["tensor"]
            latent_idx = item["latent_idx"]
            offset = item["offset"]
            abs_pos = item["abs_pos"]
            
            if tensor.requires_grad:
                hook = tensor.register_hook(
                    lambda grad, lidx=latent_idx, off=offset, pos=abs_pos: self._capture_gradient(
                        grad, lidx, off, epoch, batch_idx, grad_type="context"
                    )
                )
                self.hooks.append(hook)
    
    def register_parameter_hooks(
        self,
        model,
        epoch: int,
        batch_idx: int,
        num_latents: int = 0,
    ):
        """
        Register hooks to transformer layers to track parameter gradients.
        
        Args:
            model: The model whose layers to track
            epoch: Current epoch number
            batch_idx: Current batch index
            num_latents: Number of latent tokens in current stage (for logging)
        """
        # Get the transformer blocks
        if hasattr(model, 'module'):
            # DDP wrapped model
            transformer = model.module.transformer
        else:
            transformer = model.transformer
        
        # Track first, middle, and last layers' parameters
        num_layers = len(transformer.h)
        layers_to_track = [0, num_layers // 2, num_layers - 1]
        
        for layer_idx in layers_to_track:
            if layer_idx >= num_layers:
                continue
            
            block = transformer.h[layer_idx]
            
            # Attention weights
            if hasattr(block.attn, 'c_attn') and block.attn.c_attn.weight.requires_grad:
                hook = block.attn.c_attn.weight.register_hook(
                    lambda grad, idx=layer_idx, nl=num_latents: self._capture_gradient(
                        grad, 
                        latent_idx=nl,  # Use num_latents as a proxy for "stage"
                        position_or_offset=idx,  # layer_idx
                        epoch=epoch, 
                        batch_idx=batch_idx,
                        grad_type='param_attn'
                    )
                )
                self.hooks.append(hook)
            
            # MLP weights
            if hasattr(block.mlp, 'c_fc') and block.mlp.c_fc.weight.requires_grad:
                hook = block.mlp.c_fc.weight.register_hook(
                    lambda grad, idx=layer_idx, nl=num_latents: self._capture_gradient(
                        grad,
                        latent_idx=nl,
                        position_or_offset=idx,  # layer_idx
                        epoch=epoch,
                        batch_idx=batch_idx,
                        grad_type='param_mlp'
                    )
                )
                self.hooks.append(hook)
    
    def _capture_gradient(
        self,
        grad: torch.Tensor,
        latent_idx: int,
        position_or_offset: int,
        epoch: int,
        batch_idx: int,
        grad_type: str = "latent",
    ):
        """
        Capture gradient statistics.
        
        Args:
            grad: Gradient tensor
            latent_idx: Index of the latent token (0, 1, 2, ... which latent)
            position_or_offset: Absolute position (for latent) or offset (for context)
            epoch: Current epoch
            batch_idx: Current batch index
            grad_type: Type of gradient ("latent" or "context")
        """
        if grad is None:
            return
        
        # Compute gradient statistics
        grad_np = grad.detach().cpu().float()
        grad_norm = torch.norm(grad_np).item()
        grad_mean = torch.mean(grad_np).item()
        grad_std = torch.std(grad_np).item()
        grad_max = torch.max(grad_np).item()
        grad_min = torch.min(grad_np).item()
        
        # Store in buffer
        self.gradient_buffer.append({
            'step': self.global_step,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'latent_idx': latent_idx,
            'position_or_offset': position_or_offset,
            'grad_type': grad_type,
            'grad_norm': grad_norm,
            'grad_mean': grad_mean,
            'grad_std': grad_std,
            'grad_max': grad_max,
            'grad_min': grad_min,
            'training_type': self.training_type,
        })
    
    def log_to_tensorboard(self, writer: SummaryWriter):
        """
        Log gradient statistics to TensorBoard.
        
        Args:
            writer: TensorBoard SummaryWriter instance
        """
        if not self.gradient_buffer:
            return
        
        # Separate latent and context gradients
        latent_aggregated = defaultdict(list)
        context_aggregated = defaultdict(list)
        
        for entry in self.gradient_buffer:
            grad_type = entry.get('grad_type', 'latent')
            latent_idx = entry['latent_idx']
            grad_norm = entry['grad_norm']
            
            if grad_type == 'latent':
                # Key: latent{idx}
                key = f"latent{latent_idx}"
                latent_aggregated[key].append(grad_norm)
            elif grad_type == 'context':
                # Key: latent{idx}_offset{offset}
                offset = entry['position_or_offset']
                key = f"latent{latent_idx}_offset{offset:+d}"
                context_aggregated[key].append(grad_norm)
        
        # Log latent gradients
        for key, norms in latent_aggregated.items():
            mean_norm = np.mean(norms)
            writer.add_scalar(f'grad_norm/{key}', mean_norm, self.global_step)
        
        # Log context gradients
        for key, norms in context_aggregated.items():
            mean_norm = np.mean(norms)
            writer.add_scalar(f'context_grad_norm/{key}', mean_norm, self.global_step)
    
    def save_to_csv(self):
        """Save gradient buffer to CSV file."""
        if not self.gradient_buffer:
            return
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for entry in self.gradient_buffer:
                writer.writerow([
                    entry['step'],
                    entry['epoch'],
                    entry['batch_idx'],
                    entry['latent_idx'],
                    entry['position_or_offset'],
                    entry.get('grad_type', 'latent'),
                    entry['grad_norm'],
                    entry['grad_mean'],
                    entry['grad_std'],
                    entry['grad_max'],
                    entry['grad_min'],
                    entry['training_type'],
                ])
        
        # Clear buffer after saving
        self.gradient_buffer.clear()
    
    def step(self, writer: Optional[SummaryWriter] = None):
        """
        Update step counter and perform logging/saving based on frequency.
        
        Args:
            writer: Optional TensorBoard writer for logging
        """
        self.global_step += 1
        
        # Log to TensorBoard if needed
        if writer is not None and self.global_step % self.log_freq == 0:
            self.log_to_tensorboard(writer)
        
        # Save to CSV if needed
        if self.global_step % self.csv_save_freq == 0:
            self.save_to_csv()
            if self.logger:
                self.logger.log(f"Saved gradient norms to CSV (step {self.global_step})")
        
        # Clear hooks after each step to avoid memory leaks
        self.clear_hooks()
    
    def clear_hooks(self):
        """Clear all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def finalize(self):
        """Finalize tracking and save any remaining data."""
        # Save any remaining data in buffer
        if self.gradient_buffer:
            self.save_to_csv()
        
        # Clear hooks
        self.clear_hooks()
        
        if self.logger:
            self.logger.log(f"LatentGradientTracker finalized. Logs saved to {self.csv_path}")


