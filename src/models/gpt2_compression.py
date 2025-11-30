"""
GPT-2 model with MSE-based latent compression.

This module implements a dual-decoder architecture for compressing reasoning chains
into latent representations using Mean Squared Error (MSE) loss.

Architecture:
- Teacher Model: Pre-trained vanilla CoT model (frozen)
- Student Model: Latent GPT2 model (trainable)

Loss:
- Answer Loss: CrossEntropy on final answer prediction
- Compression Loss: MSE between latent hidden states and step hidden states
- Total Loss: answer_loss + compression_weight * compression_loss

Key Feature:
- Uses flexible mode: num_latents automatically matches num_steps
- Teacher provides target hidden states from reasoning steps
- Student learns to compress reasoning into latent representations
"""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .gpt2 import LatentGPT2LMHeadModel


class LatentGPT2WithCompression(nn.Module):
    """
    GPT-2 with MSE-based compression loss.
    
    Architecture:
    - Teacher Model: Pre-trained vanilla CoT GPT2LMHeadModel
    - Student Model: LatentGPT2LMHeadModel with latent tokens
    
    Training Process:
    1. Teacher forward: Process question + steps, extract step hidden states
    2. Student forward: Process question + latents + answer, extract latent hidden states
    3. Compute losses:
       - Answer loss: Standard CE on answer prediction (student)
       - Compression loss: MSE(student_latent_hiddens, teacher_step_hiddens)
    
    Note: Only flexible mode is supported (num_latents = num_steps)
    """
    
    def __init__(
        self,
        student_model: LatentGPT2LMHeadModel,
        teacher_model: Optional[GPT2LMHeadModel],
        tokenizer: GPT2Tokenizer,
        compression_weight: float = 1.0,
        latent_token_id: int = None,
        freeze_teacher: bool = True,
        freeze_student: bool = False,
        use_teacher_hiddens: bool = True,
    ):
        """
        Initialize compression model.
        
        Args:
            student_model: LatentGPT2LMHeadModel (processes latent tokens)
            teacher_model: Pre-trained GPT2LMHeadModel (vanilla CoT), can be None if use_teacher_hiddens=False
            tokenizer: GPT2Tokenizer for encoding/decoding
            compression_weight: Weight for MSE compression loss
            latent_token_id: Token ID for <|latent|>
            freeze_teacher: If True, freeze teacher model parameters
            freeze_student: If True, freeze student model parameters
            use_teacher_hiddens: If True, use teacher hidden states; if False, use token embedding averages
        """
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.compression_weight = compression_weight
        self.latent_token_id = latent_token_id
        self.use_teacher_hiddens = use_teacher_hiddens
        
        # Validate: if using teacher hiddens, teacher_model must be provided
        if self.use_teacher_hiddens and self.teacher_model is None:
            raise ValueError("teacher_model must be provided when use_teacher_hiddens=True")
        
        # Freeze teacher model (we don't train it)
        if self.teacher_model is not None and freeze_teacher:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
        
        # Optionally freeze student model (e.g., for debugging)
        if freeze_student:
            for param in self.student_model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        labels: torch.LongTensor = None,
        step_input_ids: Optional[torch.LongTensor] = None,
        step_attention_mask: Optional[torch.LongTensor] = None,
        step_positions: Optional[List[List[int]]] = None,
        step_ranges: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with compression loss.
        
        Args:
            input_ids: Student input (question + latents + answer) [batch, seq_len]
            attention_mask: Student attention mask [batch, seq_len]
            position_ids: Student position IDs [batch, seq_len]
            labels: Student labels (for answer loss) [batch, seq_len]
            step_input_ids: Teacher input (question + steps) [batch, teacher_seq_len]
            step_attention_mask: Teacher attention mask [batch, teacher_seq_len]
            step_positions: List of step end positions for each sample
                           [[step1_end, step2_end, ...], ...] (batch_size lists)
                           Used when use_teacher_hiddens=True
            step_ranges: Tensor of step token ranges [batch, max_num_steps, 2]
                        Each range is (start, end) where -1 indicates padding
                        Used when use_teacher_hiddens=False
        
        Returns:
            Dictionary with:
                - loss: Total loss (answer + compression)
                - answer_loss: CrossEntropy loss on answer
                - compression_loss: MSE loss between latent and step representations
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # ========================================
        # Step 1: Get Target Step Representations
        # ========================================
        
        if self.use_teacher_hiddens:
            # ========================================
            # Mode A: Use Teacher Model Hidden States
            # ========================================
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=step_input_ids,
                    attention_mask=step_attention_mask,
                    output_hidden_states=True,
                )
                # Extract last layer hidden states
                teacher_hidden = teacher_outputs.hidden_states[-1]  # [batch, teacher_seq_len, hidden_dim]
            
            # Extract step hidden states based on step_positions
            # step_positions: [[pos1, pos2, ...], [pos1, pos2, ...], ...]
            step_hidden_states_list = []
            max_num_steps = 0
            
            for batch_idx in range(batch_size):
                batch_step_hiddens = []
                positions = step_positions[batch_idx]
                
                for pos in positions:
                    if pos >= 0:  # Valid position
                        batch_step_hiddens.append(teacher_hidden[batch_idx, pos])
                
                if len(batch_step_hiddens) > 0:
                    step_hidden_states_list.append(torch.stack(batch_step_hiddens))
                    max_num_steps = max(max_num_steps, len(batch_step_hiddens))
                else:
                    # No valid steps, create dummy tensor
                    step_hidden_states_list.append(torch.zeros(1, teacher_hidden.size(-1), device=device))
        
        else:
            # ========================================
            # Mode B: Use Token Embedding Averages
            # ========================================
            # Get embedding layer from student model
            embedding_layer = self.student_model.transformer.wte
            hidden_dim = embedding_layer.embedding_dim
            
            step_hidden_states_list = []
            max_num_steps = 0
            
            for batch_idx in range(batch_size):
                batch_step_hiddens = []
                # step_ranges: [batch, max_num_steps, 2]
                ranges = step_ranges[batch_idx]  # [max_num_steps, 2]
                
                for range_idx in range(ranges.size(0)):
                    start = ranges[range_idx, 0].item()
                    end = ranges[range_idx, 1].item()
                    
                    if start >= 0 and end > start:
                        # Get token ids for this step
                        step_tokens = step_input_ids[batch_idx, start:end]
                        
                        # Filter out padding tokens
                        valid_mask = step_tokens != self.tokenizer.pad_token_id
                        valid_tokens = step_tokens[valid_mask]
                        
                        if len(valid_tokens) > 0:
                            # Get embeddings and compute average
                            step_embeds = embedding_layer(valid_tokens)  # [num_tokens, hidden_dim]
                            avg_embed = step_embeds.mean(dim=0)  # [hidden_dim]
                            batch_step_hiddens.append(avg_embed)
                
                if len(batch_step_hiddens) > 0:
                    step_hidden_states_list.append(torch.stack(batch_step_hiddens))
                    max_num_steps = max(max_num_steps, len(batch_step_hiddens))
                else:
                    # No valid steps, create dummy tensor
                    step_hidden_states_list.append(torch.zeros(1, hidden_dim, device=device))
        
        # ========================================
        # Pad step representations to max_num_steps
        # ========================================
        padded_step_hiddens = []
        step_mask = []
        
        for batch_step_hiddens in step_hidden_states_list:
            num_steps = batch_step_hiddens.size(0)
            if num_steps < max_num_steps:
                # Pad with zeros
                padding = torch.zeros(
                    max_num_steps - num_steps, 
                    batch_step_hiddens.size(-1), 
                    device=device
                )
                padded = torch.cat([batch_step_hiddens, padding], dim=0)
                mask = torch.cat([
                    torch.ones(num_steps, device=device),
                    torch.zeros(max_num_steps - num_steps, device=device)
                ], dim=0)
            else:
                padded = batch_step_hiddens
                mask = torch.ones(num_steps, device=device)
            
            padded_step_hiddens.append(padded)
            step_mask.append(mask)
        
        # Stack: [batch, max_num_steps, hidden_dim]
        target_hidden_states = torch.stack(padded_step_hiddens, dim=0)
        step_mask = torch.stack(step_mask, dim=0)  # [batch, max_num_steps]
        
        # ========================================
        # Step 2: Student Forward (Get Latent Hidden States)
        # ========================================
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,
        )
        
        # Extract latent token hidden states
        student_hidden = student_outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        
        # Find latent token positions (only where attention_mask is 1 to avoid padding)
        latent_mask = (input_ids == self.latent_token_id) & (attention_mask == 1)  # [batch, seq_len]
        
        # Extract latent hidden states for each sample
        latent_hidden_states_list = []
        
        for batch_idx in range(batch_size):
            batch_latent_positions = torch.where(latent_mask[batch_idx])[0]
            if len(batch_latent_positions) > 0:
                # Ensure positions are within bounds
                valid_positions = batch_latent_positions[batch_latent_positions < student_hidden.size(1)]
                if len(valid_positions) > 0:
                    batch_latent_hiddens = student_hidden[batch_idx, valid_positions]
                    latent_hidden_states_list.append(batch_latent_hiddens)
                else:
                    # No valid positions
                    latent_hidden_states_list.append(torch.zeros(1, student_hidden.size(-1), device=device))
            else:
                # No latent tokens, create dummy
                latent_hidden_states_list.append(torch.zeros(1, student_hidden.size(-1), device=device))
        
        # Pad to max_num_steps (should match in flexible mode)
        padded_latent_hiddens = []
        latent_mask_list = []
        
        for batch_latent_hiddens in latent_hidden_states_list:
            num_latents = batch_latent_hiddens.size(0)
            if num_latents < max_num_steps:
                # Pad with zeros
                padding = torch.zeros(
                    max_num_steps - num_latents, 
                    batch_latent_hiddens.size(-1), 
                    device=device
                )
                padded = torch.cat([batch_latent_hiddens, padding], dim=0)
                mask = torch.cat([
                    torch.ones(num_latents, device=device),
                    torch.zeros(max_num_steps - num_latents, device=device)
                ], dim=0)
            else:
                padded = batch_latent_hiddens[:max_num_steps]  # Truncate if too long
                mask = torch.ones(min(num_latents, max_num_steps), device=device)
            
            padded_latent_hiddens.append(padded)
            latent_mask_list.append(mask)
        
        # Stack: [batch, max_num_steps, hidden_dim]
        pred_hidden_states = torch.stack(padded_latent_hiddens, dim=0)
        latent_mask_tensor = torch.stack(latent_mask_list, dim=0)  # [batch, max_num_steps]
        
        # ========================================
        # Step 3: Compute Losses
        # ========================================
        
        # Answer loss (from student model)
        answer_loss = student_outputs.loss
        
        # Compression loss (MSE between latent and step hidden states)
        # Only compute on valid positions (where both have content)
        valid_mask = step_mask * latent_mask_tensor  # [batch, max_num_steps]
        
        if valid_mask.sum() > 0:
            # Compute MSE only on valid positions
            mse_per_position = F.mse_loss(
                pred_hidden_states, 
                target_hidden_states, 
                reduction='none'
            ).mean(dim=-1)  # [batch, max_num_steps]
            
            # Mask and average
            masked_mse = mse_per_position * valid_mask
            compression_loss = masked_mse.sum() / valid_mask.sum()
        else:
            # No valid positions (shouldn't happen in normal training)
            compression_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = answer_loss + self.compression_weight * compression_loss
        
        return {
            'loss': total_loss,
            'answer_loss': answer_loss,
            'compression_loss': compression_loss,
        }
    
    def generate(self, *args, **kwargs):
        """
        Generate using student model (for inference).
        """
        return self.student_model.generate(*args, **kwargs)
    
    def save_student(self, save_path: str):
        """
        Save only the student model (for inference).
        
        Args:
            save_path: Path to save the student model checkpoint
        """
        torch.save(self.student_model.state_dict(), save_path)
    
    def load_student(self, load_path: str):
        """
        Load student model weights.
        
        Args:
            load_path: Path to student model checkpoint
        """
        self.student_model.load_state_dict(torch.load(load_path))

