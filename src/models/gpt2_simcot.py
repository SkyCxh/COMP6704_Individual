"""
GPT-2 model with SIM-CoT step-level reconstruction loss.

This module implements a dual-decoder architecture for supervised implicit
chain-of-thought reasoning, using reconstruction loss on latent tokens.

✅ CORRECT IMPLEMENTATION:
- Uses single forward pass through base model
- Extracts processed inputs_embeds (containing latent hidden states)
- Computes reconstruction loss from true latent hidden states
- Maintains unified computation graph for gradient flow
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .gpt2 import LatentGPT2LMHeadModel


class LatentGPT2WithReconstruction(nn.Module):
    """
    GPT-2 with step-level reconstruction loss (SIM-CoT).
    
    Architecture:
    - Base Model: LatentGPT2LMHeadModel (main latent CoT model)
    - Auxiliary Decoder: GPT2LMHeadModel or Projection layer (for step reconstruction)
    
    Loss:
    - Answer Loss: CrossEntropy on final answer
    - Reconstruction Loss: CrossEntropy on each latent token → step mapping
    - Total Loss: answer_loss + reconstruction_weight * reconstruction_loss
    
    Key Feature:
    - ✅ Uses processed inputs_embeds from base model
    - ✅ Latent tokens contain true hidden states (not original embeddings)
    - ✅ Single computation graph for proper gradient flow
    """
    
    def __init__(
        self,
        base_model: LatentGPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        use_auxiliary_gpt2: bool = True,
        reconstruction_weight: float = 1.0,
        latent_token_id: int = None,
        eos_token_id: int = None,
        freeze_base_model: bool = False,
        freeze_auxiliary: bool = False,
    ):
        """
        Initialize SIM-CoT model.
        
        Args:
            base_model: The base LatentGPT2LMHeadModel
            tokenizer: GPT2Tokenizer for encoding/decoding
            use_auxiliary_gpt2: If True, use full GPT-2 as auxiliary decoder;
                               if False, use simple projection layer
            reconstruction_weight: Weight for reconstruction loss
            latent_token_id: Token ID for <|latent|>
            eos_token_id: Token ID for EOS
            freeze_base_model: If True, freeze base model parameters
            freeze_auxiliary: If True, freeze auxiliary decoder parameters
        """
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.reconstruction_weight = reconstruction_weight
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.use_auxiliary_gpt2 = use_auxiliary_gpt2
        
        # Shared embedding layer
        self.embedding = self.base_model.transformer.wte
        
        if use_auxiliary_gpt2:
            # Independent GPT-2 as auxiliary decoder
            self.auxiliary_decoder = GPT2LMHeadModel(base_model.config)
            # Share embeddings and LM head to save parameters
            self.auxiliary_decoder.transformer.wte = self.embedding
            self.auxiliary_decoder.lm_head.weight = self.base_model.lm_head.weight
        else:
            # Simple projection layer (lightweight alternative)
            hidden_size = base_model.config.hidden_size
            vocab_size = base_model.config.vocab_size
            self.projection = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Freeze settings
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        if freeze_auxiliary:
            if use_auxiliary_gpt2:
                for param in self.auxiliary_decoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.projection.parameters():
                    param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        step_labels: Optional[torch.LongTensor] = None,
        step_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with answer loss and reconstruction loss.
        
        ✅ CORRECT IMPLEMENTATION:
        1. Single forward pass through base_model
        2. Get processed inputs_embeds (latent tokens = hidden states)
        3. Extract latent hidden states from inputs_embeds
        4. Compute reconstruction loss using true hidden states
        5. All in same computation graph!
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
            labels: Labels for answer loss (batch_size, seq_len)
            step_labels: Labels for reconstruction loss (batch_size, num_latent, max_step_len)
            step_attention_mask: Attention mask for steps (batch_size, num_latent, max_step_len)
        
        Returns:
            Dictionary with:
                - loss: Total loss
                - answer_loss: Answer-only loss
                - reconstruction_loss: Step-level reconstruction loss
                - logits: Model logits
        """
        # 1. ✅ Single forward pass through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,  # Need hidden states for reconstruction
            **kwargs
        )
        
        answer_loss = outputs.loss
        
        # 2. ✅ Get processed inputs_embeds (contains latent hidden states)
        inputs_embeds = outputs.inputs_embeds
        
        # 3. Compute reconstruction loss if step_labels provided
        if step_labels is not None and inputs_embeds is not None:
            reconstruction_loss = self._compute_reconstruction_loss(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,  # ✅ Processed embeddings with latent hidden states
                step_labels=step_labels,
                step_attention_mask=step_attention_mask,
            )
            total_loss = answer_loss + self.reconstruction_weight * reconstruction_loss
        else:
            reconstruction_loss = torch.tensor(0.0, device=answer_loss.device)
            total_loss = answer_loss
        
        return {
            'loss': total_loss,
            'answer_loss': answer_loss,
            'reconstruction_loss': reconstruction_loss,
            'logits': outputs.logits,
        }
    
    def _compute_reconstruction_loss(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,  # ✅ Processed embeddings
        step_labels: torch.LongTensor,
        step_attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss from latent hidden states to steps.
        
        ✅ KEY INSIGHT:
        - inputs_embeds at latent token positions contain HIDDEN STATES
        - NOT original <|latent|> token embeddings
        - These are the true latent representations we want to supervise
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            inputs_embeds: Processed embeddings (batch_size, seq_len, hidden_size)
                          At latent positions: contains hidden states from previous tokens
            step_labels: Ground truth steps (batch_size, num_latent, max_step_len)
            step_attention_mask: Attention mask for steps (batch_size, num_latent, max_step_len)
        
        Returns:
            Reconstruction loss (scalar)
        """
        batch_size = input_ids.shape[0]
        num_latent = step_labels.shape[1]
        max_step_len = step_labels.shape[2]
        
        # Find latent token positions in input_ids
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        
        # Group by batch
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(batch_size)
        ]
        
        # ✅ Extract latent hidden states from inputs_embeds
        # These are NOT original embeddings, but replaced hidden states!
        latent_hidden_states = []
        for batch_idx in range(batch_size):
            batch_latent_hiddens = []
            for latent_idx in range(num_latent):
                if latent_idx < len(latent_lists[batch_idx]):
                    token_pos = latent_lists[batch_idx][latent_idx]
                    # ✅ This is the HIDDEN STATE at latent position
                    hidden = inputs_embeds[batch_idx, token_pos, :]
                    batch_latent_hiddens.append(hidden)
                else:
                    # Padding (for samples with fewer latent tokens)
                    hidden = torch.zeros_like(inputs_embeds[batch_idx, 0, :])
                    batch_latent_hiddens.append(hidden)
            latent_hidden_states.append(torch.stack(batch_latent_hiddens))
        
        latent_hidden_states = torch.stack(latent_hidden_states)  # (bs, num_latent, hidden_size)
        
        # Decode each latent hidden state to step text
        if self.use_auxiliary_gpt2:
            # Use full GPT-2 as auxiliary decoder
            reconstruction_loss = self._reconstruction_with_gpt2(
                latent_hidden_states=latent_hidden_states,
                step_labels=step_labels,
                step_attention_mask=step_attention_mask,
            )
        else:
            # Use simple projection layer
            reconstruction_loss = self._reconstruction_with_projection(
                latent_hidden_states=latent_hidden_states,
                step_labels=step_labels,
                step_attention_mask=step_attention_mask,
            )
        
        return reconstruction_loss
    
    def _reconstruction_with_gpt2(
        self,
        latent_hidden_states: torch.FloatTensor,  # (bs, num_latent, hidden_size)
        step_labels: torch.LongTensor,  # (bs, num_latent, max_step_len)
        step_attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss using auxiliary GPT-2 decoder.
        
        For each latent hidden state:
        1. Use it as the first token embedding
        2. Concatenate with step token embeddings
        3. Forward through auxiliary GPT-2
        4. Compute CrossEntropy loss against step labels
        """
        batch_size, num_latent, hidden_size = latent_hidden_states.shape
        max_step_len = step_labels.shape[2]
        
        total_loss = 0.0
        num_valid_steps = 0
        
        # Process each latent token's reconstruction
        for latent_idx in range(num_latent):
            # Get hidden state for this latent position across batch
            latent_hidden = latent_hidden_states[:, latent_idx, :]  # (bs, hidden_size)
            
            # Get step labels for this position
            step_label = step_labels[:, latent_idx, :]  # (bs, max_step_len)
            
            # Skip if all labels are padding (-100)
            if (step_label == -100).all():
                continue
            
            # Get step embeddings (shift right for teacher forcing)
            # step_label: [token1, token2, token3, eos]
            # We want to predict: token1, token2, token3, eos
            # Given: <latent_hidden>, token1, token2, token3
            
            # Replace -100 with pad_token_id for embedding lookup
            step_label_for_embed = step_label.clone()
            step_label_for_embed[step_label_for_embed == -100] = self.tokenizer.pad_token_id
            
            # Create input: [BOS] + step_tokens[:-1]
            # We'll use latent_hidden as the "BOS"
            step_input_ids = step_label_for_embed[:, :-1]  # Remove last token (bs, max_step_len-1)
            step_input_embeds = self.embedding(step_input_ids)  # (bs, max_step_len-1, hidden_size)
            
            # Concatenate: [latent_hidden] + [step_embeddings]
            latent_hidden_expanded = latent_hidden.unsqueeze(1)  # (bs, 1, hidden_size)
            inputs_embeds = torch.cat([latent_hidden_expanded, step_input_embeds], dim=1)  # (bs, max_step_len, hidden_size)
            
            # Create attention mask
            if step_attention_mask is not None:
                attn_mask = step_attention_mask[:, latent_idx, :]  # (bs, max_step_len)
            else:
                attn_mask = (step_label != -100).float()  # (bs, max_step_len)
            
            # Forward through auxiliary decoder
            aux_outputs = self.auxiliary_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
            )
            
            logits = aux_outputs.logits  # (bs, max_step_len, vocab_size)
            
            # Compute loss
            # Shift: predict step_label from inputs
            shift_logits = logits  # (bs, max_step_len, vocab_size)
            shift_labels = step_label  # (bs, max_step_len)
            
            # Flatten and compute loss
            loss_fct = CrossEntropyLoss(reduction='sum')
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
            
            # Count valid tokens (not -100)
            num_valid = (shift_labels != -100).sum()
            if num_valid > 0:
                total_loss += loss
                num_valid_steps += num_valid
        
        # Average over all valid tokens
        if num_valid_steps > 0:
            return total_loss / num_valid_steps
        else:
            return torch.tensor(0.0, device=latent_hidden_states.device)
    
    def _reconstruction_with_projection(
        self,
        latent_hidden_states: torch.FloatTensor,  # (bs, num_latent, hidden_size)
        step_labels: torch.LongTensor,  # (bs, num_latent, max_step_len)
        step_attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss using simple projection layer.
        
        Directly project latent hidden state to vocabulary and predict first token of step.
        (Simpler, faster, but less powerful than full GPT-2)
        """
        batch_size, num_latent, hidden_size = latent_hidden_states.shape
        
        # Project to vocabulary
        logits = self.projection(latent_hidden_states)  # (bs, num_latent, vocab_size)
        
        # Get first token of each step as target
        first_token_labels = step_labels[:, :, 0]  # (bs, num_latent)
        
        # Compute loss
        loss_fct = CrossEntropyLoss(reduction='sum')
        loss = loss_fct(
            logits.reshape(-1, logits.size(-1)),
            first_token_labels.reshape(-1)
        )
        
        # Average over valid tokens
        num_valid = (first_token_labels != -100).sum()
        if num_valid > 0:
            return loss / num_valid
        else:
            return torch.tensor(0.0, device=latent_hidden_states.device)
    
    def generate(self, *args, **kwargs):
        """
        Generate text using only the base model.
        
        During inference, we only use the base model for generation.
        The auxiliary decoder is only used during training.
        """
        return self.base_model.generate(*args, **kwargs)
