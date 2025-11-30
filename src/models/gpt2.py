"""
GPT-2 model with latent thought capabilities.

This module implements a GPT-2 model that can process latent thought tokens
by replacing their embeddings with hidden states from previous tokens.
"""

from typing import Optional, Tuple, Union, List, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, ModelOutput


@dataclass
class LatentCausalLMOutput(ModelOutput):
    """
    Latent CoT model output with inputs_embeds.
    
    Extended from CausalLMOutputWithCrossAttentions to include
    the processed inputs_embeds for SIM-CoT reconstruction loss.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    inputs_embeds: Optional[torch.FloatTensor] = None  # New field for SIM-CoT
    latent_hiddens: Optional[List[Dict]] = None  # Latent token hidden states for gradient tracking
    context_hiddens: Optional[List[Dict]] = None  # Context token hidden states for gradient tracking


class LatentGPT2Config(GPT2Config):
    """
    Configuration class for LatentGPT2LMHeadModel.
    
    Extends GPT2Config with latent thought specific parameters.
    """
    
    model_type = "latent_gpt2"
    
    def __init__(
        self,
        latent_token_id: int = None,
        start_latent_id: int = None,
        end_latent_id: int = None,
        use_projection: bool = False,
        use_layernorm: bool = False,
        projector_dropout: float = 0.1,
        projector_hidden_size: int = 2048,
        **kwargs
    ):
        """
        Initialize LatentGPT2Config.
        
        Args:
            latent_token_id: Token ID for <|latent|>
            start_latent_id: Token ID for <|start-latent|>
            end_latent_id: Token ID for <|end-latent|>
            use_projection: Whether to use projection layer on hidden states
            use_layernorm: Whether to use LayerNorm in projection
            projector_dropout: Dropout rate for projector
            projector_hidden_size: Hidden size for projector middle layer
            **kwargs: Additional GPT2Config arguments
        """
        super().__init__(**kwargs)
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.use_projection = use_projection
        self.use_layernorm = use_layernorm
        self.projector_dropout = projector_dropout
        self.projector_hidden_size = projector_hidden_size


class LatentGPT2LMHeadModel(GPT2LMHeadModel):
    """
    GPT-2 Language Model with latent thought processing.
    
    This model extends GPT2LMHeadModel to handle latent thought tokens by:
    1. Processing the sequence until latent tokens
    2. Replacing latent token embeddings with hidden states from previous tokens
    3. Optionally applying projection and LayerNorm to hidden states
    4. Continuing processing with the updated embeddings
    """
    
    config_class = LatentGPT2Config
    
    def __init__(self, config: LatentGPT2Config):
        super().__init__(config)
        self.config = config
        
        # Optional projection module for hidden states
        if config.use_projection:
            layers = [
                nn.Dropout(config.projector_dropout),
                nn.Linear(config.hidden_size, config.projector_hidden_size),
                nn.GELU(),
                nn.Linear(config.projector_hidden_size, config.hidden_size),
            ]
            if config.use_layernorm:
                layers.append(nn.LayerNorm(config.hidden_size))
            
            self.projector = nn.Sequential(*layers)
        else:
            self.projector = None
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        Forward pass with latent thought processing.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
            past_key_values: Cached key-values for generation
            inputs_embeds: Input embeddings (optional)
            labels: Labels for language modeling loss (batch_size, seq_len)
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return ModelOutput object
        
        Returns:
            CausalLMOutputWithCrossAttentions or tuple
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get initial embeddings
        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        
        # Check if we have latent tokens to process
        if input_ids is not None and self.config.latent_token_id is not None:
            has_latent = (input_ids == self.config.latent_token_id).any()
        else:
            has_latent = False
        
        if has_latent:
            # Process with latent thought replacement
            outputs = self._forward_with_latent(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            # Standard forward pass without latent thoughts
            outputs = super().forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross entropy loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
        
        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return LatentCausalLMOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            inputs_embeds=inputs_embeds if has_latent else None,  # Return processed embeddings
            latent_hiddens=outputs.latent_hiddens if hasattr(outputs, 'latent_hiddens') else None,
            context_hiddens=outputs.context_hiddens if hasattr(outputs, 'context_hiddens') else None,
        )
    
    def _forward_with_latent(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass with latent thought token replacement.
        
        This method:
        1. Finds all latent token positions
        2. Processes sequence in passes, one latent token at a time
        3. Replaces each latent token embedding with the previous token's hidden state
        4. Optionally applies projection to hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Find latent token positions for each sample in batch
        # Only search in non-padding regions (where attention_mask == 1)
        if attention_mask is not None:
            latent_mask = (input_ids == self.config.latent_token_id) & (attention_mask == 1)
        else:
            latent_mask = (input_ids == self.config.latent_token_id)
        
        latent_indices = latent_mask.nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(batch_size)
        ]
        
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
        
        if max_n_latents == 0:
            # No latent tokens, use standard forward
            return super().forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # Initialize processing range (start at 0, end before first latent token)
        next_compute_range = (0, latent_indices[:, 1].min().item())
        kv_cache = None
        all_logits = []
        
        # For gradient tracking: collect latent hidden states
        latent_slot_hiddens = []  # Hidden states at latent token positions (filled embeddings)
        
        # Process each latent token one at a time
        for pass_idx in range(max_n_latents):
            
            if kv_cache is None:
                # First forward pass
                outputs = self.transformer(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]] if attention_mask is not None else None,
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
                    output_hidden_states=output_hidden_states,
                )
                hidden_states_offset = 0
            else:
                # Use cached key-values from previous passes
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                
                outputs = self.transformer(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
                    past_key_values=past_key_values,
                    output_hidden_states=output_hidden_states,
                )
                hidden_states_offset = next_compute_range[0]
            
            # Get logits from LM head
            hidden_states = outputs.last_hidden_state
            lm_logits = self.lm_head(hidden_states)
            all_logits.append(lm_logits)
            
            # Update KV cache
            kv_cache = outputs.past_key_values
            
            # Update compute range for next pass
            next_compute_range = (
                next_compute_range[1],
                seq_len if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1,
            )
            
            # Get last layer hidden states for replacement
            last_hidden_states = outputs.last_hidden_state
            
            # Apply optional projection
            if self.projector is not None:
                last_hidden_states = self.projector(last_hidden_states)
            
            # Find which latent tokens to fill in this pass
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            
            # Replace latent token embeddings with hidden states
            # Clone创建一个新的 tensor以避免修改原始数据
            inputs_embeds = inputs_embeds.clone()
            
            for batch_idx, token_idx in filling_indices:
                # Get latent index (which latent token is this? 0, 1, 2, ...)
                latent_idx = pass_idx
                
                # Replace with the preceding token's hidden state
                relative_pos = token_idx - 1 - hidden_states_offset
                hidden_for_latent = last_hidden_states[batch_idx, relative_pos, :]
                
                # 🔥 关键修复：追踪 hidden_for_latent（源hidden state），不是填充后的位置！
                # 这个 tensor 有完整的计算图连接！
                if self.training:
                    hidden_for_latent.retain_grad()
                    latent_slot_hiddens.append({
                        "batch": batch_idx,
                        "latent_idx": latent_idx,
                        "abs_pos": token_idx,
                        "pass_idx": pass_idx,
                        "tensor": hidden_for_latent,  # 追踪源 hidden state！
                    })
                
                # Now perform the in-place assignment
                inputs_embeds[batch_idx, token_idx, :] = hidden_for_latent
        
        # Final pass for remaining tokens after all latent tokens
        chunk_start, chunk_end = next_compute_range
        
        if kv_cache is not None:
            past_key_values = [
                (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                for k, v in kv_cache
            ]
        else:
            past_key_values = None
        
        outputs = self.transformer(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        
        # Note: latent hidden states are already collected in the loop above
        # No need to collect in final pass
        
        # Get final logits
        final_hidden = outputs.last_hidden_state
        final_logits = self.lm_head(final_hidden)
        all_logits.append(final_logits)
        
        # Concatenate all logits
        logits = torch.cat(all_logits, dim=1)
        
        return LatentCausalLMOutput(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            inputs_embeds=inputs_embeds,  # Return processed embeddings for SIM-CoT
            latent_hiddens=latent_slot_hiddens if latent_slot_hiddens else None,
            context_hiddens=None,  # Not tracking context anymore
        )

