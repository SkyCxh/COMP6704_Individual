"""Model architectures for latent CoT training"""

from .gpt2 import LatentGPT2Config, LatentGPT2LMHeadModel
from .llama import LatentLlamaConfig, LatentLlamaForCausalLM
from .gpt2_simcot import LatentGPT2WithReconstruction
from .gpt2_compression import LatentGPT2WithCompression

__all__ = [
    "LatentGPT2Config",
    "LatentGPT2LMHeadModel",
    "LatentLlamaConfig",
    "LatentLlamaForCausalLM",
    "LatentGPT2WithReconstruction",
    "LatentGPT2WithCompression",
]







