"""Small Hugging Face causal language models used by the lab."""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .device import best_device

DEFAULT_MODEL = "sshleifer/tiny-gpt2"


@dataclass
class ModelBundle:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def load_causal_lm(model_name: str = DEFAULT_MODEL) -> ModelBundle:
    """Load a compact pretrained model and place it on the best available device."""
    device = best_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: Any = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval().to(device)
    return ModelBundle(model=model, tokenizer=tokenizer, device=device)

