"""Extraction and flattening of transformer residual-stream states."""

from dataclasses import dataclass

import numpy as np
import torch

from .linear_maps import Array
from .models import ModelBundle


@dataclass
class HiddenStateBatch:
    layers: list[Array]
    attention_mask: np.ndarray
    input_ids: np.ndarray


@torch.inference_mode()
def extract_hidden_states(bundle: ModelBundle, prompts: list[str]) -> HiddenStateBatch:
    tokenizer = bundle.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    outputs = bundle.model(**encoded, output_hidden_states=True, return_dict=True)
    layers = [state.detach().float().cpu().numpy().astype(np.float64) for state in outputs.hidden_states]
    return HiddenStateBatch(
        layers=layers,
        attention_mask=encoded["attention_mask"].cpu().numpy(),
        input_ids=encoded["input_ids"].cpu().numpy(),
    )


def last_token_states(batch: HiddenStateBatch) -> list[Array]:
    indices = batch.attention_mask.sum(axis=1) - 1
    rows = np.arange(len(indices))
    return [layer[rows, indices] for layer in batch.layers]


def mean_pool_states(batch: HiddenStateBatch) -> list[Array]:
    """Mean-pool only non-padding token states for one vector per prompt."""
    mask = batch.attention_mask[..., None]
    counts = np.maximum(mask.sum(axis=1), 1)
    return [(layer * mask).sum(axis=1) / counts for layer in batch.layers]


def flatten_valid_tokens(batch: HiddenStateBatch) -> list[Array]:
    mask = batch.attention_mask.astype(bool)
    return [layer[mask] for layer in batch.layers]
