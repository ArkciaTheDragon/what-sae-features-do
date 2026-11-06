"""Loading model weights and layer-specific SAEs."""

import logging
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .config import ModelSpec

LOGGER = logging.getLogger(__name__)
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model(
    spec: ModelSpec,
    model_path: str | None,
    device: str,
    dtype: str,
) -> PreTrainedModel:
    source = model_path or spec.model_id
    LOGGER.info("Loading %s", source)
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=DTYPES[dtype], attn_implementation="eager"
    )
    if len(model.model.layers) != spec.num_layers:
        raise ValueError(f"Model at {source} does not match {spec.name}")
    model.requires_grad_(False)
    return model.eval().to(device)


def load_tokenizer(spec: ModelSpec, model_path: str | None):
    return AutoTokenizer.from_pretrained(model_path or spec.model_id)


def load_sae(spec: ModelSpec, layer: int, device: str, dtype: torch.dtype):
    from sae_lens import SAE

    LOGGER.info("Loading %s / %s", spec.sae_release, spec.sae_id(layer))
    sae = SAE.from_pretrained(
        release=spec.sae_release, sae_id=spec.sae_id(layer), device=device
    )[0]
    sae.requires_grad_(False)
    return sae.eval().to(device=device, dtype=dtype)


def head_weights(
    model: PreTrainedModel, layer: int
) -> tuple[Tensor, Tensor, nn.Module]:
    attention = model.model.layers[layer].self_attn
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    queries = attention.q_proj.weight.detach()
    keys = attention.k_proj.weight.detach()
    head_dim = queries.shape[0] // num_heads
    queries = queries.reshape(num_heads, head_dim, -1).transpose(1, 2)
    keys = keys.reshape(num_kv_heads, head_dim, -1).transpose(1, 2)
    keys = keys.repeat_interleave(num_heads // num_kv_heads, dim=0)
    return queries, keys, model.model.layers[layer].input_layernorm


def save_run_settings(path: Path, spec: ModelSpec, args: dict) -> None:
    from dataclasses import asdict

    from .storage import write_json

    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in args.items()
    }
    write_json(path, {"model": asdict(spec), "arguments": settings})
