"""Decoder-write interventions for semantic and attention features."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .attention import load_heads
from .config import ModelSpec, Role
from .http import get_json
from .models import load_sae
from .semantic import criterion_masks, semantic_mask, thresholds
from .storage import (
    attention_dir,
    model_dir,
    semantic_path,
    write_frame,
    write_json,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ForwardPass:
    logits: Tensor
    activations: Tensor
    attention: Tensor | None


def zero_ablation_hook(sae, feature_id: int, position: int):
    def hook(module, inputs, output):
        residual = output if isinstance(output, Tensor) else output[0]
        if not 0 <= position < residual.shape[1]:
            raise ValueError(
                f"Intervention position {position} is outside the sequence"
            )
        z = sae.encode(residual.to(sae.W_enc.dtype))[:, position, feature_id]
        changed = residual.clone()
        changed[:, position, :] -= (z.unsqueeze(-1) * sae.W_dec[feature_id]).to(
            residual.dtype
        )
        return changed if isinstance(output, Tensor) else (changed, *output[1:])

    return hook


@torch.inference_mode()
def forward_pass(
    model,
    sae,
    tokens: Tensor,
    layer: int,
    *,
    head: int | None = None,
    feature_id: int | None = None,
    position: int | None = None,
) -> ForwardPass:
    if feature_id is not None and position is None:
        raise ValueError("An intervention position is required")
    activations = []

    def capture(module, inputs, output):
        residual = output if isinstance(output, Tensor) else output[0]
        activations.append(sae.encode(residual.to(sae.W_enc.dtype))[0].detach())

    handles = [model.model.layers[layer].register_forward_hook(capture)]
    if feature_id is not None:
        handles.append(
            model.model.layers[layer].register_forward_hook(
                zero_ablation_hook(sae, feature_id, position)
            )
        )
    try:
        result = model(
            input_ids=tokens,
            output_attentions=head is not None,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if not activations:
        raise RuntimeError(f"No residual activation captured at layer {layer}")
    attention = None
    if head is not None:
        if result.attentions is None or result.attentions[layer + 1] is None:
            raise RuntimeError("The model did not return eager attention weights")
        attention = result.attentions[layer + 1][0, head].detach()
    return ForwardPass(result.logits[0, -1].detach(), activations[0], attention)


def prompt_text(
    client: httpx.Client,
    cache: Path,
    spec: ModelSpec,
    layer: int,
    feature_id: int,
    tokenizer,
) -> str:
    import json

    source = spec.neuronpedia_source.format(layer=layer)
    path = cache / spec.neuronpedia_model / source / f"{feature_id}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        url = (
            f"https://www.neuronpedia.org/api/feature/"
            f"{spec.neuronpedia_model}/{source}/{feature_id}"
        )
        data = get_json(client, url)
        write_json(path, data)
    activations = data["activations"]
    if not activations or not activations[0]["tokens"]:
        raise ValueError(
            f"No activating prompt for {spec.name} layer {layer} feature {feature_id}"
        )
    tokens = list(activations[0]["tokens"])
    if tokens[0] == tokenizer.bos_token:
        tokens = tokens[1:]
    text = "".join(
        token.replace("\u2581", " ").replace("\u0120", " ") for token in tokens
    )
    if not text:
        raise ValueError(f"Empty prompt for feature {feature_id}")
    return text


def specialist_frame(metrics: pd.DataFrame, heads: list[dict]) -> pd.DataFrame:
    frame = metrics.copy().sort_values("feature_id").reset_index(drop=True)
    for role in ("query", "key"):
        weights = np.stack([head[f"post_{role}_topk"] for head in heads], axis=1)
        scores = weights.sum(axis=-1) if role == "query" else weights.mean(axis=-1)
        if len(scores) != len(frame):
            raise ValueError("Semantic and attention feature counts differ")
        frame[f"{role}_score"] = scores.max(axis=1)
        frame[f"{role}_head"] = scores.argmax(axis=1)
    return frame


def select_matched(
    frame: pd.DataFrame,
    candidate_mask: pd.Series,
    feature_id: int,
    activations: Tensor,
    used: set[int],
    seed: int,
) -> int:
    indexed = frame.set_index("feature_id")
    target_norm = float(indexed.loc[feature_id, "decoder_norm"])
    ids = frame["feature_id"].to_numpy()
    z = activations.float().cpu().numpy()
    mask = (
        candidate_mask
        & frame["decoder_norm"].between(target_norm * 0.95, target_norm * 1.05)
        & ~frame["feature_id"].isin(used)
        & (z[ids] >= max(0.1 * float(z[feature_id]), 1.0))
    )
    pool = frame[mask]
    if pool.empty:
        raise ValueError(f"No matched feature for feature {feature_id}")
    return int(pool.sample(n=1, random_state=seed)["feature_id"].iloc[0])


def logit_metrics(clean: Tensor, intervened: Tensor, token_id: int) -> dict[str, float]:
    clean_logp = clean.double().log_softmax(dim=-1)
    intervened_logp = intervened.double().log_softmax(dim=-1)
    return {
        "target_logit_clean": float(clean[token_id]),
        "target_logit_intervened": float(intervened[token_id]),
        "logit_drop": float(clean[token_id] - intervened[token_id]),
        "vocab_kl": float((clean_logp.exp() * (clean_logp - intervened_logp)).sum()),
    }


def attention_metrics(
    clean: Tensor, intervened: Tensor, role: Role, position: int
) -> dict[str, float]:
    if role == "query":
        p = clean[position, : position + 1].double()
        q = intervened[position, : position + 1].double()
        return {
            "attention_kl": float((p * ((p + 1e-12).log() - (q + 1e-12).log())).sum())
        }
    if role != "key":
        raise ValueError(f"Unknown attention role: {role}")
    if position >= clean.shape[0] - 1:
        raise ValueError("Key intervention requires later query positions")
    p = clean[position + 1 :, position].double()
    q = intervened[position + 1 :, position].double()
    return {
        "incoming_attention_clean": float(p.mean()),
        "incoming_attention_intervened": float(q.mean()),
        "incoming_attention_drop": float((p - q).mean()),
        "n_later_queries": len(p),
    }


def paired_summary(frame: pd.DataFrame, role: Role) -> dict:
    metric = {
        "semantic": "logit_drop",
        "query": "attention_kl",
        "key": "incoming_attention_drop",
    }[role]
    pairs = frame.pivot(index="pair_id", columns="group", values=metric)
    pairs = pairs.clip(lower=0) if role == "query" else pairs.abs()
    matched = pairs.dropna(subset=["flagged", "matched"])
    denominator = (
        matched["matched"].replace(0, np.nan)
        if role == "semantic"
        else matched["matched"].clip(lower=1e-9)
    )
    result = {
        "n_pairs": len(matched),
        "flagged_median": float(pairs["flagged"].median()),
        "matched_median": float(matched["matched"].median()),
        "median_pair_ratio": float((matched["flagged"] / denominator).median()),
        "win_rate": float((matched["flagged"] > matched["matched"]).mean()),
    }
    if role == "semantic":
        kl = (
            frame.pivot(index="pair_id", columns="group", values="vocab_kl")
            .clip(lower=0)
            .loc[matched.index]
        )
        result["flagged_kl_median"] = float(kl["flagged"].median())
        result["matched_kl_median"] = float(kl["matched"].median())
    return result


@torch.inference_mode()
def run_interventions(
    model,
    tokenizer,
    spec: ModelSpec,
    output: Path,
    cache: Path,
    layers: list[int],
    role: Role,
    n_features: int,
    seed: int,
) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    cutoffs = thresholds(
        pd.read_parquet(semantic_path(output, spec, "dec_u", spec.decoder_calibration))
    )
    api_key = os.environ.get("NEURONPEDIA_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    directory = model_dir(output, spec) / "intervention" / role
    summaries = []
    with httpx.Client(headers=headers, timeout=30) as client:
        for layer in layers:
            spec.check_layer(layer, attention=role != "semantic")
            sae = load_sae(spec, layer, str(device), dtype)
            frame = pd.read_parquet(semantic_path(output, spec, "dec_u", layer))
            frame = frame.sort_values("feature_id").reset_index(drop=True)
            if role == "semantic":
                flagged_mask = semantic_mask(frame, cutoffs)
                candidate_mask = (~criterion_masks(frame, cutoffs)).all(axis=1)
            else:
                frame = specialist_frame(
                    frame, load_heads(attention_dir(output, spec, layer))
                )
                flagged_mask = frame[f"{role}_score"] > (
                    0.95 if role == "query" else 0.25
                )
                candidate_mask = (frame["query_score"] <= 0.95) & (
                    frame["key_score"] <= 0.25
                )
            pool = frame[
                flagged_mask
                & (frame["decoder_norm"] >= frame["decoder_norm"].quantile(0.05))
            ]
            if len(pool) < n_features:
                raise ValueError(
                    f"Layer {layer} has {len(pool)} eligible features; requested {n_features}"
                )
            sampled = pool.sample(n=n_features, random_state=seed)
            rows = []
            skipped = []
            used: set[int] = set()
            for pair_id, (_, feature) in enumerate(sampled.iterrows()):
                feature_id = int(feature["feature_id"])
                head = None if role == "semantic" else int(feature[f"{role}_head"])
                try:
                    text = prompt_text(
                        client, cache, spec, layer, feature_id, tokenizer
                    )
                    tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(
                        device
                    )
                    clean = forward_pass(model, sae, tokens, layer, head=head)
                    if len(clean.activations) <= 1:
                        raise ValueError("Prompt has no token after BOS")
                    position = int(clean.activations[1:, feature_id].argmax()) + 1
                    if role != "key":
                        tokens = tokens[:, : position + 1]
                        clean = forward_pass(model, sae, tokens, layer, head=head)
                    if float(clean.activations[position, feature_id]) < 1:
                        raise ValueError(
                            "Feature activation is below 1 at the selected position"
                        )
                    if role == "query" and position < 2:
                        raise ValueError(
                            "Query prompt has fewer than two preceding tokens"
                        )
                    if role == "key" and tokens.shape[1] - position - 1 < 5:
                        raise ValueError("Key prompt has fewer than five later queries")
                    matched_id = select_matched(
                        frame,
                        candidate_mask,
                        feature_id,
                        clean.activations[position],
                        used,
                        seed + 200_000 + pair_id,
                    )
                except (httpx.HTTPError, KeyError, ValueError) as error:
                    LOGGER.warning("Layer %s feature %s: %s", layer, feature_id, error)
                    skipped.append({"feature_id": feature_id, "reason": str(error)})
                    continue
                used.add(matched_id)
                for group, intervention_id in (
                    ("flagged", feature_id),
                    ("matched", matched_id),
                ):
                    intervened = forward_pass(
                        model,
                        sae,
                        tokens,
                        layer,
                        head=head,
                        feature_id=intervention_id,
                        position=position,
                    )
                    row = {
                        "model": spec.name,
                        "layer": layer,
                        "role": role,
                        "pair_id": pair_id,
                        "group": group,
                        "flagged_feature_id": feature_id,
                        "matched_feature_id": matched_id,
                        "intervention_feature_id": intervention_id,
                        "head": head,
                        "position": position,
                        "activation": float(
                            clean.activations[position, intervention_id]
                        ),
                        "decoder_norm": float(
                            sae.W_dec[intervention_id].float().norm()
                        ),
                        "seed": seed,
                    }
                    if role == "semantic":
                        token_id = int(feature["top_k_token_ids"][0])
                        row["target_token_id"] = token_id
                        row.update(
                            logit_metrics(clean.logits, intervened.logits, token_id)
                        )
                    else:
                        if clean.attention is None or intervened.attention is None:
                            raise RuntimeError("Attention weights were not captured")
                        row.update(
                            attention_metrics(
                                clean.attention, intervened.attention, role, position
                            )
                        )
                    rows.append(row)
            write_json(directory / f"skipped_layer_{layer}.json", skipped)
            if not rows:
                raise RuntimeError(f"No completed intervention pairs at layer {layer}")
            results = pd.DataFrame(rows)
            write_frame(directory / f"layer_{layer}.parquet", results)
            summaries.append({"layer": layer, **paired_summary(results, role)})
            del sae
    write_frame(directory / "summary.csv", pd.DataFrame(summaries))
