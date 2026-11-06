"""Feature attention statistics and specialization thresholds."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .config import KEY_THRESHOLDS, QUERY_THRESHOLDS, TOP_K, ModelSpec
from .models import head_weights, load_sae
from .storage import attention_dir, model_dir, write_frame

LOGGER = logging.getLogger(__name__)


def exclude_self_topk(weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    has_self = (indices == np.arange(len(weights))[:, None]).any(axis=1)
    return np.where(has_self[:, None], 0, weights)


@torch.inference_mode()
def attention_statistics(
    queries: Tensor, keys: Tensor, batch_size: int, top_k: int = TOP_K
) -> dict[str, np.ndarray]:
    if queries.shape != keys.shape or queries.shape[0] <= top_k:
        raise ValueError("Q and K must have matching shapes with more than top_k rows")
    size, head_dim = queries.shape
    result: dict[str, Tensor] = {
        "pre_query_sum": torch.empty(size, device=queries.device),
        "pre_key_sum": torch.zeros(size, device=queries.device),
    }
    columns = {}
    column_indices = {}
    for stage in ("pre", "post"):
        result[f"{stage}_query_topk"] = torch.empty(size, top_k, device=queries.device)
        result[f"{stage}_query_topk_indices"] = torch.empty(
            size, top_k, device=queries.device, dtype=torch.long
        )
        columns[stage] = torch.full((top_k, size), -torch.inf, device=queries.device)
        column_indices[stage] = torch.full(
            (top_k, size), -1, device=queries.device, dtype=torch.long
        )
    for start in range(0, size, batch_size):
        end = min(start + batch_size, size)
        scores = (queries[start:end].float() @ keys.float().T) / head_dim**0.5
        result["pre_query_sum"][start:end] = scores.sum(dim=1)
        result["pre_key_sum"] += scores.sum(dim=0)
        probabilities = scores.softmax(dim=-1)
        for stage, values in (("pre", scores), ("post", probabilities)):
            query = values.topk(top_k, dim=1)
            result[f"{stage}_query_topk"][start:end] = query.values
            result[f"{stage}_query_topk_indices"][start:end] = query.indices
            candidates = torch.cat((columns[stage], values), dim=0)
            indices = torch.cat(
                (
                    column_indices[stage],
                    torch.arange(start, end, device=queries.device)[:, None].expand(
                        -1, size
                    ),
                ),
                dim=0,
            )
            selected = candidates.topk(top_k, dim=0)
            columns[stage] = selected.values
            column_indices[stage] = indices.gather(0, selected.indices)
    for stage in ("pre", "post"):
        result[f"{stage}_key_topk"] = columns[stage].T
        result[f"{stage}_key_topk_indices"] = column_indices[stage].T
    arrays = {name: values.cpu().numpy() for name, values in result.items()}
    for stage in ("pre", "post"):
        for role in ("query", "key"):
            name = f"{stage}_{role}_topk"
            arrays[f"{name}_no_self"] = exclude_self_topk(
                arrays[name], arrays[f"{name}_indices"]
            )
    return arrays


def specialization_counts(heads: list[dict[str, np.ndarray]], layer: int) -> list[dict]:
    rows = []
    for suffix in ("", "_no_self"):
        for role, cutoffs, axis in (
            ("query", QUERY_THRESHOLDS, "query"),
            ("key", KEY_THRESHOLDS, "key"),
        ):
            weights = np.stack(
                [head[f"post_{axis}_topk{suffix}"] for head in heads], axis=1
            )
            scores = weights.sum(axis=-1) if role == "query" else weights.mean(axis=-1)
            for cutoff in cutoffs:
                passes = scores > cutoff
                rows.append(
                    {
                        "layer": layer,
                        "role": role,
                        "threshold": cutoff,
                        "exclude_self": bool(suffix),
                        "n_features": len(scores),
                        "mean_head_count": int((scores.mean(axis=1) > cutoff).sum()),
                        "median_head_count": int(
                            (np.median(scores, axis=1) > cutoff).sum()
                        ),
                        "any_head_count": int(passes.any(axis=1).sum()),
                        "any_head_rate": float(passes.any(axis=1).mean()),
                    }
                )
    return rows


def load_heads(directory: Path) -> list[dict[str, np.ndarray]]:
    paths = sorted(
        directory.glob("head_*.npz"), key=lambda path: int(path.stem.split("_")[-1])
    )
    if not paths:
        raise FileNotFoundError(f"No attention statistics in {directory}")
    expected = list(range(len(paths)))
    actual = [int(path.stem.split("_")[-1]) for path in paths]
    if actual != expected:
        raise ValueError(f"Incomplete attention heads in {directory}: {actual}")
    heads = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            heads.append({name: archive[name] for name in archive.files})
    return heads


@torch.inference_mode()
def run_attention(
    model, spec: ModelSpec, output: Path, layers: list[int], batch_size: int
) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for layer in layers:
        spec.check_layer(layer, attention=True)
        LOGGER.info("%s attention layer %s -> %s", spec.name, layer, layer + 1)
        sae = load_sae(spec, layer, str(device), dtype)
        query_weights, key_weights, norm = head_weights(model, layer + 1)
        normalized = norm(sae.W_dec)
        directory = attention_dir(output, spec, layer)
        directory.mkdir(parents=True, exist_ok=True)
        for head in range(len(query_weights)):
            statistics = attention_statistics(
                normalized @ query_weights[head],
                normalized @ key_weights[head],
                batch_size,
            )
            np.savez_compressed(directory / f"head_{head}.npz", **statistics)
        del sae, normalized
    summarize_attention(output, spec)


def summarize_attention(output: Path, spec: ModelSpec) -> None:
    counts = []
    magnitudes = []
    base = model_dir(output, spec) / "attention"
    for directory in sorted(base.glob("layer_*")):
        layer = int(directory.name.split("_")[-1])
        heads = load_heads(directory)
        counts.extend(specialization_counts(heads, layer))
        for head, values in enumerate(heads):
            magnitudes.append(
                {
                    "layer": layer,
                    "head": head,
                    "pre_softmax_sum_of_means": float(
                        values["pre_query_sum"].sum() / len(values["pre_query_sum"])
                    ),
                }
            )
    counts_frame = pd.DataFrame(counts)
    write_frame(base / "specialization_counts.csv", counts_frame)
    write_frame(base / "pre_softmax_magnitudes.csv", pd.DataFrame(magnitudes))
    rates = counts_frame.groupby(
        ["role", "threshold", "exclude_self"], as_index=False
    ).agg(mean_pass_rate=("any_head_rate", "mean"))
    write_frame(base / "threshold_rates.csv", rates)
