"""Encoder-decoder alignment and normalization rank statistics."""

from pathlib import Path

import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

from .config import ModelSpec
from .models import head_weights, load_sae
from .storage import model_dir, write_frame


@torch.inference_mode()
def run_alignment(model, spec: ModelSpec, output: Path, layers: list[int]) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    rows = []
    for layer in layers:
        sae = load_sae(spec, layer, str(device), dtype)
        similarities = F.cosine_similarity(sae.W_enc.T.float(), sae.W_dec.float())
        rows.append(
            {
                "layer": layer,
                "mean": float(similarities.mean()),
                "median": float(similarities.quantile(0.5)),
                "q25": float(similarities.quantile(0.25)),
                "q75": float(similarities.quantile(0.75)),
            }
        )
        write_frame(
            model_dir(output, spec) / "alignment" / f"layer_{layer}.parquet",
            pd.DataFrame(
                {
                    "feature_id": range(len(similarities)),
                    "cosine_similarity": similarities.cpu().numpy(),
                }
            ),
        )
        del sae
    write_frame(
        model_dir(output, spec) / "alignment" / "summary.csv", pd.DataFrame(rows)
    )


@torch.inference_mode()
def run_normalization(model, spec: ModelSpec, output: Path, layers: list[int]) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    rows = []
    for layer in layers:
        spec.check_layer(layer, attention=True)
        sae = load_sae(spec, layer, str(device), dtype)
        queries, keys, norm = head_weights(model, layer + 1)
        normalized = norm(sae.W_dec)
        for head in range(len(queries)):
            aggregates = []
            for weights in (sae.W_dec, normalized):
                q = (weights @ queries[head]).float()
                k = (weights @ keys[head]).float()
                aggregates.append(
                    (
                        (q @ k.sum(dim=0)).cpu().numpy(),
                        (k @ q.sum(dim=0)).cpu().numpy(),
                    )
                )
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "query_spearman": float(
                        spearmanr(aggregates[0][0], aggregates[1][0]).statistic
                    ),
                    "key_spearman": float(
                        spearmanr(aggregates[0][1], aggregates[1][1]).statistic
                    ),
                }
            )
        del sae, normalized
    write_frame(
        model_dir(output, spec) / "normalization" / "rank_correlations.csv",
        pd.DataFrame(rows),
    )
