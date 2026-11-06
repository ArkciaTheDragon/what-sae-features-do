"""Semantic feature classification, metric ablation, and threshold sweeps."""

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .config import METRICS, PERCENTILES, ModelSpec, Projection
from .metrics import feature_metrics, token_strings
from .models import load_sae
from .storage import (
    load_semantics,
    semantic_dir,
    semantic_path,
    write_frame,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def thresholds(calibration: pd.DataFrame, percentile: int = 50) -> dict[str, float]:
    quantiles = (percentile / 100, percentile / 100, 1 - percentile / 100)
    return {
        metric: float(calibration[metric].quantile(quantile))
        for metric, quantile in zip(METRICS, quantiles)
    }


def criterion_masks(frame: pd.DataFrame, cutoffs: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            METRICS[0]: frame[METRICS[0]] >= cutoffs[METRICS[0]],
            METRICS[1]: frame[METRICS[1]] >= cutoffs[METRICS[1]],
            METRICS[2]: frame[METRICS[2]] <= cutoffs[METRICS[2]],
        },
        index=frame.index,
    )


def semantic_mask(frame: pd.DataFrame, cutoffs: dict[str, float]) -> pd.Series:
    return criterion_masks(frame, cutoffs).all(axis=1)


def summarize_semantics(output: Path, spec: ModelSpec, projection: Projection) -> None:
    frames = load_semantics(output, spec, projection)
    calibration_layer = spec.calibration(projection)
    calibration = frames[calibration_layer]
    directory = semantic_dir(output, spec, projection)
    baseline = thresholds(calibration)
    write_json(
        directory / "thresholds.json",
        {"calibration_layer": calibration_layer, "thresholds": baseline},
    )
    profiles = []
    subsets = []
    selected_correlations = []
    for layer, frame in sorted(frames.items()):
        masks = criterion_masks(frame, baseline)
        for size in range(1, 4):
            for subset in combinations(METRICS, size):
                subsets.append(
                    {
                        "layer": layer,
                        "metrics": "+".join(subset),
                        "pass_rate": float(masks[list(subset)].all(axis=1).mean()),
                    }
                )
        for percentile in PERCENTILES:
            mask = semantic_mask(frame, thresholds(calibration, percentile))
            profiles.append(
                {
                    "layer": layer,
                    "percentile": percentile,
                    "n_features": len(frame),
                    "n_semantic": int(mask.sum()),
                    "pass_rate": float(mask.mean()),
                }
            )
        candidates = frame.select_dtypes(include="number").drop(
            columns=[
                column
                for column in ("layer", "feature_id", "decoder_norm")
                if column in frame
            ]
        )
        correlation = candidates.corr(method="spearman")
        correlation.to_csv(directory / f"metric_correlations_layer_{layer}.csv")
        for left, right in combinations(METRICS, 2):
            selected_correlations.append(
                {
                    "layer": layer,
                    "left": left,
                    "right": right,
                    "spearman": correlation.loc[left, right],
                }
            )
    profile = pd.DataFrame(profiles)
    base = profile[profile["percentile"] == 50].set_index("layer")["pass_rate"]
    sweep = []
    for percentile, group in profile.groupby("percentile"):
        rates = group.set_index("layer")["pass_rate"].reindex(base.index)
        correlation = (
            float(spearmanr(base, rates).statistic) if len(base) >= 2 else np.nan
        )
        sweep.append(
            {
                "percentile": int(percentile),
                "mean_pass_rate": float(rates.mean()),
                "spearman_vs_p50": correlation,
            }
        )
    sweep_frame = pd.DataFrame(sweep)
    stability = []
    for low, high in ((40, 60), (25, 75)):
        band = sweep_frame[sweep_frame["percentile"].between(low, high)]
        stability.append(
            {
                "percentile_low": low,
                "percentile_high": high,
                "min_spearman_vs_p50": band["spearman_vs_p50"].min(),
            }
        )
    write_frame(directory / "pass_rates.csv", profile[profile["percentile"] == 50])
    write_frame(directory / "metric_subsets.csv", pd.DataFrame(subsets))
    write_frame(
        directory / "selected_metric_correlations.csv",
        pd.DataFrame(selected_correlations),
    )
    write_frame(directory / "threshold_profiles.csv", profile)
    write_frame(directory / "threshold_sweep.csv", sweep_frame)
    write_frame(directory / "threshold_stability.csv", pd.DataFrame(stability))


@torch.inference_mode()
def run_semantic(
    model,
    tokenizer,
    spec: ModelSpec,
    output: Path,
    layers: list[int],
    projections: list[Projection],
    batch_size: int,
) -> None:
    device = next(model.parameters()).device
    embeddings = model.get_input_embeddings().weight.detach()
    vocab_size = min(len(tokenizer), embeddings.shape[0])
    embeddings = embeddings[:vocab_size]
    tokens, valid = token_strings(tokenizer, vocab_size)
    for projection in projections:
        selected_layers = sorted(set(layers) | {spec.calibration(projection)})
        for layer in selected_layers:
            spec.check_layer(layer)
            LOGGER.info("%s %s layer %s", spec.name, projection, layer)
            sae = load_sae(spec, layer, str(device), embeddings.dtype)
            weights = sae.W_dec if projection == "dec_u" else sae.W_enc.T
            basis = (
                model.get_output_embeddings().weight[:vocab_size].detach()
                if projection == "dec_u"
                else embeddings
            )
            rows = []
            for start in range(0, weights.shape[0], batch_size):
                batch = weights[start : start + batch_size]
                projected = model.model.norm(batch) if projection == "dec_u" else batch
                scores = projected @ basis.T
                metrics = feature_metrics(scores, embeddings, tokens, valid)
                for offset, row in enumerate(metrics):
                    feature_id = start + offset
                    rows.append(
                        {
                            "layer": layer,
                            "feature_id": feature_id,
                            "decoder_norm": float(sae.W_dec[feature_id].float().norm()),
                            **row,
                        }
                    )
            write_frame(
                semantic_path(output, spec, projection, layer), pd.DataFrame(rows)
            )
            del sae
        summarize_semantics(output, spec, projection)
