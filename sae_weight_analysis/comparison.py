"""Semantic and computational specialization across layers."""

from pathlib import Path

import numpy as np
import pandas as pd

from .attention import load_heads
from .config import ModelSpec, Projection
from .semantic import semantic_mask, thresholds
from .storage import (
    attention_dir,
    model_dir,
    semantic_path,
    write_frame,
    write_json,
)


def run_comparison(
    output: Path, spec: ModelSpec, projection: Projection, layers: list[int] | None
) -> None:
    calibration = pd.read_parquet(
        semantic_path(output, spec, projection, spec.calibration(projection))
    )
    cutoffs = thresholds(calibration)
    if layers is None:
        layers = sorted(
            int(path.name.split("_")[-1])
            for path in (model_dir(output, spec) / "attention").glob("layer_*")
        )
    if spec.name == "llama-3.1-8b":
        layers = [
            layer
            for layer in layers
            if (layer < 11 if projection == "enc_e" else layer >= 11)
        ]
    if not layers:
        raise ValueError("No layers selected for semantic-computational comparison")
    frames = []
    for layer in layers:
        metrics = pd.read_parquet(semantic_path(output, spec, projection, layer))
        metrics = metrics.sort_values("feature_id").reset_index(drop=True)
        heads = load_heads(attention_dir(output, spec, layer))
        frame = metrics[["layer", "feature_id"]].copy()
        frame["semantic"] = semantic_mask(metrics, cutoffs)
        for role in ("query", "key"):
            weights = np.stack([head[f"pre_{role}_topk"].sum(axis=1) for head in heads])
            if weights.shape[1] != len(metrics):
                raise ValueError(f"Feature count differs at layer {layer}")
            frame[f"{role}_weight"] = weights.sum(axis=0)
        frames.append(frame)
    features = pd.concat(frames, ignore_index=True)
    summaries = []
    p75 = {}
    for role in ("query", "key"):
        column = f"{role}_weight"
        cutoff = float(features[column].quantile(0.75))
        p75[role] = cutoff
        for (layer, semantic), group in features.groupby(["layer", "semantic"]):
            values = group[column]
            summaries.append(
                {
                    "layer": int(layer),
                    "semantic": bool(semantic),
                    "role": role,
                    "n_features": len(group),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "above_global_p75": float((values > cutoff).mean()),
                }
            )
    directory = model_dir(output, spec) / "comparison" / projection
    write_frame(directory / "features.parquet", features)
    write_frame(directory / "summary.csv", pd.DataFrame(summaries))
    write_json(directory / "global_p75.json", p75)
