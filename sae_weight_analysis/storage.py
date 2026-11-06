"""Experiment output paths and serialization."""

import json
from pathlib import Path

import pandas as pd

from .config import ModelSpec, Projection


def model_dir(output: Path, spec: ModelSpec) -> Path:
    return output / spec.name


def semantic_dir(output: Path, spec: ModelSpec, projection: Projection) -> Path:
    return model_dir(output, spec) / "semantic" / projection


def semantic_path(
    output: Path, spec: ModelSpec, projection: Projection, layer: int
) -> Path:
    return semantic_dir(output, spec, projection) / f"layer_{layer}.parquet"


def attention_dir(output: Path, spec: ModelSpec, layer: int) -> Path:
    return model_dir(output, spec) / "attention" / f"layer_{layer}"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def load_semantics(
    output: Path, spec: ModelSpec, projection: Projection
) -> dict[int, pd.DataFrame]:
    directory = semantic_dir(output, spec, projection)
    paths = sorted(directory.glob("layer_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No semantic metrics in {directory}")
    return {int(path.stem.split("_")[-1]): pd.read_parquet(path) for path in paths}
