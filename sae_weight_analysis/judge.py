"""LLM-judge token coherence experiment."""

import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .config import ModelSpec, Projection
from .semantic import criterion_masks, thresholds
from .storage import model_dir, semantic_path, write_frame

LOGGER = logging.getLogger(__name__)
SYSTEM = (
    "You are a careful linguistic analyst. You will be given a list of tokens "
    "(words, subwords, or symbols) drawn from a language model's vocabulary. "
    "Decide whether the tokens collectively point to one coherent meaning. "
    "A coherent meaning includes:\n"
    " - one word and its morphological variants (e.g., 'run', 'runs', 'running');\n"
    " - the same word across casing or scripts ('Cat', 'CAT', 'cat');\n"
    " - the same word across languages ('cat', 'gato', 'chat', '\u732b');\n"
    " - a tight semantic family (e.g., kitchen appliances, planets, basketball terms).\n"
    "It is NOT coherent when the list is dominated by:\n"
    " - random unrelated words across topics;\n"
    " - generic high-frequency vocabulary ('the', 'and', 'a');\n"
    " - punctuation, code symbols, or fragments with no shared meaning;\n"
    " - tokens that only share a few letters but mean different things.\n"
    "Answer only with 'yes' or 'no'."
)
FEW_SHOT = (
    ([" Pull", "Pull", " PULL", " pull", " pulled"], "yes"),
    ([" Alaska", "Alaska", " Alaskan", " Anchorage", " Juneau"], "yes"),
    ([" red", "red", " blue", " green", " yellow"], "yes"),
    ([" the", "-", " and", ".", " a"], "no"),
    (["keur", "\u00e9u", " protoimpl", "StoreMessageInfo", "neo"], "no"),
    ([" auf", " Bedeutung", " carrots", " ng\u00e0y", " \u2618"], "no"),
)


def token_message(tokens: list[str]) -> str:
    return (
        "Tokens: "
        + ", ".join(json.dumps(token, ensure_ascii=False) for token in tokens)
        + "\n\nDo these tokens collectively express one coherent meaning? "
        "Answer 'yes' or 'no'."
    )


def judge_messages(tokens: list[str]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM}]
    for example, verdict in FEW_SHOT:
        messages.extend(
            [
                {"role": "user", "content": token_message(example)},
                {"role": "assistant", "content": verdict},
            ]
        )
    messages.append({"role": "user", "content": token_message(tokens[:5])})
    return messages


async def judge_tokens(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    judge_model: str,
    tokens: list[str],
    seed: int,
) -> str:
    payload = {
        "model": judge_model,
        "messages": judge_messages(tokens),
        "temperature": 0.0,
        "max_tokens": 3,
        "seed": seed,
        "guided_choice": ["yes", "no"],
    }
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.post(
                    endpoint.rstrip("/") + "/chat/completions", json=payload
                )
                response.raise_for_status()
                verdict = (
                    response.json()["choices"][0]["message"]["content"].strip().lower()
                )
                if verdict not in ("yes", "no"):
                    raise ValueError(f"Unexpected judge verdict: {verdict!r}")
                return verdict
            except httpx.HTTPStatusError as error:
                if (
                    error.response.status_code < 500
                    and error.response.status_code != 429
                ):
                    raise
                if attempt == 2:
                    raise
                LOGGER.warning(
                    "Retrying judge after HTTP %s", error.response.status_code
                )
            except httpx.TransportError:
                if attempt == 2:
                    raise
                LOGGER.warning("Retrying judge after a transport error", exc_info=True)
            await asyncio.sleep(attempt + 1)
    raise RuntimeError("Judge request did not complete")


def judge_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["label"].eq("semantic")
    prediction = frame["verdict"].eq("yes")
    n = len(frame)
    positives = int(truth.sum())
    negatives = n - positives
    if not positives or not negatives:
        raise ValueError("Judge summary requires both sample groups")
    tp = int((truth & prediction).sum())
    tn = int((~truth & ~prediction).sum())
    agreement = (tp + tn) / n
    predicted_positive = float(prediction.mean())
    expected = (positives / n) * predicted_positive + (negatives / n) * (
        1 - predicted_positive
    )
    return {
        "n_semantic": positives,
        "n_nonsemantic": negatives,
        "sensitivity": tp / positives,
        "specificity": tn / negatives,
        "agreement": agreement,
        "cohen_kappa": (agreement - expected) / (1 - expected),
    }


async def run_judge(
    output: Path,
    spec: ModelSpec,
    projections: list[Projection],
    layers: list[int] | None,
    endpoint: str,
    judge_model: str,
    n_per_group: int,
    concurrency: int,
    seed: int,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    api_key = os.environ.get("JUDGE_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    summaries = []
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        for projection in projections:
            cutoffs = thresholds(
                pd.read_parquet(
                    semantic_path(
                        output, spec, projection, spec.calibration(projection)
                    )
                )
            )
            for layer in (
                layers if layers is not None else spec.judge_layers(projection)
            ):
                frame = pd.read_parquet(semantic_path(output, spec, projection, layer))
                masks = criterion_masks(frame, cutoffs)
                groups = (
                    ("semantic", frame[masks.all(axis=1)]),
                    ("nonsemantic", frame[(~masks).all(axis=1)]),
                )
                rng = np.random.default_rng(seed)
                selected = []
                for label, group in groups:
                    if group.empty:
                        raise ValueError(f"Layer {layer} has no {label} features")
                    count = min(n_per_group, len(group))
                    LOGGER.info(
                        "%s %s layer %s %s: sampling %s of %s features",
                        spec.name,
                        projection,
                        layer,
                        label,
                        count,
                        len(group),
                    )
                    indices = rng.choice(len(group), size=count, replace=False)
                    for _, feature in group.iloc[indices].iterrows():
                        selected.append(
                            {
                                "model": spec.name,
                                "projection": projection,
                                "layer": layer,
                                "feature_id": int(feature["feature_id"]),
                                "label": label,
                                "tokens": list(feature["top_k_tokens"][:5]),
                                "seed": seed,
                                "judge_model": judge_model,
                            }
                        )
                verdicts = await asyncio.gather(
                    *(
                        judge_tokens(
                            client,
                            semaphore,
                            endpoint,
                            judge_model,
                            row["tokens"],
                            seed,
                        )
                        for row in selected
                    )
                )
                results = pd.DataFrame(
                    [
                        {**row, "verdict": verdict}
                        for row, verdict in zip(selected, verdicts)
                    ]
                )
                write_frame(
                    model_dir(output, spec)
                    / "judge"
                    / projection
                    / f"layer_{layer}.parquet",
                    results,
                )
                summaries.append(
                    {"projection": projection, "layer": layer, **judge_summary(results)}
                )
    summary = pd.DataFrame(summaries)
    directory = model_dir(output, spec) / "judge"
    write_frame(directory / "per_layer.csv", summary)
    write_frame(
        directory / "summary.csv",
        summary.drop(columns="layer").groupby("projection", as_index=False).mean(),
    )
