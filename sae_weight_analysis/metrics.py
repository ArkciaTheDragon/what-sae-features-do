"""Token-level semantic metrics."""

import logging
import re
from collections import Counter
from itertools import combinations

import Levenshtein
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)
ENTROPY_SCALES = (10, 25, 50, 100)


def ratio(numerator: float, denominator: float, name: str) -> float:
    if denominator == 0:
        LOGGER.warning("%s has a zero denominator", name)
        return float("nan")
    return float(numerator / denominator)


def token_strings(tokenizer, vocab_size: int) -> tuple[list[str], Tensor]:
    tokens = []
    valid = []
    for token_id in range(vocab_size):
        decoded = tokenizer.decode([token_id])
        token = (
            tokenizer.convert_ids_to_tokens(token_id)
            if "\ufffd" in decoded
            else decoded
        )
        tokens.append(token)
        valid.append(
            re.fullmatch(r"<unused\d+>|<\|reserved_special_token_\d+\|>", decoded)
            is None
        )
    return tokens, torch.tensor(valid, dtype=torch.bool)


def case_pattern(token: str) -> str:
    if token.islower():
        return "lower"
    if token.isupper():
        return "upper"
    if token.istitle():
        return "title"
    return "mixed"


def lexical_metrics(tokens: list[str]) -> dict[str, float]:
    pairs = [
        1 - Levenshtein.distance(left, right) / max(len(left), len(right))
        for left, right in combinations(tokens, 2)
        if left or right
    ]
    lengths = np.array([len(token) for token in tokens])
    return {
        "levenshtein_similarity": float(np.mean(pairs)),
        "overlap_score": 1
        - len({re.sub(r"[^\w]", "", token.lower().strip()) for token in tokens})
        / len(tokens),
        "token_length_mean": float(lengths.mean()),
        "token_length_std": float(lengths.std()),
        "token_length_max": float(lengths.max()),
        "prefix_diversity": len({token[:3] for token in tokens}) / len(tokens),
        "suffix_diversity": len({token[-3:] for token in tokens}) / len(tokens),
        "case_consistency": max(Counter(map(case_pattern, tokens)).values())
        / len(tokens),
        "whitespace_pattern": sum(
            bool(token) and token[0].isspace() for token in tokens
        )
        / len(tokens),
    }


def entropy(scores: Tensor) -> Tensor:
    log_probabilities = scores.log_softmax(dim=-1)
    return -(log_probabilities.exp() * log_probabilities).sum(dim=-1)


@torch.inference_mode()
def feature_metrics(
    scores: Tensor,
    embeddings: Tensor,
    tokens: list[str],
    valid: Tensor,
) -> list[dict]:
    valid = valid.to(scores.device)
    if int(valid.sum()) < 100:
        raise ValueError("Semantic metrics require at least 100 valid tokens")
    scores = scores.float()
    top_scores, top_ids = scores.masked_fill(~valid, -torch.inf).topk(100, dim=-1)
    normalized = F.normalize(embeddings[top_ids[:, :10]].float(), dim=-1)
    similarities = normalized @ normalized.transpose(-1, -2)
    cosine = (
        similarities.sum(dim=(-1, -2))
        - similarities.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    ) / 90
    valid_scores = scores[:, valid]
    entropies = {
        f"entropy_top{k}": entropy(top_scores[:, :k]).cpu().numpy()
        for k in ENTROPY_SCALES
    }
    entropies["entropy_full"] = entropy(valid_scores).cpu().numpy()
    positive_mass = valid_scores.clamp_min(0).sum(dim=-1).cpu().numpy()
    l2 = (valid_scores.norm(dim=-1) / valid_scores.shape[-1] ** 0.5).cpu().numpy()
    top_scores = top_scores.cpu().numpy()
    top_ids = top_ids.cpu().numpy()
    cosine = cosine.cpu().numpy()
    rows = []
    for index, (values, ids) in enumerate(zip(top_scores, top_ids)):
        text = [tokens[int(token_id)] for token_id in ids[:10]]
        ascending = np.sort(values)
        row = {
            **lexical_metrics(text),
            **{name: float(values_[index]) for name, values_ in entropies.items()},
            "cosine_similarity": float(cosine[index]),
            "topk_cv": ratio(values[:10].std(), values[:10].mean(), "topk_cv"),
            "max_min_ratio": ratio(values[0], values[9], "max_min_ratio"),
            "gini_coefficient": ratio(
                2 * np.dot(np.arange(1, 101), ascending),
                100 * ascending.sum(),
                "gini_coefficient",
            )
            - 101 / 100,
            "l2_norm": float(l2[index]),
            "top_k_token_ids": ids[:10].tolist(),
            "top_k_tokens": text,
            "top_k_scores": values[:10].tolist(),
        }
        for k in ENTROPY_SCALES:
            row[f"mass_concentration_top{k}"] = ratio(
                values[:k].sum(), positive_mass[index], "mass_concentration"
            )
        rows.append(row)
    return rows
