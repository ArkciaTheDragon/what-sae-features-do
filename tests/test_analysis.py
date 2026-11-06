import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import Gemma2Config, Gemma2ForCausalLM, LlamaConfig, LlamaForCausalLM

from sae_weight_analysis.attention import attention_statistics, specialization_counts
from sae_weight_analysis.cli import main, parser
from sae_weight_analysis.config import GEMMA_LAYERS, METRICS, MODELS
from sae_weight_analysis.intervention import (
    ForwardPass,
    attention_metrics,
    forward_pass,
    logit_metrics,
    paired_summary,
    prompt_text,
    run_interventions,
    select_matched,
    zero_ablation_hook,
)
from sae_weight_analysis.judge import (
    judge_messages,
    judge_summary,
    judge_tokens,
    run_judge,
)
from sae_weight_analysis.metrics import feature_metrics
from sae_weight_analysis.models import head_weights
from sae_weight_analysis.semantic import semantic_mask, summarize_semantics, thresholds
from sae_weight_analysis.storage import attention_dir, semantic_path, write_frame


class TinySAE(nn.Module):
    def __init__(self, layer=0):
        super().__init__()
        generator = torch.Generator().manual_seed(20 + layer)
        self.W_enc = nn.Parameter(torch.randn(16, 12, generator=generator) / 10)
        self.W_dec = nn.Parameter(torch.randn(12, 16, generator=generator) / 10)

    def encode(self, residual):
        return torch.relu(residual @ self.W_enc + 1)


class TinyTokenizer:
    bos_token = "<bos>"

    def __len__(self):
        return 128

    def decode(self, ids):
        return "".join(f" token{token_id}" for token_id in ids)

    def convert_ids_to_tokens(self, token_id):
        return f"token{token_id}"


def tiny_model(architecture="llama"):
    torch.manual_seed(7)
    config_type, model_type = (
        (LlamaConfig, LlamaForCausalLM)
        if architecture == "llama"
        else (Gemma2Config, Gemma2ForCausalLM)
    )
    config = config_type(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        attention_dropout=0,
    )
    config._attn_implementation = "eager"
    return model_type(config).eval()


class SettingsTests(unittest.TestCase):
    def test_paper_layer_counts(self):
        specs = [MODELS["gemma-2-9b"], MODELS["llama-3.1-8b"]]
        self.assertEqual(sum(len(spec.semantic_layers) for spec in specs), 16)
        self.assertEqual(sum(len(spec.query_layers) for spec in specs), 18)
        self.assertEqual(sum(len(spec.key_layers) for spec in specs), 17)
        self.assertEqual(
            MODELS["gemma-2-9b"].sae_id(27), "layer_27/width_16k/canonical"
        )
        self.assertEqual(MODELS["llama-3.1-8b"].sae_id(20), "l20r_8x")
        self.assertEqual(GEMMA_LAYERS, (5, 10, 15, 20, 25, 30, 35, 40))

    def test_cli_defaults(self):
        args = parser().parse_args(["judge", "--model", "llama-3.1-8b"])
        self.assertEqual(args.n_per_group, 1000)
        self.assertEqual(args.projection, "both")
        args = parser().parse_args(
            ["intervene", "--model", "gemma-2-9b", "--role", "semantic"]
        )
        self.assertEqual(args.n_features, 100)
        with self.assertRaises(ValueError):
            MODELS["llama-3.1-8b"].check_layer(31, attention=True)


class MetricTests(unittest.TestCase):
    def test_all_metrics_and_projection_statistics(self):
        torch.manual_seed(3)
        scores = torch.randn(2, 128) + 5
        embeddings = torch.randn(128, 16)
        tokens = [f" token{i}" for i in range(128)]
        valid = torch.ones(128, dtype=torch.bool)
        valid[127] = False
        rows = feature_metrics(scores, embeddings, tokens, valid)
        self.assertEqual(len(rows), 2)
        numeric = [name for name, value in rows[0].items() if isinstance(value, float)]
        self.assertEqual(len(numeric), 23)
        self.assertNotIn(127, rows[0]["top_k_token_ids"])
        top, ids = scores[0].masked_fill(~valid, -torch.inf).topk(100)
        p = top.softmax(dim=0)
        self.assertAlmostEqual(
            rows[0]["entropy_top100"], float(-(p * p.log()).sum()), places=5
        )
        normalized = torch.nn.functional.normalize(embeddings[ids[:10]], dim=-1)
        pairwise = normalized @ normalized.T
        expected = pairwise[~torch.eye(10, dtype=torch.bool)].mean()
        self.assertAlmostEqual(rows[0]["cosine_similarity"], float(expected), places=6)
        self.assertAlmostEqual(
            rows[0]["l2_norm"],
            float(scores[0, valid].norm() / valid.sum().sqrt()),
            places=5,
        )

    def test_thresholds(self):
        frame = pd.DataFrame(
            {
                METRICS[0]: [0.1, 0.3, 0.5, 0.7],
                METRICS[1]: [0.2, 0.4, 0.6, 0.8],
                METRICS[2]: [4.0, 3.0, 2.0, 1.0],
            }
        )
        cutoff = thresholds(frame)
        np.testing.assert_array_equal(
            semantic_mask(frame, cutoff), [False, False, True, True]
        )
        self.assertGreater(thresholds(frame, 75)[METRICS[0]], cutoff[METRICS[0]])
        self.assertLess(thresholds(frame, 75)[METRICS[2]], cutoff[METRICS[2]])


class AttentionTests(unittest.TestCase):
    def test_chunked_statistics_match_dense(self):
        torch.manual_seed(11)
        q = torch.randn(13, 4)
        k = torch.randn(13, 4)
        pre = q @ k.T / 2
        post = pre.softmax(dim=-1)
        for batch_size in (1, 4, 20):
            with self.subTest(batch_size=batch_size):
                result = attention_statistics(q, k, batch_size)
                np.testing.assert_allclose(
                    result["pre_query_sum"], pre.sum(dim=1), atol=1e-6
                )
                np.testing.assert_allclose(
                    result["pre_key_sum"], pre.sum(dim=0), atol=1e-6
                )
                for stage, matrix in (("pre", pre), ("post", post)):
                    for suffix in ("", "_no_self"):
                        query = matrix.topk(10, dim=1)
                        key = matrix.topk(10, dim=0)
                        query_values = query.values.clone()
                        key_values = key.values.T.clone()
                        if suffix:
                            query_values[
                                (query.indices == torch.arange(13)[:, None]).any(dim=1)
                            ] = 0
                            key_values[
                                (key.indices.T == torch.arange(13)[:, None]).any(dim=1)
                            ] = 0
                        np.testing.assert_allclose(
                            result[f"{stage}_query_topk{suffix}"],
                            query_values,
                            atol=1e-6,
                        )
                        np.testing.assert_allclose(
                            result[f"{stage}_key_topk{suffix}"],
                            key_values,
                            atol=1e-6,
                        )
        rows = specialization_counts([result, result], 2)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(0 <= row["any_head_rate"] <= 1 for row in rows))

    def test_grouped_query_head_mapping(self):
        model = tiny_model()
        queries, keys, norm = head_weights(model, 1)
        self.assertEqual(tuple(queries.shape), (4, 16, 4))
        torch.testing.assert_close(keys[0], keys[1])
        torch.testing.assert_close(keys[2], keys[3])
        self.assertIs(norm, model.model.layers[1].input_layernorm)

    def test_head_aggregated_scores_are_thresholded(self):
        heads = []
        for values in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]):
            weights = np.zeros((3, 10))
            weights[:, 0] = values
            heads.append(
                {
                    f"post_{role}_topk{suffix}": weights
                    for role in ("query", "key")
                    for suffix in ("", "_no_self")
                }
            )
        row = next(
            row
            for row in specialization_counts(heads, 0)
            if row["role"] == "query"
            and row["threshold"] == 0.95
            and not row["exclude_self"]
        )
        self.assertEqual(row["mean_head_count"], 0)
        self.assertEqual(row["median_head_count"], 1)
        self.assertEqual(row["any_head_count"], 2)


class InterventionTests(unittest.TestCase):
    def test_single_position_ablation(self):
        sae = TinySAE()
        residual = torch.ones(1, 4, 16)
        hook = zero_ablation_hook(sae, 2, 1)
        result = hook(None, None, (residual, "metadata"))
        expected = residual.clone()
        expected[:, 1] -= sae.encode(residual)[:, 1, 2, None] * sae.W_dec[2]
        torch.testing.assert_close(result[0], expected)
        torch.testing.assert_close(result[0][:, 0], residual[:, 0])
        torch.testing.assert_close(result[0][:, 2:], residual[:, 2:])
        self.assertEqual(result[1], "metadata")

    def test_native_model_intervention(self):
        for architecture in ("llama", "gemma"):
            with self.subTest(architecture=architecture):
                model = tiny_model(architecture)
                sae = TinySAE()
                tokens = torch.tensor([[1, 7, 9, 12, 16, 18, 19, 20]])
                clean = forward_pass(model, sae, tokens, 0, head=1)
                intervened = forward_pass(
                    model, sae, tokens, 0, head=1, feature_id=2, position=2
                )
                self.assertEqual(tuple(clean.activations.shape), (8, 12))
                self.assertEqual(tuple(clean.attention.shape), (8, 8))
                self.assertFalse(torch.allclose(clean.logits, intervened.logits))
                metrics = logit_metrics(clean.logits, intervened.logits, 7)
                self.assertGreaterEqual(metrics["vocab_kl"], -1e-12)
                self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)
                query = attention_metrics(
                    clean.attention, intervened.attention, "query", 2
                )
                self.assertGreaterEqual(query["attention_kl"], -1e-10)
                key = attention_metrics(clean.attention, intervened.attention, "key", 2)
                self.assertEqual(key["n_later_queries"], 5)

    def test_hook_cleanup(self):
        model = tiny_model()
        sae = TinySAE()
        tokens = torch.tensor([[1, 7, 9]])
        with self.assertRaises(ValueError):
            forward_pass(model, sae, tokens, 0, feature_id=2)
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)
        with self.assertRaises(ValueError):
            forward_pass(model, sae, tokens, 0, feature_id=2, position=9)
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)

    def test_matched_sampling_and_pair_summary(self):
        frame = pd.DataFrame(
            {"feature_id": [0, 1, 2], "decoder_norm": [1.0, 1.04, 1.2]}
        )
        selected = select_matched(
            frame,
            pd.Series([False, True, True]),
            0,
            torch.tensor([3.0, 2.0, 5.0]),
            set(),
            42,
        )
        self.assertEqual(selected, 1)
        with self.assertRaises(ValueError):
            select_matched(
                frame,
                pd.Series([False, True, True]),
                0,
                torch.tensor([3.0, 0.0, 5.0]),
                set(),
                42,
            )
        pairs = pd.DataFrame(
            {
                "pair_id": [0, 0, 1, 1, 2],
                "group": ["flagged", "matched", "flagged", "matched", "flagged"],
                "logit_drop": [10.0, 2.0, -6.0, 3.0, 100.0],
                "vocab_kl": [0.2, 0.1, 0.4, 0.1, 10.0],
            }
        )
        summary = paired_summary(pairs, "semantic")
        self.assertEqual(summary["median_pair_ratio"], 3.5)
        self.assertEqual(summary["win_rate"], 1.0)
        self.assertEqual(summary["n_pairs"], 2)
        self.assertEqual(summary["flagged_median"], 10.0)
        self.assertAlmostEqual(summary["flagged_kl_median"], 0.3)

    def test_prompt_cache(self):
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "activations": [{"tokens": ["<bos>", "\u2581hello", "\u2581world"]}]
                },
            )

        with tempfile.TemporaryDirectory() as temporary:
            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                for _ in range(2):
                    result = prompt_text(
                        client,
                        Path(temporary),
                        MODELS["gemma-2-9b"],
                        5,
                        3,
                        TinyTokenizer(),
                    )
                    self.assertEqual(result, " hello world")
            self.assertEqual(len(calls), 1)


class JudgeTests(unittest.TestCase):
    def test_prompt_and_metrics(self):
        messages = judge_messages([" first", "second", "third", "fourth", "fifth"])
        self.assertEqual(len(messages), 14)
        self.assertIn('" first"', messages[-1]["content"])
        self.assertNotIn("semantic", messages[-1]["content"])
        frame = pd.DataFrame(
            {
                "label": ["semantic", "semantic", "nonsemantic", "nonsemantic"],
                "verdict": ["yes", "yes", "yes", "no"],
            }
        )
        result = judge_summary(frame)
        self.assertEqual(result["agreement"], 0.75)
        self.assertEqual(result["cohen_kappa"], 0.5)

    def test_response_validation(self):
        async def run(verdict):
            def respond(request):
                payload = json.loads(request.content)
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["guided_choice"], ["yes", "no"])
                return httpx.Response(
                    200, json={"choices": [{"message": {"content": verdict}}]}
                )

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(respond)
            ) as client:
                return await judge_tokens(
                    client,
                    asyncio.Semaphore(1),
                    "http://judge/v1",
                    "judge",
                    [" token"],
                    42,
                )

        self.assertEqual(asyncio.run(run("yes")), "yes")
        with self.assertRaises(ValueError):
            asyncio.run(run("maybe"))


class PipelineTests(unittest.TestCase):
    def test_attention_intervention_pipelines(self):
        class PromptTokenizer:
            def __call__(self, text, return_tensors):
                return {"input_ids": torch.arange(10).unsqueeze(0)}

        model = tiny_model()
        sae = TinySAE()
        frame = pd.DataFrame(
            {
                "feature_id": range(12),
                "decoder_norm": np.ones(12),
                METRICS[0]: [0.1] * 6 + [0.9] * 6,
                METRICS[1]: [0.1] * 6 + [0.9] * 6,
                METRICS[2]: [4.0] * 6 + [0.0] * 6,
            }
        )
        query = np.full((12, 10), 0.02)
        key = np.full((12, 10), 0.01)
        query[6:] = 0.1
        key[6:] = 0.3
        heads = [{"post_query_topk": query, "post_key_topk": key}]
        spec = MODELS["gemma-2-9b"]
        for role in ("query", "key"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for layer in (0, spec.decoder_calibration):
                    write_frame(semantic_path(root, spec, "dec_u", layer), frame)

                def passes(
                    model,
                    sae,
                    tokens,
                    layer,
                    *,
                    head=None,
                    feature_id=None,
                    position=None,
                ):
                    size = tokens.shape[1]
                    activations = torch.ones(size, 12)
                    activations[4] = 3
                    attention = torch.ones(size, size).tril()
                    attention /= attention.sum(dim=1, keepdim=True)
                    if feature_id is not None:
                        delta = 0.05 if feature_id >= 6 else 0.005
                        if role == "query":
                            attention[position, 0] -= delta
                            attention[position, 1] += delta
                        else:
                            attention[position + 1 :, position] -= delta
                            attention[position + 1 :, 0] += delta
                    return ForwardPass(torch.zeros(128), activations, attention)

                with (
                    patch(
                        "sae_weight_analysis.intervention.load_sae", return_value=sae
                    ),
                    patch(
                        "sae_weight_analysis.intervention.load_heads",
                        return_value=heads,
                    ),
                    patch(
                        "sae_weight_analysis.intervention.prompt_text",
                        return_value=" text",
                    ),
                    patch(
                        "sae_weight_analysis.intervention.forward_pass",
                        side_effect=passes,
                    ),
                ):
                    run_interventions(
                        model,
                        PromptTokenizer(),
                        spec,
                        root,
                        root / "cache",
                        [0],
                        role,
                        1,
                        42,
                    )
                directory = root / spec.name / "intervention" / role
                results = pd.read_parquet(directory / "layer_0.parquet")
                self.assertEqual(len(results), 2)
                summary = pd.read_csv(directory / "summary.csv")
                self.assertEqual(float(summary["win_rate"].iloc[0]), 1.0)

    def test_judge_pipeline(self):
        spec = MODELS["gemma-2-9b"]
        frame = pd.DataFrame(
            {
                "feature_id": [0, 1, 2, 3],
                METRICS[0]: [0.1, 0.2, 0.8, 0.9],
                METRICS[1]: [0.1, 0.2, 0.8, 0.9],
                METRICS[2]: [4.0, 3.0, 1.0, 0.0],
                "top_k_tokens": [[" token"] * 5] * 4,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for layer in (5, spec.decoder_calibration):
                write_frame(semantic_path(root, spec, "dec_u", layer), frame)
            with patch("sae_weight_analysis.judge.judge_tokens", return_value="yes"):
                asyncio.run(
                    run_judge(
                        root, spec, ["dec_u"], [5], "http://judge/v1", "judge", 1, 1, 42
                    )
                )
            result = pd.read_parquet(root / spec.name / "judge/dec_u/layer_5.parquet")
            self.assertEqual(len(result), 2)
            summary = pd.read_csv(root / spec.name / "judge/summary.csv")
            self.assertEqual(float(summary["agreement"].iloc[0]), 0.5)

    def test_judge_available_groups(self):
        spec = MODELS["llama-3.1-8b"]
        calibration = pd.DataFrame(
            {
                "feature_id": [0, 1, 2, 3],
                METRICS[0]: [0.1, 0.2, 0.8, 0.9],
                METRICS[1]: [0.1, 0.2, 0.8, 0.9],
                METRICS[2]: [4.0, 3.0, 1.0, 0.0],
                "top_k_tokens": [[" token"] * 5] * 4,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_frame(
                semantic_path(root, spec, "enc_e", spec.encoder_calibration),
                calibration,
            )
            frame = calibration.iloc[[0, 1, 3]]
            write_frame(semantic_path(root, spec, "enc_e", 10), frame)
            with patch(
                "sae_weight_analysis.judge.judge_tokens", return_value="yes"
            ) as judge:
                asyncio.run(
                    run_judge(
                        root,
                        spec,
                        ["enc_e"],
                        [10],
                        "http://judge/v1",
                        "judge",
                        1000,
                        1,
                        42,
                    )
                )
            result = pd.read_parquet(root / spec.name / "judge/enc_e/layer_10.parquet")
            self.assertEqual(len(result), 3)
            self.assertEqual(set(result["feature_id"]), {0, 1, 3})
            self.assertEqual(judge.call_count, 3)
            per_layer = pd.read_csv(root / spec.name / "judge/per_layer.csv")
            self.assertEqual(per_layer["n_semantic"].item(), 1)
            self.assertEqual(per_layer["n_nonsemantic"].item(), 2)

    def test_judge_empty_group(self):
        spec = MODELS["gemma-2-9b"]
        frame = pd.DataFrame(
            {
                "feature_id": [0, 1],
                METRICS[0]: [0.1, 0.9],
                METRICS[1]: [0.1, 0.9],
                METRICS[2]: [4.0, 0.0],
                "top_k_tokens": [[" token"] * 5] * 2,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_frame(
                semantic_path(root, spec, "dec_u", spec.decoder_calibration), frame
            )
            write_frame(semantic_path(root, spec, "dec_u", 5), frame.iloc[[1]])
            with patch("sae_weight_analysis.judge.judge_tokens") as judge:
                with self.assertRaisesRegex(ValueError, "no nonsemantic features"):
                    asyncio.run(
                        run_judge(
                            root,
                            spec,
                            ["dec_u"],
                            [5],
                            "http://judge/v1",
                            "judge",
                            1000,
                            1,
                            42,
                        )
                    )
                judge.assert_not_called()

    def test_intervention_pipeline(self):
        class PromptTokenizer:
            def __call__(self, text, return_tensors):
                return {"input_ids": torch.tensor([[1, 7, 9, 12, 16, 18, 19, 20]])}

        model = tiny_model()
        sae = TinySAE()
        with torch.no_grad():
            sae.W_enc.fill_(0.1)
            sae.W_dec.div_(sae.W_dec.norm(dim=1, keepdim=True))
        frame = pd.DataFrame(
            {
                "feature_id": range(12),
                "decoder_norm": sae.W_dec.detach().norm(dim=1).numpy(),
                METRICS[0]: [0.1] * 6 + [0.9] * 6,
                METRICS[1]: [0.1] * 6 + [0.9] * 6,
                METRICS[2]: [4.0] * 6 + [0.0] * 6,
                "top_k_token_ids": [[7, 9, 12, 16, 18]] * 12,
            }
        )
        spec = MODELS["gemma-2-9b"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for layer in (0, spec.decoder_calibration):
                write_frame(semantic_path(root, spec, "dec_u", layer), frame)
            with (
                patch("sae_weight_analysis.intervention.load_sae", return_value=sae),
                patch(
                    "sae_weight_analysis.intervention.prompt_text", return_value=" text"
                ),
            ):
                run_interventions(
                    model,
                    PromptTokenizer(),
                    spec,
                    root,
                    root / "cache",
                    [0],
                    "semantic",
                    1,
                    42,
                )
            result = pd.read_parquet(
                root / spec.name / "intervention/semantic/layer_0.parquet"
            )
            self.assertEqual(set(result["group"]), {"flagged", "matched"})
            self.assertEqual(len(result), 2)

    def test_cli_computation_pipeline(self):
        model = tiny_model()
        spec = replace(
            MODELS["gemma-2-9b"],
            num_layers=3,
            decoder_calibration=1,
            encoder_calibration=0,
        )

        def load(spec, layer, device, dtype):
            return TinySAE(layer).to(device=device, dtype=dtype).eval()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = [
                "--model",
                spec.name,
                "--output",
                str(root),
                "--layers",
                "0",
                "1",
            ]
            with (
                patch.dict(MODELS, {spec.name: spec}),
                patch("sae_weight_analysis.models.load_model", return_value=model),
                patch(
                    "sae_weight_analysis.models.load_tokenizer",
                    return_value=TinyTokenizer(),
                ),
                patch("sae_weight_analysis.semantic.load_sae", side_effect=load),
                patch("sae_weight_analysis.attention.load_sae", side_effect=load),
                patch("sae_weight_analysis.validation.load_sae", side_effect=load),
            ):
                main(["semantic", *arguments, "--device", "cpu", "--batch-size", "4"])
                main(["attention", *arguments, "--device", "cpu", "--batch-size", "4"])
                main(["compare", *arguments, "--projection", "dec_u"])
                main(["compare", *arguments, "--projection", "enc_e"])
                main(["alignment", *arguments, "--device", "cpu"])
                main(["normalization", *arguments, "--device", "cpu"])
            base = root / spec.name
            semantic = pd.read_parquet(semantic_path(root, spec, "dec_u", 0))
            self.assertEqual(len(semantic), 12)
            self.assertTrue((base / "semantic/dec_u/threshold_sweep.csv").exists())
            self.assertTrue((base / "attention/threshold_rates.csv").exists())
            self.assertTrue((base / "comparison/dec_u/features.parquet").exists())
            self.assertTrue((base / "comparison/enc_e/features.parquet").exists())
            compared = pd.read_parquet(base / "comparison/dec_u/features.parquet")
            expected = []
            for head in range(4):
                with np.load(attention_dir(root, spec, 0) / f"head_{head}.npz") as data:
                    expected.append(data["pre_query_topk"].sum(axis=1))
            np.testing.assert_allclose(
                compared.loc[compared["layer"] == 0, "query_weight"],
                np.stack(expected).sum(axis=0),
            )
            cutoff = json.loads((base / "comparison/dec_u/global_p75.json").read_text())
            self.assertAlmostEqual(
                cutoff["query"], float(compared["query_weight"].quantile(0.75))
            )
            self.assertTrue((base / "alignment/summary.csv").exists())
            self.assertTrue((base / "normalization/rank_correlations.csv").exists())
            self.assertTrue((attention_dir(root, spec, 0) / "head_3.npz").exists())

    def test_summary_sweep_shape(self):
        spec = MODELS["gemma-2-9b"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for layer in (5, 10, spec.decoder_calibration):
                values = np.linspace(0, 1, 100)
                offset = layer / 100
                frame = pd.DataFrame(
                    {
                        "layer": layer,
                        "feature_id": range(100),
                        "decoder_norm": np.ones(100),
                        METRICS[0]: values + offset,
                        METRICS[1]: values + offset,
                        METRICS[2]: 1 - values - offset,
                    }
                )
                write_frame(semantic_path(root, spec, "dec_u", layer), frame)
            summarize_semantics(root, spec, "dec_u")
            directory = root / spec.name / "semantic/dec_u"
            profiles = pd.read_csv(directory / "threshold_profiles.csv")
            subsets = pd.read_csv(directory / "metric_subsets.csv")
            self.assertEqual(len(profiles), 33)
            self.assertEqual(len(subsets), 21)
            self.assertEqual(len(pd.read_csv(directory / "threshold_sweep.csv")), 11)


if __name__ == "__main__":
    unittest.main()
