"""Command-line entry points for the paper experiments."""

import argparse
import asyncio
import logging
from pathlib import Path

from .config import GEMMA_LAYERS, MODELS, SEED


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Weight-based SAE feature analysis")
    commands = root.add_subparsers(dest="command", required=True)
    descriptions = {
        "semantic": "Compute token projections, semantic metrics, and threshold sweeps",
        "attention": "Compute feature QK statistics and specialization counts",
        "compare": "Compare semantic and computational specialization",
        "summarize": "Aggregate saved semantic metrics",
        "judge": "Run the token coherence judge",
        "intervene": "Run decoder-write interventions",
        "alignment": "Compute encoder-decoder cosine statistics",
        "normalization": "Compute normalization rank correlations",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        choices = list(MODELS)
        if name in ("intervene", "alignment"):
            choices = ["gemma-2-9b", "llama-3.1-8b"]
        elif name == "normalization":
            choices = ["gemma-2-9b"]
        command.add_argument("--model", choices=choices, required=True)
        command.add_argument("--output", type=Path, default=Path("output"))
        command.add_argument("--layers", nargs="+", type=int)
        if name in ("semantic", "compare", "summarize", "judge"):
            projections = ["dec_u", "enc_e"]
            if name != "compare":
                projections.append("both")
            command.add_argument(
                "--projection",
                choices=projections,
                default="dec_u" if name == "compare" else "both",
            )
        if name in ("semantic", "attention", "intervene", "alignment", "normalization"):
            command.add_argument(
                "--model-path", help="Local model directory or Hugging Face model ID"
            )
            command.add_argument("--device", default="auto")
            command.add_argument(
                "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
            )
        if name in ("semantic", "attention"):
            command.add_argument(
                "--batch-size",
                type=positive_int,
                default=64 if name == "semantic" else 256,
            )
        if name in ("judge", "intervene"):
            command.add_argument("--seed", type=int, default=SEED)
        if name == "judge":
            command.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
            command.add_argument(
                "--judge-model", default="meta-llama/Meta-Llama-3-70B-Instruct"
            )
            command.add_argument(
                "--n-per-group",
                type=positive_int,
                default=1000,
                help="Maximum number of features sampled from each group",
            )
            command.add_argument("--concurrency", type=positive_int, default=64)
        if name == "intervene":
            command.add_argument(
                "--role", choices=["semantic", "query", "key"], required=True
            )
            command.add_argument("--n-features", type=positive_int, default=100)
            command.add_argument(
                "--cache", type=Path, default=Path("output/prompt_cache")
            )
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    spec = MODELS[args.model]
    if args.layers is not None:
        for layer in args.layers:
            spec.check_layer(
                layer,
                attention=args.command in ("attention", "normalization")
                or (args.command == "intervene" and args.role != "semantic"),
            )
        if len(set(args.layers)) != len(args.layers):
            raise ValueError("Layer selection contains duplicate indices")
    from .models import save_run_settings
    from .storage import model_dir

    identifier = args.command
    if hasattr(args, "projection"):
        identifier += f"_{args.projection}"
    if hasattr(args, "role"):
        identifier += f"_{args.role}"
    save_run_settings(
        model_dir(args.output, spec) / "runs" / f"{identifier}.json", spec, vars(args)
    )
    if args.command in ("compare", "summarize", "judge"):
        if args.command == "compare":
            from .comparison import run_comparison

            run_comparison(args.output, spec, args.projection, args.layers)
        elif args.command == "summarize":
            from .semantic import summarize_semantics

            projections = (
                ["dec_u", "enc_e"] if args.projection == "both" else [args.projection]
            )
            for projection in projections:
                summarize_semantics(args.output, spec, projection)
        else:
            from .judge import run_judge

            projections = (
                ["dec_u", "enc_e"] if args.projection == "both" else [args.projection]
            )
            asyncio.run(
                run_judge(
                    args.output,
                    spec,
                    projections,
                    args.layers,
                    args.endpoint,
                    args.judge_model,
                    args.n_per_group,
                    args.concurrency,
                    args.seed,
                )
            )
        return
    import torch

    from .models import load_model, load_tokenizer

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    if args.command == "intervene":
        layers = (
            args.layers
            if args.layers is not None
            else list(spec.intervention_layers(args.role))
        )
    elif args.command == "normalization":
        layers = args.layers if args.layers is not None else list(GEMMA_LAYERS)
    else:
        layers = (
            args.layers
            if args.layers is not None
            else list(range(spec.num_layers - int(args.command == "attention")))
        )
    model = load_model(spec, args.model_path, device, args.dtype)
    if args.command == "semantic":
        from .semantic import run_semantic

        projections = (
            ["dec_u", "enc_e"] if args.projection == "both" else [args.projection]
        )
        run_semantic(
            model,
            load_tokenizer(spec, args.model_path),
            spec,
            args.output,
            layers,
            projections,
            args.batch_size,
        )
    elif args.command == "attention":
        from .attention import run_attention

        run_attention(model, spec, args.output, layers, args.batch_size)
    elif args.command == "intervene":
        from .intervention import run_interventions

        run_interventions(
            model,
            load_tokenizer(spec, args.model_path),
            spec,
            args.output,
            args.cache,
            layers,
            args.role,
            args.n_features,
            args.seed,
        )
    elif args.command == "alignment":
        from .validation import run_alignment

        run_alignment(model, spec, args.output, layers)
    elif args.command == "normalization":
        from .validation import run_normalization

        run_normalization(model, spec, args.output, layers)
