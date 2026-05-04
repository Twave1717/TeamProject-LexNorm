from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from member_ky.src.data import load_local_env, read_split
from member_ky.src.evaluation import SUMMARY_FIELDS
from member_ky.src.model.public_byt5 import (
    PUBLIC_MODELS,
    SUPPORTED_2026_LANGS,
    make_token_examples,
    predict_examples,
    write_lang_outputs,
)
from member_ky.src.model.registry import (
    AVAILABLE_MODELS,
    canonical_model_name,
    print_available_models,
)
from member_ky.src.model.rule_based import run_rule_baseline
from member_ky.src.util import write_csv


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more LexNorm model runners with shared output format."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rule_based"],
        help="Model runners to execute. Aliases: rule/mfr, byt5/t5.",
    )
    parser.add_argument("--list-models", action="store_true", help="Show available model runners.")
    parser.add_argument("--langs", nargs="+", help="Language codes to run. Defaults depend on model.")
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    parser.add_argument("--train-split", default="train", choices=["train"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "temp" / "models",
        help="Root directory for model outputs.",
    )

    byt5 = parser.add_argument_group("public_byt5 options")
    byt5.add_argument("--model-id", help="Override Hugging Face checkpoint for public_byt5.")
    byt5.add_argument("--limit-tokens", type=int, help="Smoke-test limit per language.")
    byt5.add_argument("--batch-size", type=int, default=4)
    byt5.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    byt5.add_argument("--num-beams", type=int, default=4)
    byt5.add_argument("--max-new-tokens", type=int, default=32)
    byt5.add_argument("--postprocess", default="alnum", choices=["none", "alnum"])
    return parser.parse_args()


def available_langs(df) -> list[str]:
    return sorted(str(lang) for lang in df["lang"].dropna().unique())


def resolve_langs(eval_df, requested: list[str] | None, model_name: str) -> list[str]:
    present = set(available_langs(eval_df))
    if requested:
        langs = [str(lang) for lang in requested]
    elif model_name == "public_byt5":
        langs = [lang for lang in SUPPORTED_2026_LANGS if lang in present]
    else:
        langs = sorted(present)

    missing = [lang for lang in langs if lang not in present]
    if missing:
        raise ValueError(f"{model_name}: missing languages in eval split: {', '.join(missing)}")
    return langs


def run_rule_based(args: argparse.Namespace, eval_df, langs: list[str]) -> None:
    print(f"\n[rule_based] langs={', '.join(langs)}")
    train_df = read_split(args.train_split)
    output_dir = args.output_root / "rule_based"
    run_rule_baseline(
        train_df=train_df,
        eval_df=eval_df,
        langs=langs,
        output_dir=output_dir,
        model_name="rule_based",
    )
    print(f"[rule_based] outputs -> {output_dir}")


def run_public_byt5(args: argparse.Namespace, eval_df, langs: list[str]) -> None:
    print(f"\n[public_byt5] langs={', '.join(langs)}")
    output_dir = args.output_root / "public_byt5"
    summaries = []

    for lang in langs:
        model_id = args.model_id or PUBLIC_MODELS.get(lang)
        if not model_id:
            raise ValueError(
                f"public_byt5 has no default checkpoint for '{lang}'. "
                "Use --model-id to provide one explicitly."
            )

        examples = make_token_examples(eval_df, lang, args.limit_tokens)
        token_rows = predict_examples(
            examples,
            model_id,
            batch_size=args.batch_size,
            device_name=args.device,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            postprocess=args.postprocess,
        )
        summary = write_lang_outputs(output_dir, lang, model_id, token_rows)
        summaries.append(summary)
        print(
            f"{lang}: accuracy={summary['accuracy']:.4f}, "
            f"ERR={summary['err']:.4f}, "
            f"action TP/FP/FN={summary['action_tp']}/"
            f"{summary['action_fp']}/{summary['action_fn']}"
        )

    write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    print(f"[public_byt5] outputs -> {output_dir}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        print_available_models()
        return

    model_names = [canonical_model_name(name) for name in args.models]
    invalid = sorted(set(model_names) - set(AVAILABLE_MODELS))
    if invalid:
        raise ValueError(f"Unavailable models: {', '.join(invalid)}")

    load_local_env()
    eval_df = read_split(args.eval_split)

    for model_name in dict.fromkeys(model_names):
        langs = resolve_langs(eval_df, args.langs, model_name)
        if model_name == "rule_based":
            run_rule_based(args, eval_df, langs)
        elif model_name == "public_byt5":
            run_public_byt5(args, eval_df, langs)
        else:
            raise ValueError(f"No runner is registered for {model_name}")


if __name__ == "__main__":
    main()
