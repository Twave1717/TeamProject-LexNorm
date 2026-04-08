import argparse
import os
from pathlib import Path

try:
    from baseline.shared import (
        DATASET_NAMES,
        get_submission_predictions,
        load_local_env,
        load_public_dataset,
        maybe_filter_lang,
        predict_with_mfr,
        write_submission,
    )
    from baseline.utils import evaluate
except ImportError:
    from shared import (
        DATASET_NAMES,
        get_submission_predictions,
        load_local_env,
        load_public_dataset,
        maybe_filter_lang,
        predict_with_mfr,
        write_submission,
    )
    from utils import evaluate


def parse_args():
    load_local_env()
    parser = argparse.ArgumentParser(
        description="Run the MultiLexNorm 2026 MFR baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Train on the train split and evaluate on validation.",
    )
    add_common_args(eval_parser)
    eval_parser.add_argument(
        "--verbose-errors",
        action="store_true",
        help="Print token-level errors during evaluation.",
    )

    submit_parser = subparsers.add_parser(
        "submit",
        help="Train on train+validation and create predictions.json plus a zip.",
    )
    add_common_args(submit_parser)
    submit_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store predictions. Defaults to outputs/submission_<phase>.",
    )

    return parser.parse_args()


def add_common_args(parser):
    parser.add_argument(
        "--phase",
        choices=sorted(DATASET_NAMES),
        default="dev",
        help="Which public dataset release to use.",
    )
    parser.add_argument(
        "--lang",
        help="Optional language code for single-language evaluation.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HF_TOEKN"),
        help=(
            "Hugging Face access token. Defaults to the HF_TOKEN environment "
            "variable. Required if the dataset is gated."
        ),
    )
def run_evaluate(args):
    data = load_public_dataset(args.phase, args.hf_token)
    train = maybe_filter_lang(data["train"], args.lang)
    validation = maybe_filter_lang(data["validation"], args.lang)

    predictions = predict_with_mfr(train, validation)
    print(
        "Evaluating",
        f"phase={args.phase}",
        f"lang={args.lang or 'all'}",
        f"rows={len(predictions)}",
    )
    evaluate(
        raw=predictions["raw"].tolist(),
        gold=predictions["norm"].tolist(),
        pred=predictions["pred"].tolist(),
        verbose=args.verbose_errors,
    )


def run_submit(args):
    predictions = get_submission_predictions(args.phase, args.hf_token, args.lang)
    output_dir = args.output_dir or Path("outputs") / f"submission_{args.phase}"
    predictions_path, zip_path = write_submission(predictions, args.phase, output_dir)
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved zip archive to: {zip_path}")


def main():
    args = parse_args()
    if args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "submit":
        run_submit(args)


if __name__ == "__main__":
    main()
