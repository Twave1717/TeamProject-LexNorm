from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from member_ky.src.evaluation.compare import compare_multiple
from member_ky.src.model.registry import canonical_model_name, print_available_models
from member_ky.src.util import write_csv


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare token-level predictions from two or more model runners."
    )
    parser.add_argument("--models", nargs="+", help="Models to compare.")
    parser.add_argument("--list-models", action="store_true", help="Show registered model runners.")
    parser.add_argument("--langs", nargs="+", help="Language codes to compare. Defaults to common langs.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "temp" / "models",
        help="Root used by run_model.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "temp" / "model_compare",
        help="Directory for comparison CSV files.",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help=(
            "Override a model output path. PATH can be a model dir, lang dir, "
            "or token_predictions.csv. Can be repeated."
        ),
    )
    return parser.parse_args()


def parse_model_paths(entries: list[str]) -> dict[str, Path]:
    paths = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--model-path must use MODEL=PATH: {entry}")
        model, path = entry.split("=", 1)
        paths[canonical_or_raw(model)] = Path(path)
    return paths


def canonical_or_raw(name: str) -> str:
    try:
        return canonical_model_name(name)
    except ValueError:
        return name.strip()


def token_csv_lang(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as in_file:
        reader = csv.DictReader(in_file)
        first = next(reader, None)
    if not first:
        return None
    return first.get("lang")


def model_base(model_name: str, output_root: Path, overrides: dict[str, Path]) -> Path:
    return overrides.get(model_name, output_root / model_name)


def infer_langs(model_names: list[str], output_root: Path, overrides: dict[str, Path]) -> list[str]:
    lang_sets = []
    for model_name in model_names:
        base = model_base(model_name, output_root, overrides)
        if base.is_file():
            lang = token_csv_lang(base)
            lang_sets.append({lang} if lang else set())
        elif (base / "token_predictions.csv").exists():
            lang = token_csv_lang(base / "token_predictions.csv")
            lang_sets.append({lang} if lang else set())
        elif base.exists():
            lang_sets.append(
                {
                    child.name
                    for child in base.iterdir()
                    if child.is_dir() and (child / "token_predictions.csv").exists()
                }
            )
        else:
            lang_sets.append(set())

    common = set.intersection(*lang_sets) if lang_sets else set()
    if not common:
        raise ValueError(
            "No common languages found. Run models first or pass --langs and --model-path."
        )
    return sorted(common)


def resolve_token_csv(base: Path, lang: str) -> Path:
    if base.is_file():
        return base
    if (base / "token_predictions.csv").exists():
        return base / "token_predictions.csv"
    return base / lang / "token_predictions.csv"


def main() -> None:
    args = parse_args()
    if args.list_models:
        print_available_models()
        return

    if not args.models:
        raise ValueError("--models is required unless --list-models is used.")

    model_names = [canonical_or_raw(name) for name in args.models]
    if len(dict.fromkeys(model_names)) < 2:
        raise ValueError("Compare needs at least two distinct models.")
    model_names = list(dict.fromkeys(model_names))

    overrides = parse_model_paths(args.model_path)
    langs = args.langs or infer_langs(model_names, args.output_root, overrides)

    index_rows = []
    for lang in langs:
        model_paths = {}
        for model_name in model_names:
            path = resolve_token_csv(model_base(model_name, args.output_root, overrides), lang)
            if not path.exists():
                raise FileNotFoundError(f"Missing token prediction CSV for {model_name}/{lang}: {path}")
            model_paths[model_name] = path

        lang_output_dir = args.output_dir / lang
        result = compare_multiple(model_paths, lang_output_dir)
        index_rows.append(
            {
                "lang": lang,
                "models": ",".join(model_names),
                "shared_tokens": result["shared_tokens"],
                "output_dir": str(lang_output_dir),
            }
        )
        print(f"{lang}: shared tokens={result['shared_tokens']} -> {lang_output_dir}")

    write_csv(
        args.output_dir / "comparison_index.csv",
        index_rows,
        ["lang", "models", "shared_tokens", "output_dir"],
    )


if __name__ == "__main__":
    main()
