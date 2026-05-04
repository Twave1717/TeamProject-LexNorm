from __future__ import annotations

import csv
import itertools
import re
from pathlib import Path

from member_ky.src.evaluation import evaluate_token_rows, token_confusion
from member_ky.src.util import write_csv


BASE_COLUMNS = ["lang", "sent_id", "token_id", "raw", "gold"]


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "model"


def unique_slugs(model_names: list[str]) -> dict[str, str]:
    counts: dict[str, int] = {}
    slugs = {}
    for model_name in model_names:
        base = slugify(model_name)
        counts[base] = counts.get(base, 0) + 1
        slugs[model_name] = base if counts[base] == 1 else f"{base}_{counts[base]}"
    return slugs


def read_token_predictions(path: Path) -> dict[tuple[str, int, int], dict]:
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            key = (row["lang"], int(row["sent_id"]), int(row["token_id"]))
            rows[key] = {
                "lang": row["lang"],
                "sent_id": int(row["sent_id"]),
                "token_id": int(row["token_id"]),
                "raw": row["raw"],
                "gold": row["gold"],
                "pred": row["pred"],
            }
    return rows


def summarize_model_rows(
    model_name: str,
    path: Path,
    rows: dict,
    shared_keys: list[tuple[str, int, int]],
) -> dict:
    token_rows = []
    for key in shared_keys:
        row = rows[key]
        token_rows.append(
            {
                "lang": row["lang"],
                "sent_id": row["sent_id"],
                "token_id": row["token_id"],
                "raw": row["raw"],
                "gold": row["gold"],
                "pred": row["pred"],
                **token_confusion(row["raw"], row["gold"], row["pred"]),
            }
        )
    metrics = evaluate_token_rows(token_rows)
    return {"model": model_name, "token_csv": str(path), **metrics}


def combined_case_type(correct_models: list[str], all_models: list[str]) -> str:
    if len(correct_models) == len(all_models):
        return "all_correct"
    if not correct_models:
        return "all_fail"
    if len(correct_models) == 1:
        return f"winner_{slugify(correct_models[0])}"
    return "partial_disagreement"


def compare_multiple(model_paths: dict[str, Path], output_dir: Path) -> dict:
    model_rows = {model_name: read_token_predictions(path) for model_name, path in model_paths.items()}
    shared_keys = sorted(set.intersection(*(set(rows) for rows in model_rows.values())))
    model_names = list(model_paths)
    slugs = unique_slugs(model_names)

    combined_columns = [*BASE_COLUMNS]
    for model_name in model_names:
        combined_columns.extend([f"{slugs[model_name]}_pred", f"{slugs[model_name]}_correct"])
    combined_columns.extend(["correct_models", "wrong_models", "case_type"])

    compared = []
    for key in shared_keys:
        base = next(iter(model_rows.values()))[key]
        correct_models = []
        wrong_models = []
        row = {column: base[column] for column in BASE_COLUMNS}

        for model_name in model_names:
            pred = model_rows[model_name][key]["pred"]
            is_correct = pred == base["gold"]
            if is_correct:
                correct_models.append(model_name)
            else:
                wrong_models.append(model_name)
            row[f"{slugs[model_name]}_pred"] = pred
            row[f"{slugs[model_name]}_correct"] = is_correct

        row["correct_models"] = ",".join(correct_models)
        row["wrong_models"] = ",".join(wrong_models)
        row["case_type"] = combined_case_type(correct_models, model_names)
        compared.append(row)

    write_csv(output_dir / "combined_token_comparison.csv", compared, combined_columns)

    summary_rows = [
        summarize_model_rows(model_name, model_paths[model_name], model_rows[model_name], shared_keys)
        for model_name in model_names
    ]
    summary_fields = [
        "model",
        "token_csv",
        "tokens",
        "gold_changed_tokens",
        "accuracy",
        "lai",
        "err",
        "action_tp",
        "action_fp",
        "action_fn",
        "action_tn",
        "correction_tp",
        "correction_fp",
        "correction_fn",
        "correction_tn",
        "wrong_change_on_changed_gold",
    ]
    write_csv(output_dir / "comparison_summary.csv", summary_rows, summary_fields)
    write_pairwise_case_files(compared, model_names, slugs, output_dir)
    return {"shared_tokens": len(shared_keys), "summary": summary_rows}


def write_pairwise_case_files(
    rows: list[dict],
    model_names: list[str],
    slugs: dict[str, str],
    output_dir: Path,
) -> None:
    for left, right in itertools.combinations(model_names, 2):
        left_slug = slugs[left]
        right_slug = slugs[right]
        pair_dir = output_dir / "pairs" / f"{left_slug}_vs_{right_slug}"
        pair_columns = [
            *BASE_COLUMNS,
            f"{left_slug}_pred",
            f"{right_slug}_pred",
            "case_type",
        ]
        buckets = {
            f"{left_slug}_gt_{right_slug}": [],
            f"{right_slug}_gt_{left_slug}": [],
            "both_fail": [],
            "both_correct": [],
        }

        for row in rows:
            left_correct = row[f"{left_slug}_correct"] in [True, "True", "true"]
            right_correct = row[f"{right_slug}_correct"] in [True, "True", "true"]
            if left_correct and not right_correct:
                case_type = f"{left_slug}_gt_{right_slug}"
            elif right_correct and not left_correct:
                case_type = f"{right_slug}_gt_{left_slug}"
            elif left_correct and right_correct:
                case_type = "both_correct"
            else:
                case_type = "both_fail"

            buckets[case_type].append(
                {
                    **{column: row[column] for column in BASE_COLUMNS},
                    f"{left_slug}_pred": row[f"{left_slug}_pred"],
                    f"{right_slug}_pred": row[f"{right_slug}_pred"],
                    "case_type": case_type,
                }
            )

        for case_type, case_rows in buckets.items():
            write_csv(pair_dir / f"{case_type}.csv", case_rows, pair_columns)
