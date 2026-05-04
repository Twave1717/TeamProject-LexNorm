from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from member_ky.src.data import normalize_tokens
from member_ky.src.evaluation import SUMMARY_FIELDS, TOKEN_FIELDS
from member_ky.src.evaluation import evaluate_token_rows, token_confusion
from member_ky.src.util import assemble_full_predictions, write_csv, write_jsonl


def build_mfr_counts(train_df: pd.DataFrame) -> dict[str, dict[str, Counter]]:
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for _, row in train_df.iterrows():
        lang = str(row["lang"])
        raw_tokens = normalize_tokens(row["raw"])
        norm_tokens = normalize_tokens(row["norm"])
        for raw_token, norm_token in zip(raw_tokens, norm_tokens):
            counts[lang][raw_token][norm_token] += 1
    return counts


def mfr_predict(raw_tokens: list[str], counts: dict[str, Counter]) -> list[str]:
    predictions = []
    for raw_token in raw_tokens:
        if raw_token in counts:
            predictions.append(counts[raw_token].most_common(1)[0][0])
        else:
            predictions.append(raw_token)
    return predictions


def make_rule_token_rows(
    eval_df: pd.DataFrame,
    lang: str,
    counts_by_lang: dict[str, dict[str, Counter]],
) -> list[dict]:
    lang_eval = eval_df.loc[eval_df["lang"] == lang].reset_index(drop=True)
    token_rows = []

    for sent_id, row in lang_eval.iterrows():
        raw_tokens = normalize_tokens(row["raw"])
        gold_tokens = normalize_tokens(row["norm"])
        pred_tokens = mfr_predict(raw_tokens, counts_by_lang.get(lang, {}))

        for token_id, (raw, gold, pred) in enumerate(zip(raw_tokens, gold_tokens, pred_tokens)):
            token_rows.append(
                {
                    "lang": lang,
                    "sent_id": sent_id,
                    "token_id": token_id,
                    "raw": raw,
                    "gold": gold,
                    "pred": pred,
                    **token_confusion(raw, gold, pred),
                }
            )

    return token_rows


def write_prediction_outputs(
    output_dir: Path,
    lang: str,
    model_name: str,
    token_rows: list[dict],
) -> dict:
    lang_dir = output_dir / lang
    full_rows = assemble_full_predictions(token_rows)

    write_jsonl(lang_dir / "full_predictions.jsonl", full_rows)
    write_csv(lang_dir / "token_predictions.csv", token_rows, TOKEN_FIELDS)

    summary = {"lang": lang, "model": model_name, **evaluate_token_rows(token_rows)}
    write_csv(lang_dir / "summary.csv", [summary], SUMMARY_FIELDS)
    return summary


def run_rule_baseline(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    langs: list[str],
    output_dir: Path,
    model_name: str = "mfr_rule_based",
) -> list[dict]:
    counts_by_lang = build_mfr_counts(train_df)
    summaries = []

    for lang in langs:
        token_rows = make_rule_token_rows(eval_df, lang, counts_by_lang)
        summary = write_prediction_outputs(output_dir, lang, model_name, token_rows)
        summaries.append(summary)
        print(
            f"{lang}: accuracy={summary['accuracy']:.4f}, "
            f"ERR={summary['err']:.4f}, "
            f"action TP/FP/FN={summary['action_tp']}/"
            f"{summary['action_fp']}/{summary['action_fn']}"
        )

    write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    return summaries
