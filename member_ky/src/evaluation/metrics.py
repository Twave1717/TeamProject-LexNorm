from __future__ import annotations

from collections import Counter


TOKEN_FIELDS = [
    "lang",
    "sent_id",
    "token_id",
    "raw",
    "gold",
    "pred",
    "gold_changed",
    "pred_changed",
    "correct",
    "action_confusion",
    "correction_confusion",
]

SUMMARY_FIELDS = [
    "lang",
    "model",
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


def token_confusion(raw: str, gold: str, pred: str) -> dict[str, str | bool]:
    gold_changed = raw != gold
    pred_changed = raw != pred
    correct = pred == gold

    if gold_changed and pred_changed:
        action = "TP"
    elif not gold_changed and pred_changed:
        action = "FP"
    elif gold_changed and not pred_changed:
        action = "FN"
    else:
        action = "TN"

    if gold_changed and correct:
        correction = "TP"
    elif not gold_changed and pred_changed:
        correction = "FP"
    elif gold_changed and pred == raw:
        correction = "FN"
    elif gold_changed and pred_changed and not correct:
        correction = "FP_FN"
    else:
        correction = "TN"

    return {
        "gold_changed": gold_changed,
        "pred_changed": pred_changed,
        "correct": correct,
        "action_confusion": action,
        "correction_confusion": correction,
    }


def evaluate_token_rows(token_rows: list[dict]) -> dict[str, float | int]:
    total = len(token_rows)
    correct = sum(row["correct"] for row in token_rows)
    changed = sum(row["gold_changed"] for row in token_rows)
    accuracy = correct / total if total else 0.0
    lai = (total - changed) / total if total else 0.0
    err = (accuracy - lai) / (1 - lai) if changed else 0.0

    action_counts = Counter(row["action_confusion"] for row in token_rows)
    correction_counts = Counter(row["correction_confusion"] for row in token_rows)

    return {
        "tokens": total,
        "gold_changed_tokens": changed,
        "accuracy": accuracy,
        "lai": lai,
        "err": err,
        "action_tp": action_counts["TP"],
        "action_fp": action_counts["FP"],
        "action_fn": action_counts["FN"],
        "action_tn": action_counts["TN"],
        "correction_tp": correction_counts["TP"],
        "correction_fp": correction_counts["FP"] + correction_counts["FP_FN"],
        "correction_fn": correction_counts["FN"] + correction_counts["FP_FN"],
        "correction_tn": correction_counts["TN"],
        "wrong_change_on_changed_gold": correction_counts["FP_FN"],
    }
