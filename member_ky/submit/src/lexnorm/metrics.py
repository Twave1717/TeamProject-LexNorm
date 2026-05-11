from __future__ import annotations

from typing import Dict, Iterable, List, Any
import pandas as pd


def error_mode(raw: str, gold: str, pred: str) -> str:
    raw, gold, pred = str(raw), str(gold), str(pred)
    gold_changed = raw != gold
    pred_changed = raw != pred
    if gold_changed and pred == gold:
        return "TP"
    if not gold_changed and pred_changed:
        return "over_normalization"
    if gold_changed and pred == raw:
        return "under_normalization"
    if gold_changed and pred != raw and pred != gold:
        return "wrong_candidate"
    if not gold_changed and pred == raw:
        return "correct_keep"
    return "other"


def evaluate_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, float | int]:
    rows = list(rows)
    n = len(rows)
    correct = 0
    tp = fp = fn = 0
    over = under = wrong = correct_keep = 0
    changed_total = unchanged_total = changed_correct = unchanged_correct = 0
    for r in rows:
        raw = str(r.get("raw_token", r.get("target_token", "")))
        gold = str(r.get("gold_norm", ""))
        pred = str(r.get("pred_norm", r.get("normalized", "")))
        mode = error_mode(raw, gold, pred)
        correct += int(pred == gold)
        if raw != gold:
            changed_total += 1
            changed_correct += int(pred == gold)
        else:
            unchanged_total += 1
            unchanged_correct += int(pred == raw)
        if mode == "TP":
            tp += 1
        elif mode == "over_normalization":
            fp += 1; over += 1
        elif mode == "under_normalization":
            fn += 1; under += 1
        elif mode == "wrong_candidate":
            fn += 1; wrong += 1
        elif mode == "correct_keep":
            correct_keep += 1
    acc = correct / n if n else 0.0
    err = (tp - fp) / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": n,
        "accuracy": acc,
        "ERR": err,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "over_normalization": over,
        "under_normalization": under,
        "wrong_candidate": wrong,
        "correct_keep": correct_keep,
        "changed_accuracy": changed_correct / changed_total if changed_total else 0.0,
        "unchanged_accuracy": unchanged_correct / unchanged_total if unchanged_total else 0.0,
    }


def add_error_modes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error_mode"] = [error_mode(r, g, p) for r, g, p in zip(out["raw_token"], out["gold_norm"], out["pred_norm"])]
    return out
