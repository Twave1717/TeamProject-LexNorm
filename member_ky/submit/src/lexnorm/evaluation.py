from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .metrics import add_error_modes, evaluate_rows
from .utils import ensure_dir


def _judge_summary(df: pd.DataFrame) -> Dict:
    if len(df) == 0:
        return {"n": 0, "accept_rate": 0.0, "reject_rate": 0.0, "softened_rate": 0.0, "social_context_needed_rate": 0.0, "mean_confidence": 0.0}
    return {
        "n": int(len(df)),
        "accept_rate": float((df["decision"] == "Accept").mean()) if "decision" in df else 0.0,
        "reject_rate": float((df["decision"] == "Reject").mean()) if "decision" in df else 0.0,
        "softened_rate": float(df["softened"].astype(bool).mean()) if "softened" in df else 0.0,
        "social_context_needed_rate": float(df["social_context_needed"].astype(bool).mean()) if "social_context_needed" in df else 0.0,
        "mean_confidence": float(df["judge_confidence"].mean()) if "judge_confidence" in df else 0.0,
    }


def evaluate_system_outputs(cases_csv: str | Path, preds_csv: str | Path, judge_csv: str | Path | None, output_json: str | Path) -> Dict:
    cases = pd.read_csv(cases_csv)
    preds = pd.read_csv(preds_csv)
    cols = ["case_id", "lang", "raw_token", "target_token", "gold_norm", "gold_changed"]
    for optional in ["sample_group", "social_context_candidate"]:
        if optional in cases.columns:
            cols.append(optional)
    merged = preds.merge(cases[cols], on="case_id", how="left")
    merged = add_error_modes(merged)
    official = evaluate_rows(merged.to_dict("records"))
    result: Dict = {"official": official}
    if "sample_group" in merged.columns:
        result["official_by_sample_group"] = {
            str(group): evaluate_rows(g.to_dict("records")) for group, g in merged.groupby("sample_group")
        }
    if "social_context_candidate" in merged.columns:
        result["official_by_social_candidate"] = {
            str(group): evaluate_rows(g.to_dict("records")) for group, g in merged.groupby("social_context_candidate")
        }
    if judge_csv:
        judge = pd.read_csv(judge_csv)
        jm = merged.merge(judge, on="case_id", how="left")
        result["judge"] = _judge_summary(jm)
        if "social_context_needed" in jm.columns:
            result["judge_by_social_context_needed"] = {
                str(group): _judge_summary(g) for group, g in jm.groupby(jm["social_context_needed"].astype(bool))
            }
        if "sample_group" in jm.columns:
            result["judge_by_sample_group"] = {
                str(group): _judge_summary(g) for group, g in jm.groupby("sample_group")
            }
    ensure_dir(Path(output_json).parent)
    Path(output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
