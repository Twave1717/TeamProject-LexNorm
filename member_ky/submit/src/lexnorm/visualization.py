from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .utils import ensure_dir, safe_json_loads


def _label(path: str) -> str:
    return Path(path).stem.replace("summary_", "")


def plot_summary(summary_csv: str | Path, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    df = pd.read_csv(summary_csv)
    out = ensure_dir(output_dir)
    if {"summary_path", "ERR"}.issubset(df.columns):
        labels = [_label(x) for x in df["summary_path"]]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, df["ERR"])
        plt.ylabel("ERR")
        plt.title("Official ERR by system")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out / "err_by_system.png", dpi=180)
        plt.close()
    if "judge_accept_rate" in df.columns:
        labels = [_label(x) for x in df["summary_path"]]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, df["judge_accept_rate"])
        plt.ylabel("Judge Accept Rate")
        plt.title("Acceptability by system")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out / "judge_accept_by_system.png", dpi=180)
        plt.close()
    if {"ERR", "judge_accept_rate"}.issubset(df.columns):
        labels = [_label(x) for x in df["summary_path"]]
        plt.figure(figsize=(5, 4))
        plt.scatter(df["ERR"], df["judge_accept_rate"])
        for label, x, y in zip(labels, df["ERR"], df["judge_accept_rate"]):
            plt.annotate(label, (x, y))
        plt.xlabel("ERR")
        plt.ylabel("Judge Accept Rate")
        plt.title("ERR vs Acceptability")
        plt.tight_layout()
        plt.savefig(out / "err_vs_acceptability.png", dpi=180)
        plt.close()
    if {"summary_path", "ERR", "judge_accept_rate"}.issubset(df.columns):
        labels = [_label(x) for x in df["summary_path"]]
        x = range(len(labels))
        plt.figure(figsize=(8, 4))
        plt.bar([i - 0.18 for i in x], df["ERR"], width=0.36, label="ERR")
        plt.bar([i + 0.18 for i in x], df["judge_accept_rate"], width=0.36, label="Accept")
        plt.xticks(list(x), labels, rotation=25, ha="right")
        plt.legend()
        plt.title("Pure vs Pair Few-Shot vs Metadata-RAG")
        plt.tight_layout()
        plt.savefig(out / "pure_pair_metadata.png", dpi=180)
        plt.close()


def plot_from_input_dir(input_dir: str | Path, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    input_dir = Path(input_dir)
    out = ensure_dir(output_dir)
    summaries = sorted(input_dir.glob("summary_*.json"))
    rows = []
    for p in summaries:
        obj = json.loads(p.read_text(encoding="utf-8"))
        row = {"summary_path": str(p), "system": p.stem.replace("summary_", "")}
        row.update(obj.get("official", {}))
        for k, v in obj.get("judge", {}).items():
            row[f"judge_{k}"] = v
        rows.append(row)
    if rows:
        summary_csv = input_dir / "combined_summary.csv"
        pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        plot_summary(summary_csv, out)

    # Fig 2: ordinary vs social-context-needed acceptability from judge files.
    split_rows = []
    for p in sorted(input_dir.glob("judge_*.csv")):
        df = pd.read_csv(p)
        if {"decision", "social_context_needed"}.issubset(df.columns):
            for group, g in df.groupby(df["social_context_needed"].astype(bool)):
                split_rows.append({"system": p.stem.replace("judge_", ""), "social_context_needed": bool(group), "accept_rate": float((g["decision"] == "Accept").mean())})
    if split_rows:
        df = pd.DataFrame(split_rows)
        pivot = df.pivot(index="system", columns="social_context_needed", values="accept_rate").fillna(0)
        pivot = pivot.rename(columns={False: "ordinary", True: "social_context_needed"})
        pivot.plot(kind="bar", figsize=(8, 4))
        plt.ylabel("Judge Accept Rate")
        plt.title("Ordinary vs Social-Context-Needed Performance")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out / "ordinary_vs_social_context.png", dpi=180)
        plt.close()

    # Fig 4: over-softening penalty distribution.
    penalty_rows = []
    for p in sorted(input_dir.glob("judge_*.csv")):
        df = pd.read_csv(p)
        if "scores_json" not in df.columns:
            continue
        for s in df["scores_json"]:
            scores = safe_json_loads(s, {}) or {}
            penalty_rows.append({"system": p.stem.replace("judge_", ""), "penalty": scores.get("over_softening_penalty", 0)})
    if penalty_rows:
        df = pd.DataFrame(penalty_rows)
        pivot = pd.crosstab(df["system"], df["penalty"], normalize="index")
        pivot.plot(kind="bar", stacked=True, figsize=(8, 4))
        plt.ylabel("Proportion")
        plt.title("Softening Penalty Distribution")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out / "softening_penalty_distribution.png", dpi=180)
        plt.close()
