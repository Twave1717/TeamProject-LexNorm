from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from .openai_io import call_openai_json, get_openai_client
from .utils import ensure_dir, load_metadata_jsonl, read_json, read_jsonl, read_text, safe_json_loads


def retrieve_examples(index_df: pd.DataFrame, lang: str, raw_token: str, k: int = 5) -> List[Dict[str, Any]]:
    if index_df is None or len(index_df) == 0:
        return []
    sub = index_df[index_df["lang"] == lang].copy()
    if len(sub) == 0:
        return []
    exact = sub[sub["raw_token"] == raw_token]
    rows = exact.to_dict("records") if len(exact) else []
    if len(rows) < k:
        sub["sim"] = sub["raw_token"].map(lambda x: SequenceMatcher(None, str(x), str(raw_token)).ratio())
        rows += sub.sort_values(["sim", "count" if "count" in sub.columns else "total"], ascending=False).head(k).to_dict("records")
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        cands = safe_json_loads(row.get("candidates_json"), []) or []
        exs = safe_json_loads(row.get("examples_json"), []) or []
        if row.get("norm_token") and (row.get("raw_token"), row.get("norm_token")) not in seen:
            out.append({"raw": row.get("raw_token"), "norm": row.get("norm_token"), "count": row.get("count")})
            seen.add((row.get("raw_token"), row.get("norm_token")))
        for c in cands[:3]:
            key = (row.get("raw_token"), c.get("norm"))
            if key not in seen:
                out.append({"raw": row.get("raw_token"), "norm": c.get("norm"), "count": c.get("count"), "prob": c.get("prob")})
                seen.add(key)
        for e in exs[:2]:
            out.append({"sentence": e.get("sentence"), "target_index": e.get("target_index"), "norm": e.get("norm")})
        if len(out) >= k:
            break
    return out[:k]


def build_metadata_user(row: pd.Series, examples: List[Dict[str, Any]], include_gold: bool = False) -> str:
    payload = {
        "case_id": row["case_id"],
        "lang": row["lang"],
        "tokens": safe_json_loads(row["tokens_json"], []),
        "target_index": int(row["target_index"]),
        "target_token": row["target_token"],
        "retrieved_examples": examples,
    }
    if include_gold:
        payload["gold_norm"] = row.get("gold_norm", None)
    return json.dumps(payload, ensure_ascii=False, indent=2)



def generate_metadata(
    cases_csv: str | Path,
    output_jsonl: str | Path,
    prompt_path: str | Path,
    schema_path: str | Path,
    model: str = "gpt-4.1-mini",
    index_csv: str | Path | None = None,
    limit: int | None = None,
    include_gold: bool = False,
) -> pd.DataFrame:
    cases = pd.read_csv(cases_csv)
    if limit:
        cases = cases.head(limit)
    index_df = pd.read_csv(index_csv) if index_csv else pd.DataFrame()
    system = read_text(prompt_path)
    schema = read_json(schema_path)
    client = get_openai_client()
    out_path = Path(output_jsonl)
    ensure_dir(out_path.parent)
    existing = {r.get("case_id") for r in read_jsonl(out_path) if r.get("case_id")}
    rows = []
    with out_path.open("a", encoding="utf-8") as f:
        for _, row in tqdm(cases.iterrows(), total=len(cases)):
            if row["case_id"] in existing:
                continue
            examples = retrieve_examples(index_df, row["lang"], row["target_token"])
            user = build_metadata_user(row, examples, include_gold=include_gold)
            out = call_openai_json(client, model, system, user, schema)
            out["case_id"] = row["case_id"]
            rows.append(out)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
    return pd.DataFrame(rows)


def build_normalizer_user(row: pd.Series, mode: str, examples: List[Dict[str, Any]], metadata: Dict[str, Any] | None, softening_policy: str) -> str:
    payload = {
        "case_id": row["case_id"],
        "lang": row["lang"],
        "tokens": safe_json_loads(row["tokens_json"], []),
        "target_index": int(row["target_index"]),
        "target_token": row["target_token"],
        "softening_policy": softening_policy,
    }
    if mode in {"pair_fewshot", "metadata_rag"}:
        payload["retrieved_lexnorm_examples"] = examples
    if mode == "metadata_rag":
        payload["metadata"] = metadata
    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_normalizer(
    cases_csv: str | Path,
    output_csv: str | Path,
    prompt_path: str | Path,
    schema_path: str | Path,
    mode: str,
    model: str = "gpt-4.1-mini",
    index_csv: str | Path | None = None,
    metadata_jsonl: str | Path | None = None,
    softening_policy: str = "preserve_force",
    limit: int | None = None,
) -> pd.DataFrame:
    cases = pd.read_csv(cases_csv)
    if limit:
        cases = cases.head(limit)
    index_df = pd.read_csv(index_csv) if index_csv else pd.DataFrame()
    metadata_map = load_metadata_jsonl(metadata_jsonl)
    system = read_text(prompt_path)
    schema = read_json(schema_path)
    client = get_openai_client()
    out_path = Path(output_csv)
    ensure_dir(out_path.parent)
    existing_df = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    done = set(existing_df["case_id"]) if "case_id" in existing_df else set()
    rows = []
    for _, row in tqdm(cases.iterrows(), total=len(cases)):
        if row["case_id"] in done:
            continue
        examples = retrieve_examples(index_df, row["lang"], row["target_token"]) if mode != "pure" else []
        md = metadata_map.get(row["case_id"]) if mode == "metadata_rag" else None
        user = build_normalizer_user(row, mode, examples, md, softening_policy)
        out = call_openai_json(client, model, system, user, schema)
        rows.append({
            "case_id": row["case_id"],
            "system": f"{model}_{mode}_{softening_policy}",
            "mode": mode,
            "softening_policy": softening_policy,
            "pred_norm": out.get("normalized", row["target_token"]),
            "edit": out.get("edit", False),
            "confidence": out.get("confidence", 0.0),
            "error": out.get("error", ""),
        })
    new_df = pd.DataFrame(rows)
    df = pd.concat([existing_df, new_df], ignore_index=True) if len(existing_df) else new_df
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def build_judge_user(row: pd.Series, pred: pd.Series, metadata: Dict[str, Any] | None) -> str:
    payload = {
        "case_id": row["case_id"],
        "lang": row["lang"],
        "tokens": safe_json_loads(row["tokens_json"], []),
        "target_index": int(row["target_index"]),
        "target_token": row["target_token"],
        "gold": row["gold_norm"],
        "candidate": pred["pred_norm"],
        "metadata": metadata,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def judge_candidates(
    cases_csv: str | Path,
    preds_csv: str | Path,
    output_csv: str | Path,
    prompt_path: str | Path,
    schema_path: str | Path,
    model: str = "gpt-4.1-mini",
    metadata_jsonl: str | Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    cases = pd.read_csv(cases_csv)
    preds = pd.read_csv(preds_csv)
    if limit:
        cases = cases.head(limit)
        preds = preds[preds["case_id"].isin(set(cases["case_id"]))]
    case_map = {r["case_id"]: r for _, r in cases.iterrows()}
    meta_map = load_metadata_jsonl(metadata_jsonl)
    system = read_text(prompt_path)
    schema = read_json(schema_path)
    client = get_openai_client()
    out_path = Path(output_csv)
    ensure_dir(out_path.parent)
    existing_df = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    done = set(existing_df["case_id"]) if "case_id" in existing_df else set()
    rows = []
    for _, pred in tqdm(preds.iterrows(), total=len(preds)):
        if pred["case_id"] in done:
            continue
        row = case_map[pred["case_id"]]
        md = meta_map.get(pred["case_id"])
        user = build_judge_user(row, pred, md)
        out = call_openai_json(client, model, system, user, schema)
        rows.append({
            "case_id": pred["case_id"],
            "decision": out.get("decision", "Reject"),
            "judge_confidence": out.get("confidence", 0.0),
            "social_context_needed": out.get("social_context_needed", False),
            "softened": out.get("softened", False),
            "scores_json": json.dumps(out.get("scores", {}), ensure_ascii=False),
            "evidence": out.get("evidence", ""),
        })
    new_df = pd.DataFrame(rows)
    df = pd.concat([existing_df, new_df], ignore_index=True) if len(existing_df) else new_df
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df
