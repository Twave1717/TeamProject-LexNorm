from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from .utils import as_tokens, ensure_dir, env_value, load_known_env_files, official_alnum_filter, tokens_json


def hf_token() -> str | None:
    token = env_value("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token", "HF_token")
    if not token:
        load_known_env_files()
        token = env_value("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token", "HF_token")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def require_hf_token() -> str:
    token = hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Put HF_TOKEN=... or hf_token=... in Drive .env, then rerun 00_setup_and_data.ipynb.")
    return token


KO_SOCIAL_PATTERNS = [
    r"[ㅋㅎ]{2,}",
    r"[ㅠㅜ]{2,}",
    r"^[ㄱ-ㅎ]{2,}$",
    r"[가-힣][.·_\-][가-힣]",
]
KO_SOCIAL_TERMS = {
    "ㅅㅂ", "ㅂㅅ", "ㅈㄴ", "존나", "개웃", "개웃기", "개좋", "개빡", "미친", "ㄱㅅ", "ㅇㅋ", "낼", "걍",
}
EN_SOCIAL_TERMS = {
    "lol", "lmao", "lmfao", "rofl", "idk", "imo", "imho", "wtf", "omg", "bruh", "bro", "sis",
    "stan", "slay", "cuz", "ya", "yall", "ur", "u", "r", "af", "tf",
}


def is_social_context_candidate(token: str, sentence: str = "", lang: str | None = None) -> bool:
    """감사용 샘플링 휴리스틱이다. 최종 라벨은 human/judge가 결정한다."""
    tok = str(token).strip()
    low = tok.lower()
    if re.search(r"\w[._*·-]\w", tok):
        return True
    if lang == "ko" or re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", tok):
        if low in KO_SOCIAL_TERMS:
            return True
        return any(re.search(p, tok) for p in KO_SOCIAL_PATTERNS)
    if lang == "en" or re.search(r"[A-Za-z]", tok):
        if low in EN_SOCIAL_TERMS:
            return True
        if re.search(r"(.)\1{2,}", low):
            return True
    return False


def build_target_examples_for_lang(split_ds, lang: str, add_lang_prefix: bool = False, target_filter: str = "alnum") -> Tuple[Dataset, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for row_id, row in enumerate(split_ds):
        if row["lang"] != lang:
            continue
        raw = as_tokens(row["raw"])
        norm = as_tokens(row["norm"])
        if len(raw) != len(norm):
            skipped.append({"row_id": row_id, "lang": lang, "reason": "length_mismatch", "raw": raw, "norm": norm})
            continue
        for i, (r, g) in enumerate(zip(raw, norm)):
            if target_filter == "alnum" and not official_alnum_filter(r):
                continue
            marked = raw[:i] + ["<extra_id_0>", r, "<extra_id_1>"] + raw[i + 1 :]
            input_text = " ".join(marked)
            if add_lang_prefix:
                input_text = f"<lang={lang}> " + input_text
            rows.append({
                "example_id": f"{lang}_{row_id}_{i}",
                "lang": lang,
                "row_id": row_id,
                "target_index": i,
                "raw_token": r,
                "gold_norm": g,
                "gold_changed": r != g,
                "tokens_json": tokens_json(raw),
                "input_text": input_text,
                "target_text": g,
            })
    return Dataset.from_list(rows), skipped


def build_target_examples_per_lang(
    dataset_name: str,
    output_root: str | Path,
    langs: List[str],
    add_lang_prefix: bool = False,
    target_filter: str = "alnum",
) -> Dict[str, Dict[str, int]]:
    ds = load_dataset(dataset_name, token=require_hf_token())
    actual = sorted(set(ds["train"]["lang"]))
    print("actual dataset languages:", actual)
    root = ensure_dir(output_root)
    summary: Dict[str, Dict[str, int]] = {}
    for lang in langs:
        if lang not in actual:
            print(f"[skip] {lang}: not in dataset. actual={actual}")
            continue
        dd = {}
        skipped_all = {}
        for split in ds.keys():
            built, skipped = build_target_examples_for_lang(ds[split], lang, add_lang_prefix, target_filter)
            dd[split] = built
            skipped_all[split] = skipped
        out_dir = root / (f"{lang}_with_lang_prefix" if add_lang_prefix else lang)
        DatasetDict(dd).save_to_disk(str(out_dir))
        with open(out_dir / "skipped_alignment_cases.json", "w", encoding="utf-8") as f:
            json.dump(skipped_all, f, ensure_ascii=False, indent=2)
        summary[lang] = {split: len(dd[split]) for split in dd}
        print(lang, summary[lang], "->", out_dir)
    return summary


def _take_with_social_mix(bucket_social: List[Dict[str, Any]], bucket_ordinary: List[Dict[str, Any]], n: int, social_ratio: float) -> List[Dict[str, Any]]:
    n_social = min(len(bucket_social), int(round(n * social_ratio)))
    picked = bucket_social[:n_social]
    picked += bucket_ordinary[: max(0, n - len(picked))]
    if len(picked) < n:
        picked += bucket_social[n_social : n_social + (n - len(picked))]
    return picked[:n]


def collect_audit_cases(split_ds, lang: str, n: int, changed_ratio: float, seed: int, social_ratio: float = 0.25) -> List[Dict[str, Any]]:
    changed_social: List[Dict[str, Any]] = []
    changed_ordinary: List[Dict[str, Any]] = []
    unchanged_social: List[Dict[str, Any]] = []
    unchanged_ordinary: List[Dict[str, Any]] = []
    for row_id, row in enumerate(split_ds):
        if row["lang"] != lang:
            continue
        raw = as_tokens(row["raw"])
        norm = as_tokens(row["norm"])
        if len(raw) != len(norm):
            continue
        sentence = " ".join(raw)
        for i, (r, g) in enumerate(zip(raw, norm)):
            social_candidate = is_social_context_candidate(r, sentence, lang)
            item = {
                "case_id": f"{lang}_{row_id}_{i}",
                "split": "validation",
                "lang": lang,
                "row_id": row_id,
                "tokens_json": tokens_json(raw),
                "gold_tokens_json": tokens_json(norm),
                "sentence_raw": sentence,
                "sentence_norm": " ".join(norm),
                "target_index": i,
                "target_token": r,
                "raw_token": r,
                "gold_norm": g,
                "gold_changed": r != g,
                "social_context_candidate": social_candidate,
                "gold_accept": "",
                "human_confidence": "",
            }
            if r != g and social_candidate:
                changed_social.append(item)
            elif r != g:
                changed_ordinary.append(item)
            elif social_candidate:
                unchanged_social.append(item)
            else:
                unchanged_ordinary.append(item)
    rng = random.Random(seed)
    for bucket in [changed_social, changed_ordinary, unchanged_social, unchanged_ordinary]:
        rng.shuffle(bucket)
    n_changed = int(round(n * changed_ratio))
    n_unchanged = n - n_changed
    picked = _take_with_social_mix(changed_social, changed_ordinary, n_changed, social_ratio)
    picked += _take_with_social_mix(unchanged_social, unchanged_ordinary, n_unchanged, social_ratio)
    if len(picked) < n:
        rest = changed_social + changed_ordinary + unchanged_social + unchanged_ordinary
        used = {x["case_id"] for x in picked}
        rest = [x for x in rest if x["case_id"] not in used]
        rng.shuffle(rest)
        picked += rest[: n - len(picked)]
    rng.shuffle(picked)
    for x in picked:
        prefix = "changed" if x["gold_changed"] else "unchanged"
        suffix = "social_context_candidate" if x["social_context_candidate"] else "ordinary"
        x["sample_group"] = f"{prefix}_{suffix}"
    return picked[:n]


def build_audit_cases(
    dataset_name: str,
    output: str | Path,
    n_ko: int = 1000,
    n_en: int = 300,
    split: str = "validation",
    changed_ratio: float = 0.7,
    seed: int = 42,
    social_ratio: float = 0.25,
) -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split, token=require_hf_token())
    actual = sorted(set(ds["lang"]))
    print("actual dataset languages:", actual)
    rows: List[Dict[str, Any]] = []
    if "ko" in actual:
        rows += collect_audit_cases(ds, "ko", n_ko, changed_ratio, seed, social_ratio)
    else:
        print("[skip] ko not in split")
    if "en" in actual:
        rows += collect_audit_cases(ds, "en", n_en, changed_ratio, seed + 1, social_ratio)
    else:
        print("[skip] en not in split")
    df = pd.DataFrame(rows)
    ensure_dir(Path(output).parent)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return df


def build_retrieval_index(dataset_name: str, output: str | Path, langs: List[str], split: str = "train") -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split, token=require_hf_token())
    actual = sorted(set(ds["lang"]))
    print("actual dataset languages:", actual)
    keep = set(langs)
    counts = defaultdict(Counter)
    examples = defaultdict(list)
    for row_id, row in enumerate(ds):
        lang = row["lang"]
        if keep and lang not in keep:
            continue
        raw = as_tokens(row["raw"])
        norm = as_tokens(row["norm"])
        if len(raw) != len(norm):
            continue
        for i, (r, g) in enumerate(zip(raw, norm)):
            key = (lang, r)
            counts[key][g] += 1
            if len(examples[key]) < 10:
                examples[key].append({"sentence": " ".join(raw), "target_index": i, "norm": g})
    rows = []
    for (lang, raw_tok), cnt in counts.items():
        total = sum(cnt.values())
        cands = [{"norm": n, "count": c, "prob": c / total} for n, c in cnt.most_common()]
        example_sentence = examples[(lang, raw_tok)][0]["sentence"] if examples[(lang, raw_tok)] else ""
        for cand in cands:
            rows.append({
                "lang": lang,
                "raw_token": raw_tok,
                "norm_token": cand["norm"],
                "count": cand["count"],
                "example_sentence": example_sentence,
                "total": total,
                "top_norm": cands[0]["norm"],
                "top_prob": cands[0]["prob"],
                "num_candidates": len(cands),
                "candidates_json": json.dumps(cands, ensure_ascii=False),
                "examples_json": json.dumps(examples[(lang, raw_tok)], ensure_ascii=False),
            })
    columns = ["lang", "raw_token", "norm_token", "count", "example_sentence", "total", "top_norm", "top_prob", "num_candidates", "candidates_json", "examples_json"]
    df = pd.DataFrame(rows, columns=columns)
    if len(df):
        df = df.sort_values(["lang", "raw_token", "count"], ascending=[True, True, False])
    ensure_dir(Path(output).parent)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return df
