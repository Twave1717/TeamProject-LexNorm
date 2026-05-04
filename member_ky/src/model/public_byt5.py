from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from member_ky.src.data import normalize_tokens
from member_ky.src.evaluation import SUMMARY_FIELDS, TOKEN_FIELDS
from member_ky.src.evaluation import evaluate_token_rows, token_confusion
from member_ky.src.util import assemble_full_predictions, batched, write_csv, write_jsonl


PUBLIC_MODELS = {
    "da": "ufal/byt5-small-multilexnorm2021-da",
    "de": "ufal/byt5-small-multilexnorm2021-de",
    "en": "ufal/byt5-small-multilexnorm2021-en",
    "es": "ufal/byt5-small-multilexnorm2021-es",
    "hr": "ufal/byt5-small-multilexnorm2021-hr",
    "iden": "ufal/byt5-small-multilexnorm2021-iden",
    "it": "ufal/byt5-small-multilexnorm2021-it",
    "nl": "ufal/byt5-small-multilexnorm2021-nl",
    "sr": "ufal/byt5-small-multilexnorm2021-sr",
    "tr": "ufal/byt5-small-multilexnorm2021-tr",
}

SUPPORTED_2026_LANGS = ["de", "en", "hr", "iden", "nl", "sr"]


def require_transformers():
    try:
        import torch
        from transformers import ByT5Tokenizer, T5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Missing ByT5 inference dependencies. Install once with:\n"
            "uv pip install -r member_ky/requirements_byt5.txt"
        ) from exc
    return torch, ByT5Tokenizer, T5ForConditionalGeneration


def marked_input(raw_tokens: list[str], token_id: int) -> str:
    marked = (
        raw_tokens[:token_id]
        + ["<extra_id_0>", raw_tokens[token_id], "<extra_id_1>"]
        + raw_tokens[token_id + 1 :]
    )
    return " ".join(marked)


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def alnum_postprocess(raw: str, pred: str) -> str:
    if raw.isdigit() and len(raw) > 1:
        return raw
    if not raw.replace("'", "").isalnum():
        return raw
    return pred


def make_token_examples(df: pd.DataFrame, lang: str, limit_tokens: int | None) -> list[dict]:
    lang_df = df.loc[df["lang"] == lang].reset_index(drop=True)
    examples = []
    for sent_id, row in lang_df.iterrows():
        raw_tokens = normalize_tokens(row["raw"])
        gold_tokens = normalize_tokens(row["norm"])
        for token_id, (raw, gold) in enumerate(zip(raw_tokens, gold_tokens)):
            examples.append(
                {
                    "lang": lang,
                    "sent_id": sent_id,
                    "token_id": token_id,
                    "raw": raw,
                    "gold": gold,
                    "input": marked_input(raw_tokens, token_id),
                }
            )
            if limit_tokens and len(examples) >= limit_tokens:
                return examples
    return examples


def predict_examples(
    examples: list[dict],
    model_id: str,
    *,
    batch_size: int,
    device_name: str,
    num_beams: int,
    max_new_tokens: int,
    postprocess: str,
) -> list[dict]:
    torch, ByT5Tokenizer, T5ForConditionalGeneration = require_transformers()
    device = torch.device(choose_device(torch, device_name))

    tokenizer = ByT5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    token_rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(batched(examples, batch_size), start=1):
            encoded = tokenizer(
                [row["input"] for row in batch],
                padding=True,
                truncation=False,
                pad_to_multiple_of=8,
                return_attention_mask=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model.generate(
                **encoded,
                num_beams=num_beams,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.0,
                length_penalty=1.0,
            )
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for source_row, pred in zip(batch, predictions):
                pred = pred.replace("\n", "").replace("\t", " ").strip()
                if postprocess == "alnum":
                    pred = alnum_postprocess(source_row["raw"], pred)
                token_rows.append(
                    {
                        "lang": source_row["lang"],
                        "sent_id": source_row["sent_id"],
                        "token_id": source_row["token_id"],
                        "raw": source_row["raw"],
                        "gold": source_row["gold"],
                        "pred": pred,
                        **token_confusion(source_row["raw"], source_row["gold"], pred),
                    }
                )

            print(
                f"{model_id}: batch {batch_index} / "
                f"{math.ceil(len(examples) / batch_size)}",
                flush=True,
            )

    return token_rows


def write_lang_outputs(
    output_dir: Path,
    lang: str,
    model_id: str,
    token_rows: list[dict],
) -> dict:
    lang_dir = output_dir / lang
    full_rows = assemble_full_predictions(token_rows)

    write_jsonl(lang_dir / "full_predictions.jsonl", full_rows)
    write_csv(lang_dir / "token_predictions.csv", token_rows, TOKEN_FIELDS)

    summary = {"lang": lang, "model": model_id, **evaluate_token_rows(token_rows)}
    write_csv(lang_dir / "summary.csv", [summary], SUMMARY_FIELDS)
    return summary


def print_public_models() -> None:
    print("Public ÚFAL MultiLexNorm 2021 checkpoints:")
    for lang, model_id in sorted(PUBLIC_MODELS.items()):
        marker = " (2026 overlap)" if lang in SUPPORTED_2026_LANGS else ""
        print(f"- {lang}: {model_id}{marker}")
