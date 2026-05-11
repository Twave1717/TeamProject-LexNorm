from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from .metrics import add_error_modes, evaluate_rows
from .utils import ensure_dir, official_alnum_postprocess


class FixedExamplesPerEpochTrainer(Seq2SeqTrainer):
    """매 epoch마다 지정한 개수만큼 train example을 샘플링한다."""

    def __init__(self, *args, train_examples_per_epoch: int | None = None, **kwargs):
        self.train_examples_per_epoch = train_examples_per_epoch
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if self.train_examples_per_epoch and dataset is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.args.seed))
            replacement = len(dataset) < self.train_examples_per_epoch
            return RandomSampler(
                dataset,
                replacement=replacement,
                num_samples=self.train_examples_per_epoch,
                generator=generator,
            )
        return super()._get_train_sampler(train_dataset)


def load_tokenizer(model_name: str, trust_remote_code: bool = False):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)


def fine_tune_seq2seq(
    model_name_or_path: str,
    data_dir: str | Path,
    output_dir: str | Path,
    train_split: str = "train",
    eval_split: str = "validation",
    learning_rate: float = 1e-4,
    num_train_epochs: float = 10,
    per_device_train_batch_size: int = 128,
    per_device_eval_batch_size: int = 64,
    gradient_accumulation_steps: int = 1,
    train_examples_per_epoch: int | None = None,
    max_source_length: int = 200,
    max_target_length: int = 32,
    fp16: bool = False,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    seed: int = 42,
    trust_remote_code: bool = False,
    eval_accumulation_steps: int | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    logging_steps: int = 100,
    lr_scheduler_type: str = "constant",
) -> None:
    set_seed(seed)
    ds = load_from_disk(str(data_dir))
    if eval_split not in ds or len(ds[eval_split]) == 0:
        raise ValueError(f"No non-empty eval split {eval_split} in {data_dir}")
    if train_split not in ds or len(ds[train_split]) == 0:
        raise ValueError(f"No non-empty train split {train_split} in {data_dir}")
    train_ds = ds[train_split]
    eval_ds = ds[eval_split]

    effective_train_len = train_examples_per_epoch or len(train_ds)
    updates_per_epoch = math.ceil(effective_train_len / (per_device_train_batch_size * gradient_accumulation_steps))
    total_updates = math.ceil(updates_per_epoch * num_train_epochs)
    print(
        f"[train] model={model_name_or_path} data={data_dir} output={output_dir} "
        f"train_pool={len(train_ds)} train_examples_per_epoch={effective_train_len} eval={len(eval_ds)} epochs={num_train_epochs} "
        f"batch={per_device_train_batch_size} grad_accum={gradient_accumulation_steps} "
        f"effective_batch={per_device_train_batch_size * gradient_accumulation_steps} "
        f"updates_per_epoch~{updates_per_epoch} total_updates~{total_updates}",
        flush=True,
    )

    print("[train] loading tokenizer/model", flush=True)
    tokenizer = load_tokenizer(model_name_or_path, trust_remote_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    def preprocess(batch):
        x = tokenizer(batch["input_text"], max_length=max_source_length, truncation=True)
        y = tokenizer(text_target=batch["target_text"], max_length=max_target_length, truncation=True)
        x["labels"] = y["input_ids"]
        return x

    print("[train] tokenizing train/eval splits", flush=True)
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)
    print("[train] tokenization complete", flush=True)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_wandb = bool(wandb_project and os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_entity:
            os.environ["WANDB_ENTITY"] = wandb_entity
        if wandb_run_name:
            os.environ["WANDB_NAME"] = wandb_run_name
        if wandb_group:
            os.environ["WANDB_RUN_GROUP"] = wandb_group

    ta_kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # 제출용 출력은 간결하게 유지한다. 중간 epoch checkpoint는 저장하지 않는다.
        save_strategy="no",
        predict_with_generate=True,
        generation_num_beams=1,
        generation_max_length=max_target_length,
        optim="adafactor",
        fp16=fp16,
        bf16=bf16,
        report_to="wandb" if use_wandb else "none",
        logging_strategy="steps",
        logging_steps=logging_steps,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
    )
    if eval_accumulation_steps is not None:
        ta_kwargs["eval_accumulation_steps"] = eval_accumulation_steps
    ta_kwargs["eval_strategy"] = "epoch"
    args = Seq2SeqTrainingArguments(**ta_kwargs)

    trainer = FixedExamplesPerEpochTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        processing_class=tokenizer,
        train_examples_per_epoch=train_examples_per_epoch,
    )
    print("[train] trainer.train() start", flush=True)
    try:
        trainer.train()
        print("[train] trainer.train() complete", flush=True)
        out = ensure_dir(output_dir)
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))
        (out / "train_complete.json").write_text('{"complete": true}', encoding="utf-8")
        print(f"[train] saved final model -> {out}", flush=True)
    finally:
        if use_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
                    print("[train] wandb run finished", flush=True)
            except Exception as exc:
                print(f"[train] wandb finish skipped: {exc}", flush=True)


@torch.no_grad()
def predict_seq2seq(
    model_name_or_path: str,
    data_dir: str | Path,
    output_csv: str | Path,
    split: str = "validation",
    batch_size: int = 32,
    max_source_length: int = 200,
    max_new_tokens: int = 32,
    num_beams: int = 1,
    trust_remote_code: bool = False,
    postprocess: str = "alnum",
) -> pd.DataFrame:
    ds = load_from_disk(str(data_dir))[split]
    tokenizer = load_tokenizer(model_name_or_path, trust_remote_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def collate(batch):
        enc = tokenizer([x["input_text"] for x in batch], padding=True, truncation=True, max_length=max_source_length, return_tensors="pt")
        return batch, enc

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    rows: List[Dict[str, Any]] = []
    for batch, enc in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(**enc, num_beams=num_beams, max_new_tokens=max_new_tokens, do_sample=False)
        preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for ex, pred in zip(batch, preds):
            pred = str(pred).strip()
            if postprocess == "alnum":
                pred = official_alnum_postprocess(ex["raw_token"], pred)
            rows.append({
                "example_id": ex["example_id"],
                "lang": ex["lang"],
                "row_id": ex["row_id"],
                "target_index": ex["target_index"],
                "raw_token": ex["raw_token"],
                "gold_norm": ex["gold_norm"],
                "gold_changed": ex["gold_changed"],
                "pred_norm": pred,
                "pred_changed": pred != ex["raw_token"],
                "correct": pred == ex["gold_norm"],
            })
    df = add_error_modes(pd.DataFrame(rows))
    ensure_dir(Path(output_csv).parent)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def evaluate_prediction_csv(
    pred_csv: str | Path,
    out_csv: str | Path,
    model: str | None = None,
    lang: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    result = evaluate_rows(df.to_dict("records"))
    if model is None:
        model = Path(pred_csv).parent.name
    if lang is None and "lang" in df.columns and len(df):
        langs = sorted(str(x) for x in df["lang"].dropna().unique())
        lang = langs[0] if len(langs) == 1 else ",".join(langs)
    result = {"model": model, "lang": lang or "", **result}
    out = pd.DataFrame([result])
    ensure_dir(Path(out_csv).parent)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out
