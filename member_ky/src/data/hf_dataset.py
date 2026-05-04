from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd


DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


def load_local_env() -> None:
    for env_file in [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]:
        if not env_file.exists():
            continue
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)
            if key.lower() == "hf_token":
                os.environ.setdefault("HF_TOKEN", value)
            if key.lower() == "hugging_face_hub_token":
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", value)
        break


def read_split(split: str, dataset_name: str = DATASET_NAME) -> pd.DataFrame:
    parquet_path = "hf://datasets/" + dataset_name + "/" + SPLITS[split]
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("hf_token")
        or os.environ.get("hugging_face_hub_token")
    )
    storage_options = {"token": token} if token else None
    try:
        return pd.read_parquet(parquet_path, storage_options=storage_options)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {split} from {dataset_name}. Check member_ky/.env "
            "and Hugging Face dataset access."
        ) from exc


def normalize_tokens(value) -> list[str]:
    if isinstance(value, str):
        return value.split()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [str(token) for token in list(value)]
