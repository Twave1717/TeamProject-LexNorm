from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_env_file(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not os.environ.get(key):
            os.environ[key] = value
    return True


def load_known_env_files() -> None:
    paths = []
    try:
        paths.append(Path.cwd() / ".env")
    except FileNotFoundError:
        pass
    paths.extend([
        Path("/content/lexnorm_submit/.env"),
        Path("/drive/MyDrive/AI개론_박진영/.env"),
        Path("/content/drive/MyDrive/AI개론_박진영/.env"),
    ])
    for path in paths:
        load_env_file(path)


def env_value(*names: str) -> str | None:
    lowered = {key.lower(): value for key, value in os.environ.items()}
    for name in names:
        value = os.environ.get(name) or lowered.get(name.lower())
        if value:
            return value
    return None


def drive_output_root() -> Path | None:
    """Google Drive mirror root. Returns None outside mounted Colab Drive."""
    for root in [
        Path("/drive/MyDrive/AI개론_박진영"),
        Path("/content/drive/MyDrive/AI개론_박진영"),
    ]:
        if root.exists():
            return root / "lexnorm_outputs"
    return None


def sync_to_drive(path: str | Path, local_root: str | Path = "outputs") -> Path | None:
    """Copy one output file/directory to Drive, preserving the path under outputs/."""
    src = Path(path)
    if not src.exists():
        print(f"[drive sync skip] missing: {src}")
        return None
    drive_root = drive_output_root()
    if drive_root is None:
        print("[drive sync skip] Google Drive is not mounted.")
        return None

    local_root = Path(local_root)
    try:
        rel = src.relative_to(local_root)
    except ValueError:
        rel = Path(src.name)
    dst = drive_root / rel
    ensure_dir(dst.parent)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f"[drive sync] {src} -> {dst}")
    return dst


def sync_many_to_drive(paths: List[str | Path], local_root: str | Path = "outputs") -> List[Path]:
    copied: List[Path] = []
    for path in paths:
        dst = sync_to_drive(path, local_root)
        if dst is not None:
            copied.append(dst)
    return copied


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_tokens(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(t) for t in x]
    if isinstance(x, tuple):
        return [str(t) for t in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("["):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(t) for t in obj]
            except Exception:
                pass
        return s.split()
    try:
        return [str(t) for t in list(x)]
    except Exception:
        return [str(x)]


def tokens_json(tokens: List[str]) -> str:
    return json.dumps(tokens, ensure_ascii=False)


def safe_json_loads(s: Any, default=None):
    if isinstance(s, (dict, list)):
        return s
    if pd.isna(s):
        return default
    try:
        return json.loads(str(s))
    except Exception:
        return default


def official_alnum_filter(word: str) -> bool:
    """UFAL-style target filter: remove apostrophe, hyphen, spaces, then check alnum.

    This follows the 2021 UFAL MultiLexNorm dataset filter style.
    """
    w = str(word).replace("'", "").replace("-", "").replace(" ", "")
    return bool(w) and w.isalnum()


def official_alnum_postprocess(raw: str, pred: str) -> str:
    """UFAL-style conservative postprocess for digits / non-alnum tokens."""
    raw = str(raw)
    pred = str(pred).strip()
    if raw.isdigit() and len(raw) > 1:
        return raw
    w = raw.replace("'", "").replace("-", "").replace(" ", "")
    if not w.isalnum():
        return raw
    if pred == "":
        return raw
    return pred


def load_metadata_jsonl(path: str | Path | None) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    return {row.get("case_id"): row for row in read_jsonl(path) if row.get("case_id")}
