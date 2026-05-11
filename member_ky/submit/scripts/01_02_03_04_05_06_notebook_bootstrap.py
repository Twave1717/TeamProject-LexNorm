from __future__ import annotations

import os
import sys
from pathlib import Path


def _safe_cwd() -> Path | None:
    try:
        return Path.cwd()
    except FileNotFoundError:
        return None


def setup_project() -> Path:
    # Colab에서 /content/lexnorm_submit을 재압축 해제하면 이전 cwd가 삭제될 수 있다.
    # 그래서 cwd보다 bootstrap 파일 위치와 고정 Colab 경로를 우선 사용한다.
    candidates: list[Path] = [Path(__file__).resolve().parents[1]]
    cwd = _safe_cwd()
    if cwd is not None:
        candidates.append(cwd)
    candidates.append(Path("/content/lexnorm_submit"))

    seen: set[str] = set()
    for root in candidates:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if (root / "src" / "lexnorm").exists():
            root = root.resolve()
            os.chdir(root)
            src = root / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            print("PROJECT_ROOT =", root)
            return root

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError("Run 00_setup_and_data.ipynb first. Could not find src/lexnorm under:\n" + searched)
