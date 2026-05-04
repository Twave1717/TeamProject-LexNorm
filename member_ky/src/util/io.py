from __future__ import annotations

import csv
import json
from pathlib import Path


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def assemble_full_predictions(token_rows: list[dict]) -> list[dict]:
    sentences: dict[tuple[str, int], dict] = {}
    for row in token_rows:
        key = (row["lang"], int(row["sent_id"]))
        sentence = sentences.setdefault(
            key,
            {
                "lang": row["lang"],
                "sent_id": int(row["sent_id"]),
                "raw": [],
                "gold": [],
                "pred": [],
            },
        )
        sentence["raw"].append(row["raw"])
        sentence["gold"].append(row["gold"])
        sentence["pred"].append(row["pred"])
    return [sentences[key] for key in sorted(sentences)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        for row in rows:
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
