from __future__ import annotations

import argparse
import csv
import math
import os
import random
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import pandas as pd


DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

AUDIT_LABELS = ["G", "P", "GP", "?"]
AUDIT_HEADERS = {
    "ko": ["라벨", "raw", "norm", "full sentence"],
    "en": ["Label", "raw", "norm", "full sentence"],
}


@dataclass(frozen=True)
class Assignment:
    reviewer: str
    lang: str
    rows: list[dict[str, str]]
    output_path: Path


@dataclass(frozen=True)
class RichText:
    before: str
    target: str
    after: str


def load_local_env() -> None:
    env_files = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
    ]

    for env_file in env_files:
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
            if key and key not in os.environ:
                os.environ[key] = value
            if key.lower() == "hf_token" and "HF_TOKEN" not in os.environ:
                os.environ["HF_TOKEN"] = value
            if (
                key.lower() == "hugging_face_hub_token"
                and "HUGGING_FACE_HUB_TOKEN" not in os.environ
            ):
                os.environ["HUGGING_FACE_HUB_TOKEN"] = value
        break


def read_split(dataset_name: str, split: str) -> pd.DataFrame:
    parquet_path = "hf://datasets/" + dataset_name + "/" + SPLITS[split]

    # This keeps the same loading style as the shared snippet. If the gated
    # dataset needs an explicit token, retry with the token loaded from .env.
    try:
        return pd.read_parquet(parquet_path)
    except Exception as first_exc:
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("hf_token")
            or os.environ.get("hugging_face_hub_token")
        )
        if not token:
            raise RuntimeError(
                "Failed to read the gated Hugging Face parquet. Create a root .env "
                "or member_ky/.env file with HF_TOKEN=your_token_here, and make "
                f"sure the same account has access to {dataset_name}."
            ) from first_exc
        try:
            return pd.read_parquet(parquet_path, storage_options={"token": token})
        except Exception as second_exc:
            raise RuntimeError(
                "Failed to read the gated Hugging Face parquet even with HF_TOKEN. "
                "Check that the token is valid and the token account has accepted "
                f"access to {dataset_name}."
            ) from second_exc


def normalize_tokens(value) -> list[str]:
    if isinstance(value, str):
        return value.split()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [str(token) for token in list(value)]


def join_tokens(tokens: Iterable[str]) -> str:
    return " ".join(str(token) for token in tokens)


def sentence_parts(tokens: list[str], target_index: int) -> tuple[str, str, str]:
    before = join_tokens(tokens[:target_index])
    target = tokens[target_index] if 0 <= target_index < len(tokens) else ""
    after = join_tokens(tokens[target_index + 1 :])
    if before:
        before += " "
    if after:
        after = " " + after
    return before, target, after


def build_review_candidates(
    df: pd.DataFrame,
    split: str,
) -> list[dict[str, str]]:
    required_columns = {"raw", "norm", "lang"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    candidates: list[dict[str, str]] = []
    for source_row, row in df.reset_index(drop=True).iterrows():
        raw_tokens = normalize_tokens(row["raw"])
        norm_tokens = normalize_tokens(row["norm"])
        lang = str(row["lang"])

        if len(raw_tokens) != len(norm_tokens):
            raise ValueError(
                f"raw/norm token length mismatch at source row {source_row}: "
                f"{len(raw_tokens)} != {len(norm_tokens)}"
            )

        for token_index, (raw_token, gold_norm) in enumerate(zip(raw_tokens, norm_tokens)):
            before, target, after = sentence_parts(raw_tokens, token_index)
            candidates.append(
                {
                    "split": split,
                    "lang": lang,
                    "raw": raw_token,
                    "norm": gold_norm,
                    "changed": "Y" if raw_token != gold_norm else "N",
                    "sentence_before": before,
                    "sentence_target": target,
                    "sentence_after": after,
                }
            )

    return candidates


def shuffled_lang_pool(
    candidates: list[dict[str, str]],
    lang: str,
    seed: int,
) -> list[dict[str, str]]:
    pool = [row for row in candidates if row["lang"] == lang]
    rng = random.Random(seed + sum(ord(char) for char in lang))
    rng.shuffle(pool)
    return pool


def reviewer_name(lang: str, index: int) -> str:
    return f"{lang}_reviewer_{index:02d}"


def split_evenly(items: list[dict[str, str]], count: int) -> list[list[dict[str, str]]]:
    base, remainder = divmod(len(items), count)
    chunks = []
    start = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


def split_round_robin(items: list[dict[str, str]], count: int) -> list[list[dict[str, str]]]:
    chunks = [[] for _ in range(count)]
    for index, item in enumerate(items):
        chunks[index % count].append(item)
    return chunks


def fill_per_reviewer_chunks(
    changed: list[dict[str, str]],
    unchanged: list[dict[str, str]],
    reviewer_count: int,
    item_count: int,
) -> list[list[dict[str, str]]]:
    chunks = [[] for _ in range(reviewer_count)]
    cursor = 0

    for pool in [changed, unchanged]:
        for row in pool:
            for _ in range(reviewer_count):
                target = cursor % reviewer_count
                cursor += 1
                if len(chunks[target]) < item_count:
                    chunks[target].append(row)
                    break
            if all(len(chunk) >= item_count for chunk in chunks):
                return chunks

    return chunks


def allocate_assignments(
    candidates: list[dict[str, str]],
    *,
    lang: str,
    reviewer_count: int,
    item_count: int,
    allocation: str,
    seed: int,
    output_dir: Path,
) -> list[Assignment]:
    pool = shuffled_lang_pool(candidates, lang, seed)
    changed_pool = [row for row in pool if row.get("changed") == "Y"]
    unchanged_pool = [row for row in pool if row.get("changed") != "Y"]
    prioritized_pool = changed_pool + unchanged_pool

    if allocation == "per-reviewer":
        needed = reviewer_count * item_count
        chunks = fill_per_reviewer_chunks(
            changed_pool,
            unchanged_pool,
            reviewer_count,
            item_count,
        )
    elif allocation == "total":
        needed = item_count
        chunks = split_round_robin(prioritized_pool[:item_count], reviewer_count)
    elif allocation == "same":
        needed = item_count
        shared = prioritized_pool[:item_count]
        chunks = [shared for _ in range(reviewer_count)]
    else:
        raise ValueError(f"Unknown allocation mode: {allocation}")

    if len(pool) < needed:
        raise ValueError(f"Not enough {lang} candidates. Need {needed}, found {len(pool)}.")

    assignments = []
    for index, rows in enumerate(chunks, start=1):
        reviewer = reviewer_name(lang, index)
        rows_with_reviewer = [{**row, "담당자": reviewer} for row in rows]
        output_path = output_dir / f"gold_audit_{lang}_reviewer_{index:02d}.xlsx"
        assignments.append(
            Assignment(
                reviewer=reviewer,
                lang=lang,
                rows=rows_with_reviewer,
                output_path=output_path,
            )
        )
    return assignments


def column_letter(column_number: int) -> str:
    result = ""
    while column_number:
        column_number, remainder = divmod(column_number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def text_node(value) -> str:
    text = escape(safe_text(value), {'"': "&quot;"})
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return f"<t{preserve}>{text}</t>"


def rich_text_runs(value: RichText) -> str:
    runs = []
    if value.before:
        runs.append(f"<r>{text_node(value.before)}</r>")
    runs.append(
        "<r><rPr><b/><color rgb=\"FFFF0000\"/></rPr>"
        f"{text_node(value.target)}</r>"
    )
    if value.after:
        runs.append(f"<r>{text_node(value.after)}</r>")
    return "".join(runs)


def inline_cell(row_index: int, col_index: int, value, style: int | None = None) -> str:
    ref = f"{column_letter(col_index)}{row_index}"
    style_attr = f' s="{style}"' if style is not None else ""
    if isinstance(value, RichText):
        return (
            f'<c r="{ref}" t="inlineStr"{style_attr}><is>'
            f"{rich_text_runs(value)}</is></c>"
        )
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is>{text_node(value)}</is></c>'


def worksheet_xml(
    rows,
    *,
    column_widths: dict[int, int] | None = None,
    freeze_header: bool = False,
    filter_header: bool = False,
    validation_column: int | None = None,
    validation_values: list[str] | None = None,
) -> str:
    column_widths = column_widths or {}
    max_column = max((len(row) for row in rows), default=1)
    max_row = max(len(rows), 1)

    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
    ]
    if freeze_header:
        parts.append(
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '<selection pane="bottomLeft"/>'
            "</sheetView></sheetViews>"
        )
    if column_widths:
        parts.append("<cols>")
        for col_index, width in sorted(column_widths.items()):
            parts.append(
                f'<col min="{col_index}" max="{col_index}" width="{width}" customWidth="1"/>'
            )
        parts.append("</cols>")

    parts.append("<sheetData>")
    for row_index, row in enumerate(rows, start=1):
        parts.append(f'<row r="{row_index}">')
        for col_index, value in enumerate(row, start=1):
            if value == "":
                continue
            style = 1 if row_index == 1 else 2
            parts.append(inline_cell(row_index, col_index, value, style))
        parts.append("</row>")
    parts.append("</sheetData>")

    if filter_header and rows:
        last_cell = f"{column_letter(max_column)}{max_row}"
        parts.append(f'<autoFilter ref="A1:{last_cell}"/>')

    if validation_column and validation_values and max_row >= 2:
        col = column_letter(validation_column)
        values = ",".join(validation_values)
        parts.append(
            '<dataValidations count="1">'
            '<dataValidation type="list" allowBlank="1" showErrorMessage="1" '
            f'sqref="{col}2:{col}{max_row}">'
            f"<formula1>&quot;{escape(values)}&quot;</formula1>"
            "</dataValidation>"
            "</dataValidations>"
        )

    parts.append(
        '<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" '
        'header="0.3" footer="0.3"/>'
    )
    parts.append("</worksheet>")
    return "".join(parts)


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = []
    for index, sheet_name in enumerate(sheet_names, start=1):
        sheets.append(
            f'<sheet name="{escape(sheet_name)}" sheetId="{index}" r:id="rId{index}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        + "".join(sheets)
        + "</sheets></workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for index in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for index in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + "</Types>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font>'
        "</fonts>"
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF305496"/></patternFill></fill>'
        "</fills>"
        '<borders count="2">'
        "<border><left/><right/><top/><bottom/><diagonal/></border>"
        '<border><left style="thin"/><right style="thin"/><top style="thin"/>'
        '<bottom style="thin"/><diagonal/></border>'
        "</borders>"
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="3">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" '
        'applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">'
        '<alignment horizontal="center" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" '
        'applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )


def app_xml(sheet_names: list[str]) -> str:
    titles = "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name in sheet_names)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Microsoft Excel</Application>"
        "<DocSecurity>0</DocSecurity>"
        "<ScaleCrop>false</ScaleCrop>"
        "<HeadingPairs><vt:vector size=\"2\" baseType=\"variant\">"
        "<vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>"
        f"<vt:variant><vt:i4>{len(sheet_names)}</vt:i4></vt:variant>"
        "</vt:vector></HeadingPairs>"
        f"<TitlesOfParts><vt:vector size=\"{len(sheet_names)}\" baseType=\"lpstr\">"
        + titles
        + "</vt:vector></TitlesOfParts>"
        "</Properties>"
    )


def core_xml() -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:creator>TeamProject-LexNorm</dc:creator>"
        "<cp:lastModifiedBy>TeamProject-LexNorm</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def sheet_name_for_lang(lang: str) -> str:
    return "Audit" if lang == "en" else "검수"


def audit_headers_for_lang(lang: str) -> list[str]:
    return AUDIT_HEADERS.get(lang, AUDIT_HEADERS["en"])


def audit_sheet_row(row: dict[str, str]) -> list:
    return [
        "",
        row["raw"],
        row["norm"],
        RichText(
            before=row["sentence_before"],
            target=row["sentence_target"],
            after=row["sentence_after"],
        ),
    ]


def write_xlsx(path: Path, review_rows: list[dict[str, str]], lang: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = [sheet_name_for_lang(lang)]
    review_sheet_rows = [audit_headers_for_lang(lang)] + [
        audit_sheet_row(row) for row in review_rows
    ]

    review_widths = {
        1: 12,
        2: 18,
        3: 18,
        4: 100,
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types_xml(len(sheet_names)))
        workbook.writestr("_rels/.rels", root_rels_xml())
        workbook.writestr("docProps/app.xml", app_xml(sheet_names))
        workbook.writestr("docProps/core.xml", core_xml())
        workbook.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheet_names)))
        workbook.writestr("xl/styles.xml", styles_xml())
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            worksheet_xml(
                review_sheet_rows,
                column_widths=review_widths,
                freeze_header=True,
                filter_header=True,
                validation_column=1,
                validation_values=AUDIT_LABELS,
            ),
        )


def write_manifest(path: Path, assignments: list[Assignment]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["file", "reviewer", "lang", "rows"],
        )
        writer.writeheader()
        for assignment in assignments:
            writer.writerow(
                {
                    "file": assignment.output_path.name,
                    "reviewer": assignment.reviewer,
                    "lang": assignment.lang,
                    "rows": len(assignment.rows),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Excel gold-label review sheets for MultiLexNorm."
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLITS),
        default=["train", "validation"],
        help="Dataset splits to sample from.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("member_ky/temp"))
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument(
        "--changed-only",
        action="store_true",
        dest="changed_only",
        help="Only include tokens where raw_token and gold_norm differ.",
    )
    parser.add_argument(
        "--include-agreements",
        action="store_false",
        dest="changed_only",
        help="Allow raw_token == gold_norm rows when there are not enough disagreements.",
    )
    parser.set_defaults(changed_only=True)
    parser.add_argument("--ko-reviewers", type=int, default=3)
    parser.add_argument("--ko-items", type=int, default=1000)
    parser.add_argument(
        "--ko-allocation",
        choices=["per-reviewer", "total", "same"],
        default="total",
    )
    parser.add_argument("--en-reviewers", type=int, default=1)
    parser.add_argument("--en-items", type=int, default=300)
    parser.add_argument(
        "--en-allocation",
        choices=["per-reviewer", "total", "same"],
        default="total",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env()

    candidates = []
    for split in args.splits:
        df = read_split(args.dataset, split)
        candidates.extend(build_review_candidates(df, split))

    if args.changed_only:
        candidates = [row for row in candidates if row["changed"] == "Y"]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments = []
    assignments.extend(
        allocate_assignments(
            candidates,
            lang="ko",
            reviewer_count=args.ko_reviewers,
            item_count=args.ko_items,
            allocation=args.ko_allocation,
            seed=args.seed,
            output_dir=output_dir,
        )
    )
    assignments.extend(
        allocate_assignments(
            candidates,
            lang="en",
            reviewer_count=args.en_reviewers,
            item_count=args.en_items,
            allocation=args.en_allocation,
            seed=args.seed,
            output_dir=output_dir,
        )
    )

    for assignment in assignments:
        write_xlsx(assignment.output_path, assignment.rows, assignment.lang)
        print(
            f"Wrote {assignment.output_path} "
            f"({assignment.lang}, {assignment.reviewer}, {len(assignment.rows)} rows)"
        )

    manifest_path = output_dir / "gold_audit_manifest.csv"
    write_manifest(manifest_path, assignments)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
