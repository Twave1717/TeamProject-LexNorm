# Gold Audit Excel Sheets

이 폴더는 human audit용 엑셀 파일을 생성하는 위치입니다.

루트 디렉터리에서 실행:

```bash
uv run python member_ky/create_gold_review_sheets.py
```

데이터셋이 gated라서 루트 `.env` 또는 `member_ky/.env`에 접근 승인된
Hugging Face 토큰이 필요합니다.

```text
HF_TOKEN=your_token_here
```

기본 생성 파일:

- `gold_audit_ko_reviewer_01.xlsx`
- `gold_audit_ko_reviewer_02.xlsx`
- `gold_audit_ko_reviewer_03.xlsx`
- `gold_audit_en_reviewer_01.xlsx`
- `gold_audit_manifest.csv`

기본값은 `train` + `validation`에서 raw와 norm이 다른 disagreement만
샘플링합니다. 한국어는 총 1000개를 3명에게 나누고, 영어는 300개를
1명에게 배정합니다.

엑셀 컬럼은 4개만 사용합니다.

- `Label` / `라벨`
- `raw`
- `norm`
- `full sentence`

라벨 값:

- `G`: gold is wrong
- `P`: sLLM pred is wrong
- `GP`: both are wrong
- `?`: uncertain

한국어 같은 1000개를 3명이 모두 보게 하려면 다음처럼 실행합니다.

```bash
uv run python member_ky/create_gold_review_sheets.py --ko-allocation same
```

raw와 norm이 같은 항목도 포함하려면 다음처럼 실행합니다.

```bash
uv run python member_ky/create_gold_review_sheets.py --include-agreements
```
