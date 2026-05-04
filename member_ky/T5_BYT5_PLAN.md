# T5 / ByT5 Baseline Plan

## Current Baseline Reference

- Professor demo repository: `WeerayutBu/MultiLexNorm2026`
- The rule-based baseline in that demo is MFR, Most Frequent Replacement.
- Current local runner: `member_ky/run_model.py --models rule_based`

## Local Code Layout

- `member_ky/src/data/`: Hugging Face parquet loading and token normalization
- `member_ky/src/evaluation/`: shared Accuracy, ERR, TP/FP/FN, and case comparison
- `member_ky/src/model/`: MFR and public ByT5 runners
- `member_ky/src/util/`: CSV, JSONL, batching, and full prediction assembly

The two CLI scripts in `member_ky/` are intentionally thin wrappers around
`src/`: `run_model.py` for prediction/evaluation and `compare_model.py` for
token-level case comparison. Rule-based and ByT5 outputs use the same
token-level columns and summary metrics.

## Public Fine-Tuned Checkpoints

Official 2021 ÚFAL checkpoints are public on Hugging Face:

- `ufal/byt5-small-multilexnorm2021-da`
- `ufal/byt5-small-multilexnorm2021-de`
- `ufal/byt5-small-multilexnorm2021-en`
- `ufal/byt5-small-multilexnorm2021-es`
- `ufal/byt5-small-multilexnorm2021-hr`
- `ufal/byt5-small-multilexnorm2021-iden`
- `ufal/byt5-small-multilexnorm2021-it`
- `ufal/byt5-small-multilexnorm2021-nl`
- `ufal/byt5-small-multilexnorm2021-sr`
- `ufal/byt5-small-multilexnorm2021-tr`

Overlap with the 2026 dev dataset currently in this repo:

- directly usable: `de`, `en`, `hr`, `iden`, `nl`, `sr`
- no direct 2021 checkpoint found: `id`, `ja`, `ko`, `sl`, `th`, `vi`

The local inference runner is:

```bash
uv run python member_ky/run_model.py --list-models
```

Install the extra inference dependencies once:

```bash
uv pip install -r member_ky/requirements_byt5.txt
```

Small smoke run:

```bash
uv run python member_ky/run_model.py \
  --models public_byt5 \
  --langs en \
  --limit-tokens 50 \
  --batch-size 4
```

This has been smoke-tested locally with `--limit-tokens 5`. The English
checkpoint cache is about 1.1 GB.

Full English validation run:

```bash
uv run python member_ky/run_model.py --models public_byt5 --langs en --batch-size 8
```

Outputs go to `member_ky/temp/models/public_byt5/<lang>/`:

- `full_predictions.jsonl`
- `token_predictions.csv`
- `summary.csv`

The summary includes:

- Accuracy
- LAI
- ERR
- action TP/FP/FN/TN: whether the model decided to change a token
- correction TP/FP/FN/TN: whether the produced normalization is correct

## Model Size Check

T5 family:

- `t5-small`: about 60M parameters
- `t5-base`: about 220M parameters
- `t5-large`: about 770M parameters
- `t5-3b`: about 3B parameters
- `t5-11b`: about 11B parameters

ByT5 family:

- `google/byt5-small`: 300M parameters
- `google/byt5-base`: 580M parameters
- `google/byt5-large`: 1.2B parameters
- `google/byt5-xl`: 3.7B parameters
- `google/byt5-xxl`: 13B parameters

Practical starting point: `google/byt5-small`. It is already 300M parameters and
the PyTorch checkpoint is roughly 1.2 GB, so it is the smallest sensible ByT5
run for a local baseline.

## Comparison Outputs

After selected models create token-level prediction CSVs under
`member_ky/temp/models/<model>/<lang>/token_predictions.csv`, run:

```bash
uv run python member_ky/compare_model.py \
  --models rule_based public_byt5 \
  --langs en
```

This creates:

- `combined_token_comparison.csv`
- `comparison_summary.csv`
- `pairs/<model_a>_vs_<model_b>/<case_type>.csv`

## Notes

The 2021 winning paper uses ByT5 with each target word marked in context and
predicts the normalization for that word. To keep the first pass comparable,
start by running an existing published ByT5 implementation/checkpoint as-is
before changing prompts, synthetic data, or fine-tuning settings.
