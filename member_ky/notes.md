# KY Notes

## Hypothesis

- Rule-based MFR baseline gives a reproducible floor for MultiLexNorm 2026.
- T5/ByT5 should be compared token-by-token against MFR to find where neural
  context helps and where memorized replacement is still stronger.

## Changes

- Added `run_model.py` for professor-style MFR and public ÚFAL ByT5 inference.
- Added `compare_model.py` for later Rule > T5 / T5 > Rule / both fail
  case extraction, with support for more than two models.
- Added `T5_BYT5_PLAN.md` with public model list, model size notes, and
  comparison workflow.
- Refactored experiment code under `member_ky/src/{data,evaluation,model,util}`.
- Unified rule-based and ByT5 evaluation metrics.
- Consolidated model entrypoints into `run_model.py` and `compare_model.py`.

## Results

- Rule-based baseline outputs are under `member_ky/temp/models/rule_based/`.
- Korean validation: Accuracy 0.9223, ERR 0.1205, action TP/FP/FN 31/7/135.
- English validation: Accuracy 0.9737, ERR 0.6193, action TP/FP/FN 443/38/190.
- Public 2021 ByT5 checkpoints exist for `de`, `en`, `hr`, `iden`, `nl`, `sr`
  among our 2026 languages. No direct public checkpoint found for `ko`, `ja`,
  `sl`, `th`, `vi`, or monolingual `id`.
- Installed ByT5 inference dependencies in the local `.venv`.
- Smoke-tested `ufal/byt5-small-multilexnorm2021-en` on 5 English validation
  tokens and verified comparison output generation.
