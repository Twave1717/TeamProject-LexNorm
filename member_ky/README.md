# KY Experiments

Personal workspace for KY's hypotheses, scripts, notes, and local artifacts.

Recommended files:

- `run_model.py`: run one or more model baselines
- `compare_model.py`: compare token-level outputs from two or more models
- `notes.md`: experiment log, scores, and next actions
- `artifacts/`: local outputs that should usually remain untracked
- `src/data/`: dataset loading and token normalization helpers
- `src/evaluation/`: Accuracy, ERR, TP/FP/FN, and comparison utilities
- `src/model/`: rule-based and public ByT5 model runners
- `src/util/`: CSV/JSONL and shared IO helpers

Run from the repository root:

```bash
uv run python member_ky/run_model.py --list-models
```

Run multiple models with the same output schema:

```bash
uv run python member_ky/run_model.py --models rule_based public_byt5 --langs en
```

Compare selected model outputs:

```bash
uv run python member_ky/compare_model.py --models rule_based public_byt5 --langs en
```
