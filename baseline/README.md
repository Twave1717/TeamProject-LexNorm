# MultiLexNorm 2026 Baseline

This workspace contains a runnable implementation of the official MFR
(Most-Frequent-Replacement) baseline for the MultiLexNorm shared task.

- Notebook reference: `demo.ipynb`
- CLI entrypoint: `run_baseline.py`
- Example submissions: `outputs/submission_dev.zip`, `outputs/submission_full.zip`
- Importable helpers: `baseline.shared`, `baseline.utils`

## Set up with uv

```bash
cd ..
uv sync --python 3.13
```

The Hugging Face datasets are currently gated, so you need access permission and
either:

```bash
export HF_TOKEN=your_token
```

or pass `--hf-token your_token` to the CLI.

If you prefer activating the environment manually:

```bash
source ../.venv/bin/activate
```

## Evaluate on validation

Run the baseline on the validation split:

```bash
uv run python baseline/run_baseline.py evaluate --phase dev
```

Evaluate only one language:

```bash
uv run python baseline/run_baseline.py evaluate --phase dev --lang ko
```

You can also import the shared helpers from personal experiment folders:

```python
from baseline.shared import load_public_dataset, predict_with_mfr, write_submission
from baseline.utils import evaluate
```

## Create a submission

Train on `train + validation`, predict the `test` split, and create a flat zip:

```bash
uv run python baseline/run_baseline.py submit --phase dev
```

For the final public release:

```bash
uv run python baseline/run_baseline.py submit --phase full
```

The command writes:

- `outputs/submission_<phase>/predictions.json`
- `outputs/submission_<phase>.zip`

## Dataset names

- `dev`: `weerayut/multilexnorm2026-dev-pub`
- `full`: `weerayut/multilexnorm2026-full-pub`
