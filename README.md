# AI Intro MultiLexNorm 2026 Workspace

Korean version: [README_KR.md](README_KR.md)

This repository is the shared workspace for a 4-person team project based on the
MultiLexNorm 2026 task for the AI Intro course. The task is multilingual lexical
normalization: converting noisy, non-standard tokens into their standard forms.

This repository is intentionally organized in a "Kaggle-style team workflow":

- shared code stays small and stable
- each member runs independent experiments in a personal folder at the repo root
- the team compares validation results and leaderboard movement every week or
  every two weeks
- good ideas are merged back into the shared path for final submission and
  reporting

## What Is In This Repo

- `baseline/`: shared dataset loading, baseline logic, evaluation, and
  submission helpers
- `ky/`, `member1/`, `member2/`, `member3/`: personal experiment folders for
  each team member
- `outputs/`: generated submission files
- `Assignment Guideline.pdf`: assignment requirements
- `Assignment Paper Writing Guideline.pdf`: report writing guide

The current personal folders are:

- `ky/`
- `member1/`
- `member2/`
- `member3/`

If you want to rename folders to real names later, that is fine. The important
rule is that each person should work in a separate folder at the repository
root.

## First-Time Setup

1. Install the root environment with `uv`.

```bash
uv sync --python 3.13
```

2. Request access to the gated Hugging Face datasets:
   `weerayut/multilexnorm2026-dev-pub` and, if needed later,
   `weerayut/multilexnorm2026-full-pub`.

3. Create a root `.env` file and put your Hugging Face token there.

```bash
HF_TOKEN=your_token_here
```

4. Make sure you clicked the access/agree button on the dataset page itself.
   A valid token is not enough if the account has not been approved.

## Quick Start

Run the shared baseline evaluation on Korean:

```bash
uv run python baseline/run_baseline.py evaluate --phase dev --lang ko
```

Create a submission file for the public dev phase:

```bash
uv run python baseline/run_baseline.py submit --phase dev
```

Important: `submit` does not upload anything. It only creates the prediction
files you can later upload to the leaderboard.

## Repository Structure

```text
AI_intro/
  baseline/
    run_baseline.py
    shared.py
    utils.py
    demo.ipynb
  ky/
  member1/
  member2/
  member3/
  outputs/
  .env
  pyproject.toml
  README.md
  README_KR.md
```

## Shared Code

The shared code in `baseline/` is the part every team member can reuse.

- [baseline/run_baseline.py](/home/lgs/KiostKY/STUDY26/AI_intro/baseline/run_baseline.py): CLI entrypoint for evaluation and submission file generation
- [baseline/shared.py](/home/lgs/KiostKY/STUDY26/AI_intro/baseline/shared.py): importable helpers for dataset loading, prediction, and writing submission files
- [baseline/utils.py](/home/lgs/KiostKY/STUDY26/AI_intro/baseline/utils.py): token counting, MFR prediction, ERR evaluation
- [baseline/README.md](/home/lgs/KiostKY/STUDY26/AI_intro/baseline/README.md): baseline-specific usage notes

Useful shared imports:

```python
from baseline.shared import load_local_env, load_public_dataset, predict_with_mfr
from baseline.shared import get_submission_predictions, write_submission
from baseline.utils import counting, evaluate, mfr
```

## Personal Experiment Workflow

Each team member should work mostly inside their own folder at the repository
root.

Suggested workflow:

1. Start from the shared baseline or copy logic into your own experiment script.
2. Change only your personal experiment code.
3. Record your hypothesis, code changes, and results in `notes.md`.
4. Compare your validation score or leaderboard result with the team regularly.
5. Only move stable, useful logic back into shared code after team discussion.

To run a personal experiment from the repository root:

```bash
uv run python -m ky.run_experiment
uv run python -m member1.run_experiment
```

If you add another teammate or want a fresh experiment folder, copy one of the
existing member folders and adjust the contents.

## Shared Team Rules

- Run commands from the repository root unless a script explicitly says otherwise.
- Keep shared logic in `baseline/` focused on reusable parts:
  data access, metrics, shared utilities, submission formatting.
- Keep risky or experimental code in your own folder first.
- Use one agreed team leaderboard account for official submissions.
- Track what you changed and when, because the assignment requires individual
  contribution reporting later.

## Common Commands

Sync the environment:

```bash
uv sync --python 3.13
```

Evaluate the baseline on all languages:

```bash
uv run python baseline/run_baseline.py evaluate --phase dev
```

Evaluate the baseline on Korean only:

```bash
uv run python baseline/run_baseline.py evaluate --phase dev --lang ko
```

Create a dev submission zip:

```bash
uv run python baseline/run_baseline.py submit --phase dev
```

Create a full submission zip:

```bash
uv run python baseline/run_baseline.py submit --phase full
```

Run one member's experiment:

```bash
uv run python -m member2.run_experiment
```

## Outputs

Shared submission commands write files under `outputs/` from the repository
root:

- `outputs/submission_dev/predictions.json`
- `outputs/submission_dev.zip`
- `outputs/submission_full/predictions.json`
- `outputs/submission_full.zip`

These files are generated artifacts, not source code.

## Common Confusions

- `submit` means "create submission files", not "upload to leaderboard".
- A Hugging Face token alone is not enough; the account behind the token must be
  approved for the gated dataset.
- The root `.venv` is the main environment. The old `baseline/.venv` is legacy
  local residue and should not be treated as the official project environment.
- Validation score and leaderboard score are both useful, but the assignment
  report and reproducibility also matter.

## Recommended Weekly / Bi-Weekly Process

1. Each member runs one or more hypotheses in a personal folder.
2. Each member logs changes and results in `notes.md`.
3. The team meets weekly or bi-weekly.
4. The team compares:
   validation scores, leaderboard changes, failure cases, and implementation cost.
5. The team decides which ideas should move into the shared path and final report.

## Where To Start If You Are New

If this is your first time in the project, do this in order:

1. Read this file.
2. Read [baseline/README.md](/home/lgs/KiostKY/STUDY26/AI_intro/baseline/README.md).
3. Run the Korean baseline evaluation once.
4. Open your personal folder at the repository root.
5. Edit `run_experiment.py` and `notes.md`.
6. Test one small hypothesis before attempting a larger model change.
