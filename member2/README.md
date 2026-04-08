# Personal Experiment Template

Keep this folder at the repository root for member-specific scripts, notes,
checkpoints, and result summaries.

Suggested conventions:

- `run_experiment.py`: main experiment entrypoint
- `notes.md`: hypothesis, changes, and observed results
- `artifacts/`: generated local outputs that should usually stay untracked

Run from the repository root:

```bash
uv run python -m member2.run_experiment
```
