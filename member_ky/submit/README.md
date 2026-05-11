# MultiLexNorm2026 Code Package

This is the runnable code package. Run the numbered notebooks in this directory in order. Core functions live under `src/lexnorm/`; `scripts/` are CLI wrappers.

## Tracks

- **Track A: Official ERR baseline**
  - Fine-tune `ufal/byt5-small-multilexnorm2021-en` on new languages: `id, ja, ko, th, vi`.
  - Evaluate with Accuracy, ERR, TP/FP/FN.

- **Track B: Social-context RAG + LLM-as-a-Judge**
  - Build Ko/En audit cases.
  - Generate metadata cards.
  - Compare `pure`, `pair_fewshot`, and `metadata_rag` prompting.
  - Evaluate with both official ERR and judge acceptability.

## Execution Style

- Notebooks call `src/lexnorm` functions directly. This is the preferred Colab path because traceback is clearer.
- `scripts/` files are thin CLI wrappers around the same functions, so every step can also be run from a terminal.

## Quick Start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export HF_TOKEN=...
```

Run notebooks in order:

```text
00_setup_and_data.ipynb
01_byt5_exact_baseline.ipynb
02_build_audit_and_index.ipynb
03_metadata_generation.ipynb
04_s0_s1_s2_normalization.ipynb
05_llm_as_judge.ipynb
06_evaluate_and_visualize.ipynb
```

Package layout:

```text
src/      -> notebooks and scripts both call this implementation
scripts/  -> command-line wrappers around src/lexnorm
prompts/  -> OpenAI prompt text
schemas/  -> structured JSON output schemas
```

노트북은 `src/lexnorm/` 함수를 직접 호출합니다. `scripts/`는 같은 함수를 터미널에서 실행하기 위한 얇은 wrapper입니다.

Prompt/schema filenames are prefixed by the notebook step that uses them:

```text
03_* -> metadata generation
04_* -> S0/S1/S2 normalization
05_* -> LLM-as-a-Judge
```

Equivalent smoke commands:

```bash
python scripts/01_build_target_examples_per_lang.py \
  --dataset weerayut/multilexnorm2026-dev-pub \
  --langs id,ja,ko,th,vi \
  --output-root outputs/target_examples

python scripts/01_train_seq2seq.py \
  --model-name-or-path ufal/byt5-small-multilexnorm2021-en \
  --data-dir outputs/target_examples/ko \
  --output-dir outputs/byt5/ByT52021EN_to_ko \
  --epochs 1 \
  --learning-rate 1e-4 \
  --train-batch-size 25 \
  --eval-batch-size 16 \
  --grad-accum 4 \
  --train-examples-per-epoch 50000 \
  --max-source-length 200 \
  --max-target-length 32 \
  --fp16 false
```


## Output Layout

Notebook outputs are written locally and mirrored to Google Drive when Drive is mounted:

```text
/content/lexnorm_submit/outputs/...
/drive/MyDrive/AI개론_박진영/lexnorm_outputs/...
```

The mirrored structure stays concise: `target_examples/`, `byt5/`, and `context_rag/`.

## Important Rules

- Do not mix Track A and Track B.
- Do not use LLM-as-a-Judge as official ERR replacement.
- Do not reveal candidate source to judge.
- Use `--limit 20` smoke runs before full API runs.
- ByT5 transfer should start with fp16 disabled.

For dependency setup:

```bash
uv sync
# or, in Colab/plain pip environments:
pip install -r requirements.txt
```
