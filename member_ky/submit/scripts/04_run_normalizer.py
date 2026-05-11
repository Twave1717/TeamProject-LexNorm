from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.rag import run_normalizer


def main():
    ap = argparse.ArgumentParser(description='Run pure / pair few-shot / metadata-RAG normalization.')
    ap.add_argument('--cases', required=True)
    ap.add_argument('--index', default=None)
    ap.add_argument('--metadata', default=None)
    ap.add_argument('--mode', choices=['pure', 'pair_fewshot', 'metadata_rag'], required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default='gpt-4.1-mini')
    ap.add_argument('--softening-policy', choices=['preserve_force', 'neutralize'], default='preserve_force')
    ap.add_argument('--schema', default='schemas/04_normalizer_schema.json')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    prompt_map = {'pure': 'prompts/04_normalizer_pure.txt', 'pair_fewshot': 'prompts/04_normalizer_pair_fewshot.txt', 'metadata_rag': 'prompts/04_normalizer_metadata_rag.txt'}
    run_normalizer(args.cases, args.output, ROOT / prompt_map[args.mode], ROOT / args.schema, args.mode, args.model, args.index, args.metadata, args.softening_policy, args.limit)

if __name__ == '__main__':
    main()
