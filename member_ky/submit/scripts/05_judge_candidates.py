from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.rag import judge_candidates


def main():
    ap = argparse.ArgumentParser(description='Evaluate candidates with LLM-as-a-Judge.')
    ap.add_argument('--cases', required=True)
    ap.add_argument('--preds', required=True)
    ap.add_argument('--metadata', default=None)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default='gpt-4.1-mini')
    ap.add_argument('--prompt', default='prompts/05_judge_system.txt')
    ap.add_argument('--schema', default='schemas/05_judge_schema.json')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    judge_candidates(args.cases, args.preds, args.output, ROOT / args.prompt, ROOT / args.schema, args.model, args.metadata, args.limit)

if __name__ == '__main__':
    main()
