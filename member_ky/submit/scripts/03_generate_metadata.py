from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.rag import generate_metadata


def main():
    ap = argparse.ArgumentParser(description='Generate metadata cards for audit cases with a large LLM.')
    ap.add_argument('--cases', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default='gpt-4.1-mini')
    ap.add_argument('--index', default=None)
    ap.add_argument('--prompt', default='prompts/03_metadata_generator_system.txt')
    ap.add_argument('--schema', default='schemas/03_metadata_schema.json')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--include-gold', action='store_true', help='Gold-aware metadata is for audit analysis only; do not use it for S2 comparison.')
    args = ap.parse_args()
    generate_metadata(args.cases, args.output, ROOT / args.prompt, ROOT / args.schema, args.model, args.index, args.limit, args.include_gold)

if __name__ == '__main__':
    main()
