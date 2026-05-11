from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.data import build_retrieval_index


def main():
    ap = argparse.ArgumentParser(description='Build raw-token to norm candidate retrieval index.')
    ap.add_argument('--dataset', default='weerayut/multilexnorm2026-dev-pub')
    ap.add_argument('--split', default='train')
    ap.add_argument('--output', required=True)
    ap.add_argument('--langs', default='ko,en')
    args = ap.parse_args()
    langs = [x.strip() for x in args.langs.split(',') if x.strip()]
    df = build_retrieval_index(args.dataset, args.output, langs, args.split)
    print('saved', args.output, len(df))

if __name__ == '__main__':
    main()
