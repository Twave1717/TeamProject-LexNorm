from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.data import build_target_examples_per_lang


def main():
    ap = argparse.ArgumentParser(description='Build UFAL-style target-token examples per language.')
    ap.add_argument('--dataset', default='weerayut/multilexnorm2026-dev-pub')
    ap.add_argument('--output-root', default='outputs/target_examples')
    ap.add_argument('--langs', default='id,ja,ko,th,vi')
    ap.add_argument('--add-lang-prefix', action='store_true')
    ap.add_argument('--target-filter', choices=['alnum', 'none'], default='alnum')
    args = ap.parse_args()
    langs = [x.strip() for x in args.langs.split(',') if x.strip()]
    build_target_examples_per_lang(args.dataset, args.output_root, langs, args.add_lang_prefix, args.target_filter)

if __name__ == '__main__':
    main()
