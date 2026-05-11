from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.data import build_audit_cases


def main():
    ap = argparse.ArgumentParser(description='Build Ko/En audit/evaluation cases.')
    ap.add_argument('--dataset', default='weerayut/multilexnorm2026-dev-pub')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--output', required=True)
    ap.add_argument('--n-ko', type=int, default=1000)
    ap.add_argument('--n-en', type=int, default=300)
    ap.add_argument('--changed-ratio', type=float, default=0.7)
    ap.add_argument('--social-ratio', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    df = build_audit_cases(args.dataset, args.output, args.n_ko, args.n_en, args.split, args.changed_ratio, args.seed, args.social_ratio)
    print('saved', args.output, len(df))
    if len(df) and 'sample_group' in df:
        print(df['sample_group'].value_counts().to_string())

if __name__ == '__main__':
    main()
