from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.visualization import plot_from_input_dir, plot_summary


def main():
    ap = argparse.ArgumentParser(description='Create figures from context-RAG outputs.')
    ap.add_argument('--input-dir', default=None)
    ap.add_argument('--summary-csv', default=None)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    if args.input_dir:
        plot_from_input_dir(args.input_dir, args.output_dir)
    elif args.summary_csv:
        plot_summary(args.summary_csv, args.output_dir)
    else:
        raise SystemExit('Provide --input-dir or --summary-csv')

if __name__ == '__main__':
    main()
