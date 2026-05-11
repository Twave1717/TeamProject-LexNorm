from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.seq2seq import evaluate_prediction_csv


def main():
    ap = argparse.ArgumentParser(description='Evaluate prediction CSV with ERR and token metrics.')
    ap.add_argument('--pred-csv', required=True)
    ap.add_argument('--out-csv', '--summary-csv', dest='out_csv', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--lang', default=None)
    args = ap.parse_args()
    evaluate_prediction_csv(args.pred_csv, args.out_csv, args.model, args.lang)

if __name__ == '__main__':
    main()
