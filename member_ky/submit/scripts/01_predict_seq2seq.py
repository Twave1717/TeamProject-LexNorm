from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.seq2seq import predict_seq2seq


def main():
    ap = argparse.ArgumentParser(description='Predict with a target-token seq2seq normalizer.')
    ap.add_argument('--model-name-or-path', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-source-length', type=int, default=200)
    ap.add_argument('--max-new-tokens', type=int, default=32)
    ap.add_argument('--num-beams', type=int, default=1)
    ap.add_argument('--trust-remote-code', action='store_true')
    ap.add_argument('--postprocess', choices=['alnum', 'none'], default='alnum')
    args = ap.parse_args()
    predict_seq2seq(**vars(args))

if __name__ == '__main__':
    main()
