from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.evaluation import evaluate_system_outputs


def main():
    ap = argparse.ArgumentParser(description='Evaluate system outputs with ERR and judge metrics.')
    ap.add_argument('--cases', required=True)
    ap.add_argument('--preds', required=True)
    ap.add_argument('--judge', default=None)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    result = evaluate_system_outputs(args.cases, args.preds, args.judge, args.output)
    print(result)

if __name__ == '__main__':
    main()
