from __future__ import annotations
import argparse
from pathlib import Path
import runpy
ROOT = runpy.run_path(str(Path(__file__).with_name('01_02_03_04_05_06_cli_bootstrap.py')))['ROOT']
from lexnorm.seq2seq import fine_tune_seq2seq


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if v in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'expected boolean value, got {value!r}')


def main():
    ap = argparse.ArgumentParser(description='Fine-tune a target-token seq2seq normalizer.')
    ap.add_argument('--model-name-or-path', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--train-split', default='train')
    ap.add_argument('--eval-split', default='validation')
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--num-train-epochs', '--epochs', dest='num_train_epochs', type=float, default=10)
    ap.add_argument('--per-device-train-batch-size', '--train-batch-size', dest='per_device_train_batch_size', type=int, default=128)
    ap.add_argument('--per-device-eval-batch-size', '--eval-batch-size', dest='per_device_eval_batch_size', type=int, default=64)
    ap.add_argument('--gradient-accumulation-steps', '--grad-accum', dest='gradient_accumulation_steps', type=int, default=1)
    ap.add_argument('--train-examples-per-epoch', type=int, default=None)
    ap.add_argument('--max-source-length', type=int, default=200)
    ap.add_argument('--max-target-length', type=int, default=32)
    ap.add_argument('--fp16', nargs='?', const=True, default=False, type=str_to_bool)
    ap.add_argument('--bf16', nargs='?', const=True, default=False, type=str_to_bool)
    ap.add_argument('--gradient-checkpointing', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trust-remote-code', action='store_true')
    ap.add_argument('--eval-accumulation-steps', type=int, default=None)
    ap.add_argument('--wandb-project', default=None)
    ap.add_argument('--wandb-entity', default=None)
    ap.add_argument('--wandb-run-name', default=None)
    ap.add_argument('--wandb-group', default=None)
    ap.add_argument('--logging-steps', type=int, default=100)
    ap.add_argument('--lr-scheduler-type', default='constant')
    args = ap.parse_args()
    fine_tune_seq2seq(**vars(args))

if __name__ == '__main__':
    main()
