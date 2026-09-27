"""Compare ordinary, block-repeated, and memory Transformers on multiplication.

Run with the same Python environment used for transformer.py:

  .venv/bin/python experiment_multiplication.py

The default sweep trains each model on 1x1, 2x1, and 2x2 digit problems. Each case uses a
mixture of operand sizes up to its stated width. Runs with the same case and seed see the same
training and validation examples. Results and full logs go into --output-dir; completed runs
are skipped when the script is restarted with the same settings. Use --dry-run to list commands.
"""

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


CASES = {'1x1': (1, 1), '2x1': (2, 1), '2x2': (2, 2)}
MODELS = {
  'plain': ('transformer', 1, 1),
  'repeated4': ('transformer', 4, 1),
  'memory2': ('memory', 1, 2),
  'memory4': ('memory', 1, 4),
  'memory8': ('memory', 1, 8),
}
PARAMS_RE = re.compile(r'^Trainable parameters: ([\d,]+)$')
LOSS_RE = re.compile(r'^Step (\d+), Loss .*?: ([\d.eE+-]+),')
ACCURACY_RE = re.compile(r'^Validation accuracy: ([\d.]+)%$')
SUMMARY_FIELDS = (
  'case', 'model', 'seed', 'steps', 'block_calls', 'parameters', 'final_accuracy',
  'best_accuracy', 'final_loss', 'seconds', 'log', 'checkpoint_dir')


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--output-dir', type=Path, default=Path('data/multiplication_experiment'))
  parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
  parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
  parser.add_argument('--seeds', nargs='+', type=int, default=[0],
                      help='Repeat each comparison with these seeds; e.g. --seeds 0 1 2.')
  parser.add_argument('--steps', nargs=3, type=int, default=[2000, 5000, 10000],
                      metavar=('ONE_BY_ONE', 'TWO_BY_ONE', 'TWO_BY_TWO'))
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--num-samples', type=int, default=1000)
  parser.add_argument('--d-model', type=int, default=64)
  parser.add_argument('--n-heads', type=int, default=4)
  parser.add_argument('--n-layers', type=int, default=2)
  parser.add_argument('--hidden-dim', type=int, default=128)
  parser.add_argument('--learning-rate', type=float, default=0.003)
  parser.add_argument('--steps-per-chunk', type=int, default=250)
  parser.add_argument('--dry-run', action='store_true', help='Print commands without training.')
  args = parser.parse_args()
  if any(n < 1 for n in args.steps):
    parser.error('--steps values must be positive')
  if args.d_model < 1 or args.n_heads < 1 or args.d_model % args.n_heads:
    parser.error('--d-model must be positive and divisible by --n-heads')
  if args.batch_size < 1 or args.num_samples < 1 or args.steps_per_chunk < 1:
    parser.error('batch size, validation samples, and steps per chunk must be positive')
  return args


def command_for(args, case, model, seed, checkpoint_dir):
  length1, length2 = CASES[case]
  steps = dict(zip(CASES, args.steps))[case]
  model_type, recurrent_steps, memory_passes = MODELS[model]
  warmup_steps = min(200, steps // 10)
  return [
    sys.executable, str(Path(__file__).with_name('transformer.py')),
    '--task', 'multiply', '--model-type', model_type,
    '--recurrent-steps', str(recurrent_steps), '--memory-passes', str(memory_passes),
    '--sequence-length', str(length1), str(length2),
    '--num-length-schedule', f'{length1}:{length2}:{steps}', '--num-steps', str(steps),
    '--batch-size', str(args.batch_size), '--num-samples', str(args.num_samples),
    '--d-model', str(args.d_model), '--n-heads', str(args.n_heads),
    '--n-layers', str(args.n_layers), '--hidden-dims', str(args.hidden_dim),
    '--optimizer', 'adamw', '--learning-rate', str(args.learning_rate),
    '--lr-schedule', 'cosine', '--warmup-steps', str(warmup_steps),
    '--lr-min', str(args.learning_rate / 10),
    '--seed', str(seed), '--steps-per-chunk', str(args.steps_per_chunk),
    '--checkpoint-dir', str(checkpoint_dir), '--checkpoint-every', str(steps + 1),
  ]


def read_metrics(log_path):
  parameters = None
  losses = {}
  accuracies = []
  step = None
  with log_path.open() as log:
    for raw_line in log:
      line = raw_line.strip()
      if match := PARAMS_RE.match(line):
        parameters = int(match.group(1).replace(',', ''))
      elif match := LOSS_RE.match(line):
        step = int(match.group(1))
        losses[step] = float(match.group(2))
      elif match := ACCURACY_RE.match(line):
        if step is not None:
          accuracies.append((step, float(match.group(1)) / 100))
  if parameters is None or not accuracies:
    raise ValueError(f'No completed training metrics found in {log_path}')
  final_step, final_accuracy = accuracies[-1]
  return {
    'parameters': parameters,
    'final_accuracy': final_accuracy,
    'best_accuracy': max(accuracy for _, accuracy in accuracies),
    'final_loss': losses[final_step],
  }


def write_summary(output_dir, results):
  output_dir.mkdir(parents=True, exist_ok=True)
  with (output_dir / 'summary.csv').open('w', newline='') as summary:
    writer = csv.DictWriter(summary, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(results)


def main():
  args = parse_args()
  output_dir = args.output_dir.resolve()
  results = []
  planned = [(case, model, seed) for case in args.cases
             for model in args.models for seed in args.seeds]

  for index, (case, model, seed) in enumerate(planned, 1):
    # Include the full training settings in the directory name. Changing settings never silently
    # reuses a result from an earlier run, while restarting an identical sweep skips completed runs.
    provisional = command_for(args, case, model, seed, Path('CHECKPOINT_DIR'))
    digest = hashlib.sha256(json.dumps(provisional).encode()).hexdigest()[:10]
    run_dir = output_dir / f'{case}_{model}_seed{seed}_{digest}'
    checkpoint_dir = run_dir / 'checkpoints'
    command = command_for(args, case, model, seed, checkpoint_dir)
    result_path = run_dir / 'result.json'
    log_path = run_dir / 'train.log'

    print(f'[{index}/{len(planned)}] {case} {model} seed={seed}', flush=True)
    if args.dry_run:
      print(shlex.join(command), flush=True)
      continue
    if result_path.exists():
      with result_path.open() as result_file:
        result = json.load(result_file)
      print(f"  already complete: {result['final_accuracy']:.2%}", flush=True)
    else:
      run_dir.mkdir(parents=True, exist_ok=True)
      with (run_dir / 'command.txt').open('w') as command_file:
        command_file.write(shlex.join(command) + '\n')
      start = time.monotonic()
      with log_path.open('w') as log_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
          for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end='', flush=True)
          returncode = process.wait()
        except KeyboardInterrupt:
          process.terminate()
          process.wait()
          raise
      if returncode:
        raise RuntimeError(f'{case} {model} seed={seed} failed; see {log_path}')
      steps = dict(zip(CASES, args.steps))[case]
      _, recurrent_steps, memory_passes = MODELS[model]
      result = {
        'case': case, 'model': model, 'seed': seed, 'steps': steps,
        'block_calls': args.n_layers * recurrent_steps * memory_passes,
        **read_metrics(log_path), 'seconds': round(time.monotonic() - start, 1),
        'log': str(log_path), 'checkpoint_dir': str(checkpoint_dir),
      }
      with result_path.open('w') as result_file:
        json.dump(result, result_file, indent=2)
      print(f"  final={result['final_accuracy']:.2%}, best={result['best_accuracy']:.2%}", flush=True)
    results.append(result)
    write_summary(output_dir, results)

  if not args.dry_run:
    print(f'Results: {output_dir / "summary.csv"}', flush=True)


if __name__ == '__main__':
  main()
