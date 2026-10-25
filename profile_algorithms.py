r"""Profile a training run of transformer.py or d3pm.py: where does its wall-clock time go?

Pass the trainer and its own arguments after `--`, exactly as you would train:

  python profile_algorithms.py --name plain_L8 -- transformer.py --task multiply \
    --sequence-length 5 5 --num-length-schedule 2:2:5000 5:5:20000 --d-model 128 --n-layers 8 ...
  python profile_algorithms.py --name d3pm --batch-sizes 256 1024 -- d3pm.py --task multiply

The profiler builds the model, optimizer, train_chunk and validate with the trainer's own
build_training(), so every option (aux probe, MTP, memory passes, CoT, embedding lookup,
LR schedule, Muon, ...) is profiled as it is trained. It never writes checkpoints.

At the first batch size (default: the trainer's --batch-size) it measures, at the last
curriculum stage and the trainer's --steps-per-chunk (further batch sizes time only `full`,
which saves most of their compilation):

  data      one step's batch construction (under CoT this includes sampling the thoughts)
  fwd       the training loss on pre-generated batches
  fwd+bwd   loss and gradients (bwd = fwd+bwd - fwd)
  step      fwd+bwd plus the optimizer update (optimizer = step - fwd+bwd)
  full      the trainer's jitted train_chunk itself (other = full - data - step)

Every variant is a jitted lax.scan over the chunk, like train_chunk, timed with
block_until_ready (median of --repeats, after a warmup call). It also measures the compile time
of train_chunk and validate, the validation time, and XLA's FLOP/byte estimate, and it counts
the kernels in the compiled chunk. The trainer's loop is then replayed without running it:
every distinct curriculum/chunk signature recompiles train_chunk, every chunk validates (and
transformer.py validates once more per chunk of a non-final stage), which projects the run's
wall-clock time split into compilation, training and validation. Flags are printed for likely
bottlenecks (e.g. a backward pass much slower than the forward).

Outputs, under --output-dir/<name>/: result.json, report.txt, the optimized HLO of train_chunk
and validate, and (with --trace) Perfetto traces of one chunk and one validation, captured
after timing: open traces/*/plugins/profile/*/perfetto_trace.json.gz in https://ui.perfetto.dev.
One row per batch size is appended to --output-dir/summary.csv, so several invocations (e.g.
one per model) can share a directory. XLA compilation is CPU-bound: give the job several CPUs.
"""

import argparse
import csv
import fcntl
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import optax


TRAINERS = ('transformer', 'd3pm')
SUMMARY_FIELDS = (
  'name', 'trainer', 'batch_size', 'steps_per_chunk', 'parameters', 'device', 'cpus', 'status',
  'full_ms_per_step', 'data_ms', 'fwd_ms', 'bwd_ms', 'optimizer_ms', 'other_ms', 'bwd_fwd_ratio',
  'achieved_tflops', 'achieved_gb_per_s', 'train_compile_s', 'validation_s',
  'validation_compile_s', 'stage_validation_s', 'stage_validation_compile_s',
  'validation_per_chunk_ratio', 'projected_hours', 'projected_compile_frac',
  'projected_train_frac', 'projected_validation_frac', 'peak_memory_mb', 'flags', 'output_dir',
)
HLO_PATTERNS = {
  'fusion': r'= \S+ fusion\(', 'gemm': r'custom-call\(.*gemm|= \S+ dot\(', 'copy': r'= \S+ copy\(',
  'transpose': r'= \S+ transpose\(', 'while': r'= \S+ while\(',
  'scatter': r'= \S+ scatter\(', 'gather': r'= \S+ gather\(',
}


def parse_args(argv=None):
  argv = sys.argv[1:] if argv is None else argv
  if '--' not in argv:
    raise SystemExit('usage: profile_algorithms.py [options] -- {transformer.py,d3pm.py} TRAIN_ARGS...')
  split = argv.index('--')
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--name', help='Name of this configuration (default: the trainer).')
  p.add_argument('--output-dir', type=Path, default=Path('data/profiles'))
  p.add_argument('--batch-sizes', type=int, nargs='+', help="Default: the trainer's --batch-size.")
  p.add_argument('--repeats', type=int, default=3, help='Timed calls per measurement (median).')
  p.add_argument('--matmul-precisions', nargs='*', default=[],
                 choices=['bfloat16', 'tensorfloat32', 'float32'],
                 help='Also time full chunks under these default matmul precisions (first batch size).')
  p.add_argument('--skip-breakdown', action='store_true', help='Only time the full chunk.')
  p.add_argument('--skip-validation', action='store_true')
  p.add_argument('--trace', action=argparse.BooleanOptionalAction, default=False,
                 help='Capture Perfetto traces (off by default: CUPTI has crashed processes when '
                      'several processes on one node traced at once).')
  a = p.parse_args(argv[:split])
  if not argv[split + 1:]:
    p.error('name the trainer after --')
  a.trainer = Path(argv[split + 1]).stem
  if a.trainer not in TRAINERS:
    p.error(f'the trainer must be one of {TRAINERS}, got {argv[split + 1]}')
  a.train_argv = argv[split + 2:]
  a.name = a.name or a.trainer
  if a.repeats < 1 or min(a.batch_sizes or [1]) < 1:
    p.error('--repeats and batch sizes must be positive')
  return a


def aot_compile(jitted, *args, **kwargs):
  """Lower and compile separately (both are timed). The executable takes the dynamic args only."""
  start = time.perf_counter()
  lowered = jitted.lower(*args, **kwargs)
  lowered_at = time.perf_counter()
  compiled = lowered.compile()
  return compiled, {'lower_s': lowered_at - start, 'compile_s': time.perf_counter() - lowered_at}


def timed(invoke, repeats):
  """Median seconds of repeats blocking calls after one warmup call."""
  jax.block_until_ready(invoke())
  times = []
  for _ in range(repeats):
    start = time.perf_counter()
    jax.block_until_ready(invoke())
    times.append(time.perf_counter() - start)
  return {'median_s': statistics.median(times), 'min_s': min(times), 'max_s': max(times)}


def cost_estimate(compiled):
  try:
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, (list, tuple)) else cost
    return {'flops': cost.get('flops'), 'bytes_accessed': cost.get('bytes accessed')}
  except Exception as error:  # Not every backend implements the analysis.
    return {'error': str(error)}


def memory_estimate(compiled):
  try:
    analysis = compiled.memory_analysis()
    return {k: getattr(analysis, k) for k in dir(analysis) if k.endswith('_in_bytes')}
  except Exception as error:
    return {'error': str(error)}


def save_hlo(compiled, path):
  try:
    text = compiled.as_text()
  except Exception as error:
    return {'error': str(error)}
  path.write_text(text)
  return {k: len(re.findall(pattern, text)) for k, pattern in HLO_PATTERNS.items()}


def capture_trace(path, invoke):
  with jax.profiler.trace(str(path), create_perfetto_trace=True):
    with jax.profiler.StepTraceAnnotation(path.name, step_num=0):
      jax.block_until_ready(invoke())
  files = sorted(path.rglob('perfetto_trace.json.gz'))
  info = {'directory': str(path), 'perfetto_files': [str(f) for f in files]}
  if files:
    import gzip
    with gzip.open(files[0], 'rt') as file:
      events = json.load(file).get('traceEvents', [])
    gpu_pids = {e.get('pid') for e in events if e.get('name') == 'process_name'
                and 'gpu' in e.get('args', {}).get('name', '').lower()}
    info['gpu_duration_events'] = sum(e.get('pid') in gpu_pids and e.get('ph') == 'X' for e in events)
  return info


def checksum(tree):
  """A scalar depending on every non-key leaf, so XLA cannot drop the work that produced them."""
  leaves = [x for x in jax.tree_util.tree_leaves(tree)
            if not jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)]
  return sum(jnp.sum(x).astype(jnp.float32) for x in leaves)


def loop_plan(schedule, steps_per_chunk, full_width, stage_validation):
  """Replays the trainer's chunk loop: its calls and distinct jit signatures."""
  plan = {'chunks': 0, 'steps': 0, 'train_signatures': set(), 'stage_chunks': 0, 'stage_signatures': set()}
  for n1, n2, iters, max_digit in schedule:
    done = 0
    while done < iters:
      n = min(steps_per_chunk, iters - done)
      done += n
      plan['chunks'] += 1
      plan['steps'] += n
      plan['train_signatures'].add((n1, n2, max_digit, n))
      if stage_validation and (n1, n2) != tuple(full_width):
        plan['stage_chunks'] += 1
        plan['stage_signatures'].add((n1, n2))
  return plan


def environment_info(device):
  packages = {}
  for name in ('jax', 'jaxlib', 'flax', 'optax', 'jax-cuda12-plugin'):
    try:
      packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      pass
  info = {'python': sys.version, 'packages': packages, 'device': str(device),
          'device_kind': device.device_kind, 'backend': device.platform,
          'cpus': len(os.sched_getaffinity(0)),
          'environment': {k: v for k, v in os.environ.items() if k.startswith(('XLA_', 'JAX_'))
                          or k in ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID')}}
  for key, command in (('git_revision', ['git', 'rev-parse', 'HEAD']),
                       ('git_status', ['git', 'status', '--short', '--untracked-files=no'])):
    try:
      proc = subprocess.run(command, cwd=Path(__file__).resolve().parent, text=True,
                            capture_output=True, timeout=10)
      info[key] = proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
      pass
  return info


def profile_batch(a, module, batch_size, directory, device, validation_cache, detailed):
  targs = module.parse_args(a.train_argv + ['--batch-size', str(batch_size)])
  if getattr(targs, 'checkpoint', None):
    raise ValueError('Profile training: remove --checkpoint from the d3pm.py arguments')
  run = module.build_training(targs)
  n1, n2, _, max_digit = run.num_length_schedule[-1]
  chunk = run.steps_per_chunk
  step = run.train_step
  loss = step.loss_fn if a.trainer == 'd3pm' else (lambda params, batch: step.loss_fn(params, *batch))
  make_batch = lambda params, key: step.make_batch(params, key, n1, n2, max_digit)
  params = run.opt[0]
  result = {'batch_size': batch_size, 'steps_per_chunk': chunk, 'stage': [n1, n2, max_digit],
            'parameters': sum(p.size for p in jax.tree_util.tree_leaves(params)),
            'hparams': run.hparams, 'timings_ms_per_step': {}, 'compile': {}}
  per_step = lambda t: 1000 * t['median_s'] / chunk
  keys = jax.random.split(jax.random.fold_in(run.rng_key, 1), chunk)

  if detailed and not a.skip_breakdown:
    batches = jax.jit(lambda params, keys: jax.lax.map(lambda k: make_batch(params, k), keys))(params, keys)
    jax.block_until_ready(batches)
    first = jax.tree_util.tree_map(lambda x: x[0], batches)

    def v_data(params, keys, batches):
      return jax.lax.scan(lambda c, k: (c + checksum(make_batch(params, k)), None), 0., keys)[0]

    def v_fwd(params, keys, batches):
      return jax.lax.scan(lambda c, b: (c + loss(params, b)[0], None), 0., batches)[0]

    def v_fwdbwd(params, keys, batches):
      def body(c, b):
        (value, _), grads = jax.value_and_grad(loss, has_aux=True)(params, b)
        return c + value + optax.global_norm(grads), None
      return jax.lax.scan(body, 0., batches)[0]

    def v_step(opt, keys, batches):
      def body(carry, b):
        p, s = carry
        (value, _), grads = jax.value_and_grad(loss, has_aux=True)(p, b)
        updates, s = run.optimizer.update(grads, s, p)
        return (optax.apply_updates(p, updates), s), value
      carry, values = jax.lax.scan(body, opt, batches)
      return carry, values.sum()

    step_args = (params, keys, batches)
    for name, fn in (('data', v_data), ('fwd', v_fwd), ('fwd+bwd', v_fwdbwd), ('step', v_step)):
      args = (run.opt, keys, batches) if name == 'step' else step_args
      compiled, result['compile'][name] = aot_compile(jax.jit(fn), *args)
      t = timed(lambda: compiled(*args), a.repeats)
      result['timings_ms_per_step'][name] = per_step(t)
      print(f'  batch {batch_size:>5} {name:<8} {per_step(t):9.3f} ms/step', flush=True)
      del compiled
    # FLOP and byte estimates of one step's loss and gradients, for the achieved rate.
    one_step, _ = aot_compile(jax.jit(jax.value_and_grad(loss, has_aux=True)), params, first)
    result['fwd_bwd_cost_per_step'] = cost_estimate(one_step)
    del batches, first, one_step

  # The trainer's own train_chunk (donates opt, so the state is threaded through the calls).
  opt = run.opt
  key = jax.random.fold_in(run.rng_key, 2)
  compiled, result['compile']['train_chunk'] = aot_compile(
    run.train_chunk, opt, key, n1, n2, max_digit, chunk)
  result['train_chunk_memory_estimate'] = memory_estimate(compiled)
  result['train_chunk_hlo_counts'] = save_hlo(compiled, directory / f'train_chunk_b{batch_size}.optimized_hlo.txt')

  def invoke_train():
    nonlocal opt
    out = compiled(opt, key)
    opt = out[0]
    return out
  t = timed(invoke_train, a.repeats)
  result['timings_ms_per_step']['full'] = per_step(t)
  result['chunk_seconds'] = t['median_s']
  print(f'  batch {batch_size:>5} {"full":<8} {per_step(t):9.3f} ms/step '
        f'(train_chunk compile {result["compile"]["train_chunk"]["compile_s"]:.1f} s)', flush=True)
  for precision in a.matmul_precisions if detailed else []:
    with jax.default_matmul_precision(precision):
      variant, _ = aot_compile(run.train_chunk, opt, key, n1, n2, max_digit, chunk)
    def invoke_variant():
      nonlocal opt
      out = variant(opt, key)
      opt = out[0]
      return out
    t = timed(invoke_variant, a.repeats)
    result['timings_ms_per_step'][f'full/{precision}'] = per_step(t)
    print(f'  batch {batch_size:>5} full/{precision:<9} {per_step(t):9.3f} ms/step', flush=True)

  # Validation does not depend on the batch size: measure it once per configuration.
  if not a.skip_validation and not validation_cache:
    val_key = jax.random.fold_in(run.rng_key, 3)
    # A copy: the next train_chunk call (e.g. the trace) donates opt, deleting its buffers.
    params_now = jax.tree_util.tree_map(jnp.copy, opt[0])
    if a.trainer == 'd3pm':
      val, info = aot_compile(run.validate, params_now)
      invoke_val = lambda: val(params_now)
    else:
      val, info = aot_compile(run.validate, params_now, val_key)
      invoke_val = lambda: val(params_now, val_key)
    validation_cache['validation'] = {**timed(invoke_val, a.repeats), 'compile': info,
                                      'hlo_counts': save_hlo(val, directory / 'validate.optimized_hlo.txt')}
    validation_cache['invoke'] = invoke_val
    print(f'  validation {validation_cache["validation"]["median_s"]:.3f} s '
          f'(compile {info["compile_s"]:.1f} s)', flush=True)
    stages = [(s1, s2) for s1, s2, _, _ in run.num_length_schedule if (s1, s2) != tuple(run.sequence_length)]
    if a.trainer == 'transformer' and stages:
      s1, s2 = stages[-1]
      stage_val, info = aot_compile(run.validate, params_now, val_key, num_length1=s1, num_length2=s2)
      validation_cache['stage_validation'] = {
        **timed(lambda: stage_val(params_now, val_key), a.repeats), 'compile': info, 'stage': [s1, s2]}
      print(f'  stage validation {validation_cache["stage_validation"]["median_s"]:.3f} s '
            f'(compile {info["compile_s"]:.1f} s)', flush=True)

  if a.trace and 'train_trace' not in validation_cache:
    # Captured by main() after the results are saved: tracing has crashed the process (CUPTI).
    validation_cache['train_trace'] = True
    result['_trace'] = [(name, invoke) for name, invoke in (
      ('train_chunk', invoke_train), ('validation', validation_cache.get('invoke'))) if invoke]
  try:
    result['device_memory_stats'] = device.memory_stats()
  except Exception as error:
    result['device_memory_stats'] = {'error': str(error)}
  result['plan'] = loop_plan(run.num_length_schedule, chunk, run.sequence_length, a.trainer == 'transformer')
  result['total_steps'] = result['plan']['steps']
  return result


def analyze(result, validation, trainer):
  """Derived shares, the projected wall-clock split, and bottleneck flags."""
  t, flags = result['timings_ms_per_step'], []
  derived = {}
  full = t['full']
  if 'fwd+bwd' in t:
    derived['bwd_ms'] = t['fwd+bwd'] - t['fwd']
    derived['optimizer_ms'] = t['step'] - t['fwd+bwd']
    derived['other_ms'] = full - t['data'] - t['step']
    derived['bwd_fwd_ratio'] = derived['bwd_ms'] / t['fwd'] if t['fwd'] > 0 else math.nan
    if derived['bwd_fwd_ratio'] > 4:
      scatters = result.get('train_chunk_hlo_counts', {}).get('scatter')
      flags.append(f"backward is {derived['bwd_fwd_ratio']:.1f}x the forward (expect ~2x): look for "
                   f"scatter-adds (train_chunk has {scatters}; slow when indices collide) or "
                   "recomputation in the HLO/trace")
    for part, value in (('data', t['data']), ('optimizer', derived['optimizer_ms'])):
      if value > 0.2 * full:
        flags.append(f'{part} takes {100 * value / full:.0f}% of a training step')
    if derived['other_ms'] > 0.2 * full:
      flags.append(f"{100 * derived['other_ms'] / full:.0f}% of the chunk is outside data/fwd/bwd/optimizer")
    cost = result.get('fwd_bwd_cost_per_step', {})
    fwd_bwd_s = t['fwd+bwd'] / 1000
    if cost.get('flops') and fwd_bwd_s > 0:
      derived['achieved_tflops'] = cost['flops'] / fwd_bwd_s / 1e12
      derived['achieved_gb_per_s'] = (cost.get('bytes_accessed') or 0) / fwd_bwd_s / 1e9
  plan, compile_s = result['plan'], result['compile']['train_chunk']
  chunk_compile = compile_s['lower_s'] + compile_s['compile_s']
  seconds = {'compile': len(plan['train_signatures']) * chunk_compile,
             'train': plan['steps'] * full / 1000, 'validation': 0.}
  if validation.get('validation'):
    v = validation['validation']
    # d3pm.py also validates once before training.
    calls = plan['chunks'] + (trainer == 'd3pm')
    seconds['validation'] += calls * v['median_s']
    seconds['compile'] += v['compile']['lower_s'] + v['compile']['compile_s']
    derived['validation_per_chunk_ratio'] = v['median_s'] / result['chunk_seconds']
    stage = validation.get('stage_validation')
    if stage and plan['stage_chunks']:
      seconds['validation'] += plan['stage_chunks'] * stage['median_s']
      seconds['compile'] += len(plan['stage_signatures']) * (stage['compile']['lower_s'] + stage['compile']['compile_s'])
  total = sum(seconds.values())
  derived['projected_seconds'] = seconds
  derived['projected_hours'] = total / 3600
  for part, value in seconds.items():
    derived[f'projected_{part}_frac'] = value / total if total else math.nan
  if derived['projected_validation_frac'] > 0.2:
    flags.append(f"validation is {100 * derived['projected_validation_frac']:.0f}% of the projected run "
                 "(generation recomputes the full prefix for every token)")
  if derived['projected_compile_frac'] > 0.1:
    flags.append(f"compilation is {100 * derived['projected_compile_frac']:.0f}% of the projected run "
                 f"({len(plan['train_signatures'])} train_chunk signatures x {chunk_compile:.0f} s; "
                 "more CPUs per job, fewer curriculum stages, or chunk sizes dividing each stage help)")
  derived['flags'] = flags
  return derived


def report_lines(a, env, result, validation, derived):
  t = result['timings_ms_per_step']
  lines = [f"== {a.name} ({a.trainer}), batch {result['batch_size']}, {result['parameters']:,} params, "
           f"{env['device_kind']}, {env['cpus']} CPUs",
           f"   chunk of {result['steps_per_chunk']} steps at stage {result['stage']}"]
  for name, value in t.items():
    lines.append(f'   {name:<22}{value:10.3f} ms/step')
  for name in ('bwd_ms', 'optimizer_ms', 'other_ms'):
    if name in derived:
      lines.append(f"   {'= ' + name[:-3]:<22}{derived[name]:10.3f} ms/step")
  if 'bwd_fwd_ratio' in derived:
    lines.append(f"   bwd/fwd ratio {derived['bwd_fwd_ratio']:.2f}")
  if 'achieved_tflops' in derived:
    lines.append(f"   fwd+bwd achieves {derived['achieved_tflops']:.2f} TFLOP/s, "
                 f"{derived['achieved_gb_per_s']:.0f} GB/s (XLA estimates)")
  lines.append(f"   train_chunk compile {result['compile']['train_chunk']['compile_s']:.1f} s "
               f"(+ lower {result['compile']['train_chunk']['lower_s']:.1f} s); HLO {result.get('train_chunk_hlo_counts')}")
  if validation.get('validation'):
    v = validation['validation']
    lines.append(f"   validation {v['median_s']:.3f} s = {derived['validation_per_chunk_ratio']:.2f} x a chunk; "
                 f"compile {v['compile']['compile_s']:.1f} s")
  if validation.get('stage_validation'):
    v = validation['stage_validation']
    lines.append(f"   stage validation {v['median_s']:.3f} s; compile {v['compile']['compile_s']:.1f} s")
  s, plan = derived['projected_seconds'], result['plan']
  lines.append(f"   projected run: {derived['projected_hours']:.2f} h = compile {s['compile'] / 60:.1f} min "
               f"({len(plan['train_signatures'])} train_chunk signatures), train {s['train'] / 60:.1f} min, "
               f"validation {s['validation'] / 60:.1f} min over {plan['chunks']} chunks")
  lines += [f'   FLAG: {flag}' for flag in derived['flags']] or ['   no flags']
  return lines


def append_summary(path, row):
  with path.open('a', newline='') as file:
    fcntl.flock(file, fcntl.LOCK_EX)
    writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
    if file.tell() == 0:
      writer.writeheader()
    writer.writerow(row)


def summary_row(a, env, directory, result=None, validation=None, derived=None, error=None):
  row = {'name': a.name, 'trainer': a.trainer, 'device': env['device_kind'], 'cpus': env['cpus'],
         'output_dir': str(directory), 'status': 'ok' if error is None else f'failed: {error}'}
  if result is None:
    return row
  t, validation = result['timings_ms_per_step'], validation or {}
  row.update(batch_size=result['batch_size'], steps_per_chunk=result['steps_per_chunk'],
             parameters=result['parameters'], full_ms_per_step=t.get('full'), data_ms=t.get('data'),
             fwd_ms=t.get('fwd'), train_compile_s=result['compile']['train_chunk']['compile_s'],
             flags=' | '.join(derived['flags']))
  for key in ('bwd_ms', 'optimizer_ms', 'other_ms', 'bwd_fwd_ratio', 'achieved_tflops', 'achieved_gb_per_s',
              'validation_per_chunk_ratio', 'projected_hours', 'projected_compile_frac',
              'projected_train_frac', 'projected_validation_frac'):
    row[key] = derived.get(key)
  for key in ('validation', 'stage_validation'):
    if validation.get(key):
      row[f'{key}_s'] = validation[key]['median_s']
      row[f'{key}_compile_s'] = validation[key]['compile']['compile_s']
  peak = result.get('device_memory_stats', {}).get('peak_bytes_in_use')
  row['peak_memory_mb'] = peak / 2**20 if peak else None
  return row


def main():
  a = parse_args()
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  module = importlib.import_module(a.trainer)
  device = jax.devices()[0]
  env = environment_info(device)
  directory = (a.output_dir / a.name).resolve()
  directory.mkdir(parents=True, exist_ok=True)
  batch_sizes = a.batch_sizes or [module.parse_args(a.train_argv).batch_size]
  print(f"Profiling {a.name}: {a.trainer}.py {' '.join(a.train_argv)}\n"
        f"on {env['device_kind']} with {env['cpus']} CPUs, batch sizes {batch_sizes}", flush=True)
  output = {'name': a.name, 'trainer': a.trainer, 'train_argv': a.train_argv, 'environment': env,
            'profiler_args': {k: v for k, v in vars(a).items() if k != 'output_dir'}, 'runs': []}
  validation, report = {}, []
  for batch_size in batch_sizes:
    try:
      result = profile_batch(a, module, batch_size, directory, device, validation,
                             detailed=batch_size == batch_sizes[0])
    except Exception as error:  # e.g. out of memory at a large batch size: keep the others.
      message = f'{type(error).__name__}: {str(error)[:300]}'
      print(f'  batch {batch_size} failed: {message}', flush=True)
      output['runs'].append({'batch_size': batch_size, 'error': message})
      append_summary(a.output_dir / 'summary.csv', {**summary_row(a, env, directory, error=message),
                                                    'batch_size': batch_size})
      continue
    to_trace = result.pop('_trace', [])
    stored = {k: v for k, v in validation.items() if k in ('validation', 'stage_validation')}
    derived = analyze(result, stored, a.trainer)
    result['plan'] = {k: sorted(v) if isinstance(v, set) else v for k, v in result['plan'].items()}
    output['runs'].append({**result, 'validation': stored, 'derived': derived})
    lines = report_lines(a, env, result, stored, derived)
    report += lines
    print('\n'.join(lines), flush=True)
    append_summary(a.output_dir / 'summary.csv', summary_row(a, env, directory, result, stored, derived))
    (directory / 'report.txt').write_text('\n'.join(report) + '\n')
    (directory / 'result.json').write_text(json.dumps(output, indent=2, default=str) + '\n')
    for name, invoke in to_trace:
      try:
        trace = capture_trace(directory / 'traces' / name, invoke)
      except Exception as error:
        trace = {'error': f'{type(error).__name__}: {error}'}
      output['runs'][-1].setdefault('traces', {})[name] = trace
      (directory / 'result.json').write_text(json.dumps(output, indent=2, default=str) + '\n')
  if any('error' in run for run in output['runs']):
    raise SystemExit(f'Some batch sizes failed; see {directory / "result.json"}')


if __name__ == '__main__':
  main()
