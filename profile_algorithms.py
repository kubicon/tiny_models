r"""Profile the actual arithmetic training and generation routines on a GPU.

Run using the same GPU-enabled Python environment as transformer.py and d3pm.py:

  python profile_algorithms.py --models plain repeated4 memory2 d3pm-mask d3pm-uniform
  python profile_algorithms.py --models plain d3pm-mask --cot-steps 0 4 \
    --batch-sizes 64 256 --chunk-sizes 1 250
  python profile_algorithms.py --task add --sequence-length 5 1 --num-length 3 1 \
    --models memory4 --optimizer muon --pos-embed rope

Default architecture matches experiment_multiplication.py (64/4 heads/2 layers/128).
Select the architecture and operand widths of your intended experiment. Each model,
batch size, chunk size, and thought count runs in a fresh, serial subprocess. GPU
is required by default; --backend cpu is available for a local smoke test. Use
--dry-run to inspect the matrix without importing JAX or creating files.

Measurements exclude compilation, warmup, profiler overhead, logging, validation
scoring, and checkpoint I/O. Training includes on-device data generation, thoughts,
loss/gradients, clipping, and optimizer updates using the trainers' own step functions.
A constant learning rate is used. Generation measures the trainers' decoding routines
(including thoughts), with device-resident prompts; prompt preparation is excluded.
Both synchronized chunk latency and queued training throughput are recorded. Host
enqueue latency is not GPU execution time. Problems/s counts original arithmetic
problems; CoT additionally processes cot_samples scratchpads per training problem.
Different chunk sizes perform different numbers of updates: losses are sanity checks,
not convergence results. Use experiment_multiplication.py for multiplication
convergence comparisons.

Outputs: summary.csv, per-run result.json/log, compile times, latency distributions,
compiler memory/cost estimates, StableHLO and optimized HLO, device memory snapshot,
and separate warmed training/generation traces. Trace capture occurs AFTER timing.
Copy a perfetto_trace.json.gz file to your laptop and open it in https://ui.perfetto.dev
(no server or SSH tunnel required). For XProf, install xprof in your viewing environment
and point its viewer at the per-run traces directory. See official instructions:
https://docs.jax.dev/en/latest/profiling.html
https://docs.jax.dev/en/latest/benchmarking.html

Inspect GPU stream gaps, many tiny kernels, host dispatch, RNG/data-generation work,
repeated passes, and optimizer kernels. Compare chunk=1 vs chunk=250 and batch sizes
to identify overhead that can be amortized. HLO source scopes include train_step,
data_generation, thought_generation, loss_and_gradients, and optimizer_update;
compiler fusion can combine stages. Compiler estimates are not measured peak VRAM.
A memory snapshot reports live allocations, not
peak usage. GPU stream traces require a working GPU profiling runtime (CUPTI on CUDA);
a host-only trace is insufficient for kernel analysis. Consult the JAX profiling
troubleshooting guide above if device events are missing. --no-trace still benchmarks.
"""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

from experiment_multiplication import MODELS


SUMMARY_FIELDS = (
  'model', 'task', 'batch_size', 'chunk_size', 'cot_steps', 'parameters', 'backend',
  'device', 'train_compile_seconds', 'train_latency_ms_per_step',
  'train_queued_steps_per_second', 'train_queued_problems_per_second',
  'generation_ms_per_batch', 'generation_problems_per_second', 'warnings', 'status', 'run_dir',
)


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
  p.add_argument('--task', choices=['add', 'multiply'], default='multiply')
  p.add_argument('--sequence-length', nargs='+', type=int, default=[2, 2])
  p.add_argument('--num-length', nargs='+', type=int, help='Curriculum caps; defaults to rendered widths.')
  p.add_argument('--batch-sizes', nargs='+', type=int, default=[256])
  p.add_argument('--chunk-sizes', nargs='+', type=int, default=[1, 250])
  p.add_argument('--cot-steps', nargs='+', type=int, default=[0])
  p.add_argument('--cot-samples', type=int, default=2)
  p.add_argument('--cot-pg-weight', type=float, default=0.1)
  p.add_argument('--cot-sampling-steps', type=int)
  p.add_argument('--diffusion-steps', type=int, default=16)
  p.add_argument('--sampling-steps', type=int)
  p.add_argument('--sample-final', action='store_true')
  p.add_argument('--aux-loss-weight', type=float, default=0.1)
  p.add_argument('--reverse-operands', action='store_true')
  p.add_argument('--early-eos', action='store_true', help='AR only; incompatible with D3PM.')
  p.add_argument('--d-model', type=int, default=64)
  p.add_argument('--n-heads', type=int, default=4)
  p.add_argument('--n-layers', type=int, default=2)
  p.add_argument('--hidden-dims', nargs='+', type=int, default=[128])
  p.add_argument('--recurrent-steps', type=int, help='Override the recurrence of all selected presets.')
  p.add_argument('--memory-passes', type=int, help='Override passes of memory presets.')
  p.add_argument('--use-bias', action='store_true')
  p.add_argument('--activation', choices=['gelu', 'relu', 'swish', 'silu', 'mish', 'tanh', 'sigmoid', 'none'], default='silu')
  p.add_argument('--normalization', choices=['layer', 'rms', 'none'], default='rms')
  p.add_argument('--pos-embed', choices=['learned', 'rope'], default='learned')
  p.add_argument('--rope-base', type=float, default=10000.)
  p.add_argument('--qk-norm', action='store_true')
  p.add_argument('--n-kv-heads', type=int, default=0)
  p.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw')
  p.add_argument('--learning-rate', type=float, default=0.003)
  p.add_argument('--grad-clip-norm', type=float, default=1.)
  p.add_argument('--weight-decay', type=float, default=0.01)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--backend', choices=['gpu', 'cpu'], default='gpu')
  p.add_argument('--device-index', type=int, default=0, help='Index among visible devices of the chosen backend.')
  p.add_argument('--warmup-chunks', type=int, default=2)
  p.add_argument('--timing-chunks', type=int, default=5)
  p.add_argument('--throughput-chunks', type=int, default=5)
  p.add_argument('--trace-chunks', type=int, default=1)
  p.add_argument('--num-samples', type=int, default=256, help='Problems per generation batch.')
  p.add_argument('--inference-repeats', type=int, default=5)
  p.add_argument('--skip-inference', action='store_true')
  p.add_argument('--trace', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--output-dir', type=Path, default=Path('data/profiles'))
  p.add_argument('--dry-run', action='store_true')
  p.add_argument('--worker-config', type=Path, help=argparse.SUPPRESS)
  a = p.parse_args(argv)
  if a.worker_config:
    return a
  for field in ('sequence_length', 'num_length'):
    values = getattr(a, field)
    if values is None:
      continue
    if len(values) not in (1, 2) or min(values) < 1:
      p.error(f'--{field.replace("_", "-")} requires one or two positive widths')
    if len(values) == 1:
      setattr(a, field, values * 2)
  a.num_length = a.num_length or list(a.sequence_length)
  if any(n > width for n, width in zip(a.num_length, a.sequence_length)):
    p.error('--num-length caps must fit --sequence-length')
  counts = ('d_model', 'n_heads', 'n_layers', 'diffusion_steps', 'warmup_chunks',
            'timing_chunks', 'throughput_chunks', 'trace_chunks', 'num_samples', 'inference_repeats')
  if any(getattr(a, f) < 1 for f in counts) or min(a.batch_sizes + a.chunk_sizes + a.hidden_dims) < 1:
    p.error('dimensions, batch sizes, chunks, repetitions and sample counts must be positive')
  if a.d_model % a.n_heads or a.n_kv_heads < 0 or (a.n_kv_heads and a.n_heads % a.n_kv_heads):
    p.error('--d-model must be divisible by --n-heads; --n-kv-heads must divide --n-heads')
  if min(a.cot_steps) < 0 or (max(a.cot_steps) and a.cot_samples < 2):
    p.error('thought counts must be nonnegative; thoughts require --cot-samples >= 2')
  if a.learning_rate <= 0 or min(a.cot_pg_weight, a.aux_loss_weight, a.weight_decay, a.grad_clip_norm) < 0:
    p.error('learning rate must be positive; weights and clipping must be nonnegative')
  for field in ('recurrent_steps', 'memory_passes', 'sampling_steps', 'cot_sampling_steps'):
    value = getattr(a, field)
    if value is not None and value < 1:
      p.error(f'--{field.replace("_", "-")} must be positive')
  if any(v is not None and v > a.diffusion_steps for v in (a.sampling_steps, a.cot_sampling_steps)):
    p.error('sampling steps cannot exceed --diffusion-steps')
  if a.device_index < 0:
    p.error('--device-index must be nonnegative')
  if a.early_eos and any(MODELS[name][0] == 'd3pm' for name in a.models):
    p.error('--early-eos is unsupported by D3PM; select only AR presets')
  return a


def configurations(args):
  shared = vars(args).copy()
  for key in ('models', 'batch_sizes', 'chunk_sizes', 'output_dir', 'dry_run', 'worker_config'):
    shared.pop(key)
  for model, batch, chunk, thoughts in itertools.product(
      dict.fromkeys(args.models), dict.fromkeys(args.batch_sizes),
      dict.fromkeys(args.chunk_sizes), dict.fromkeys(args.cot_steps)):
    model_type, recurrence, passes = MODELS[model]
    yield {**shared, 'model': model, 'model_type': model_type, 'batch_size': batch,
           'chunk_size': chunk, 'cot_steps': thoughts,
           'recurrent_steps': args.recurrent_steps or recurrence,
           'memory_passes': args.memory_passes or passes,
           'corruption': 'uniform' if model == 'd3pm-uniform' else 'mask'}


def write_json(path, value):
  path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')


def latency_stats(seconds):
  values = sorted(seconds)
  def percentile(q):
    offset = (len(values) - 1) * q
    low = int(offset)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (offset - low)
  return {'samples_seconds': seconds, 'median_seconds': statistics.median(values),
          'min_seconds': values[0], 'p10_seconds': percentile(.1), 'p90_seconds': percentile(.9)}


def compile_workload(jitted, arguments, directory, name):
  start = time.perf_counter()
  lowered = jitted.lower(*arguments)
  lower_seconds = time.perf_counter() - start
  start = time.perf_counter()
  executable = lowered.compile()
  compile_seconds = time.perf_counter() - start
  info = {'lower_seconds': lower_seconds, 'compile_seconds': compile_seconds}
  for filename, get_text in (
      (f'{name}.stablehlo.txt', lambda: lowered.as_text()),
      (f'{name}.optimized_hlo.txt', executable.as_text)):
    try:
      (directory / filename).write_text(get_text())
    except (AttributeError, NotImplementedError, RuntimeError) as error:
      info[filename + '_error'] = str(error)
  for name_, analyze in (('cost_estimate', executable.cost_analysis), ('memory_estimate', executable.memory_analysis)):
    try:
      estimate = analyze()
      if name_ == 'memory_estimate' and estimate is not None:
        estimate = {key: getattr(estimate, key) for key in dir(estimate) if key.endswith('_in_bytes')}
      info[name_] = estimate
    except (AttributeError, NotImplementedError, RuntimeError) as error:
      info[name_ + '_error'] = str(error)
  return executable, info


def capture_trace(directory, name, count, invoke):
  import jax
  path = directory / 'traces' / name
  with jax.profiler.trace(str(path), create_perfetto_trace=True):
    for i in range(count):
      with jax.profiler.StepTraceAnnotation(name, step_num=i):
        jax.block_until_ready(invoke())
  files = sorted(path.rglob('perfetto_trace.json.gz'))
  info = {'directory': str(path), 'perfetto_files': [str(p) for p in files]}
  # Metadata alone is insufficient: verify that GPU lanes contain duration events.
  # This inspection is outside all measured regions.
  if files:
    with gzip.open(files[0], 'rt') as file:
      events = json.load(file).get('traceEvents', [])
    gpu_pids = {e.get('pid') for e in events if e.get('name') == 'process_name'
                and 'gpu' in e.get('args', {}).get('name', '').lower()}
    info['gpu_duration_events'] = sum(e.get('pid') in gpu_pids and e.get('ph') == 'X' for e in events)
  return info


def environment_info(device):
  import jax
  import os
  packages = {}
  for name in ('jax', 'jaxlib', 'flax', 'optax', 'numpy', 'jax-cuda12-plugin', 'jax-cuda13-plugin'):
    try:
      packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      pass
  info = {'python': sys.version, 'packages': packages, 'device': str(device),
          'device_kind': device.device_kind, 'backend': device.platform,
          'visible_devices': [str(d) for d in jax.devices()],
          'jax_enable_x64': jax.config.jax_enable_x64,
          'environment': {k: os.environ[k] for k in (
            'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'XLA_FLAGS', 'JAX_PLATFORMS',
            'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
            'JAX_ENABLE_COMPILATION_CACHE', 'JAX_COMPILATION_CACHE_DIR') if k in os.environ}}
  for key, command in (
      ('git_revision', ['git', 'rev-parse', 'HEAD']),
      ('git_status', ['git', 'status', '--short']),
      ('nvidia_smi', ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,utilization.gpu', '--format=csv'])):
    try:
      proc = subprocess.run(command, cwd=Path(__file__).resolve().parent, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
      if proc.returncode == 0:
        info[key] = proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
      pass
  return info


def run_worker(config_path):
  import jax
  import jax.numpy as jnp
  import optax
  import d3pm
  import transformer as ar
  from task import TASKS, generate_episode, generate_prompts

  c = json.loads(config_path.read_text())
  directory = config_path.parent
  try:
    devices = jax.devices(c['backend'])
  except RuntimeError as error:
    raise RuntimeError(f"Requested {c['backend']} backend is unavailable. Run this script in your "
                       "GPU-enabled JAX environment, or explicitly use --backend cpu for a smoke test.") from error
  if c['device_index'] >= len(devices):
    raise ValueError(f"Device index {c['device_index']} exceeds the {len(devices)} visible {c['backend']} devices")
  device = devices[c['device_index']]
  with jax.default_device(device):
    task = TASKS[c['task']]()
    width1, width2 = c['sequence_length']
    answer_length = task.answer_length(width1, width2)
    diffusion = c['model_type'] == 'd3pm'
    base_vocab = task.vocab_size + (2 if c['cot_steps'] else 0)
    hparams = {**c, 'vocab_size': base_vocab + int(diffusion and c['corruption'] == 'mask'),
               'max_seq_len': task.episode_length(width1, width2) - int(diffusion) +
                              (c['cot_steps'] + 2 if c['cot_steps'] else 0)}
    model = ar.build_transformer(hparams)
    key = jax.random.key(c['seed'])
    key, data_key = jax.random.split(key)
    key, init_key = jax.random.split(key)
    if diffusion:
      process = d3pm.D3PM(c['diffusion_steps'], c['corruption'])
      prompts, _, _ = generate_prompts(task, width1, width2, 1, data_key)
      if c['cot_steps']:
        prompts = d3pm.thought_prefix(prompts, jnp.zeros((1, c['cot_steps']), dtype=jnp.int32), task)
      example = jnp.concatenate((prompts[0], jnp.zeros((answer_length,), dtype=jnp.int32)))
      params = model.init(init_key, example, timestep=jnp.array(process.num_steps))
    else:
      example = generate_episode(task, width1, width2, 1, data_key,
                                 reverse_operands=c['reverse_operands'], early_eos=c['early_eos'])[0]
      if c['cot_steps']:
        end = task.answer_start_index(width1, width2) + 1
        example = jnp.concatenate((example[:end], jnp.array([task.think_token]),
                                   jnp.zeros((c['cot_steps'],), dtype=jnp.int32),
                                   jnp.array([task.answer_token]), example[end:]))
      if isinstance(model, ar.MemoryTransformer):
        params = model.init(init_key, example, jnp.zeros((example.size, c['d_model']), dtype=jnp.float32), train=False)
      else:
        params = model.init(init_key, example, train=False)
    optimizer_fn = optax.contrib.muon if c['optimizer'] == 'muon' else optax.adamw
    optimizer = optimizer_fn(learning_rate=optax.constant_schedule(c['learning_rate']),
                             weight_decay=c['weight_decay'])
    if c['grad_clip_norm'] > 0:
      optimizer = optax.chain(optax.clip_by_global_norm(c['grad_clip_norm']), optimizer)
    carry = (params, optimizer.init(params))
    if diffusion:
      step = d3pm.make_train_step(model, process, task, optimizer, width1, width2, c['batch_size'],
                                 base_vocab, c['reverse_operands'], c['cot_steps'], c['cot_samples'],
                                 c['cot_pg_weight'], c['aux_loss_weight'], c['cot_sampling_steps'])
    else:
      step = ar.make_train_step(model, task, optimizer, width1, width2, c['batch_size'],
                               c['reverse_operands'], c['early_eos'], c['cot_steps'],
                               c['cot_samples'], c['cot_pg_weight'])

    def train_chunk(carry, key):
      keys = jax.random.split(key, c['chunk_size'])
      carry, metrics = jax.lax.scan(lambda state, k: step(state, k, *c['num_length']), carry, keys)
      if not diffusion:
        metrics = dict(zip(('loss', 'grad_norm'), metrics))
      return carry, jax.tree_util.tree_map(jnp.mean, metrics)

    jax.block_until_ready(carry)
    result = {'config': c, 'hparams': hparams, 'environment': environment_info(device),
              'parameters': sum(p.size for p in jax.tree_util.tree_leaves(params)),
              'parameter_dtypes': sorted({str(p.dtype) for p in jax.tree_util.tree_leaves(params)}),
              'warnings': []}
    print(f"Profiling {c['model']}, batch={c['batch_size']}, chunk={c['chunk_size']}, "
          f"thoughts={c['cot_steps']} on {device.device_kind}", flush=True)
    executable, compilation = compile_workload(jax.jit(train_chunk, donate_argnums=(0,)),
                                                (carry, key), directory, 'train')
    count = c['warmup_chunks'] + c['timing_chunks'] + c['throughput_chunks'] + c['trace_chunks']
    keys = list(jax.random.split(jax.random.fold_in(key, 1001), count))
    jax.block_until_ready(keys)
    key_iter = iter(keys)
    train_invocations = 0
    def invoke_train():
      nonlocal carry
      nonlocal train_invocations
      carry, metrics = executable(carry, next(key_iter))
      train_invocations += 1
      return carry, metrics
    for _ in range(c['warmup_chunks']):
      jax.block_until_ready(invoke_train())
    times, enqueues = [], []
    for _ in range(c['timing_chunks']):
      start = time.perf_counter()
      output = invoke_train()
      enqueued = time.perf_counter()
      jax.block_until_ready(output)
      times.append(time.perf_counter() - start)
      enqueues.append(enqueued - start)
    # Fence only at the end to expose sustainable dispatch/compute throughput.
    pending = []
    start = time.perf_counter()
    for _ in range(c['throughput_chunks']):
      _, metrics = invoke_train()
      pending.append(metrics)
    jax.block_until_ready((carry, pending))
    queued_seconds = time.perf_counter() - start
    steps = c['chunk_size'] * c['throughput_chunks']
    result['train'] = {'compilation': compilation, 'latency': latency_stats(times),
                       'host_enqueue': latency_stats(enqueues),
                       'median_ms_per_step': 1000 * statistics.median(times) / c['chunk_size'],
                       'queued_seconds': queued_seconds, 'queued_steps': steps,
                       'queued_steps_per_second': steps / queued_seconds,
                       'queued_problems_per_second': steps * c['batch_size'] / queued_seconds,
                       'last_metrics': jax.tree_util.tree_map(float, pending[-1]),
                       'expanded_training_batch': c['batch_size'] * (c['cot_samples'] if c['cot_steps'] else 1)}
    # Save baseline results before optional trace capture, which may fail without CUPTI.
    write_json(directory / 'result.json', result)
    print(f"Train: {result['train']['median_ms_per_step']:.3f} ms/step; "
          f"queued {result['train']['queued_problems_per_second']:.0f} problems/s", flush=True)

    generation_executable = None
    if not c['skip_inference']:
      eval_key = jax.random.fold_in(jax.random.key(c['seed']), 9001)
      prompts, _, _ = generate_prompts(task, width1, width2, c['num_samples'], eval_key,
                                      reverse_operands=c['reverse_operands'])
      jax.block_until_ready(prompts)
      def generate(params, prompts, rng):
        if diffusion:
          if c['cot_steps']:
            thought_key, rng = jax.random.split(rng)
            thoughts = d3pm.generate_thoughts(model, params, process, prompts, task, c['cot_steps'],
                                             thought_key, base_vocab, c['cot_sampling_steps'])
            prompts = d3pm.thought_prefix(prompts, thoughts, task)
          return d3pm.generate_answers(model, params, process, prompts, answer_length, rng,
                                       base_vocab, c['sampling_steps'], c['sample_final'])
        if c['cot_steps']:
          prompts = ar.generate_thoughts(model, params, prompts, task, c['cot_steps'])
          prompts = jnp.concatenate((prompts, jnp.full((prompts.shape[0], 1), task.answer_token)), axis=-1)
        return ar.generate_tokens(model, params, prompts, answer_length + int(c['early_eos']))
      generation_executable, compilation = compile_workload(jax.jit(generate),
                                       (carry[0], prompts, eval_key), directory, 'generation')
      def invoke_generation():
        return generation_executable(carry[0], prompts, eval_key)
      for _ in range(c['warmup_chunks']):
        jax.block_until_ready(invoke_generation())
      times = []
      for _ in range(c['inference_repeats']):
        start = time.perf_counter()
        jax.block_until_ready(invoke_generation())
        times.append(time.perf_counter() - start)
      median = statistics.median(times)
      result['generation'] = {'compilation': compilation, 'latency': latency_stats(times),
                              'median_ms_per_batch': 1000 * median,
                              'problems_per_second': c['num_samples'] / median,
                              'batch_size': c['num_samples']}
      print(f"Generation: {1000 * median:.3f} ms/batch", flush=True)

    if c['trace']:
      for name, invoke in [('train', invoke_train)] + ([('generation', invoke_generation)] if generation_executable else []):
        try:
          result[name]['trace'] = capture_trace(directory, name, c['trace_chunks'], invoke)
          if not result[name]['trace']['perfetto_files']:
            result['warnings'].append(f'{name}: no Perfetto file was exported; inspect worker.log')
          elif c['backend'] == 'gpu' and not result[name]['trace'].get('gpu_duration_events'):
            result['warnings'].append(f'{name}: no GPU duration events detected in Perfetto. '
                                       'Inspect the device lanes and worker.log; check GPU profiling runtime/CUPTI.')
        except Exception as error:
          result['warnings'].append(f'{name} trace capture failed: {type(error).__name__}: {error}')
    result['optimizer_steps_executed'] = train_invocations * c['chunk_size']
    try:
      jax.profiler.save_device_memory_profile(str(directory / 'device_memory.pb'), backend=c['backend'])
      result['device_memory_snapshot'] = str(directory / 'device_memory.pb')
    except Exception as error:
      result['warnings'].append(f'Device memory snapshot unavailable: {error}')
    try:
      result['device_allocator_stats'] = device.memory_stats()
    except (AttributeError, RuntimeError) as error:
      result['warnings'].append(f'Device allocator statistics unavailable: {error}')
    for warning in result['warnings']:
      print('Warning:', warning, flush=True)
    result['status'] = 'ok'
    write_json(directory / 'result.json', result)


def summary_row(result, directory):
  c = result['config']
  train = result.get('train', {})
  gen = result.get('generation', {})
  return {'model': c['model'], 'task': c['task'], 'batch_size': c['batch_size'],
          'chunk_size': c['chunk_size'], 'cot_steps': c['cot_steps'],
          'parameters': result.get('parameters'), 'backend': c['backend'],
          'device': result.get('environment', {}).get('device_kind'),
          'train_compile_seconds': train.get('compilation', {}).get('compile_seconds'),
          'train_latency_ms_per_step': train.get('median_ms_per_step'),
          'train_queued_steps_per_second': train.get('queued_steps_per_second'),
          'train_queued_problems_per_second': train.get('queued_problems_per_second'),
          'generation_ms_per_batch': gen.get('median_ms_per_batch'),
          'generation_problems_per_second': gen.get('problems_per_second'),
          'warnings': len(result.get('warnings', [])),
          'status': result.get('status', 'failed'), 'run_dir': str(directory)}


def main():
  args = parse_args()
  if args.worker_config:
    run_worker(args.worker_config)
    return
  configs = list(configurations(args))
  if args.dry_run:
    for c in configs:
      print(json.dumps(c, sort_keys=True))
    print(f'{len(configs)} serial profiling runs; no files created.')
    return
  root = args.output_dir.resolve() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
  root.mkdir(parents=True)
  print(f'{len(configs)} serial runs. Results: {root}', flush=True)
  rows, failures = [], []
  for index, c in enumerate(configs, 1):
    digest = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:8]
    directory = root / f"{c['model']}_b{c['batch_size']}_chunk{c['chunk_size']}_cot{c['cot_steps']}_{digest}"
    directory.mkdir()
    config_path = directory / 'config.json'
    write_json(config_path, c)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-config', str(config_path)]
    print(f'[{index}/{len(configs)}] {directory.name}', flush=True)
    with (directory / 'worker.log').open('w') as log:
      log.write(shlex.join(command) + '\n')
      log.flush()
      proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    result_path = directory / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {'config': c}
    if proc.returncode:
      result.update(status='failed', returncode=proc.returncode)
      write_json(result_path, result)
      failures.append(directory)
      print(f'  Failed; see {directory / "worker.log"}', flush=True)
      # A missing GPU affects every preset. Stop instead of running the same failure repeatedly.
      if 'backend is unavailable' in (directory / 'worker.log').read_text():
        print('  Requested backend unavailable. Use GPU-enabled JAX, or --backend cpu for a smoke test.', flush=True)
    row = summary_row(result, directory)
    rows.append(row)
    with (root / 'summary.csv').open('w', newline='') as file:
      writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
      writer.writeheader()
      writer.writerows(rows)
    if row['status'] == 'ok':
      print(f"  {row['train_latency_ms_per_step']:.3f} ms/step; "
            f"{row['train_queued_problems_per_second']:.0f} queued problems/s", flush=True)
      for warning in result.get('warnings', []):
        print(f'  Warning: {warning}', flush=True)
    elif 'backend is unavailable' in (directory / 'worker.log').read_text():
      break
  print(f'Summary: {root / "summary.csv"}', flush=True)
  if args.trace and any(row['status'] == 'ok' for row in rows):
    print('Open traces/*/plugins/profile/*/perfetto_trace.json.gz in https://ui.perfetto.dev')
  if failures:
    raise SystemExit(f'{len(failures)} profiling run(s) failed; see their worker.log files.')


if __name__ == '__main__':
  main()
