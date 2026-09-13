"""Continues training a saved checkpoint with L-BFGS, a full-batch quasi-Newton optimizer.

Unlike AdamW/Muon, L-BFGS relies on a line search that repeatedly re-evaluates the objective, so
it only makes sense on a fixed (non-resampled) batch rather than the fresh minibatch each step
that transformer.py's SGD-style training draws. This script therefore samples one batch up front
and runs every L-BFGS step against it.

Run from the repo root, e.g.:
  python posttrain.py --checkpoint data/transformer_777 --num-steps 500
"""

import argparse
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import optax

from task import (
  TASKS, answer_token_mask, format_answer_tokens, generate_episode, generate_validation_prompts,
)
from transformer import Transformer, generate_tokens, save_checkpoint


def load_checkpoint(checkpoint_path):
  """Accepts either a direct path to a `<step>.pkl` file, or a checkpoint directory, in which
  case the highest-numbered checkpoint in it is loaded."""

  if os.path.isdir(checkpoint_path):
    files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pkl')]
    if not files:
      raise ValueError(f"No checkpoints found in {checkpoint_path}")
    step = max(int(f[:-4]) for f in files)
    checkpoint_path = os.path.join(checkpoint_path, f"{step}.pkl")

  with open(checkpoint_path, 'rb') as f:
    checkpoint = pickle.load(f)
  return checkpoint


def build_model(hparams):
  task = TASKS[hparams['task']]()
  transformer = Transformer(
    n_heads=hparams['n_heads'],
    d_model=hparams['d_model'],
    use_bias=hparams['use_bias'],
    n_layers=hparams['n_layers'],
    recurrent_steps=hparams.get('recurrent_steps', 1),
    vocab_size=hparams['vocab_size'],
    hidden_dims=tuple(hparams['hidden_dims']),
    activation=hparams['activation'],
    normalization=hparams['normalization'],
    max_seq_len=hparams['max_seq_len'],
    pos_embed=hparams.get('pos_embed', 'learned'),
    rope_base=hparams.get('rope_base', 10000.0),
    qk_norm=hparams.get('qk_norm', False),
    n_kv_heads=hparams.get('n_kv_heads', 0),
  )
  return task, transformer


def parse_args():
  parser = argparse.ArgumentParser(description="Post-train a saved checkpoint with L-BFGS.")
  parser.add_argument('--checkpoint', type=str, required=True,
                       help="Path to a `<step>.pkl` checkpoint file, or a checkpoint directory "
                            "(loads its highest-numbered checkpoint).")
  parser.add_argument('--num-steps', type=int, default=500, help="Number of L-BFGS steps to run.")
  parser.add_argument('--batch-size', type=int, default=1024,
                       help="Size of the fixed batch L-BFGS optimizes against. Sampled once "
                            "up front, not resampled between steps.")
  parser.add_argument('--num-length', type=int, nargs='+', default=None,
                       help="Cap on the operands' digit length for the post-training batch, as in "
                            "--num-length-schedule: one value caps both operands, two values (N1 N2) "
                            "cap the first and second operand independently. Defaults to the "
                            "checkpoint's full --sequence-length (no curriculum).")
  parser.add_argument('--memory-size', type=int, default=10,
                       help="Number of past updates L-BFGS keeps for its Hessian approximation.")
  parser.add_argument('--seed', type=int, default=0, help="RNG seed for the batch and validation.")
  parser.add_argument('--num-samples', type=int, default=1000, help="Validation batch size.")
  parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help="Folder to store post-trained checkpoints in. Defaults to a "
                            "'posttrain' subfolder next to the loaded checkpoint.")
  parser.add_argument('--checkpoint-every', type=int, default=50, help="Save a checkpoint every N steps.")
  parser.add_argument('--steps-per-chunk', type=int, default=50,
                       help="Number of L-BFGS steps fused into a single jitted lax.scan call.")
  return parser.parse_args()


def main():
  args = parse_args()

  checkpoint = load_checkpoint(args.checkpoint)
  base_step = checkpoint['step']
  params = checkpoint['params']
  hparams = checkpoint['hparams']

  task, transformer = build_model(hparams)
  sequence_length1, sequence_length2 = hparams['sequence_length']
  reverse_operands = hparams.get('reverse_operands', False)
  early_eos = hparams.get('early_eos', False)
  if args.num_length is None:
    num_length1, num_length2 = sequence_length1, sequence_length2
  elif len(args.num_length) == 1:
    num_length1 = num_length2 = args.num_length[0]
  elif len(args.num_length) == 2:
    num_length1, num_length2 = args.num_length
  else:
    raise ValueError(f"--num-length takes 1 or 2 values, got {len(args.num_length)}: {args.num_length}")

  checkpoint_dir = args.checkpoint_dir
  if checkpoint_dir is None:
    base_dir = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.dirname(args.checkpoint)
    checkpoint_dir = os.path.join(base_dir, 'posttrain')
  checkpoint_every = args.checkpoint_every
  steps_per_chunk = max(1, args.steps_per_chunk)

  posttrain_hparams = dict(hparams)
  posttrain_hparams['posttrain'] = {
    'base_checkpoint': os.path.abspath(args.checkpoint),
    'base_step': base_step,
    'optimizer': 'lbfgs',
    'memory_size': args.memory_size,
    'batch_size': args.batch_size,
    'num_length': [num_length1, num_length2],
    'num_steps': args.num_steps,
    'seed': args.seed,
  }

  rng_key = jax.random.key(args.seed)
  rng_key, batch_key = jax.random.split(rng_key)
  batch = generate_episode(
    task, sequence_length1, sequence_length2, args.batch_size, batch_key, num_length1, num_length2,
    reverse_operands=reverse_operands, early_eos=early_eos)

  answer_start = task.answer_start_index(sequence_length1, sequence_length2)

  def loss_fn(params):
    def loss_fn_single(sequence):
      logits = transformer.apply(params, sequence, train=True)
      loss = optax.softmax_cross_entropy_with_integer_labels(
        logits[answer_start:-1], sequence[answer_start + 1:])
      if early_eos:
        mask = answer_token_mask(sequence[answer_start + 1:], task.eos_token)
        return (loss * mask).sum() / mask.sum()
      return loss.mean()
    return jax.vmap(loss_fn_single)(batch).mean()

  optimizer = optax.lbfgs(memory_size=args.memory_size)
  opt_state = optimizer.init(params)
  value_and_grad_fn = optax.value_and_grad_from_state(loss_fn)

  def lbfgs_step(carry, _):
    params, opt_state = carry
    value, grad = value_and_grad_fn(params, state=opt_state)
    updates, opt_state = optimizer.update(
      grad, opt_state, params, value=value, grad=grad, value_fn=loss_fn)
    params = optax.apply_updates(params, updates)
    return (params, opt_state), value

  # opt_state isn't donated: optax.lbfgs's state holds copies of params/grad that, on the first
  # call, can alias the same buffer as params, and donating both then trips XLA's "same buffer
  # donated twice" check.
  @partial(jax.jit, static_argnames=('n_steps',), donate_argnums=(0,))
  def train_chunk(params, opt_state, n_steps):
    (params, opt_state), losses = jax.lax.scan(
      lbfgs_step, (params, opt_state), None, length=n_steps)
    return params, opt_state, losses

  @jax.jit
  def validate(params, rng_key):
    prompts, target_digits, (a, b) = generate_validation_prompts(
      task, sequence_length1, sequence_length2, args.num_samples, rng_key,
      reverse_operands=reverse_operands)

    n_output_digits = task.answer_length(sequence_length1, sequence_length2)
    n_output_tokens = n_output_digits + 1 if early_eos else n_output_digits
    generated_tokens = generate_tokens(transformer, params, prompts, n_output_tokens)
    if early_eos:
      target_tokens = format_answer_tokens(task, target_digits, early_eos=True)
      target_mask = answer_token_mask(target_tokens, task.eos_token)
      correct = jnp.all((generated_tokens == target_tokens) | ~target_mask, axis=-1)
    else:
      correct = jnp.all(jnp.flip(generated_tokens, axis=-1) == target_digits, axis=-1)
    return {'accuracy': jnp.mean(correct)}

  rng_key, val_key = jax.random.split(rng_key)
  results = validate(params, val_key)
  print(f"Initial validation accuracy (step {base_step}): {float(results['accuracy']) * 100:.2f}%", flush=True)

  i = 0
  last_checkpoint_step = 0
  while i < args.num_steps:
    n_steps = min(steps_per_chunk, args.num_steps - i)
    params, opt_state, losses = train_chunk(params, opt_state, n_steps)
    i += n_steps

    rng_key, val_key = jax.random.split(rng_key)
    results = validate(params, val_key)
    print(f"Step {base_step + i} (posttrain {i}), Loss (mean over last {n_steps} steps): {losses.mean()}")
    print(f"Validation accuracy: {float(results['accuracy']) * 100:.2f}%", flush=True)

    if checkpoint_dir is not None and i - last_checkpoint_step >= checkpoint_every:
      save_checkpoint(checkpoint_dir, base_step + i, params, opt_state, posttrain_hparams)
      last_checkpoint_step = i

  if checkpoint_dir is not None:
    save_checkpoint(checkpoint_dir, base_step + i, params, opt_state, posttrain_hparams)


if __name__ == "__main__":
  main()
