"""Conditional D3PM for fixed-width arithmetic answers.

Implements the x0 parameterization and hybrid variational/auxiliary objective of
Austin et al., https://arxiv.org/abs/2107.03006. Only answer digits diffuse; the
arithmetic prompt remains visible. Answers use task.py's LSB-first digit order.

Train a masked model (defaults: two layers, d_model=64, 16 diffusion steps):
  .venv/bin/python d3pm.py --task multiply --sequence-length 2 2

Use --corruption uniform to train denoising of random digit replacements.
Use --sampling-steps 4 to evaluate four reverse steps from the 16-step schedule.
Intermediate reverse transitions are stochastic; --sample-final also samples the
last step, which otherwise selects its most likely digit. Validation uses a fixed
dataset and noise seed. This is distribution accuracy, not a held-out pair split.

Evaluate a saved model without training:
  .venv/bin/python d3pm.py --checkpoint data/d3pm_multiply_mask --sampling-steps 4

Compare with the autoregressive models:
  .venv/bin/python experiment_multiplication.py --models plain d3pm-mask d3pm-uniform

For a bidirectional one-pass baseline, train with --diffusion-steps 1
--aux-loss-weight 0. This reduces to predicting the entire answer from masks.

Optional unsupervised digit thoughts:
  .venv/bin/python d3pm.py --cot-steps 4 --cot-samples 2 --cot-pg-weight 0.1

Thoughts are sampled from a separate diffusion chain conditioned only on the
operands, then kept visible during answer denoising. They have no target labels.
A policy gradient uses the full sampled reverse-trajectory log probability and
a leave-one-out baseline from the other scratchpads for the same problem. Answer
corruption is shared within each group so reward differences reflect scratchpads.
--cot-sampling-steps controls thought generation during training and evaluation;
it defaults to --diffusion-steps. Thought transitions, including the final one,
are always sampled. Evaluation uses one scratchpad per problem with fixed noise.

This trainer uses known fixed answer widths; early EOS is unsupported. Checkpoints
store the model and process settings, and can be loaded with load_checkpoint()
and build_transformer(). CoT checkpoints use a different vocabulary and layout;
their default output directory has a _cot<N> suffix.
"""

import argparse
from functools import partial
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import optax

from task import TASKS, generate_prompts, generate_validation_prompts
from transformer import build_transformer, parse_num_length_schedule, save_checkpoint


def log_probs(probs):
  """Exact support for sampling: zero-probability states cannot be drawn."""
  return jnp.where(probs > 0, jnp.log(jnp.maximum(probs, 1e-30)), -jnp.inf)


class D3PM:
  """Small categorical transition matrices with row-source/column-target axes.

  Local states are digits 0..9, plus MASK=10 for absorbing corruption. These are
  distinct from prompt token IDs. Q_bar(t) = alpha_bar(t) I + (1-alpha_bar(t)) R,
  where R is a projection onto the terminal prior and alpha_bar(t) = 1-t/T.
  Thus q(x_T|x_0) equals the prior exactly, including with a small number of steps.
  Matrix/posterior methods accept a scalar timestep and any answer array shape.
  """

  mask_state = 10

  def __init__(self, num_steps=16, corruption='mask'):
    if num_steps < 1 or corruption not in ('mask', 'uniform'):
      raise ValueError('D3PM requires positive steps and mask or uniform corruption')
    self.num_steps = num_steps
    self.corruption = corruption
    self.num_states = 11 if corruption == 'mask' else 10
    self.identity = jnp.eye(self.num_states)
    self.prior = (jax.nn.one_hot(self.mask_state, self.num_states)
                  if corruption == 'mask' else jnp.full((10,), 0.1))
    self.reset = jnp.broadcast_to(self.prior, self.identity.shape)

  def q_bar(self, t):
    alpha = 1 - jnp.asarray(t, dtype=jnp.float32) / self.num_steps
    return alpha * self.identity + (1 - alpha) * self.reset

  def transition(self, s, t):
    """Q_{s->t}, including skipped reverse steps (0 <= s < t <= T)."""
    alpha = (self.num_steps - t) / (self.num_steps - s)
    return alpha * self.identity + (1 - alpha) * self.reset

  def sample_forward(self, clean, t, key):
    return jax.random.categorical(key, log_probs(self.q_bar(t)[clean])).astype(jnp.int32)

  def posterior(self, clean_probs, noisy, s, t):
    """Normalize (p(x0) Q_bar(s)) * Q_{s->t}[:, x_t], as in D3PM Eq. 4.

    A one-hot clean_probs gives the true forward posterior. A model distribution
    gives the learned reverse transition. Structural zeros are preserved.
    """
    prior_at_s = clean_probs @ self.q_bar(s)
    likelihood = jnp.moveaxis(self.transition(s, t)[:, noisy], 0, -1)
    weights = prior_at_s * likelihood
    normalizer = weights.sum(axis=-1, keepdims=True)
    return weights / jnp.where(normalizer > 0, normalizer, 1)

  def clean_probs(self, digit_logits):
    probs = jax.nn.softmax(digit_logits, axis=-1)
    # Keep every clean category possible even if softmax underflows. Otherwise
    # a very confident wrong prediction could erase the posterior's support.
    probs = jnp.maximum(probs, 1e-20)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    if self.corruption == 'mask':
      probs = jnp.pad(probs, [(0, 0)] * (probs.ndim - 1) + [(0, 1)])
    return probs

  def input_tokens(self, answers, mask_token):
    if self.corruption == 'mask':
      return jnp.where(answers == self.mask_state, mask_token, answers)
    return answers


def denoise_logits(model, params, process, prompts, noisy, timesteps, mask_token, train=False):
  """Batched predictions at the answer positions, restricted to clean digits."""
  sequences = jnp.concatenate((prompts, process.input_tokens(noisy, mask_token)), axis=-1)
  def apply(sequence, t):
    return model.apply(params, sequence, train=train, timestep=t)[prompts.shape[1]:, :10]
  return jax.vmap(apply)(sequences, timesteps)


def diffusion_loss(params, model, process, prompts, clean, key, mask_token, aux_weight=0.1,
                   return_per_example=False, noise_group_size=1):
  """Unbiased single-timestep estimate of sum_t L_t, averaged over digits.

  t=1 uses reconstruction NLL; t>1 uses posterior KL. The terminal prior KL is
  exactly zero for this schedule. The auxiliary loss predicts x0 at all digits.
  """
  if noise_group_size < 1 or clean.shape[0] % noise_group_size:
    raise ValueError('noise_group_size must divide the batch size')
  time_key, noise_key = jax.random.split(key)
  groups = clean.shape[0] // noise_group_size
  times = jnp.repeat(jax.random.randint(time_key, (groups,), 1, process.num_steps + 1),
                     noise_group_size, axis=0)
  noise_keys = jnp.repeat(jax.random.split(noise_key, groups), noise_group_size, axis=0)
  noisy = jax.vmap(process.sample_forward)(clean, times, noise_keys)
  logits = denoise_logits(model, params, process, prompts, noisy, times, mask_token, train=True)
  predicted_clean = process.clean_probs(logits)
  true_clean = jax.nn.one_hot(clean, process.num_states)
  posterior = jax.vmap(lambda p, x, t: process.posterior(p, x, t - 1, t))
  q = posterior(true_clean, noisy, times)
  p = posterior(predicted_clean, noisy, times)
  # Finite log floors avoid NaNs from 0 * log(0), including in gradients.
  log_q = jnp.log(jnp.maximum(q, 1e-30))
  log_p = jnp.log(jnp.maximum(p, 1e-30))
  kl = (q * (log_q - log_p)).sum(axis=-1)
  reconstruction = -jnp.take_along_axis(log_p, clean[..., None], axis=-1)[..., 0]
  vb = process.num_steps * jnp.where(times[:, None] == 1, reconstruction, kl).mean(axis=-1)
  auxiliary = optax.softmax_cross_entropy_with_integer_labels(logits, clean).mean(axis=-1)
  loss = vb + aux_weight * auxiliary
  return (loss if return_per_example else loss.mean()), {'vb': vb.mean(), 'auxiliary': auxiliary.mean()}


def generate_answers(model, params, process, prompts, answer_length, key, mask_token,
                     sampling_steps=None, sample_final=False, return_trace=False, return_logprob=False):
  """Reverse D3PM chain from its prior; returns LSB-first digits.

  Intermediate steps sample the categorical posterior. The last step takes its
  mode by default for arithmetic evaluation; sample_final=True samples the full
  generative chain. A trace includes x_T and every subsequent answer state.
  return_logprob requires full sampling and returns the sum of learned reverse
  transition log probabilities per example (the parameter-independent prior is
  omitted). Samples are discrete; gradients of this score are used for REINFORCE.
  """
  sampling_steps = process.num_steps if sampling_steps is None else sampling_steps
  if not 1 <= sampling_steps <= process.num_steps:
    raise ValueError('sampling_steps must be between 1 and diffusion_steps')
  if return_logprob and not sample_final:
    raise ValueError('Trajectory policy scores require sample_final=True')
  key, prior_key = jax.random.split(key)
  shape = (prompts.shape[0], answer_length)
  initial = jax.random.categorical(prior_key, log_probs(process.prior), shape=shape).astype(jnp.int32)
  times = jnp.array([process.num_steps - (i * process.num_steps) // sampling_steps
                    for i in range(sampling_steps + 1)], dtype=jnp.int32)

  def step(carry, pair):
    noisy, key = carry
    t, s = pair
    logits = denoise_logits(model, params, process, prompts, noisy,
                            jnp.full((prompts.shape[0],), t), mask_token)
    p = process.posterior(process.clean_probs(logits), noisy, s, t)
    key, draw_key = jax.random.split(key)
    sampled = jax.random.categorical(draw_key, log_probs(p)).astype(jnp.int32)
    next_answer = sampled if sample_final else jnp.where(s == 0, jnp.argmax(p, axis=-1), sampled)
    if return_logprob:
      score = jnp.take_along_axis(log_probs(p), next_answer[..., None], axis=-1)[..., 0].sum(axis=-1)
      return (next_answer, key), (next_answer, score)
    return (next_answer, key), next_answer

  (answers, _), trace = jax.lax.scan(step, (initial, key), (times[:-1], times[1:]))
  if return_logprob:
    trace, scores = trace
    scores = scores.sum(axis=0)
  if return_trace:
    trace = jnp.concatenate((initial[None], trace), axis=0)
    return (answers, trace, scores) if return_logprob else (answers, trace)
  if return_logprob:
    return answers, scores
  return answers


def generate_thoughts(model, params, process, prompts, task, cot_steps, key, mask_token,
                     sampling_steps=None, return_logprob=False):
  """Sample a scratchpad without access to answer tokens or answer targets."""
  if cot_steps < 1:
    raise ValueError('Thought generation requires a positive cot_steps')
  marker = jnp.full((prompts.shape[0], 1), task.think_token, dtype=prompts.dtype)
  thought_prompts = jnp.concatenate((prompts, marker), axis=-1)
  return generate_answers(model, params, process, thought_prompts, cot_steps, key,
                           mask_token, sampling_steps, sample_final=True,
                           return_logprob=return_logprob)


def thought_prefix(prompts, thoughts, task):
  """Layout for answer generation: operands = THINK thoughts ANSWER."""
  shape = (prompts.shape[0], 1)
  think = jnp.full(shape, task.think_token, dtype=prompts.dtype)
  answer = jnp.full(shape, task.answer_token, dtype=prompts.dtype)
  return jnp.concatenate((prompts, think, thoughts, answer), axis=-1)


def thought_policy_loss(answer_losses, trajectory_logprobs, cot_samples):
  """Leave-one-out REINFORCE surrogate: lower answer loss rewards a trajectory."""
  if cot_samples < 2 or answer_losses.shape[0] % cot_samples:
    raise ValueError('CoT requires at least two samples per problem in a complete group')
  grouped = answer_losses.reshape(-1, cot_samples)
  # Subtract a shared offset before comparing losses. This leaves the advantage
  # unchanged mathematically and makes identical rewards cancel exactly in float32.
  centered = grouped - grouped[:, :1]
  other_mean = (centered.sum(axis=-1, keepdims=True) - centered) / (cot_samples - 1)
  advantage = jax.lax.stop_gradient(other_mean - centered).reshape(-1)
  return -(advantage * trajectory_logprobs).mean()


def cot_diffusion_loss(params, model, process, prompts, clean, key, mask_token, task,
                       cot_steps, cot_samples=2, cot_pg_weight=0.1, aux_weight=0.1,
                       cot_sampling_steps=None):
  """Train answers and unsupervised diffusion thoughts with shared parameters.

  Discrete thoughts carry no pathwise gradient. Only their trajectory scores
  receive the policy gradient; advantages cannot backpropagate through rewards.
  Direct answer gradients train prediction conditioned on the sampled thoughts.
  """
  if cot_samples < 2:
    raise ValueError('CoT requires at least two scratchpad samples per problem')
  thought_key, answer_key = jax.random.split(key)
  prompts = jnp.repeat(prompts, cot_samples, axis=0)
  clean = jnp.repeat(clean, cot_samples, axis=0)
  with jax.named_scope('thought_generation'):
    thoughts, scores = generate_thoughts(model, params, process, prompts, task, cot_steps,
                                        thought_key, mask_token, cot_sampling_steps, return_logprob=True)
  prefix = thought_prefix(prompts, jax.lax.stop_gradient(thoughts), task)
  with jax.named_scope('answer_denoising_loss'):
    losses, metrics = diffusion_loss(params, model, process, prefix, clean, answer_key,
                                    mask_token, aux_weight, return_per_example=True,
                                    noise_group_size=cot_samples)
  policy_loss = thought_policy_loss(losses, scores, cot_samples)
  return losses.mean() + cot_pg_weight * policy_loss, {
    **metrics, 'answer_loss': losses.mean(), 'cot_policy': policy_loss,
    'thought_logprob': scores.mean(),
  }


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--task', choices=sorted(TASKS), default='multiply')
  parser.add_argument('--sequence-length', type=int, nargs='+', default=[2, 2])
  parser.add_argument('--reverse-operands', action='store_true')
  parser.add_argument('--corruption', choices=['mask', 'uniform'], default='mask')
  parser.add_argument('--diffusion-steps', type=int, default=16)
  parser.add_argument('--sampling-steps', type=int, default=None)
  parser.add_argument('--sample-final', action='store_true')
  parser.add_argument('--aux-loss-weight', type=float, default=0.1)
  parser.add_argument('--cot-steps', type=int, default=0,
                      help='Number of unsupervised diffusion thought digits; zero disables thoughts.')
  parser.add_argument('--cot-samples', type=int, default=2,
                      help='Scratchpad samples per training problem; CoT requires at least two.')
  parser.add_argument('--cot-pg-weight', type=float, default=0.1,
                      help='Weight of the scratchpad trajectory policy-gradient loss.')
  parser.add_argument('--cot-sampling-steps', type=int, default=None,
                      help='Thought reverse steps during training/evaluation; defaults to diffusion steps.')
  parser.add_argument('--d-model', type=int, default=64)
  parser.add_argument('--n-heads', type=int, default=4)
  parser.add_argument('--n-layers', type=int, default=2)
  parser.add_argument('--recurrent-steps', type=int, default=1)
  parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128])
  parser.add_argument('--use-bias', action='store_true')
  parser.add_argument('--activation', choices=['silu', 'gelu', 'relu'], default='silu')
  parser.add_argument('--normalization', choices=['rms', 'layer', 'none'], default='rms')
  parser.add_argument('--pos-embed', choices=['learned', 'rope'], default='learned')
  parser.add_argument('--rope-base', type=float, default=10000.)
  parser.add_argument('--qk-norm', action='store_true')
  parser.add_argument('--n-kv-heads', type=int, default=0)
  parser.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw')
  parser.add_argument('--learning-rate', type=float, default=0.003)
  parser.add_argument('--lr-schedule', choices=['cosine', 'constant'], default='cosine')
  parser.add_argument('--lr-min', type=float, default=0.0003)
  parser.add_argument('--warmup-steps', type=int, default=200)
  parser.add_argument('--grad-clip-norm', type=float, default=1.)
  parser.add_argument('--weight-decay', type=float, default=0.01)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--num-steps', type=int, default=10000)
  parser.add_argument('--num-length-schedule', nargs='+', default=None)
  parser.add_argument('--steps-per-chunk', type=int, default=250)
  parser.add_argument('--num-samples', type=int, default=1000)
  parser.add_argument('--validation-seed', type=int, default=None)
  parser.add_argument('--seed', type=int, default=None,
                      help='Defaults to zero for training, or the saved seed for checkpoint evaluation.')
  parser.add_argument('--checkpoint-dir', default=None,
                      help='Defaults to data/d3pm_<task>_<corruption>.')
  parser.add_argument('--checkpoint-every', type=int, default=1000)
  parser.add_argument('--checkpoint', type=Path, help='Evaluate a saved D3PM checkpoint; no training.')
  args = parser.parse_args()
  if len(args.sequence_length) not in (1, 2) or min(args.sequence_length) < 1:
    parser.error('--sequence-length requires one or two positive widths')
  if len(args.sequence_length) == 1:
    args.sequence_length *= 2
  positive = ('diffusion_steps', 'd_model', 'n_heads', 'n_layers', 'recurrent_steps',
              'batch_size', 'num_steps', 'steps_per_chunk', 'num_samples', 'checkpoint_every')
  if any(getattr(args, field) < 1 for field in positive) or min(args.hidden_dims) < 1:
    parser.error('steps, model dimensions, batch size and sample counts must be positive')
  if args.d_model % args.n_heads or args.n_kv_heads < 0 or (args.n_kv_heads and args.n_heads % args.n_kv_heads):
    parser.error('--d-model must be divisible by --n-heads; --n-kv-heads must divide --n-heads')
  if args.aux_loss_weight < 0 or args.learning_rate <= 0 or args.warmup_steps < 0:
    parser.error('auxiliary weight/warmup must be nonnegative and learning rate positive')
  if not 0 <= args.lr_min <= args.learning_rate:
    parser.error('--lr-min must lie between zero and --learning-rate')
  if args.sampling_steps is not None and args.sampling_steps < 1:
    parser.error('--sampling-steps must be positive')
  if not args.checkpoint and args.sampling_steps is not None and args.sampling_steps > args.diffusion_steps:
    parser.error('--sampling-steps cannot exceed --diffusion-steps')
  if args.cot_steps < 0 or (args.cot_steps and args.cot_samples < 2):
    parser.error('CoT requires --cot-steps >= 0 and --cot-samples >= 2')
  if args.cot_pg_weight < 0:
    parser.error('--cot-pg-weight must be nonnegative')
  if args.cot_sampling_steps is not None and args.cot_sampling_steps < 1:
    parser.error('--cot-sampling-steps must be positive')
  if not args.checkpoint and args.cot_sampling_steps is not None and args.cot_sampling_steps > args.diffusion_steps:
    parser.error('--cot-sampling-steps cannot exceed --diffusion-steps')
  return args


def load_checkpoint(path):
  path = Path(path)
  if path.is_dir():
    files = [p for p in path.glob('*.pkl') if p.stem.isdigit()]
    if not files:
      raise ValueError(f'No checkpoints in {path}')
    path = max(files, key=lambda p: int(p.stem))
  with path.open('rb') as file:
    checkpoint = pickle.load(file)
  if checkpoint['hparams'].get('model_type') != 'd3pm':
    raise ValueError('Expected a D3PM checkpoint')
  return checkpoint


def make_train_step(model, process, task, optimizer, width1, width2, batch_size,
                    mask_token, reverse_operands=False, cot_steps=0, cot_samples=2,
                    cot_pg_weight=0.1, aux_loss_weight=0.1, cot_sampling_steps=None):
  """Build the optimizer step shared by the trainer and profiling tools."""
  make_prompts = jax.named_call(generate_prompts, name="data_generation")
  def train_step(carry, step_key, n1, n2):
    params, state = carry
    if cot_steps:
      # Match the AR CoT trainer's operand RNG, independently of scratchpad draws.
      data_key, noise_key = jax.random.split(step_key)
    else:
      data_key = step_key
      noise_key = jax.random.fold_in(step_key, 101)
    batch_prompts, _, answers = make_prompts(task, width1, width2, batch_size, data_key,
                                           n1, n2, reverse_operands=reverse_operands)
    if cot_steps:
      loss_fn = lambda p: cot_diffusion_loss(
        p, model, process, batch_prompts, jnp.flip(answers, axis=-1), noise_key, mask_token,
        task, cot_steps, cot_samples, cot_pg_weight, aux_loss_weight,
        cot_sampling_steps)
    else:
      loss_fn = lambda p: diffusion_loss(p, model, process, batch_prompts, jnp.flip(answers, axis=-1),
                                         noise_key, mask_token, aux_loss_weight)
    with jax.named_scope("loss_and_gradients"):
      (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    with jax.named_scope("optimizer_update"):
      updates, state = optimizer.update(grads, state, params)
    return (optax.apply_updates(params, updates), state), {'loss': loss, **metrics}

  return jax.named_call(train_step, name="train_step")


def main():
  args = parse_args()
  if args.checkpoint:
    checkpoint = load_checkpoint(args.checkpoint)
    hparams, params = checkpoint['hparams'], checkpoint['params']
  else:
    hparams = vars(args).copy()
    hparams.pop('checkpoint')
    task = TASKS[args.task]()
    base_vocab = task.vocab_size + (2 if args.cot_steps else 0)
    hparams.update(model_type='d3pm', vocab_size=base_vocab + (args.corruption == 'mask'),
                   max_seq_len=sum(args.sequence_length) + 2 + task.answer_length(*args.sequence_length) +
                               (args.cot_steps + 2 if args.cot_steps else 0),
                   mask_token=base_vocab, early_eos=False)
    if hparams['checkpoint_dir'] is None:
      suffix = f'_cot{args.cot_steps}' if args.cot_steps else ''
      hparams['checkpoint_dir'] = f'data/d3pm_{args.task}_{args.corruption}{suffix}'
    hparams['seed'] = 0 if args.seed is None else args.seed
  seed = hparams['seed'] if args.seed is None else args.seed
  key = jax.random.key(seed)
  task = TASKS[hparams['task']]()
  width1, width2 = hparams['sequence_length']
  process = D3PM(hparams['diffusion_steps'], hparams['corruption'])
  model = build_transformer(hparams)
  mask_token = hparams['mask_token']
  answer_length = task.answer_length(width1, width2)
  sampling_steps = args.sampling_steps if args.sampling_steps is not None else hparams.get('sampling_steps')
  if sampling_steps is not None and sampling_steps > process.num_steps:
    raise ValueError('--sampling-steps cannot exceed the checkpoint diffusion_steps')
  sample_final = args.sample_final or hparams.get('sample_final', False)
  cot_steps = hparams.get('cot_steps', 0)
  cot_sampling_steps = (args.cot_sampling_steps if args.cot_sampling_steps is not None
                        else hparams.get('cot_sampling_steps'))
  if cot_sampling_steps is not None and cot_sampling_steps > process.num_steps:
    raise ValueError('--cot-sampling-steps cannot exceed the checkpoint diffusion_steps')

  # Fixed evaluation data and noise across checkpoints; separate keys keep training
  # examples unchanged when the evaluation size or generation settings change.
  validation_seed = args.validation_seed
  if validation_seed is None:
    validation_seed = hparams.get('validation_seed')
  val_data_key = jax.random.key(seed if validation_seed is None else validation_seed)
  val_noise_key = jax.random.fold_in(val_data_key, 9001)
  prompts, targets, _ = generate_validation_prompts(
    task, width1, width2, args.num_samples, val_data_key,
    reverse_operands=hparams['reverse_operands'])
  targets = jnp.flip(targets, axis=-1)

  @jax.jit
  def validate(params):
    answer_prompts, answer_key = prompts, val_noise_key
    if cot_steps:
      thought_key, answer_key = jax.random.split(val_noise_key)
      thoughts = generate_thoughts(model, params, process, prompts, task, cot_steps,
                                   thought_key, mask_token, cot_sampling_steps)
      answer_prompts = thought_prefix(prompts, thoughts, task)
    answers = generate_answers(model, params, process, answer_prompts, answer_length, answer_key,
                               mask_token, sampling_steps, sample_final)
    matches = answers == targets
    return jnp.all(matches, axis=-1).mean(), matches.mean()

  if not args.checkpoint:
    # Match transformer.py's data RNG progression for paired sweep runs.
    key, data_key = jax.random.split(key)
    key, init_key = jax.random.split(key)
    example_prompts, _, _ = generate_prompts(task, width1, width2, 1, data_key)
    if cot_steps:
      example_prompts = thought_prefix(example_prompts, jnp.zeros((1, cot_steps), dtype=jnp.int32), task)
    example = jnp.concatenate((example_prompts[0], jnp.zeros((answer_length,), dtype=jnp.int32)))
    params = model.init(init_key, example, timestep=jnp.array(process.num_steps))
  print(f'Trainable parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params)):,}', flush=True)
  accuracy, digit_accuracy = validate(params)
  print(f'Initial validation accuracy: {float(accuracy):.2%}, digit accuracy: {float(digit_accuracy):.2%}', flush=True)
  if args.checkpoint:
    start = time.monotonic()
    accuracy, digit_accuracy = validate(params)
    accuracy.block_until_ready()
    print(f'Validation accuracy: {float(accuracy):.2%}')
    print(f'Digit accuracy: {float(digit_accuracy):.2%}, validation seconds: {time.monotonic() - start:.3f}')
    return

  schedule = parse_num_length_schedule(args.num_length_schedule or [], width1, width2, args.num_steps)
  if any(not (1 <= n1 <= width1 and 1 <= n2 <= width2) or steps < 1 for n1, n2, steps in schedule):
    raise ValueError('Curriculum widths must fit --sequence-length and stage steps must be positive')
  total_steps = sum(steps for _, _, steps in schedule)
  hparams.update(num_length_schedule=schedule, total_steps=total_steps)
  if args.lr_schedule == 'constant':
    lr = optax.constant_schedule(args.learning_rate)
  elif args.warmup_steps:
    if args.warmup_steps >= total_steps:
      raise ValueError('--warmup-steps must be smaller than the training horizon')
    lr = optax.warmup_cosine_decay_schedule(0., args.learning_rate, args.warmup_steps,
                                           total_steps, end_value=args.lr_min)
  else:
    lr = optax.cosine_decay_schedule(args.learning_rate, total_steps, alpha=args.lr_min / args.learning_rate)
  optimizer_fn = optax.contrib.muon if args.optimizer == 'muon' else optax.adamw
  optimizer = optimizer_fn(learning_rate=lr, weight_decay=args.weight_decay)
  if args.grad_clip_norm > 0:
    optimizer = optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optimizer)
  state = optimizer.init(params)

  train_step = make_train_step(
    model, process, task, optimizer, width1, width2, args.batch_size, mask_token,
    args.reverse_operands, cot_steps, args.cot_samples, args.cot_pg_weight,
    args.aux_loss_weight, cot_sampling_steps)

  @partial(jax.jit, static_argnames=('n1', 'n2', 'steps'), donate_argnums=(0,))
  def train_chunk(carry, key, n1, n2, steps):
    keys = jax.random.split(key, steps)
    carry, metrics = jax.lax.scan(lambda c, k: train_step(c, k, n1, n2), carry, keys)
    return carry, jax.tree_util.tree_map(jnp.mean, metrics)

  carry = (params, state)
  step, last_save = 0, 0
  for n1, n2, stage_steps in schedule:
    print(f'Training on num_length1={n1}, num_length2={n2} for {stage_steps} steps', flush=True)
    done = 0
    while done < stage_steps:
      count = min(args.steps_per_chunk, stage_steps - done)
      key, train_key = jax.random.split(key)
      carry, metrics = train_chunk(carry, train_key, n1, n2, count)
      key, _ = jax.random.split(key)  # Same validation-key consumption as transformer.py.
      step += count
      done += count
      start = time.monotonic()
      accuracy, digit_accuracy = validate(carry[0])
      accuracy.block_until_ready()
      print(f'Step {step}, Loss (mean over last {count} steps): {float(metrics["loss"])}, VB: {float(metrics["vb"])}, auxiliary CE: {float(metrics["auxiliary"])}')
      if cot_steps:
        print(f'Answer loss: {float(metrics["answer_loss"])}, CoT policy loss: {float(metrics["cot_policy"])}, thought trajectory logprob: {float(metrics["thought_logprob"])}')
      print(f'Validation accuracy: {float(accuracy):.2%}')
      print(f'Digit accuracy: {float(digit_accuracy):.2%}, validation seconds: {time.monotonic() - start:.3f}', flush=True)
      if step - last_save >= args.checkpoint_every:
        save_checkpoint(hparams['checkpoint_dir'], step, *carry, hparams)
        last_save = step
  save_checkpoint(hparams['checkpoint_dir'], step, *carry, hparams)


if __name__ == '__main__':
  main()
