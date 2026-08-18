import argparse
import os
import pickle
from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Sequence

import optax



class Attention(nn.Module): 
  
  n_heads: int = 8 
  d_model: int = 512
  use_bias: bool = True
  
  
  @nn.compact
  def __call__(self, x, train: bool = False):
    seq_len, embed_dim = x.shape
    
    head_dim = embed_dim // self.n_heads
    
    qkv = nn.Dense(3 * embed_dim, use_bias=self.use_bias, name='qkv_proj')(x)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    
    
    # q1 = q.reshape(seq_len, self.n_heads, head_dim).transpose(1, 0, 2)
    # k1 = k.reshape(seq_len, self.n_heads, head_dim).transpose(1, 0, 2)
    # v1 = v.reshape(seq_len, self.n_heads, head_dim).transpose(1, 0, 2)
    # scale = 1.0 / jnp.sqrt(head_dim)
    # attn = jnp.matmul(q1, k1.transpose(0, 2, 1)) * scale
    # mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    # attn = jnp.where(mask == 0, -1e9, attn)

    # attn = jax.nn.softmax(attn, axis=-1)
    # out = jnp.matmul(attn, v)
    # out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

    q = q.reshape(seq_len, self.n_heads, head_dim)
    k = k.reshape(seq_len, self.n_heads, head_dim)
    v = v.reshape(seq_len, self.n_heads, head_dim)
    
    attention = jnp.einsum('ijk,ljk->ilj', q, k)
    attention = attention / jnp.sqrt(head_dim)

    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    attention = jnp.where(mask[..., None] < 0.5, -jnp.inf, attention)
    attention = jax.nn.softmax(attention, axis=1)
    
    y = jnp.einsum('ijk,jkl->ikl', attention, v)
    y = y.reshape(seq_len, embed_dim)
    
    y = nn.Dense(embed_dim, use_bias=self.use_bias, name='out_proj')(y)
    
    return y
  
  
class Activation(nn.Module):
  activation: str = 'gelu'
  
  @nn.compact
  def __call__(self, x, train: bool = False):
    if self.activation == 'gelu':
      return nn.gelu(x)
    elif self.activation == 'relu':
      return nn.relu(x)
    elif self.activation == 'swish':
      return nn.swish(x)
    elif self.activation == 'silu':
      return nn.silu(x)
    elif self.activation == 'mish':
      return nn.mish(x)
    elif self.activation == 'tanh':
      return nn.tanh(x)
    elif self.activation == 'sigmoid':
      return nn.sigmoid(x)
    elif self.activation == 'none':
      return x
    else:
      raise ValueError(f"Invalid activation: {self.activation}")

class Normalization(nn.Module):
  normalization: str = 'layer'
  
  @nn.compact
  def __call__(self, x, train: bool = False):
    if self.normalization == 'layer':
      return nn.LayerNorm()(x)
    elif self.normalization == 'rms':
      return nn.RMSNorm()(x)
    elif self.normalization == 'none':
      return x
    else:
      raise ValueError(f"Invalid normalization: {self.normalization}")

class MLP(nn.Module):
  hidden_dims: Sequence[int] = (32, 32)
  out_dim: int = 32
  activation: str = 'gelu'
  normalization: str = 'none'
  use_bias: bool = True
  
  @nn.compact
  def __call__(self, x, train: bool = False):
    for hidden_dim in self.hidden_dims:
      # GLU: one projection is gated by the activated other one, so self.activation
      # picks the variant (silu -> SwiGLU, gelu -> GEGLU, none -> plain bilinear GLU).
      gate, value = jnp.split(nn.Dense(2 * hidden_dim, use_bias=self.use_bias)(x), 2, axis=-1)
      x = Activation(self.activation)(gate, train=train) * value
      x = Normalization(self.normalization)(x, train=train)
    x = nn.Dense(self.out_dim, use_bias=self.use_bias)(x)
    return x
    
      
class Transformer(nn.Module):
  n_heads: int = 8
  d_model: int = 512
  use_bias: bool = True
  n_layers: int = 6
  vocab_size: int = 12
  hidden_dims: Sequence[int] = (32, 32)
  activation: str = 'gelu'
  normalization: str = 'none'
  max_seq_len: int = 512

  @nn.compact
  def __call__(self, x, train: bool = False):

    seq_len = x.shape[0]

    x = nn.Embed(self.vocab_size, self.d_model)(x)

    # Fixed-size table sliced to seq_len, so params stay valid across calls with different lengths
    # (needed for autoregressive generation, which calls the model on growing prefixes).
    pos_embed = nn.Embed(self.max_seq_len, self.d_model)(jnp.arange(seq_len, dtype=jnp.int32))
    
    x = x + pos_embed
    

    for _ in range(self.n_layers):
      residual = x
      x = Normalization(self.normalization)(x, train=train)
      x = Attention(n_heads=self.n_heads, d_model=self.d_model, use_bias=self.use_bias)(x, train=train)
      x = x + residual
      
      residual = x
      x = Normalization(self.normalization)(x, train=train)
      x = MLP(hidden_dims=self.hidden_dims, out_dim=self.d_model, activation=self.activation, use_bias=self.use_bias)(x, train=train)
      x = x + residual
      
    x = Normalization(self.normalization)(x, train=train)
    x = nn.Dense(self.vocab_size, use_bias=self.use_bias, name='out_proj')(x) 
    
    return x


class Task:
  """An arithmetic task: the symbols it uses and how one episode is laid out.

  The token layout is shared by every task: 0-9 are digits, followed by the task's operator,
  '=' and <EOS>, so `vocab_size` follows from the tokens rather than being configured.
  Subclasses supply the operator's meaning via `compute` and, where the task needs it, a
  different operand distribution via `sample_inputs`."""

  name = 'task'
  op_token = 10
  eq_token = 11
  eos_token = 12

  @property
  def vocab_size(self) -> int:
    return self.eos_token + 1

  def answer_length(self, max_length: int) -> int:
    """Number of answer digits produced for two `max_length`-digit operands."""
    raise NotImplementedError

  def compute(self, numbers: jnp.ndarray) -> jnp.ndarray:
    """(max_length, 2) MSB-first digit pairs -> the answer's digits, MSB-first."""
    raise NotImplementedError

  def sample_inputs(self, num_length: int, batch_size: int, rng) -> jnp.ndarray:
    return jax.random.randint(rng, (batch_size, num_length, 2), 0, 10)

  def episode_length(self, max_length: int) -> int:
    """Total token length of a generate_episode sequence: N1, op, N2, '=', answer digits, <EOS>."""
    return max_length + 1 + max_length + 1 + self.answer_length(max_length) + 1

  def answer_start_index(self, max_length: int) -> int:
    """Index of the '=' token in a generate_episode sequence.

    Layout: N1 [0, L), op [L], N2 [L+1, 2L], '=' [2L+1], answer digits, <EOS>.
    In a next-token loss, position i predicts token i+1, so slicing logits from this index gives
    exactly the answer targets (the answer digits and <EOS>) and drops the prompt positions,
    whose targets are uniform random digits and carry no learnable signal."""
    return 2 * max_length + 1


class AdditionTask(Task):
  name = 'add'

  def answer_length(self, max_length: int) -> int:
    return max_length + 1

  def compute(self, numbers: jnp.ndarray) -> jnp.ndarray:
    def add_digits(carry, x):
      result = x[0] + x[1] + carry
      carry = result // 10
      result = result % 10
      return carry, result

    carry, digits = jax.lax.scan(add_digits, 0, numbers, reverse=True)
    return jnp.concatenate((carry[None], digits))


TASKS = {task.name: task for task in (AdditionTask,)}


def pad_numbers(numbers: jnp.ndarray, max_length: int):

  return jnp.pad(numbers, (max_length - numbers.shape[0], 0))


def generate_prompts(task: Task, max_length: int, batch_size: int, rng, num_length: int = None):
  """Samples an operand pair per row and renders the prompt 'N1 op N2 ='.

  Numbers are drawn with only `num_length` digits (default: max_length) and zero-padded on the
  most-significant side up to max_length. Returns the prompts, the operand digits and the
  MSB-first answer digits, so both the training and the validation builders share one layout."""

  if num_length is None:
    num_length = max_length

  input_numbers = task.sample_inputs(num_length, batch_size, rng)
  input_numbers = jnp.pad(input_numbers, ((0, 0), (max_length - num_length, 0), (0, 0)))

  answer_digits = jax.vmap(task.compute)(input_numbers)

  op_tokens = jnp.full((batch_size, 1), task.op_token)
  eq_tokens = jnp.full((batch_size, 1), task.eq_token)

  prompts = jnp.concatenate((input_numbers[..., 0], op_tokens, input_numbers[..., 1], eq_tokens), axis=-1)

  return prompts, input_numbers, answer_digits


def generate_episode(task: Task, max_length: int, batch_size: int, rng, num_length: int = None):
  """A full training sequence: the prompt, then the answer emitted LSB-first, then <EOS>."""

  prompts, _, answer_digits = generate_prompts(task, max_length, batch_size, rng, num_length)
  eos_tokens = jnp.full((batch_size, 1), task.eos_token)

  return jnp.concatenate((prompts, jnp.flip(answer_digits, axis=-1), eos_tokens), axis=-1)


def generate_validation_prompts(task: Task, max_length: int, batch_size: int, rng):
  """Same sampling as generate_episode, but stops after '=' (no target digits)."""

  prompts, input_numbers, answer_digits = generate_prompts(task, max_length, batch_size, rng)

  return prompts, answer_digits, input_numbers


def digits_to_int(digits: jnp.ndarray, msb_first: bool = True):
  n = digits.shape[-1]
  exponents = jnp.arange(n - 1, -1, -1) if msb_first else jnp.arange(n)
  return jnp.sum(digits * (10 ** exponents), axis=-1)


def generate_tokens(transformer, params, prompts: jnp.ndarray, n_tokens: int):
  """Greedily decodes n_tokens continuations for each (unbatched-model) row in prompts."""

  apply_single = lambda seq: transformer.apply(params, seq, train=False)

  sequences = prompts
  for _ in range(n_tokens):
    logits = jax.vmap(apply_single)(sequences)
    next_tokens = jnp.argmax(logits[:, -1, :], axis=-1)
    sequences = jnp.concatenate((sequences, next_tokens[:, None]), axis=-1)

  return sequences[:, prompts.shape[1]:]


def save_checkpoint(checkpoint_dir: str, step: int, params, opt_state, hparams: dict):
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_path = os.path.join(checkpoint_dir, f"{step}.pkl")
  with open(checkpoint_path, 'wb') as f:
    pickle.dump({
      'step': step,
      'params': params,
      'opt_state': opt_state,
      'hparams': hparams,
    }, f)


def parse_num_length_schedule(schedule_args, default_length: int, default_steps: int):
  """Parses ["SIZE:ITERS", ...] into [(size, iters), ...]. Any remaining steps (default_steps
  minus the sum of the given ITERS) are appended as a final (default_length, remaining) entry,
  so the schedule doesn't need to spell out the final default_length stage explicitly."""

  schedule = []
  for item in schedule_args:
    size_str, iters_str = item.split(':')
    schedule.append((int(size_str), int(iters_str)))

  remaining_steps = default_steps - sum(iters for _, iters in schedule)
  if remaining_steps > 0:
    schedule.append((default_length, remaining_steps))

  return schedule


def parse_args():
  parser = argparse.ArgumentParser(description="Train a tiny transformer on integer arithmetic.")
  parser.add_argument('--task', type=str, default='add', choices=sorted(TASKS),
                       help="Arithmetic task to train on. The task owns its symbols, so the vocabulary "
                            "size follows from it (addition: 10 digits, +, =, <EOS> -> 13) and is not configurable.")
  parser.add_argument('--sequence-length', type=int, default=10)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--seed', type=int, default=54)
  parser.add_argument('--d-model', type=int, default=8)
  parser.add_argument('--n-heads', type=int, default=1)
  parser.add_argument('--n-layers', type=int, default=1)
  parser.add_argument('--hidden-dims', type=int, nargs='+', default=[16])
  parser.add_argument('--use-bias', action='store_true', default=False)
  parser.add_argument('--activation', type=str, default='silu', choices=['gelu', 'relu', 'swish', 'silu', 'mish', 'tanh', 'sigmoid', 'none'])
  parser.add_argument('--normalization', type=str, default='rms', choices=['layer', 'rms', 'none'])
  parser.add_argument('--num-steps', type=int, default=50000)
  parser.add_argument('--learning-rate', type=float, default=1.5e-2)
  parser.add_argument('--num-length-schedule', type=str, nargs='+', default=["3:2000", "6:5000"],
                       help="List of SIZE:ITERS pairs, e.g. --num-length-schedule 2:1000 5:2000. "
                            "Trains on `num_length=SIZE`-digit numbers (zero-padded to --sequence-length) for ITERS steps each, in order. "
                            "Any steps left over after the schedule (--num-steps minus the sum of ITERS) are trained "
                            "at --sequence-length.")
  parser.add_argument('--lr-schedule', type=str, default='cosine', choices=['cosine', 'constant'],
                       help="Learning rate schedule. 'cosine' decays --learning-rate to --lr-min "
                            "over the whole run (after any warmup).")
  parser.add_argument('--warmup-steps', type=int, default=2000,
                       help="Linear warmup from 0 to --learning-rate over this many steps, before the cosine decay.")
  parser.add_argument('--lr-min', type=float, default=1.5e-3,
                       help="Absolute learning rate the cosine decays to at the final step.")
  parser.add_argument('--num-samples', type=int, default=1000)
  parser.add_argument('--checkpoint-dir', type=str, default="data/small_model", help="Folder to store checkpoints in. If not set, no checkpoints are saved.")
  parser.add_argument('--checkpoint-every', type=int, default=1000, help="Save a checkpoint every N steps.")
  parser.add_argument('--steps-per-chunk', type=int, default=250,
                       help="Number of training steps fused into a single jitted lax.scan call. Larger values "
                            "amortize per-step dispatch/kernel-launch overhead (which dominates at this model "
                            "size on GPU) but coarsen the logging/validation granularity to this many steps.")
  return parser.parse_args() 

def main():
  args = parse_args()
  task = TASKS[args.task]()
  vocab_size = task.vocab_size
  sequence_length = args.sequence_length
  batch_size = args.batch_size
  seed = args.seed
  d_model = args.d_model
  n_heads = args.n_heads
  n_layers = args.n_layers
  hidden_dims = tuple(args.hidden_dims)
  use_bias = args.use_bias
  activation = args.activation
  normalization = args.normalization
  num_steps = args.num_steps
  learning_rate = args.learning_rate
  lr_schedule_name = args.lr_schedule
  warmup_steps = args.warmup_steps
  lr_min = args.lr_min
  num_length_schedule = parse_num_length_schedule(args.num_length_schedule, sequence_length, num_steps)
  # The curriculum is the authority on how many steps actually run (an explicit schedule may
  # overshoot --num-steps), so the LR decays over that horizon, not over --num-steps.
  total_steps = sum(iters for _, iters in num_length_schedule)
  num_samples = args.num_samples
  checkpoint_dir = args.checkpoint_dir
  checkpoint_every = args.checkpoint_every
  steps_per_chunk = max(1, args.steps_per_chunk)

  hparams = {
    'task': task.name,
    # Derived from the task, recorded so a checkpoint stays self-describing.
    'vocab_size': vocab_size,
    'sequence_length': sequence_length,
    'batch_size': batch_size,
    'seed': seed,
    'd_model': d_model,
    'n_heads': n_heads,
    'n_layers': n_layers,
    'hidden_dims': hidden_dims,
    'use_bias': use_bias,
    'activation': activation,
    'normalization': normalization,
    'learning_rate': learning_rate,
    'lr_schedule': lr_schedule_name,
    'warmup_steps': warmup_steps,
    'lr_min': lr_min,
    'total_steps': total_steps,
    'max_seq_len': task.episode_length(sequence_length),
    'num_length_schedule': num_length_schedule,
    'steps_per_chunk': steps_per_chunk,
  }

  transformer = Transformer(
    n_heads=n_heads,
    d_model=d_model,
    use_bias=use_bias,
    n_layers=n_layers,
    vocab_size=vocab_size,
    hidden_dims=hidden_dims,
    activation=activation,
    normalization=normalization,
    max_seq_len=task.episode_length(sequence_length),
  )
  
  
  example_sequence = generate_episode(task, sequence_length, 1, jax.random.key(seed))[0]
  rng_key = jax.random.key(seed)
  rng_key, init_key = jax.random.split(rng_key)
  params = transformer.init(init_key, example_sequence, train=False)

  n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
  print(f"Trainable parameters: {n_params:,}")

  # Decay over the whole run, i.e. across every num_length stage rather than restarting per stage.
  # optax threads the step count through the optimizer state, so this works unchanged inside
  # the scanned train_chunk.
  if lr_schedule_name == 'cosine':
    if warmup_steps >= total_steps:
      raise ValueError(
        f"--warmup-steps ({warmup_steps}) must be less than the {total_steps} steps the run "
        f"actually performs (the sum of --num-length-schedule iters, else --num-steps); "
        f"there would be no steps left to decay over.")
    if warmup_steps > 0:
      lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr_min,
      )
    else:
      # cosine_decay_schedule takes the floor as a fraction of init_value, not an absolute rate.
      lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps,
        alpha=lr_min / learning_rate if learning_rate > 0 else 0.0,
      )
  else:
    lr_schedule = optax.constant_schedule(learning_rate)

  optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=1e-2)
  state = optimizer.init(params)
  
  # Static: the answer span always begins at the same offset, so the loss is a compile-time
  # slice rather than a runtime mask.
  answer_start = task.answer_start_index(sequence_length)

  def train_step(opt, rng_key, num_length):
    """One optimizer step. Written as a lax.scan body (carry -> (carry, y)) so a whole chunk
    of steps can be fused into a single dispatch by train_chunk."""

    batch = generate_episode(task, sequence_length, batch_size, rng_key, num_length)
    params, state = opt

    def loss_fn(params, sequences):

      def loss_fn_single(sequence):
        logits = transformer.apply(params, sequence, train=True)
        # Score only the answer digits and <EOS>; the prompt targets (N1, N2) are uniform random
        # digits, so including them just dilutes the gradient and floors the reported loss.
        loss = optax.softmax_cross_entropy_with_integer_labels(
          logits[answer_start:-1], sequence[answer_start + 1:])
        return loss.mean()
      losses = jax.vmap(loss_fn_single)(sequences)
      return losses.mean()

    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return (params, state), loss

  @partial(jax.jit, static_argnames=('num_length', 'n_steps'), donate_argnums=(0,))
  def train_chunk(opt, rng_key, num_length, n_steps):
    """Runs n_steps training steps inside one jitted lax.scan. At this model size the per-step
    kernels are tiny, so fusing steps keeps the accelerator from being dispatch-bound; donating
    `opt` lets XLA update the params/optimizer buffers in place."""

    step_keys = jax.random.split(rng_key, n_steps)
    opt, losses = jax.lax.scan(lambda o, k: train_step(o, k, num_length), opt, step_keys)
    return opt, losses.mean()

  @jax.jit
  def validate(params, rng_key):
    prompts, target_digits, input_numbers = generate_validation_prompts(task, sequence_length, num_samples, rng_key)

    n_output_digits = task.answer_length(sequence_length)
    generated_digits = generate_tokens(transformer, params, prompts, n_output_digits)
    # The model was trained to emit the answer reversed (LSB-first); flip back to compare digit-for-digit.
    predicted_digits = jnp.flip(generated_digits, axis=-1)

    correct = jnp.all(predicted_digits == target_digits, axis=-1)

    return {
      'accuracy': jnp.mean(correct),
      'correct': correct,
      'n1': digits_to_int(input_numbers[..., 0]),
      'n2': digits_to_int(input_numbers[..., 1]),
      'target_answer': digits_to_int(target_digits),
      'predicted_answer': digits_to_int(predicted_digits),
    }

  opt = (params, state)

  i = 0
  last_checkpoint_step = 0
  for num_length, iters in num_length_schedule:
    print(f"Training on num_length={num_length} for {iters} steps", flush=True)
    steps_done = 0
    while steps_done < iters:
      # A stage whose length is not a multiple of steps_per_chunk ends with one shorter chunk,
      # which costs one extra compilation for that stage.
      n_steps = min(steps_per_chunk, iters - steps_done)
      rng_key, train_key = jax.random.split(rng_key)
      opt, loss = train_chunk(opt, train_key, num_length, n_steps)
      steps_done += n_steps
      i += n_steps

      rng_key, val_key = jax.random.split(rng_key)
      results = validate(opt[0], val_key)
      print(f"Step {i}, Loss (mean over last {n_steps} steps): {loss}, LR: {float(lr_schedule(i - 1)):.3e}")
      print(f"Validation accuracy: {float(results['accuracy']) * 100:.2f}%", flush=True)

      if checkpoint_dir is not None and i - last_checkpoint_step >= checkpoint_every:
        save_checkpoint(checkpoint_dir, i, opt[0], opt[1], hparams)
        last_checkpoint_step = i

  if checkpoint_dir is not None:
    save_checkpoint(checkpoint_dir, i, opt[0], opt[1], hparams)


if __name__ == "__main__":
  main()