import argparse
import os
import pickle
from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Sequence

import optax

from task import TASKS, generate_episode, generate_validation_prompts, digits_to_int


def apply_rope(x, base: float = 10000.0):
  """Rotary positional embedding for a (seq_len, n_heads, head_dim) tensor.

  Rotates consecutive channel pairs by an angle proportional to the position, so attention
  logits depend on relative distance only. If head_dim is odd (this model runs d_model=7 with
  one head), the trailing channel has no partner and is passed through unrotated."""

  seq_len, n_heads, head_dim = x.shape
  n_pairs = head_dim // 2
  if n_pairs == 0:
    return x

  positions = jnp.arange(seq_len, dtype=x.dtype)
  inv_freq = base ** (-jnp.arange(n_pairs, dtype=x.dtype) / n_pairs)
  angles = positions[:, None] * inv_freq[None, :]          # (seq_len, n_pairs)
  cos = jnp.cos(angles)[:, None, :]
  sin = jnp.sin(angles)[:, None, :]

  pairs = x[..., :2 * n_pairs].reshape(seq_len, n_heads, n_pairs, 2)
  even, odd = pairs[..., 0], pairs[..., 1]
  out = jnp.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1)
  out = out.reshape(seq_len, n_heads, 2 * n_pairs)

  # Odd head_dim: the leftover channel has no partner, so it passes through unrotated.
  return jnp.concatenate((out, x[..., 2 * n_pairs:]), axis=-1)


class Attention(nn.Module): 
  
  n_heads: int = 8 
  d_model: int = 512
  use_bias: bool = True
  pos_embed: str = 'learned'
  rope_base: float = 10000.0
  qk_norm: bool = False
  n_kv_heads: int = 0  # 0 means "same as n_heads", i.e. plain multi-head attention
  
  
  @nn.compact
  def __call__(self, x, train: bool = False):
    seq_len, embed_dim = x.shape
    
    head_dim = embed_dim // self.n_heads
    n_kv_heads = self.n_kv_heads or self.n_heads
    if self.n_heads % n_kv_heads != 0:
      raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

    if n_kv_heads == self.n_heads:
      # Plain MHA keeps the single fused projection, so checkpoints trained before GQA existed
      # still load under the same parameter names.
      qkv = nn.Dense(3 * embed_dim, use_bias=self.use_bias, name='qkv_proj')(x)
      q, k, v = jnp.split(qkv, 3, axis=-1)
    else:
      # GQA: k/v are narrower than q, so they need their own projections.
      q = nn.Dense(self.n_heads * head_dim, use_bias=self.use_bias, name='q_proj')(x)
      k = nn.Dense(n_kv_heads * head_dim, use_bias=self.use_bias, name='k_proj')(x)
      v = nn.Dense(n_kv_heads * head_dim, use_bias=self.use_bias, name='v_proj')(x)
    
    
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
    k = k.reshape(seq_len, n_kv_heads, head_dim)
    v = v.reshape(seq_len, n_kv_heads, head_dim)

    if self.qk_norm:
      # Qwen3-style QK-norm: an RMSNorm over the head dimension applied to q and k before RoPE.
      # The scale is shared across heads (one vector of head_dim), and it comes before the
      # rotation so the rotation stays a pure rotation of unit-scale vectors.
      q = nn.RMSNorm(name='q_norm')(q)
      k = nn.RMSNorm(name='k_norm')(k)

    if self.pos_embed == 'rope':
      q = apply_rope(q, self.rope_base)
      k = apply_rope(k, self.rope_base)

    # Each key/value head is shared by n_heads // n_kv_heads consecutive query heads. Normalizing
    # and rotating before the expansion is equivalent and cheaper, since the copies are identical.
    if n_kv_heads != self.n_heads:
      k = jnp.repeat(k, self.n_heads // n_kv_heads, axis=1)
      v = jnp.repeat(v, self.n_heads // n_kv_heads, axis=1)

    attention = jnp.einsum('ijk,ljk->ilj', q, k)
    attention = attention / jnp.sqrt(head_dim)

    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    attention = jnp.where(mask[..., None] < 0.5, -jnp.inf, attention)
    attention = jax.nn.softmax(attention, axis=1)
    # (query_pos, key_pos, head) weights, opt-in via mutable=['intermediates'] on apply();
    # a no-op otherwise, so it doesn't affect training.
    self.sow('intermediates', 'attn_weights', attention)

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
  pos_embed: str = 'learned'
  rope_base: float = 10000.0
  qk_norm: bool = False
  n_kv_heads: int = 0

  @nn.compact
  def __call__(self, x, train: bool = False):

    seq_len = x.shape[0]

    # Held as a module rather than called inline so the same table can be reused as the output
    # projection below (tied embeddings, as in the 777-parameter reference).
    tok_embed = nn.Embed(self.vocab_size, self.d_model, name='tok_embed')
    x = tok_embed(x)

    if self.pos_embed == 'learned':
      # Fixed-size table sliced to seq_len, so params stay valid across calls with different lengths
      # (needed for autoregressive generation, which calls the model on growing prefixes).
      pos_embed = nn.Embed(self.max_seq_len, self.d_model, name='pos_embed')(jnp.arange(seq_len, dtype=jnp.int32))
      x = x + pos_embed
    elif self.pos_embed != 'rope':
      # 'rope' adds nothing here; it rotates q/k inside every attention block instead.
      raise ValueError(f"Invalid pos_embed: {self.pos_embed}")


    for _ in range(self.n_layers):
      residual = x
      x = Normalization(self.normalization)(x, train=train)
      x = Attention(n_heads=self.n_heads, d_model=self.d_model, use_bias=self.use_bias,
                    pos_embed=self.pos_embed, rope_base=self.rope_base,
                    qk_norm=self.qk_norm, n_kv_heads=self.n_kv_heads)(x, train=train)
      x = x + residual
      
      residual = x
      x = Normalization(self.normalization)(x, train=train)
      x = MLP(hidden_dims=self.hidden_dims, out_dim=self.d_model, activation=self.activation, use_bias=self.use_bias)(x, train=train)
      x = x + residual
      
    x = Normalization(self.normalization)(x, train=train)
    # Tied output projection: logits = x @ tok_embed.embedding.T, with no separate decoder matrix
    # and no output bias. Saves vocab_size * d_model params and is what the reference does.
    x = tok_embed.attend(x)

    return x


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


def parse_num_length_schedule(schedule_args, default_length1: int, default_length2: int, default_steps: int):
  """Parses ["N1:N2:ITERS", ...] into [(n1, n2, iters), ...]. "N:ITERS" is shorthand for
  "N:N:ITERS" (same max size for both operands). Any remaining steps (default_steps minus the
  sum of the given ITERS) are appended as a final (default_length1, default_length2, remaining)
  entry, so the schedule doesn't need to spell out the final stage explicitly."""

  schedule = []
  for item in schedule_args:
    parts = item.split(':')
    if len(parts) == 2:
      size_str, iters_str = parts
      n1 = n2 = int(size_str)
    elif len(parts) == 3:
      n1_str, n2_str, iters_str = parts
      n1, n2 = int(n1_str), int(n2_str)
    else:
      raise ValueError(f"Invalid --num-length-schedule entry {item!r}, expected N:ITERS or N1:N2:ITERS")
    schedule.append((n1, n2, int(iters_str)))

  remaining_steps = default_steps - sum(iters for _, _, iters in schedule)
  if remaining_steps > 0:
    schedule.append((default_length1, default_length2, remaining_steps))

  return schedule


def parse_args():
  parser = argparse.ArgumentParser(description="Train a tiny transformer on integer arithmetic.")
  parser.add_argument('--task', type=str, default='add', choices=sorted(TASKS),
                       help="Arithmetic task to train on. The task owns its symbols, so the vocabulary "
                            "size follows from it (10 digits, op, =, <EOS> -> 13) and is not configurable.")
  parser.add_argument('--sequence-length', type=int, nargs='+', default=[10],
                       help="Rendered digit width of the operands: one value sizes both operands, "
                            "two values (N1 N2) size the first and second operand independently, "
                            "e.g. --sequence-length 5 1 for a single-digit second operand.")
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--d-model', type=int, default=7)
  parser.add_argument('--n-heads', type=int, default=1)
  parser.add_argument('--n-layers', type=int, default=1)
  parser.add_argument('--hidden-dims', type=int, nargs='+', default=[8])
  parser.add_argument('--use-bias', action='store_true', default=False)
  parser.add_argument('--activation', type=str, default='silu', choices=['gelu', 'relu', 'swish', 'silu', 'mish', 'tanh', 'sigmoid', 'none'])
  parser.add_argument('--normalization', type=str, default='rms', choices=['layer', 'rms', 'none'])
  parser.add_argument('--pos-embed', type=str, default='learned', choices=['learned', 'rope'],
                       help="Positional embedding. 'learned' adds a trainable per-position vector to the "
                            "token embeddings; 'rope' adds no parameters and instead rotates q/k channel "
                            "pairs by a position-dependent angle inside every attention block, so attention "
                            "logits depend on relative distance. With an odd head dimension "
                            "(d-model // n-heads) the leftover channel is left unrotated.")
  parser.add_argument('--rope-base', type=float, default=10000.0,
                       help="Base of the RoPE frequency geometric series. Only used with --pos-embed rope.")
  parser.add_argument('--n-kv-heads', type=int, default=0,
                       help="Number of key/value heads for grouped-query attention. 0 (the default) means "
                            "one k/v head per query head, i.e. plain multi-head attention. Must divide "
                            "--n-heads; each k/v head is then shared by n-heads // n-kv-heads query heads.")
  parser.add_argument('--qk-norm', action='store_true', default=False,
                       help="Apply a Qwen3-style RMSNorm over the head dimension to q and k before RoPE. "
                            "Adds 2 * (d-model // n-heads) parameters per layer and keeps attention logits "
                            "at a stable scale.")
  parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'muon'],
                       help="Optimizer to use. 'muon' orthogonalizes updates for 2D params (Newton-schulz) "
                            "and falls back to AdamW for the rest (embeddings, norms, biases).")
  parser.add_argument('--num-steps', type=int, default=27000,
                       help="Total training steps. The default matches the 777-parameter reference's "
                            "2000 + 5000 + 20000 curriculum.")
  parser.add_argument('--learning-rate', type=float, default=2e-2)
  parser.add_argument('--num-length-schedule', type=str, nargs='+', default=["3:2000", "6:5000", "9:20000"],
                       help="Curriculum, as a list of N1:N2:ITERS triples (or N:ITERS as shorthand for "
                            "N:N:ITERS), e.g. --num-length-schedule 3:2000 6:1:5000. For ITERS steps, each "
                            "example independently draws its first operand's size uniformly from 1..N1 and "
                            "its second operand's size from 1..N2 (both zero-padded to --sequence-length), "
                            "so every phase mixes sizes rather than pinning the whole phase to one. Any steps "
                            "left over after the schedule (--num-steps minus the sum of ITERS) draw from "
                            "1..--sequence-length for both operands. The default reproduces the 777-parameter "
                            "reference's phases: (1,3) x 2000, (1,6) x 5000, (1,10) x the rest.")
  parser.add_argument('--lr-schedule', type=str, default='cosine', choices=['cosine', 'constant'],
                       help="Learning rate schedule. 'cosine' decays --learning-rate to --lr-min "
                            "over the whole run (after any warmup).")
  parser.add_argument('--warmup-steps', type=int, default=1000,
                       help="Linear warmup from 0 to --learning-rate over this many steps, before the cosine decay.")
  parser.add_argument('--lr-min', type=float, default=2e-3,
                       help="Absolute learning rate the cosine decays to at the final step.")
  parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                       help="Clip gradients to this global norm before the optimizer update. Set to 0 to disable.")
  parser.add_argument('--num-samples', type=int, default=1000)
  parser.add_argument('--checkpoint-dir', type=str, default="data/transformer_777", help="Folder to store checkpoints in. If not set, no checkpoints are saved.")
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
  if len(args.sequence_length) == 1:
    sequence_length1 = sequence_length2 = args.sequence_length[0]
  elif len(args.sequence_length) == 2:
    sequence_length1, sequence_length2 = args.sequence_length
  else:
    raise ValueError(
      f"--sequence-length takes 1 or 2 values, got {len(args.sequence_length)}: {args.sequence_length}")
  batch_size = args.batch_size
  seed = args.seed
  d_model = args.d_model
  n_heads = args.n_heads
  n_layers = args.n_layers
  hidden_dims = tuple(args.hidden_dims)
  use_bias = args.use_bias
  activation = args.activation
  normalization = args.normalization
  pos_embed = args.pos_embed
  rope_base = args.rope_base
  qk_norm = args.qk_norm
  n_kv_heads = args.n_kv_heads
  num_steps = args.num_steps
  learning_rate = args.learning_rate
  lr_schedule_name = args.lr_schedule
  warmup_steps = args.warmup_steps
  lr_min = args.lr_min
  grad_clip_norm = args.grad_clip_norm
  optimizer_name = args.optimizer
  num_length_schedule = parse_num_length_schedule(
    args.num_length_schedule, sequence_length1, sequence_length2, num_steps)
  # The curriculum is the authority on how many steps actually run (an explicit schedule may
  # overshoot --num-steps), so the LR decays over that horizon, not over --num-steps.
  total_steps = sum(iters for _, _, iters in num_length_schedule)
  num_samples = args.num_samples
  checkpoint_dir = args.checkpoint_dir
  checkpoint_every = args.checkpoint_every
  steps_per_chunk = max(1, args.steps_per_chunk)

  hparams = {
    'task': task.name,
    # Derived from the task, recorded so a checkpoint stays self-describing.
    'vocab_size': vocab_size,
    'sequence_length': [sequence_length1, sequence_length2],
    'batch_size': batch_size,
    'seed': seed,
    'd_model': d_model,
    'n_heads': n_heads,
    'n_layers': n_layers,
    'hidden_dims': hidden_dims,
    'use_bias': use_bias,
    'activation': activation,
    'normalization': normalization,
    'pos_embed': pos_embed,
    'rope_base': rope_base,
    'qk_norm': qk_norm,
    'n_kv_heads': n_kv_heads,
    'learning_rate': learning_rate,
    'lr_schedule': lr_schedule_name,
    'warmup_steps': warmup_steps,
    'lr_min': lr_min,
    'grad_clip_norm': grad_clip_norm,
    'optimizer': optimizer_name,
    'total_steps': total_steps,
    'max_seq_len': task.episode_length(sequence_length1, sequence_length2),
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
    max_seq_len=task.episode_length(sequence_length1, sequence_length2),
    pos_embed=pos_embed,
    rope_base=rope_base,
    qk_norm=qk_norm,
    n_kv_heads=n_kv_heads,
  )

  rng_key = jax.random.key(seed)
  rng_key, example_key = jax.random.split(rng_key)
  example_sequence = generate_episode(task, sequence_length1, sequence_length2, 1, example_key)[0]
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

  if optimizer_name == 'muon':
    optimizer = optax.contrib.muon(learning_rate=lr_schedule, weight_decay=1e-2)
  else:
    optimizer = optax.adamw(learning_rate=lr_schedule,  weight_decay=1e-2)
  if grad_clip_norm > 0:
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)
  state = optimizer.init(params)
  
  # Static: the answer span always begins at the same offset, so the loss is a compile-time
  # slice rather than a runtime mask.
  answer_start = task.answer_start_index(sequence_length1, sequence_length2)

  def train_step(opt, rng_key, num_length1, num_length2):
    """One optimizer step. Written as a lax.scan body (carry -> (carry, y)) so a whole chunk
    of steps can be fused into a single dispatch by train_chunk."""

    batch = generate_episode(
      task, sequence_length1, sequence_length2, batch_size, rng_key, num_length1, num_length2)
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
    grad_norm = optax.global_norm(grads)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return (params, state), (loss, grad_norm)

  @partial(jax.jit, static_argnames=('num_length1', 'num_length2', 'n_steps'), donate_argnums=(0,))
  def train_chunk(opt, rng_key, num_length1, num_length2, n_steps):
    """Runs n_steps training steps inside one jitted lax.scan. At this model size the per-step
    kernels are tiny, so fusing steps keeps the accelerator from being dispatch-bound; donating
    `opt` lets XLA update the params/optimizer buffers in place."""

    step_keys = jax.random.split(rng_key, n_steps)
    opt, (losses, grad_norms) = jax.lax.scan(
      lambda o, k: train_step(o, k, num_length1, num_length2), opt, step_keys)
    return opt, losses.mean(), grad_norms.mean()

  @jax.jit
  def validate(params, rng_key):
    prompts, target_digits, (a, b) = generate_validation_prompts(
      task, sequence_length1, sequence_length2, num_samples, rng_key)

    n_output_digits = task.answer_length(sequence_length1, sequence_length2)
    generated_digits = generate_tokens(transformer, params, prompts, n_output_digits)
    # The model was trained to emit the answer reversed (LSB-first); flip back to compare digit-for-digit.
    predicted_digits = jnp.flip(generated_digits, axis=-1)

    correct = jnp.all(predicted_digits == target_digits, axis=-1)

    return {
      'accuracy': jnp.mean(correct),
      'correct': correct,
      'n1': digits_to_int(a),
      'n2': digits_to_int(b),
      'target_answer': digits_to_int(target_digits),
      'predicted_answer': digits_to_int(predicted_digits),
    }

  opt = (params, state)

  i = 0
  last_checkpoint_step = 0
  for num_length1, num_length2, iters in num_length_schedule:
    print(f"Training on num_length1={num_length1}, num_length2={num_length2} for {iters} steps", flush=True)
    steps_done = 0
    while steps_done < iters:
      # A stage whose length is not a multiple of steps_per_chunk ends with one shorter chunk,
      # which costs one extra compilation for that stage.
      n_steps = min(steps_per_chunk, iters - steps_done)
      rng_key, train_key = jax.random.split(rng_key)
      opt, loss, grad_norm = train_chunk(opt, train_key, num_length1, num_length2, n_steps)
      steps_done += n_steps
      i += n_steps

      rng_key, val_key = jax.random.split(rng_key)
      results = validate(opt[0], val_key)
      print(f"Step {i}, Loss (mean over last {n_steps} steps): {loss}, Grad norm (mean over last {n_steps} steps, pre-clip): {grad_norm}, LR: {float(lr_schedule(i - 1)):.3e}")
      print(f"Validation accuracy: {float(results['accuracy']) * 100:.2f}%", flush=True)

      if checkpoint_dir is not None and i - last_checkpoint_step >= checkpoint_every:
        save_checkpoint(checkpoint_dir, i, opt[0], opt[1], hparams)
        last_checkpoint_step = i

  if checkpoint_dir is not None:
    save_checkpoint(checkpoint_dir, i, opt[0], opt[1], hparams)


if __name__ == "__main__":
  main()