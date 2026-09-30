import argparse
import os
import pickle
from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Sequence

import optax

from task import (
  TASKS, answer_token_mask, digits_to_int, format_answer_tokens, generate_episode, generate_prompts,
  generate_validation_prompts,
)


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
  causal: bool = True
  
  
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

    if self.causal:
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
  recurrent_steps: int = 1
  vocab_size: int = 12
  hidden_dims: Sequence[int] = (32, 32)
  activation: str = 'gelu'
  normalization: str = 'none'
  max_seq_len: int = 512
  pos_embed: str = 'learned'
  rope_base: float = 10000.0
  qk_norm: bool = False
  n_kv_heads: int = 0
  causal: bool = True
  diffusion_steps: int = 0

  @nn.compact
  def __call__(self, x, train: bool = False, timestep=None):

    seq_len = x.shape[0]

    # Held as a module rather than called inline so the same table can be reused as the output
    # projection below (tied embeddings, as in the 777-parameter reference).
    tok_embed = nn.Embed(self.vocab_size, self.d_model, name='tok_embed')
    x = tok_embed(x)
    if self.diffusion_steps:
      if timestep is None:
        raise ValueError('A diffusion denoiser requires a timestep')
      x = x + nn.Embed(self.diffusion_steps + 1, self.d_model, name='time_embed')(timestep)

    if self.pos_embed == 'learned':
      # Fixed-size table sliced to seq_len, so params stay valid across calls with different lengths
      # (needed for autoregressive generation, which calls the model on growing prefixes).
      pos_embed = nn.Embed(self.max_seq_len, self.d_model, name='pos_embed')(jnp.arange(seq_len, dtype=jnp.int32))
      x = x + pos_embed
    elif self.pos_embed != 'rope':
      # 'rope' adds nothing here; it rotates q/k inside every attention block instead.
      raise ValueError(f"Invalid pos_embed: {self.pos_embed}")


    if self.recurrent_steps < 1:
      raise ValueError(f"recurrent_steps must be at least 1, got {self.recurrent_steps}")

    for _ in range(self.n_layers):
      # Construct each block's modules once, then call those same module instances repeatedly.
      # Linen consequently reuses their parameters while the activation x is refined on every
      # recurrent step. With recurrent_steps=1 this is the original, ordinary Transformer block.
      attn_norm = Normalization(self.normalization)
      attention = Attention(
        n_heads=self.n_heads, d_model=self.d_model, use_bias=self.use_bias,
        pos_embed=self.pos_embed, rope_base=self.rope_base,
        qk_norm=self.qk_norm, n_kv_heads=self.n_kv_heads, causal=self.causal)
      mlp_norm = Normalization(self.normalization)
      mlp = MLP(
        hidden_dims=self.hidden_dims, out_dim=self.d_model, activation=self.activation,
        use_bias=self.use_bias)

      for _ in range(self.recurrent_steps):
        residual = x
        x = attn_norm(x, train=train)
        x = attention(x, train=train)
        x = x + residual

        residual = x
        x = mlp_norm(x, train=train)
        x = mlp(x, train=train)
        x = x + residual
      
    x = Normalization(self.normalization)(x, train=train)
    # Tied output projection: logits = x @ tok_embed.embedding.T, with no separate decoder matrix
    # and no output bias. Saves vocab_size * d_model params and is what the reference does.
    x = tok_embed.attend(x)

    return x


class MemoryTransformer(nn.Module):
  """Causal Transformer pass with explicit, per-position recurrent memory.

  A call processes the sequence once and returns (logits, new_memory). Calling it again with
  the same tokens and new_memory refines the state; intermediate token predictions are ignored.
  Memory is aligned with token positions, so causal attention keeps every position independent
  of later tokens even when the same sequence is processed repeatedly.
  """

  n_heads: int = 8
  d_model: int = 512
  use_bias: bool = True
  n_layers: int = 6
  recurrent_steps: int = 1
  memory_passes: int = 2
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
  def __call__(self, tokens, memory, train: bool = False):
    if self.recurrent_steps < 1:
      raise ValueError(f"recurrent_steps must be at least 1, got {self.recurrent_steps}")
    if self.memory_passes < 1:
      raise ValueError(f"memory_passes must be at least 1, got {self.memory_passes}")

    seq_len = tokens.shape[0]
    tok_embed = nn.Embed(self.vocab_size, self.d_model, name='tok_embed')
    x = tok_embed(tokens)
    if memory.shape != x.shape:
      raise ValueError(f"memory must have shape {x.shape}, got {memory.shape}")

    if self.pos_embed == 'learned':
      positions = jnp.arange(seq_len, dtype=jnp.int32)
      x = x + nn.Embed(self.max_seq_len, self.d_model, name='pos_embed')(positions)
    elif self.pos_embed != 'rope':
      raise ValueError(f"Invalid pos_embed: {self.pos_embed}")

    # Reintroduce the input on every pass, while the recurrent state carries the computation.
    x = x + memory
    for _ in range(self.n_layers):
      attn_norm = Normalization(self.normalization)
      attention = Attention(
        n_heads=self.n_heads, d_model=self.d_model, use_bias=self.use_bias,
        pos_embed=self.pos_embed, rope_base=self.rope_base,
        qk_norm=self.qk_norm, n_kv_heads=self.n_kv_heads)
      mlp_norm = Normalization(self.normalization)
      mlp = MLP(
        hidden_dims=self.hidden_dims, out_dim=self.d_model, activation=self.activation,
        use_bias=self.use_bias)

      for _ in range(self.recurrent_steps):
        residual = x
        x = attn_norm(x, train=train)
        x = attention(x, train=train) + residual

        residual = x
        x = mlp_norm(x, train=train)
        x = mlp(x, train=train) + residual

    new_memory = x
    logits = Normalization(self.normalization)(new_memory, train=train)
    return tok_embed.attend(logits), new_memory


def build_transformer(hparams):
  """Builds a model from checkpoint hyperparameters (including older checkpoints)."""
  model_type = hparams.get('model_type', 'transformer')
  if model_type not in ('transformer', 'memory', 'd3pm'):
    raise ValueError(f"Invalid model_type: {model_type}")
  model_class = MemoryTransformer if model_type == 'memory' else Transformer
  kwargs = dict(
    n_heads=hparams['n_heads'], d_model=hparams['d_model'], use_bias=hparams['use_bias'],
    n_layers=hparams['n_layers'], recurrent_steps=hparams.get('recurrent_steps', 1),
    vocab_size=hparams['vocab_size'], hidden_dims=tuple(hparams['hidden_dims']),
    activation=hparams['activation'], normalization=hparams['normalization'],
    max_seq_len=hparams['max_seq_len'], pos_embed=hparams.get('pos_embed', 'learned'),
    rope_base=hparams.get('rope_base', 10000.0), qk_norm=hparams.get('qk_norm', False),
    n_kv_heads=hparams.get('n_kv_heads', 0))
  if model_type == 'memory':
    kwargs['memory_passes'] = hparams.get('memory_passes', 2)
  elif model_type == 'd3pm':
    kwargs.update(causal=False, diffusion_steps=hparams['diffusion_steps'])
  return model_class(**kwargs)


def model_logits(transformer, params, sequence: jnp.ndarray, train: bool = False):
  """Starts from zero memory and returns only the final pass's token logits.

  Gradients pass through every memory update during training. Generation starts from the same
  zero state for each growing prefix, so each prediction follows the training computation.
  """
  if getattr(transformer, 'diffusion_steps', 0):
    raise ValueError('Use d3pm.denoise_logits/generate_answers for diffusion models')
  if isinstance(transformer, MemoryTransformer):
    memory = jnp.zeros((sequence.shape[0], transformer.d_model), dtype=jnp.float32)
    def update_memory(_, state):
      _, new_memory = transformer.apply(params, sequence, state, train=train)
      return new_memory
    memory = jax.lax.fori_loop(0, transformer.memory_passes - 1, update_memory, memory)
    logits, _ = transformer.apply(params, sequence, memory, train=train)
    return logits
  return transformer.apply(params, sequence, train=train)


def generate_tokens(transformer, params, prompts: jnp.ndarray, n_tokens: int):
  """Greedily decodes n_tokens continuations for each (unbatched-model) row in prompts."""

  apply_single = lambda seq: model_logits(transformer, params, seq, train=False)

  sequences = prompts
  for _ in range(n_tokens):
    logits = jax.vmap(apply_single)(sequences)
    next_tokens = jnp.argmax(logits[:, -1, :], axis=-1)
    sequences = jnp.concatenate((sequences, next_tokens[:, None]), axis=-1)

  return sequences[:, prompts.shape[1]:]


def generate_thoughts(transformer, params, prompts, task, n_thought_tokens, rng_key=None):
  """Append <THINK> and sample a fixed number of digit thoughts.

  The padded suffix keeps shapes static inside scan. Causal attention prevents those padding
  tokens from affecting the logits used to sample each thought. Passing no key uses greedy
  decoding, useful for reproducible validation.
  """
  batch_size, prompt_length = prompts.shape
  think = jnp.full((batch_size, 1), task.think_token, dtype=prompts.dtype)
  padding = jnp.zeros((batch_size, n_thought_tokens), dtype=prompts.dtype)
  sequence = jnp.concatenate((prompts, think, padding), axis=-1)
  if rng_key is None:
    rng_key = jax.random.key(0)
    sample = False
  else:
    sample = True

  def step(carry, offset):
    sequence, key = carry
    logits = jax.vmap(lambda row: model_logits(transformer, params, row))(sequence)
    digit_logits = logits[:, prompt_length + offset, :10]
    key, draw_key = jax.random.split(key)
    token = jax.random.categorical(draw_key, digit_logits, axis=-1) if sample else jnp.argmax(digit_logits, axis=-1)
    sequence = sequence.at[:, prompt_length + offset + 1].set(token)
    return (sequence, key), None

  (sequence, _), _ = jax.lax.scan(step, (sequence, rng_key), jnp.arange(n_thought_tokens))
  return sequence


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
  parser.add_argument('--model-type', choices=['transformer', 'memory'], default='transformer',
                       help="'memory' repeatedly processes the same token sequence with an explicit "
                            "per-position state; 'transformer' is the original model.")
  parser.add_argument('--memory-passes', type=int, default=2,
                       help="Number of passes over each sequence for --model-type memory. Only the "
                            "last pass predicts tokens; gradients flow through every pass.")
  parser.add_argument('--task', type=str, default='add', choices=sorted(TASKS),
                       help="Arithmetic task to train on. The task owns its symbols, so the vocabulary "
                            "has 13 base tokens, or 15 with CoT enabled.")
  parser.add_argument('--sequence-length', type=int, nargs='+', default=[10],
                       help="Rendered digit width of the operands: one value sizes both operands, "
                            "two values (N1 N2) size the first and second operand independently, "
                            "e.g. --sequence-length 5 1 for a single-digit second operand.")
  parser.add_argument('--reverse-operands', action='store_true', default=False,
                       help="Render each operand LSB-first so its digit order agrees with the LSB-first answer. "
                            "The default keeps operands MSB-first.")
  parser.add_argument('--early-eos', action='store_true', default=False,
                       help="Emit <EOS> immediately after the most significant nonzero answer digit instead "
                            "of training on leading zero answer padding. Disabled by default.")
  parser.add_argument('--cot-steps', type=int, default=0,
                      help="Number of sampled digit thought tokens before <ANSWER>. Zero disables CoT.")
  parser.add_argument('--cot-samples', type=int, default=2,
                      help="Independent thought samples per problem. CoT requires at least two to form "
                           "a per-problem policy-gradient baseline; effective model batch size is "
                           "--batch-size times --cot-samples.")
  parser.add_argument('--cot-pg-weight', type=float, default=0.1,
                      help="Weight of the thought-token policy-gradient loss relative to answer cross-entropy.")
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--d-model', type=int, default=7)
  parser.add_argument('--n-heads', type=int, default=1)
  parser.add_argument('--n-layers', type=int, default=1)
  parser.add_argument('--recurrent-steps', type=int, default=1,
                       help="Number of times to apply each Transformer block with shared weights. "
                            "1 (the default) is an ordinary non-recurrent block; values above 1 "
                            "add iterative computation without adding block parameters.")
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
  parser.add_argument('--weight-decay', type=float, default=1e-2,
                       help="Weight decay coefficient passed to the optimizer (AdamW or Muon).")
  parser.add_argument('--num-samples', type=int, default=1000)
  parser.add_argument('--validation-seed', type=int, default=None,
                       help='Use a fixed validation set from this seed at every checkpoint.')
  parser.add_argument('--checkpoint-dir', type=str, default="data/transformer_777", help="Folder to store checkpoints in. If not set, no checkpoints are saved.")
  parser.add_argument('--checkpoint-every', type=int, default=1000, help="Save a checkpoint every N steps.")
  parser.add_argument('--steps-per-chunk', type=int, default=250,
                       help="Number of training steps fused into a single jitted lax.scan call. Larger values "
                            "amortize per-step dispatch/kernel-launch overhead (which dominates at this model "
                            "size on GPU) but coarsen the logging/validation granularity to this many steps.")
  return parser.parse_args() 

def make_train_step(transformer, task, optimizer, sequence_length1, sequence_length2,
                    batch_size, reverse_operands=False, early_eos=False, cot_steps=0,
                    cot_samples=2, cot_pg_weight=0.1):
  """Build the optimizer step shared by the trainer and profiling tools."""
  # The answer offset is static, making loss slices compile-time constants.
  answer_start = task.answer_start_index(sequence_length1, sequence_length2)
  make_prompts = jax.named_call(generate_prompts, name="data_generation")
  make_episode = jax.named_call(generate_episode, name="data_generation")
  sample_thoughts = jax.named_call(generate_thoughts, name="thought_generation")

  def train_step(opt, rng_key, num_length1, num_length2):
    """One optimizer step. Written as a lax.scan body (carry -> (carry, y)) so a whole chunk
    of steps can be fused into a single dispatch by train_chunk."""

    params, state = opt
    if cot_steps:
      data_key, thought_key = jax.random.split(rng_key)
      prompts, _, answer_digits = make_prompts(
        task, sequence_length1, sequence_length2, batch_size, data_key,
        num_length1, num_length2, reverse_operands=reverse_operands)
      answer_tokens = format_answer_tokens(task, answer_digits, early_eos)
      prompts = jnp.repeat(prompts, cot_samples, axis=0)
      answer_tokens = jnp.repeat(answer_tokens, cot_samples, axis=0)
      thought_prefix = sample_thoughts(
        transformer, params, prompts, task, cot_steps, thought_key)
      answer_marker = jnp.full((batch_size * cot_samples, 1), task.answer_token)
      batch = jnp.concatenate((thought_prefix, answer_marker, answer_tokens), axis=-1)
      answer_marker_index = answer_start + 1 + cot_steps + 1
      thought_start = answer_start + 2

      def loss_fn(params, sequences):
        def loss_fn_single(sequence):
          logits = model_logits(transformer, params, sequence, train=True)
          answer_targets = sequence[answer_marker_index + 1:]
          answer_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[answer_marker_index:-1], answer_targets)
          if early_eos:
            mask = answer_token_mask(answer_targets, task.eos_token)
            answer_loss = (answer_loss * mask).sum() / mask.sum()
          else:
            answer_loss = answer_loss.mean()
          thought_logprobs = jax.nn.log_softmax(
            logits[thought_start - 1:thought_start + cot_steps - 1, :10], axis=-1)
          thought_tokens = sequence[thought_start:thought_start + cot_steps]
          logprob = jnp.take_along_axis(
            thought_logprobs, thought_tokens[:, None], axis=-1).sum()
          return answer_loss, logprob

        answer_losses, thought_logprobs = jax.vmap(loss_fn_single)(sequences)
        grouped_losses = answer_losses.reshape(batch_size, cot_samples)
        other_mean = (grouped_losses.sum(axis=1, keepdims=True) - grouped_losses) / (cot_samples - 1)
        advantage = jax.lax.stop_gradient(other_mean - grouped_losses).reshape(-1)
        policy_loss = -(advantage * thought_logprobs).mean()
        return answer_losses.mean() + cot_pg_weight * policy_loss

    else:
      batch = make_episode(
        task, sequence_length1, sequence_length2, batch_size, rng_key, num_length1, num_length2,
        reverse_operands=reverse_operands, early_eos=early_eos)

      def loss_fn(params, sequences):
        def loss_fn_single(sequence):
          logits = model_logits(transformer, params, sequence, train=True)
          # Score only answer digits and <EOS>, never the random operand targets.
          loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[answer_start:-1], sequence[answer_start + 1:])
          if early_eos:
            mask = answer_token_mask(sequence[answer_start + 1:], task.eos_token)
            return (loss * mask).sum() / mask.sum()
          return loss.mean()
        return jax.vmap(loss_fn_single)(sequences).mean()

    with jax.named_scope("loss_and_gradients"):
      loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    grad_norm = optax.global_norm(grads)
    with jax.named_scope("optimizer_update"):
      updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return (params, state), (loss, grad_norm)

  return jax.named_call(train_step, name="train_step")


def main():
  args = parse_args()
  if args.cot_steps < 0 or (args.cot_steps and args.cot_samples < 2):
    raise ValueError("--cot-steps must be nonnegative, and CoT requires --cot-samples >= 2")
  if args.cot_pg_weight < 0:
    raise ValueError("--cot-pg-weight must be nonnegative")
  if args.memory_passes < 1:
    raise ValueError(f"--memory-passes must be at least 1, got {args.memory_passes}")
  task = TASKS[args.task]()
  vocab_size = task.vocab_size + (2 if args.cot_steps else 0)
  if len(args.sequence_length) == 1:
    sequence_length1 = sequence_length2 = args.sequence_length[0]
  elif len(args.sequence_length) == 2:
    sequence_length1, sequence_length2 = args.sequence_length
  else:
    raise ValueError(
      f"--sequence-length takes 1 or 2 values, got {len(args.sequence_length)}: {args.sequence_length}")
  batch_size = args.batch_size
  reverse_operands = args.reverse_operands
  early_eos = args.early_eos
  cot_steps = args.cot_steps
  cot_samples = args.cot_samples
  cot_pg_weight = args.cot_pg_weight
  seed = args.seed
  d_model = args.d_model
  n_heads = args.n_heads
  n_layers = args.n_layers
  recurrent_steps = args.recurrent_steps
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
  weight_decay = args.weight_decay
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
    'model_type': args.model_type,
    'memory_passes': args.memory_passes,
    'task': task.name,
    # Derived from the task, recorded so a checkpoint stays self-describing.
    'vocab_size': vocab_size,
    'sequence_length': [sequence_length1, sequence_length2],
    'batch_size': batch_size,
    'reverse_operands': reverse_operands,
    'early_eos': early_eos,
    'cot_steps': cot_steps,
    'cot_samples': cot_samples,
    'cot_pg_weight': cot_pg_weight,
    'seed': seed,
    'validation_seed': args.validation_seed,
    'd_model': d_model,
    'n_heads': n_heads,
    'n_layers': n_layers,
    'recurrent_steps': recurrent_steps,
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
    'weight_decay': weight_decay,
    'optimizer': optimizer_name,
    'total_steps': total_steps,
    'max_seq_len': task.episode_length(sequence_length1, sequence_length2) +
                   (cot_steps + 2 if cot_steps else 0),
    'num_length_schedule': num_length_schedule,
    'steps_per_chunk': steps_per_chunk,
  }

  transformer = build_transformer(hparams)

  rng_key = jax.random.key(seed)
  rng_key, example_key = jax.random.split(rng_key)
  example_sequence = generate_episode(
    task, sequence_length1, sequence_length2, 1, example_key,
    reverse_operands=reverse_operands, early_eos=early_eos)[0]
  if cot_steps:
    prompt_end = task.answer_start_index(sequence_length1, sequence_length2) + 1
    example_sequence = jnp.concatenate((
      example_sequence[:prompt_end], jnp.array([task.think_token]),
      jnp.zeros((cot_steps,), dtype=example_sequence.dtype),
      jnp.array([task.answer_token]), example_sequence[prompt_end:]))
  rng_key, init_key = jax.random.split(rng_key)
  if isinstance(transformer, MemoryTransformer):
    example_memory = jnp.zeros((example_sequence.shape[0], d_model), dtype=jnp.float32)
    params = transformer.init(init_key, example_sequence, example_memory, train=False)
  else:
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
    optimizer = optax.contrib.muon(learning_rate=lr_schedule, weight_decay=weight_decay)
  else:
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
  if grad_clip_norm > 0:
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)
  state = optimizer.init(params)
  
  train_step = make_train_step(
    transformer, task, optimizer, sequence_length1, sequence_length2, batch_size,
    reverse_operands, early_eos, cot_steps, cot_samples, cot_pg_weight)

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
    if args.validation_seed is not None:
      rng_key = jax.random.key(args.validation_seed)
    prompts, target_digits, (a, b) = generate_validation_prompts(
      task, sequence_length1, sequence_length2, num_samples, rng_key,
      reverse_operands=reverse_operands)

    if cot_steps:
      thought_prefix = generate_thoughts(transformer, params, prompts, task, cot_steps)
      answer_marker = jnp.full((num_samples, 1), task.answer_token)
      prompts = jnp.concatenate((thought_prefix, answer_marker), axis=-1)

    n_output_digits = task.answer_length(sequence_length1, sequence_length2)
    n_output_tokens = n_output_digits + 1 if early_eos else n_output_digits
    generated_tokens = generate_tokens(transformer, params, prompts, n_output_tokens)
    # The model was trained to emit the answer reversed (LSB-first); flip back to compare digit-for-digit.
    if early_eos:
      target_tokens = format_answer_tokens(task, target_digits, early_eos=True)
      target_mask = answer_token_mask(target_tokens, task.eos_token)
      correct = jnp.all((generated_tokens == target_tokens) | ~target_mask, axis=-1)
      # Keep the existing numeric diagnostic useful. EOS and anything after it decode as zero.
      before_eos = jnp.cumsum(generated_tokens == task.eos_token, axis=-1) == 0
      generated_digits = jnp.where(
        before_eos[:, :n_output_digits] & (generated_tokens[:, :n_output_digits] < 10),
        generated_tokens[:, :n_output_digits], 0)
    else:
      generated_digits = generated_tokens
      correct = jnp.all(jnp.flip(generated_digits, axis=-1) == target_digits, axis=-1)
    predicted_digits = jnp.flip(generated_digits, axis=-1)

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
