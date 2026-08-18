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
  dropout: float = 0.1 
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
    
    attention = nn.Dropout(self.dropout)(attention, deterministic=not train)
    
    y = jnp.einsum('ijk,jkl->ikl', attention, v)
    y = y.reshape(seq_len, embed_dim)
    
    y = nn.Dense(embed_dim, use_bias=self.use_bias, name='out_proj')(y)
    y = nn.Dropout(self.dropout)(y, deterministic=not train)
    
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
  dropout: float = 0.1
  use_bias: bool = True
  
  @nn.compact
  def __call__(self, x, train: bool = False):
    for hidden_dim in self.hidden_dims:
      x = nn.Dense(hidden_dim, use_bias=self.use_bias)(x)
      x = Activation(self.activation)(x, train=train)
      x = Normalization(self.normalization)(x, train=train)
    x = nn.Dense(self.out_dim, use_bias=self.use_bias)(x)
    return x
    
      
class Transformer(nn.Module):
  n_heads: int = 8
  d_model: int = 512
  dropout: float = 0.1
  use_bias: bool = True
  n_layers: int = 6
  vocab_size: int = 12
  hidden_dims: Sequence[int] = (32, 32)
  activation: str = 'gelu'
  normalization: str = 'none'
  dropout: float = 0.1
  use_bias: bool = True
  max_seq_len: int = 512

  @nn.compact
  def __call__(self, x, train: bool = False):

    seq_len = x.shape[0]

    x = nn.Embed(self.vocab_size, self.d_model)(x)

    # Fixed-size table sliced to seq_len, so params stay valid across calls with different lengths
    # (needed for autoregressive generation, which calls the model on growing prefixes).
    pos_embed = nn.Embed(self.max_seq_len, self.d_model)(jnp.arange(seq_len, dtype=jnp.int32))
    
    x = x + pos_embed
    x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
    

    for _ in range(self.n_layers):
      residual = x
      x = Normalization(self.normalization)(x, train=train)
      x = Attention(n_heads=self.n_heads, d_model=self.d_model, dropout=self.dropout, use_bias=self.use_bias)(x, train=train)
      x = x + residual
      
      residual = x
      x = Normalization(self.normalization)(x, train=train)
      x = MLP(hidden_dims=self.hidden_dims, out_dim=self.d_model, activation=self.activation, dropout=self.dropout, use_bias=self.use_bias)(x, train=train)
      x = x + residual
      
    x = Normalization(self.normalization)(x, train=train)
    x = nn.Dense(self.vocab_size, use_bias=self.use_bias, name='out_proj')(x) 
    
    return x


def generate_sum(numbers: jnp.ndarray):
  def sum_numbers(carry, x):
    result = x[0] + x[1] + carry
    carry = result // 10
    result = result % 10
    return carry, result
    
  summed_number =jax.lax.scan(sum_numbers, 0, numbers, reverse=True)
  summed_number = jnp.concatenate((summed_number[0][None], summed_number[1]))
  return summed_number
 
def pad_numbers(numbers: jnp.ndarray, max_length: int):

  return jnp.pad(numbers, (max_length - numbers.shape[0], 0))


def sample_input_numbers(max_length: int, batch_size: int, rng):
  return jax.random.randint(rng, (batch_size, max_length, 2), 0, 10)


def episode_length(max_length: int) -> int:
  """Total token length of a generate_episode sequence: N1, '+', N2, '=', sum digits, <EOS>."""
  return max_length + 1 + max_length + 1 + (max_length + 1) + 1


def generate_episode(max_length: int, batch_size: int, rng, num_length: int = None):
  """Same as before, but numbers are drawn with only `num_length` digits (default: max_length)
  and zero-padded on the most-significant side up to max_length."""

  if num_length is None:
    num_length = max_length

  input_numbers = sample_input_numbers(num_length, batch_size, rng)
  input_numbers = jnp.pad(input_numbers, ((0, 0), (max_length - num_length, 0), (0, 0)))

  output_numbers = jax.vmap(generate_sum)(input_numbers)
  output_numbers = jnp.flip(output_numbers, axis=1)

  plus_signs = jnp.full((batch_size, 1), 10)
  equals_signs = jnp.full((batch_size, 1), 11)
  eos_signs = jnp.full((batch_size, 1), 12)

  sequence = jnp.concatenate((input_numbers[..., 0], plus_signs, input_numbers[..., 1], equals_signs, output_numbers, eos_signs), axis=-1)


  return sequence


def generate_validation_prompts(max_length: int, batch_size: int, rng):
  """Same N1/N2 sampling as generate_episode, but stops after '=' (no target digits)."""

  input_numbers = sample_input_numbers(max_length, batch_size, rng)

  plus_signs = jnp.full((batch_size, 1), 10)
  equals_signs = jnp.full((batch_size, 1), 11)

  prompts = jnp.concatenate((input_numbers[..., 0], plus_signs, input_numbers[..., 1], equals_signs), axis=-1)
  target_digits = jax.vmap(generate_sum)(input_numbers)  # MSB-first, shape (batch, max_length + 1)

  return prompts, target_digits, input_numbers


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
  parser = argparse.ArgumentParser(description="Train a tiny transformer on integer addition.")
  parser.add_argument('--vocab-size', type=int, default=13, help="10 digits, +, =, <EOS>. Do not use <PAD> if not necessary")
  parser.add_argument('--sequence-length', type=int, default=10)
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--seed', type=int, default=2)
  parser.add_argument('--d-model', type=int, default=32)
  parser.add_argument('--n-heads', type=int, default=1)
  parser.add_argument('--n-layers', type=int, default=1)
  parser.add_argument('--hidden-dims', type=int, nargs='+', default=[32])
  parser.add_argument('--dropout', type=float, default=0.1)
  parser.add_argument('--use-bias', action='store_true', default=False)
  parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'relu', 'swish', 'silu', 'mish', 'tanh', 'sigmoid', 'none'])
  parser.add_argument('--normalization', type=str, default='rms', choices=['layer', 'rms', 'none'])
  parser.add_argument('--num-steps', type=int, default=100000)
  parser.add_argument('--learning-rate', type=float, default=2e-2)
  parser.add_argument('--num-length-schedule', type=str, nargs='+', default=['4:2000', '7:5000'],
                       help="List of SIZE:ITERS pairs, e.g. --num-length-schedule 2:1000 5:2000. "
                            "Trains on `num_length=SIZE`-digit numbers (zero-padded to --sequence-length) for ITERS steps each, in order. "
                            "Any steps left over after the schedule (--num-steps minus the sum of ITERS) are trained "
                            "at --sequence-length.")
  parser.add_argument('--num-samples', type=int, default=1000)
  parser.add_argument('--checkpoint-dir', type=str, default=None, help="Folder to store checkpoints in. If not set, no checkpoints are saved.")
  parser.add_argument('--checkpoint-every', type=int, default=1000, help="Save a checkpoint every N steps.")
  return parser.parse_args() 

def main():
  args = parse_args()
  vocab_size = args.vocab_size
  sequence_length = args.sequence_length
  batch_size = args.batch_size
  seed = args.seed
  d_model = args.d_model
  n_heads = args.n_heads
  n_layers = args.n_layers
  hidden_dims = tuple(args.hidden_dims)
  dropout = args.dropout
  use_bias = args.use_bias
  activation = args.activation
  normalization = args.normalization
  num_steps = args.num_steps
  learning_rate = args.learning_rate
  num_length_schedule = parse_num_length_schedule(args.num_length_schedule, sequence_length, num_steps)
  num_samples = args.num_samples
  checkpoint_dir = args.checkpoint_dir
  checkpoint_every = args.checkpoint_every

  hparams = {
    'vocab_size': vocab_size,
    'sequence_length': sequence_length,
    'batch_size': batch_size,
    'seed': seed,
    'd_model': d_model,
    'n_heads': n_heads,
    'n_layers': n_layers,
    'hidden_dims': hidden_dims,
    'dropout': dropout,
    'use_bias': use_bias,
    'activation': activation,
    'normalization': normalization,
    'learning_rate': learning_rate,
    'max_seq_len': episode_length(sequence_length),
    'num_length_schedule': num_length_schedule,
  }

  transformer = Transformer(
    n_heads=n_heads,
    d_model=d_model,
    dropout=dropout,
    use_bias=use_bias,
    n_layers=n_layers,
    vocab_size=vocab_size,
    hidden_dims=hidden_dims,
    activation=activation,
    normalization=normalization,
    max_seq_len=episode_length(sequence_length),
  )
  
  
  example_sequence = generate_episode(sequence_length, 1, jax.random.key(seed))[0]
  rng_key = jax.random.key(seed)
  rng_key, init_key = jax.random.split(rng_key)
  params = transformer.init(init_key, example_sequence, train=False)

  n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
  print(f"Trainable parameters: {n_params:,}")

  optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-2)
  state = optimizer.init(params)
  
  @partial(jax.jit, static_argnames=('num_length',))
  def train_step(opt, rng_key, num_length):

    episode_key, rng_key = jax.random.split(rng_key)
    batch = generate_episode(sequence_length, batch_size, episode_key, num_length)
    params, state = opt
     
    def loss_fn(params, sequences, rng_key):
      
      def loss_fn_single(sequence, rng_key):
        logits = transformer.apply(params, sequence, train=True, rngs={'dropout': rng_key})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits[:-1], sequence[1:])
        return loss.mean()
      rng_keys = jax.random.split(rng_key, sequences.shape[0])
      losses = jax.vmap(loss_fn_single)(sequences, rng_keys)
      return losses.mean()
    
    loss, grads = jax.value_and_grad(loss_fn)(params, batch, rng_key)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return (params, state), loss.mean()

  @jax.jit
  def validate(params, rng_key):
    prompts, target_digits, input_numbers = generate_validation_prompts(sequence_length, num_samples, rng_key)

    n_output_digits = sequence_length + 1
    generated_digits = generate_tokens(transformer, params, prompts, n_output_digits)
    # The model was trained to emit the sum reversed (LSB-first); flip back to compare digit-for-digit.
    predicted_digits = jnp.flip(generated_digits, axis=-1)

    correct = jnp.all(predicted_digits == target_digits, axis=-1)

    return {
      'accuracy': jnp.mean(correct),
      'correct': correct,
      'n1': digits_to_int(input_numbers[..., 0]),
      'n2': digits_to_int(input_numbers[..., 1]),
      'target_sum': digits_to_int(target_digits),
      'predicted_sum': digits_to_int(predicted_digits),
    }

  opt = (params, state)

  i = 0
  for num_length, iters in num_length_schedule:
    print(f"Training on num_length={num_length} for {iters} steps", flush=True)
    for _ in range(iters):
      rng_key, train_key = jax.random.split(rng_key)
      opt, loss = train_step(opt, train_key, num_length)
      if i % 100 == 0:
        rng_key, val_key = jax.random.split(rng_key)
        results = validate(opt[0], val_key)
        print(f"Step {i+1}, Loss: {loss}")
        print(f"Validation accuracy: {float(results['accuracy']) * 100:.2f}%", flush=True)

      if checkpoint_dir is not None and (i + 1) % checkpoint_every == 0:
        save_checkpoint(checkpoint_dir, i + 1, opt[0], opt[1], hparams)

      i += 1

  if checkpoint_dir is not None:
    save_checkpoint(checkpoint_dir, i, opt[0], opt[1], hparams)


if __name__ == "__main__":
  main()