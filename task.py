import jax
import jax.numpy as jnp


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


class MultiplicationTask(Task):
  name = 'multiply'

  def answer_length(self, max_length: int) -> int:
    # Two max_length-digit operands multiply to at most 2 * max_length digits.
    return 2 * max_length

  def compute(self, numbers: jnp.ndarray) -> jnp.ndarray:
    a_lsb = jnp.flip(numbers[:, 0])
    b_lsb = jnp.flip(numbers[:, 1])
    max_length = numbers.shape[0]

    # Schoolbook multiplication: for each output place k, sum every digit pair (i, j) with
    # i + j = k. max_length is static (it's an array shape), so this unrolls into a small,
    # fixed graph of elementwise multiplies/adds -- deliberately avoiding jnp.convolve, which
    # lowers to XLA's conv primitive and pulls in a cuDNN dependency this tiny sum doesn't need.
    # Each place's sum is at most max_length * 81, well within range before carry propagation.
    products = jnp.stack([
      sum(a_lsb[i] * b_lsb[k - i]
          for i in range(max(0, k - max_length + 1), min(k, max_length - 1) + 1))
      for k in range(2 * max_length - 1)
    ])

    def carry_digit(carry, x):
      total = x + carry
      carry = total // 10
      digit = total % 10
      return carry, digit

    final_carry, digits_lsb = jax.lax.scan(carry_digit, 0, products)
    # products has 2 * max_length - 1 places; the final carry supplies the top digit,
    # exactly filling answer_length's 2 * max_length digits.
    digits_lsb = jnp.concatenate((digits_lsb, final_carry[None]))
    return jnp.flip(digits_lsb)


TASKS = {task.name: task for task in (AdditionTask, MultiplicationTask)}


def pad_numbers(numbers: jnp.ndarray, max_length: int):

  return jnp.pad(numbers, (max_length - numbers.shape[0], 0))


def generate_prompts(task: Task, max_length: int, batch_size: int, rng, num_length: int = None):
  """Samples an operand pair per row and renders the prompt 'N1 op N2 ='.

  When `num_length` is given, each row independently draws its operand size uniformly from
  1..num_length; otherwise every row uses the full `max_length`. Operands are always rendered
  zero-padded on the most-significant side to `max_length`, so the token layout is identical
  whatever size a row drew.

  Mixing sizes within the batch (rather than pinning a whole curriculum stage to one size) is what
  the 777-parameter reference does: its phases are (1, 3), (1, 6), (1, 10), so even the final phase
  keeps short problems in the mix. Training on a single size instead leaves the model with no
  gradient on the digit positions that stage never populates.

  Rows are drawn at max_length and the unused leading digits are then zeroed, which is the same
  distribution as drawing n digits and padding, but keeps the shape static under jit.

  Returns the prompts, the operand digits and the MSB-first answer digits, so both the training
  and the validation builders share one layout."""

  rng, size_rng = jax.random.split(rng)
  input_numbers = task.sample_inputs(max_length, batch_size, rng)

  if num_length is not None:
    n_digits = jax.random.randint(size_rng, (batch_size,), 1, num_length + 1)
    # Keep the n_digits least-significant positions of each row, zero the leading ones.
    used = jnp.arange(max_length)[None, :] >= (max_length - n_digits)[:, None]
    input_numbers = jnp.where(used[..., None], input_numbers, 0)

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
