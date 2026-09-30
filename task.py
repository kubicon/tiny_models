import jax
import jax.numpy as jnp


class Task:
  """An arithmetic task: the symbols it uses and how one episode is laid out.

  The base token layout is shared by every task: 0-9 are digits, followed by the task's
  operator, '=' and <EOS>. Optional CoT adds <THINK> and <ANSWER> after these base tokens.
  Subclasses supply the operator's meaning via `compute` and, where the task needs it, a
  different operand distribution via `sample_inputs`.

  The two operands are independent axes throughout: `max_length1`/`max_length2` are the
  rendered (zero-padded) digit widths of the first/second operand, and generation additionally
  accepts per-call `num_length1`/`num_length2` curriculum caps. A task whose second operand
  should always be short (e.g. a single digit) just gets a small `max_length2` at the call
  site -- there's no need for a dedicated subclass."""

  name = 'task'
  op_token = 10
  eq_token = 11
  eos_token = 12
  think_token = 13
  answer_token = 14

  @property
  def vocab_size(self) -> int:
    return self.eos_token + 1

  def answer_length(self, max_length1: int, max_length2: int) -> int:
    """Number of answer digits produced for a `max_length1`-digit and `max_length2`-digit operand."""
    raise NotImplementedError

  def compute(self, a_digits: jnp.ndarray, b_digits: jnp.ndarray) -> jnp.ndarray:
    """MSB-first digits of the two operands -> the answer's digits, MSB-first."""
    raise NotImplementedError

  def place_sums(self, a_digits: jnp.ndarray, b_digits: jnp.ndarray) -> jnp.ndarray:
    """MSB-first operands -> LSB-first per-place sums before carrying (answer_length - 1 places;
    the final answer digit is the last carry)."""
    raise NotImplementedError

  def max_place_sum(self, max_length1: int, max_length2: int) -> int:
    """Upper bound of a place_sums entry, used to put auxiliary regression targets on unit scale."""
    raise NotImplementedError

  def running_sums(self, a_digits: jnp.ndarray, b_digits: jnp.ndarray):
    """LSB-first (running sum, incoming carry) at every answer place.

    running[k] = place_sums[k] + carry[k] and answer digit k is running[k] % 10, so these are the
    intermediate quantities a model has to track to emit digit k without writing them out. They
    serve as auxiliary (non-token) supervision targets."""

    sums = jnp.concatenate((self.place_sums(a_digits, b_digits), jnp.zeros((1,), a_digits.dtype)))

    def step(carry, s):
      running = s + carry
      return running // 10, (running, carry)

    _, (running, carry) = jax.lax.scan(step, jnp.zeros((), a_digits.dtype), sums)
    return running, carry

  def sample_inputs(self, max_length1: int, max_length2: int, batch_size: int, rng):
    rng1, rng2 = jax.random.split(rng)
    a = jax.random.randint(rng1, (batch_size, max_length1), 0, 10)
    b = jax.random.randint(rng2, (batch_size, max_length2), 0, 10)
    return a, b

  def episode_length(self, max_length1: int, max_length2: int) -> int:
    """Total token length of a generate_episode sequence: N1, op, N2, '=', answer digits, <EOS>."""
    return max_length1 + 1 + max_length2 + 1 + self.answer_length(max_length1, max_length2) + 1

  def significance_ids(self, max_length1: int, max_length2: int, reverse_operands: bool = False,
                       *, cot_steps: int = 0, causal: bool = True):
    """Abacus-style id of every position of a generate_episode sequence (a tuple of ints).

    An operand digit gets its significance (0 = units), whatever its token position; the answer
    positions, from '=' to the last answer digit, get the significance of the digit they *predict*
    (answer_length for the one predicting <EOS>); the op and <EOS> tokens share one extra id. So
    a_i, b_i and the position emitting product digit i share an embedding.

    CoT's '=' and <THINK> markers and thoughts use the extra id; causal answer
    significance starts at <ANSWER>.
    Diffusion instead embeds the digit being denoised at each answer position, since it
    predicts that position's clean token rather than the next token, and has no EOS."""
    n_answer = self.answer_length(max_length1, max_length2)
    other = n_answer + 1
    a = list(range(max_length1)) if reverse_operands else list(range(max_length1 - 1, -1, -1))
    b = list(range(max_length2)) if reverse_operands else list(range(max_length2 - 1, -1, -1))
    prefix = a + [other] + b
    if cot_steps:
      prefix += [other] * (cot_steps + 2)  # '=', <THINK>, thoughts
    if causal:
      return tuple(prefix + list(range(n_answer + 1)) + [other])
    return tuple(prefix + [other] + list(range(n_answer)))

  def answer_start_index(self, max_length1: int, max_length2: int) -> int:
    """Index of the '=' token in a generate_episode sequence.

    Layout: N1 [0, L1), op [L1], N2 [L1+1, L1+1+L2), '=' [L1+1+L2], answer digits, <EOS>.
    In a next-token loss, position i predicts token i+1, so slicing logits from this index gives
    exactly the answer targets (the answer digits and <EOS>) and drops the prompt positions,
    whose targets are uniform random digits and carry no learnable signal."""
    return max_length1 + 1 + max_length2


class AdditionTask(Task):
  name = 'add'

  def answer_length(self, max_length1: int, max_length2: int) -> int:
    return max(max_length1, max_length2) + 1

  def compute(self, a_digits: jnp.ndarray, b_digits: jnp.ndarray) -> jnp.ndarray:
    length = max(a_digits.shape[0], b_digits.shape[0])
    a_digits = jnp.pad(a_digits, (length - a_digits.shape[0], 0))
    b_digits = jnp.pad(b_digits, (length - b_digits.shape[0], 0))

    def add_digits(carry, xy):
      x, y = xy
      result = x + y + carry
      carry = result // 10
      result = result % 10
      return carry, result

    carry, digits = jax.lax.scan(add_digits, 0, (a_digits, b_digits), reverse=True)
    return jnp.concatenate((carry[None], digits))

  def place_sums(self, a_digits: jnp.ndarray, b_digits: jnp.ndarray) -> jnp.ndarray:
    length = max(a_digits.shape[0], b_digits.shape[0])
    a_digits = jnp.pad(a_digits, (length - a_digits.shape[0], 0))
    b_digits = jnp.pad(b_digits, (length - b_digits.shape[0], 0))
    return jnp.flip(a_digits + b_digits)

  def max_place_sum(self, max_length1: int, max_length2: int) -> int:
    return 18


class MultiplicationTask(Task):
  name = 'multiply'

  def answer_length(self, max_length1: int, max_length2: int) -> int:
    # An L1-digit and L2-digit operand multiply to at most L1 + L2 digits.
    return max_length1 + max_length2

  def place_sums(self, a_digits: jnp.ndarray, b_digits: jnp.ndarray) -> jnp.ndarray:
    a_lsb = jnp.flip(a_digits)
    b_lsb = jnp.flip(b_digits)
    length1 = a_digits.shape[0]
    length2 = b_digits.shape[0]

    # Schoolbook multiplication: for each output place k, sum every digit pair (i, j) with
    # i + j = k. length1/length2 are static (they're array shapes), so this unrolls into a small,
    # fixed graph of elementwise multiplies/adds -- deliberately avoiding jnp.convolve, which
    # lowers to XLA's conv primitive and pulls in a cuDNN dependency this tiny sum doesn't need.
    # Each place's sum is at most min(length1, length2) * 81, well within range before carry
    # propagation.
    return jnp.stack([
      sum(a_lsb[i] * b_lsb[k - i]
          for i in range(max(0, k - length2 + 1), min(k, length1 - 1) + 1))
      for k in range(length1 + length2 - 1)
    ])

  def max_place_sum(self, max_length1: int, max_length2: int) -> int:
    return 81 * min(max_length1, max_length2)

  def compute(self, a_digits: jnp.ndarray, b_digits: jnp.ndarray) -> jnp.ndarray:
    products = self.place_sums(a_digits, b_digits)

    def carry_digit(carry, x):
      total = x + carry
      carry = total // 10
      digit = total % 10
      return carry, digit

    final_carry, digits_lsb = jax.lax.scan(carry_digit, 0, products)
    # products has length1 + length2 - 1 places; the final carry supplies the top digit,
    # exactly filling answer_length's length1 + length2 digits.
    digits_lsb = jnp.concatenate((digits_lsb, final_carry[None]))
    return jnp.flip(digits_lsb)


TASKS = {task.name: task for task in (AdditionTask, MultiplicationTask)}


def generate_prompts(
    task: Task, max_length1: int, max_length2: int, batch_size: int, rng,
    num_length1: int = None, num_length2: int = None, reverse_operands: bool = False,
    full_width_prob: float = 0.0, max_digit: int = 9, mix_full_size_prob: float = 0.0):
  """Samples an operand pair per row and renders the prompt 'N1 op N2 ='.

  When `num_length1` (resp. `num_length2`) is given, each row independently draws that operand's
  size uniformly from 1..num_length1 (resp. num_length2); otherwise the operand always uses its
  full rendered width. Operands are always rendered zero-padded on the most-significant side to
  their rendered width (`max_length1` for N1, `max_length2` for N2), so the token layout is
  identical whatever size a row drew.

  Mixing sizes within the batch (rather than pinning a whole curriculum stage to one size) is what
  the 777-parameter reference does: its phases are (1, 3), (1, 6), (1, 10), so even the final phase
  keeps short problems in the mix. Training on a single size instead leaves the model with no
  gradient on the digit positions that stage never populates.

  With `full_width_prob` > 0, that fraction of rows instead uses num_length1 and num_length2 digits
  for both operands. Uniform sizes make the hardest (full-size) problems rare -- 1 in 25 rows for
  5x5 -- although they are the ones the full-width validation measures.

  Rows are drawn at the full rendered width and the unused leading digits are then zeroed, which
  is the same distribution as drawing n digits and padding, but keeps the shape static under jit.

  With `mix_full_size_prob` > 0, that fraction of rows ignores the num_length caps and draws its
  sizes from 1..max_length1 / 1..max_length2, i.e. mixes rows of the final size range into an
  earlier curriculum stage.

  With `max_digit` < 9 (a digit-value curriculum), each row instead draws a cap uniformly from
  1..max_digit and both operands' digits uniformly from 0..cap, so place sums and carries are
  small; the default draws digits from 0..9 as the task does.

  Returns the prompts, the (a, b) operand digits and the MSB-first answer digits, so both the
  training and the validation builders share one layout."""

  rng, size_rng1, size_rng2 = jax.random.split(rng, 3)
  a, b = task.sample_inputs(max_length1, max_length2, batch_size, rng)
  if max_digit < 9:
    cap_rng, digit_rng1, digit_rng2 = jax.random.split(jax.random.fold_in(rng, 2), 3)
    cap = jax.random.randint(cap_rng, (batch_size, 1), 1, max_digit + 1)
    a = jax.random.randint(digit_rng1, (batch_size, max_length1), 0, cap + 1)
    b = jax.random.randint(digit_rng2, (batch_size, max_length2), 0, cap + 1)

  if full_width_prob > 0:
    # fold_in, not split: split(rng) would reproduce the keys sample_inputs just used.
    full_width = jax.random.bernoulli(jax.random.fold_in(rng, 1), full_width_prob, (batch_size,))
  if mix_full_size_prob > 0:
    mix_full = jax.random.bernoulli(jax.random.fold_in(rng, 3), mix_full_size_prob, (batch_size,))
  if num_length1 is not None:
    n1 = jax.random.randint(size_rng1, (batch_size,), 1, num_length1 + 1)
    if full_width_prob > 0:
      n1 = jnp.where(full_width, num_length1, n1)
    if mix_full_size_prob > 0:
      n1 = jnp.where(mix_full, jax.random.randint(jax.random.fold_in(size_rng1, 1), (batch_size,), 1, max_length1 + 1), n1)
    used1 = jnp.arange(max_length1)[None, :] >= (max_length1 - n1)[:, None]
    a = jnp.where(used1, a, 0)
  if num_length2 is not None:
    n2 = jax.random.randint(size_rng2, (batch_size,), 1, num_length2 + 1)
    if full_width_prob > 0:
      n2 = jnp.where(full_width, num_length2, n2)
    if mix_full_size_prob > 0:
      n2 = jnp.where(mix_full, jax.random.randint(jax.random.fold_in(size_rng2, 1), (batch_size,), 1, max_length2 + 1), n2)
    used2 = jnp.arange(max_length2)[None, :] >= (max_length2 - n2)[:, None]
    b = jnp.where(used2, b, 0)

  answer_digits = jax.vmap(task.compute)(a, b)

  op_tokens = jnp.full((batch_size, 1), task.op_token)
  eq_tokens = jnp.full((batch_size, 1), task.eq_token)

  # The arithmetic is computed from the conventional MSB-first representation above. Only the
  # rendered prompt is reversed, making its digit order agree with the LSB-first answer order.
  prompt_a = jnp.flip(a, axis=-1) if reverse_operands else a
  prompt_b = jnp.flip(b, axis=-1) if reverse_operands else b
  prompts = jnp.concatenate((prompt_a, op_tokens, prompt_b, eq_tokens), axis=-1)

  return prompts, (a, b), answer_digits


def generate_episode(
    task: Task, max_length1: int, max_length2: int, batch_size: int, rng,
    num_length1: int = None, num_length2: int = None, reverse_operands: bool = False,
    early_eos: bool = False):
  """A full training sequence: the prompt, then the answer emitted LSB-first, then <EOS>."""

  prompts, _, answer_digits = generate_prompts(
    task, max_length1, max_length2, batch_size, rng, num_length1, num_length2,
    reverse_operands=reverse_operands)
  answer_tokens = format_answer_tokens(task, answer_digits, early_eos)

  return jnp.concatenate((prompts, answer_tokens), axis=-1)


def generate_validation_prompts(
    task: Task, max_length1: int, max_length2: int, batch_size: int, rng,
    reverse_operands: bool = False):
  """Same sampling as generate_episode, but stops after '=' (no target digits)."""

  prompts, operands, answer_digits = generate_prompts(
    task, max_length1, max_length2, batch_size, rng, reverse_operands=reverse_operands)

  return prompts, answer_digits, operands


def format_answer_tokens(task: Task, answer_digits: jnp.ndarray, early_eos: bool = False):
  """Formats MSB-first answers as fixed-shape, LSB-first autoregressive targets.

  With early_eos enabled, leading zeroes in the conventional MSB-first representation are
  omitted and EOS follows the last significant digit. The remaining fixed-width batch slots are
  also filled with EOS; callers should mask targets after the first EOS. Zero itself retains one
  output digit, so its target is ``0, EOS`` rather than an empty answer.
  """

  answer_lsb = jnp.flip(answer_digits, axis=-1)
  eos_shape = answer_digits.shape[:-1] + (1,)
  eos_tokens = jnp.full(eos_shape, task.eos_token, dtype=answer_digits.dtype)
  if not early_eos:
    return jnp.concatenate((answer_lsb, eos_tokens), axis=-1)

  answer_length = answer_digits.shape[-1]
  has_nonzero = jnp.any(answer_digits != 0, axis=-1)
  first_nonzero = jnp.argmax(answer_digits != 0, axis=-1)
  significant_length = jnp.where(has_nonzero, answer_length - first_nonzero, 1)

  # Append one safe slot for the EOS position, then replace EOS and all padding positions below.
  answer_and_slot = jnp.concatenate((answer_lsb, jnp.zeros_like(eos_tokens)), axis=-1)
  positions = jnp.arange(answer_length + 1)
  return jnp.where(positions < significant_length[..., None], answer_and_slot, task.eos_token)


def answer_token_mask(targets: jnp.ndarray, eos_token: int):
  """Mask containing all answer digits and the first EOS, but no fixed-shape EOS padding."""

  return jnp.cumsum(targets == eos_token, axis=-1) <= 1


def digits_to_int(digits: jnp.ndarray, msb_first: bool = True):
  n = digits.shape[-1]
  exponents = jnp.arange(n - 1, -1, -1) if msb_first else jnp.arange(n)
  return jnp.sum(digits * (10 ** exponents), axis=-1)
