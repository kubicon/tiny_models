import unittest

import jax
import jax.numpy as jnp

from task import MultiplicationTask, digits_to_int, generate_episode, generate_prompts


def digits(value: int, length: int):
  return [int(d) for d in str(value).zfill(length)]


class FixedInputsTask(MultiplicationTask):
  """Same as MultiplicationTask, but sample_inputs returns a fixed, caller-supplied batch
  instead of drawing from rng, so generate_episode's output is fully deterministic and can
  be checked token-for-token."""

  def __init__(self, a, b):
    self._a = a
    self._b = b

  def sample_inputs(self, max_length1, max_length2, batch_size, rng):
    assert self._a.shape == (batch_size, max_length1)
    assert self._b.shape == (batch_size, max_length2)
    return self._a, self._b


class MultiplicationEpisodeTest(unittest.TestCase):

  # (max_length1, max_length2, n1, n2)
  CASES = [
    (3, 3, 123, 45),
    (2, 2, 0, 0),
    (2, 2, 99, 99),   # largest possible pair for max_length=2: exercises the top-digit carry
    (4, 4, 7, 8),     # small operands: exercises leading zero padding in the product
    (5, 5, 12345, 999),
    (5, 1, 12345, 9),  # asymmetric operand widths, e.g. single-digit second operand
    (3, 5, 45, 12345),
  ]

  def test_generate_episode_matches_expected_tokens(self):
    for max_length1, max_length2, n1, n2 in self.CASES:
      with self.subTest(n1=n1, n2=n2, max_length1=max_length1, max_length2=max_length2):
        a = jnp.array([digits(n1, max_length1)])
        b = jnp.array([digits(n2, max_length2)])

        task = FixedInputsTask(a, b)
        episode = generate_episode(
          task, max_length1, max_length2, batch_size=1, rng=jax.random.key(0))[0]

        product_digits_msb = digits(n1 * n2, max_length1 + max_length2)
        expected = (
          digits(n1, max_length1)
          + [task.op_token]
          + digits(n2, max_length2)
          + [task.eq_token]
          + list(reversed(product_digits_msb))  # answer is emitted LSB-first
          + [task.eos_token]
        )

        self.assertEqual(episode.tolist(), expected)

  def test_running_sums_reproduce_product(self):
    task = MultiplicationTask()
    for max_length1, max_length2, n1, n2 in self.CASES:
      with self.subTest(n1=n1, n2=n2, max_length1=max_length1, max_length2=max_length2):
        a = jnp.array(digits(n1, max_length1))
        b = jnp.array(digits(n2, max_length2))
        running, carry = task.running_sums(a, b)

        # Digit k is running[k] mod 10, and running[k] // 10 is the carry into place k + 1.
        product_lsb = list(reversed(digits(n1 * n2, max_length1 + max_length2)))
        self.assertEqual((running % 10).tolist(), product_lsb)
        self.assertEqual(carry[0], 0)
        self.assertEqual(carry[1:].tolist(), (running[:-1] // 10).tolist())
        # Without the carries they absorbed, the running sums are the place sums of the product.
        self.assertEqual(sum(int(s) * 10 ** k for k, s in enumerate(running - carry)), n1 * n2)

  def test_max_digit_caps_operand_digits(self):
    task = MultiplicationTask()
    _, (a, b), answer = generate_prompts(task, 4, 4, 256, jax.random.key(1), max_digit=2)
    self.assertLessEqual(int(jnp.maximum(a.max(), b.max())), 2)
    self.assertEqual(int(jnp.maximum(a.max(), b.max())), 2)
    self.assertEqual(digits_to_int(answer).tolist(), (digits_to_int(a) * digits_to_int(b)).tolist())

  def test_mix_full_size_prob(self):
    task = MultiplicationTask()
    capped = generate_prompts(task, 4, 4, 512, jax.random.key(2), 1, 1)[1][0]
    mixed = generate_prompts(task, 4, 4, 512, jax.random.key(2), 1, 1, mix_full_size_prob=0.5)[1][0]
    self.assertEqual(int(capped[:, :3].max()), 0)  # 1-digit operands only
    mixed_rows = float(jnp.mean(jnp.any(mixed[:, :3] > 0, axis=-1)))
    self.assertGreater(mixed_rows, 0.25)  # ~0.5 * P(size > 1 and a nonzero high digit)
    self.assertLess(mixed_rows, 0.5)

  def test_significance_ids(self):
    task = MultiplicationTask()
    other = 5  # answer_length(2, 2) + 1
    # a1 a0 * b1 b0 = | '=' and the answer digits predict digits 0..3, the last one <EOS> (4).
    self.assertEqual(task.significance_ids(2, 2), (1, 0, other, 1, 0, 0, 1, 2, 3, 4, other))
    self.assertEqual(task.significance_ids(2, 2, reverse_operands=True),
                     (0, 1, other, 0, 1, 0, 1, 2, 3, 4, other))
    for max_length1, max_length2 in ((3, 3), (5, 1), (3, 5)):
      self.assertEqual(len(task.significance_ids(max_length1, max_length2)),
                       task.episode_length(max_length1, max_length2))


if __name__ == '__main__':
  unittest.main()
