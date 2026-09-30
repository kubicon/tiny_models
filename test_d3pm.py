"""Numerical and integration checks: .venv/bin/python -m unittest test_d3pm."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from d3pm import (
  D3PM, cot_diffusion_loss, diffusion_loss, generate_answers, generate_thoughts,
  log_probs, thought_policy_loss, thought_prefix,
)
from task import TASKS, generate_prompts
from transformer import Attention, Transformer, build_transformer


class D3PMTests(unittest.TestCase):
  def test_transition_products_and_terminal_prior(self):
    for corruption in ('mask', 'uniform'):
      process = D3PM(4, corruption)
      product = np.eye(process.num_states)
      for t in range(1, 5):
        q = np.asarray(process.transition(t - 1, t))
        np.testing.assert_allclose(q.sum(-1), 1, atol=1e-6)
        self.assertTrue((q >= 0).all())
        product = product @ q
        np.testing.assert_allclose(product, process.q_bar(t), atol=1e-6)
      np.testing.assert_allclose(product, process.reset, atol=1e-6)
      np.testing.assert_allclose(process.q_bar(1) @ process.transition(1, 4), product, atol=1e-6)

  def test_true_posterior_matches_bayes_enumeration(self):
    for corruption in ('mask', 'uniform'):
      process = D3PM(4, corruption)
      clean = 7
      for s, t in ((0, 1), (1, 2), (1, 4), (0, 4)):
        prior_s = np.asarray(process.q_bar(s))[clean]
        q_st = np.asarray(process.transition(s, t))
        marginal = np.asarray(process.q_bar(t))[clean]
        for noisy in range(process.num_states):
          if marginal[noisy] == 0:
            continue
          expected = prior_s * q_st[:, noisy] / marginal[noisy]
          actual = process.posterior(jax.nn.one_hot(clean, process.num_states), jnp.array(noisy), s, t)
          np.testing.assert_allclose(actual, expected, atol=1e-6)

  def test_model_posterior_matches_d3pm_x0_parameterization(self):
    for corruption in ('mask', 'uniform'):
      process = D3PM(4, corruption)
      clean_probs = np.arange(1, process.num_states + 1, dtype=np.float32)
      if corruption == 'mask':
        clean_probs[-1] = 0
      clean_probs /= clean_probs.sum()
      noisy = 10 if corruption == 'mask' else 3
      weights = (clean_probs @ np.asarray(process.q_bar(2))) * np.asarray(process.transition(2, 4))[:, noisy]
      expected = weights / weights.sum()
      actual = process.posterior(jnp.array(clean_probs), jnp.array(noisy), 2, 4)
      np.testing.assert_allclose(actual, expected, atol=1e-6)

    process = D3PM(4, 'mask')
    overconfident = process.clean_probs(jnp.array([1000.] + [-1000.] * 9))
    # A revealed digit is preserved even when the model assigns it tiny mass.
    actual = process.posterior(overconfident, jnp.array(7), 0, 1)
    np.testing.assert_array_equal(actual, jax.nn.one_hot(7, 11))

  def test_attention_is_bidirectional_only_when_requested(self):
    tokens = jax.random.normal(jax.random.key(1), (5, 8))
    causal = Attention(n_heads=2, d_model=8)
    bidirectional = Attention(n_heads=2, d_model=8, causal=False)
    params = causal.init(jax.random.key(2), tokens)
    modified = tokens.at[-1].add(jnp.arange(8) * 3)
    np.testing.assert_allclose(causal.apply(params, tokens)[0], causal.apply(params, modified)[0])
    self.assertFalse(np.allclose(bidirectional.apply(params, tokens)[0],
                                 bidirectional.apply(params, modified)[0]))

  def test_default_transformer_checkpoint_shape_is_preserved(self):
    model = Transformer(n_heads=1, d_model=7, n_layers=1, hidden_dims=(8,),
                        use_bias=False, normalization='rms', activation='silu',
                        vocab_size=13, max_seq_len=34)
    params = model.init(jax.random.key(0), jnp.zeros((34,), dtype=jnp.int32))
    self.assertNotIn('time_embed', params['params'])
    self.assertEqual(sum(p.size for p in jax.tree_util.tree_leaves(params)), 714)

  def test_loss_and_gradients_are_finite_for_both_tasks_and_corruptions(self):
    for task_name in ('add', 'multiply'):
      task = TASKS[task_name]()
      prompts, _, targets = generate_prompts(task, 2, 1, 16, jax.random.key(0))
      targets = jnp.flip(targets, axis=-1)
      for corruption in ('mask', 'uniform'):
        process = D3PM(4, corruption)
        hparams = dict(model_type='d3pm', diffusion_steps=4, n_heads=2, d_model=8,
                       n_layers=1, hidden_dims=[16], use_bias=False, vocab_size=14,
                       normalization='rms', activation='silu', max_seq_len=prompts.shape[1] + targets.shape[1])
        model = build_transformer(hparams)
        sequence = jnp.concatenate((prompts[0], targets[0]))
        params = model.init(jax.random.key(1), sequence, timestep=jnp.array(4))
        loss_fn = lambda p: diffusion_loss(p, model, process, prompts, targets, jax.random.key(2), 13)
        (loss, metrics), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertGreaterEqual(float(metrics['vb']), -1e-5)
        self.assertTrue(all(np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grads)))
        self.assertGreater(float(jnp.linalg.norm(grads['params']['time_embed']['embedding'])), 0)

  def test_oracle_generation_and_absorbing_support(self):
    task = TASKS['multiply']()
    prompts, _, targets = generate_prompts(task, 2, 1, 32, jax.random.key(0))
    targets = jnp.flip(targets, axis=-1)

    class Oracle:
      def apply(self, params, sequence, train=False, timestep=None):
        digits = jnp.flip(task.compute(sequence[:2], sequence[3:4]))
        logits = jnp.full((sequence.shape[0], 14), -100.)
        return logits.at[prompts.shape[1] + jnp.arange(3), digits].set(100.)

    for corruption in ('mask', 'uniform'):
      process = D3PM(4, corruption)
      for steps in (1, 3, 4):
        for sample_final in (False, True):
          generate = jax.jit(lambda key: generate_answers(
            Oracle(), {}, process, prompts, 3, key, 13, steps, sample_final, return_trace=True))
          answers, trace = generate(jax.random.key(2))
          np.testing.assert_array_equal(answers, targets)
          self.assertEqual(trace.shape, (steps + 1, 32, 3))
          self.assertTrue(np.all(np.asarray(answers) < 10))
          if corruption == 'mask':
            np.testing.assert_array_equal(trace[0], 10)
            previous, following = np.asarray(trace[:-1]), np.asarray(trace[1:])
            self.assertTrue(np.all((previous == 10) | (previous == following)))

  def test_invalid_steps(self):
    with self.assertRaises(ValueError):
      D3PM(0)
    with self.assertRaises(ValueError):
      generate_answers(None, {}, D3PM(4), jnp.zeros((1, 4), dtype=jnp.int32), 3,
                       jax.random.key(0), 13, sampling_steps=5)

  def test_absorbing_variational_loss_matches_weighted_masked_cross_entropy(self):
    clean = jnp.arange(24, dtype=jnp.int32).reshape(8, 3) % 10
    prompts = jnp.zeros((8, 4), dtype=jnp.int32)
    key = jax.random.key(7)

    class ConstantModel:
      def apply(self, params, sequence, train=False, timestep=None):
        return jnp.broadcast_to(jnp.arange(14) / 14, (sequence.shape[0], 14))

    ce = -jax.nn.log_softmax(jnp.arange(10) / 14)[clean]
    for steps in (1, 4):
      process = D3PM(steps)
      time_key, noise_key = jax.random.split(key)
      times = jax.random.randint(time_key, (8,), 1, steps + 1)
      noisy = jax.vmap(process.sample_forward)(clean, times, jax.random.split(noise_key, 8))
      expected_vb = steps * jnp.where(noisy == 10, ce / times[:, None], 0).mean()
      loss, metrics = diffusion_loss({}, ConstantModel(), process, prompts, clean, key, 13, 0.1)
      np.testing.assert_allclose(metrics['vb'], expected_vb, atol=1e-6)
      np.testing.assert_allclose(loss, expected_vb + 0.1 * ce.mean(), atol=1e-6)

  def test_policy_gradient_has_correct_sign_and_no_reward_gradient(self):
    rewards = jnp.array([0.2, 1.2, 3., 1.])  # Losses, so smaller is better.
    scores = jnp.array([-2., -3., -1., -4.])
    reward_grad, score_grad = jax.grad(thought_policy_loss, argnums=(0, 1))(rewards, scores, 2)
    np.testing.assert_array_equal(reward_grad, 0)
    np.testing.assert_allclose(score_grad, [-0.25, 0.25, 0.5, -0.5])

    # Enumerate every pair of categorical samples. The expected leave-one-out
    # score gradient must equal the exact gradient of expected answer cost.
    theta = jnp.arange(10, dtype=jnp.float32) / 10
    costs = jnp.array([1., 0., 4., 2., 5., 3., 2., 6., 4., 1.])
    first, second = jnp.meshgrid(jnp.arange(10), jnp.arange(10), indexing='ij')
    draws = jnp.stack((first.ravel(), second.ravel()), axis=-1)
    losses = costs[draws]
    def expected_surrogate(theta):
      lp = jax.nn.log_softmax(theta)
      weights = jax.lax.stop_gradient(jnp.exp(lp[draws].sum(axis=-1)))
      terms = jax.vmap(lambda loss, score: thought_policy_loss(loss, score, 2))(losses, lp[draws])
      return (weights * terms).sum()
    exact = jax.grad(lambda theta: (jax.nn.softmax(theta) * costs).sum())(theta)
    np.testing.assert_allclose(jax.grad(expected_surrogate)(theta), exact, atol=1e-6)

  def test_trajectory_scores_and_gradients_match_explicit_product(self):
    class ConstantModel:
      def apply(self, params, sequence, train=False, timestep=None):
        logits = jnp.pad(params['theta'], (0, 6))
        return jnp.broadcast_to(logits, (sequence.shape[0], 16))

    params = {'theta': jnp.arange(10, dtype=jnp.float32) / 10}
    prompts = jnp.zeros((2, 4), dtype=jnp.int32)
    for corruption in ('mask', 'uniform'):
      process = D3PM(3, corruption)
      def sampled_scores(params):
        return generate_answers(ConstantModel(), params, process, prompts, 2,
                                jax.random.key(2), 15, 2, sample_final=True,
                                return_trace=True, return_logprob=True)
      answers, trace, scores = sampled_scores(params)
      def explicit_scores(params):
        clean_probs = process.clean_probs(params['theta'])
        result = jnp.zeros((2,))
        for index, (t, s) in enumerate(((3, 2), (2, 0))):
          posterior = process.posterior(clean_probs, trace[index], s, t)
          selected = jnp.take_along_axis(log_probs(posterior), trace[index + 1, ..., None], axis=-1)
          result += selected[..., 0].sum(axis=-1)
        return result
      np.testing.assert_allclose(scores, explicit_scores(params), atol=1e-6)
      np.testing.assert_allclose(jax.grad(lambda p: sampled_scores(p)[2].sum())(params)['theta'],
                                 jax.grad(lambda p: explicit_scores(p).sum())(params)['theta'], atol=1e-6)
      self.assertTrue(np.all(np.asarray(answers) < 10))

  def test_thoughts_use_only_prompt_and_have_separate_markers(self):
    task = TASKS['add']()
    prompts, _, _ = generate_prompts(task, 1, 1, 2, jax.random.key(0))
    cot_steps = 3
    expected_length = prompts.shape[1] + 1 + cot_steps
    class ShapeCheckingModel:
      def apply(inner, params, sequence, train=False, timestep=None):
        self.assertEqual(sequence.shape[0], expected_length)
        return jnp.zeros((expected_length, 16))
    thoughts = generate_thoughts(ShapeCheckingModel(), {}, D3PM(2), prompts, task,
                                 cot_steps, jax.random.key(1), 15)
    prefix = thought_prefix(prompts, thoughts, task)
    np.testing.assert_array_equal(prefix[:, :prompts.shape[1]], prompts)
    np.testing.assert_array_equal(prefix[:, prompts.shape[1]], task.think_token)
    np.testing.assert_array_equal(prefix[:, -1], task.answer_token)
    self.assertEqual(thoughts.shape, (2, cot_steps))
    self.assertTrue(np.all(np.asarray(thoughts) < 10))

  def test_shared_answer_noise_makes_identical_scratchpads_equal(self):
    class ConstantModel:
      def apply(self, params, sequence, train=False, timestep=None):
        return jnp.zeros((sequence.shape[0], 16))
    prompts = jnp.zeros((6, 4), dtype=jnp.int32)
    clean = jnp.repeat(jnp.array([[2, 3], [5, 7]]), 3, axis=0)
    for corruption in ('mask', 'uniform'):
      losses, _ = diffusion_loss({}, ConstantModel(), D3PM(3, corruption), prompts,
                                 clean, jax.random.key(3), 15,
                                 return_per_example=True, noise_group_size=3)
      groups = losses.reshape(2, 3)
      np.testing.assert_array_equal(groups[:, 0], groups[:, 1])
      np.testing.assert_array_equal(groups[:, 0], groups[:, 2])
      self.assertEqual(float(thought_policy_loss(losses, jnp.arange(6.), 3)), 0)

  def test_cot_gradients_for_both_corruptions(self):
    task = TASKS['multiply']()
    prompts, _, clean = generate_prompts(task, 1, 1, 4, jax.random.key(0))
    clean = jnp.flip(clean, axis=-1)
    for corruption in ('mask', 'uniform'):
      process = D3PM(2, corruption)
      hparams = dict(model_type='d3pm', diffusion_steps=2, n_heads=2, d_model=8,
                     n_layers=1, hidden_dims=[16], use_bias=False, vocab_size=16,
                     normalization='rms', activation='silu', max_seq_len=10)
      model = build_transformer(hparams)
      example = jnp.concatenate((thought_prefix(prompts[:1], jnp.zeros((1, 2), dtype=jnp.int32), task)[0],
                                 clean[0]))
      params = model.init(jax.random.key(1), example, timestep=jnp.array(2))
      def evaluate(params, weight):
        return cot_diffusion_loss(params, model, process, prompts, clean, jax.random.key(2),
                                  15, task, 2, 3, weight, cot_sampling_steps=1)
      value_grad = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
      (loss, metrics), grads = value_grad(params, jnp.array(0.1))
      (_, no_pg_metrics), no_pg_grads = value_grad(params, jnp.array(0.))
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grads)))
      np.testing.assert_allclose(metrics['answer_loss'], no_pg_metrics['answer_loss'])
      differences = jax.tree_util.tree_map(lambda a, b: a - b, grads, no_pg_grads)
      self.assertGreater(float(sum(jnp.linalg.norm(g) for g in jax.tree_util.tree_leaves(differences))), 1e-6)
      np.testing.assert_allclose(loss, metrics['answer_loss'] + 0.1 * metrics['cot_policy'], atol=1e-6)


if __name__ == '__main__':
  unittest.main()
