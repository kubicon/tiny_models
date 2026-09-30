"""Regression checks for CoT combined with the additional training features."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from d3pm import D3PM, generate_thoughts as diffusion_thoughts, thought_prefix
from task import TASKS, generate_episode, generate_prompts
from transformer import (
  MemoryTransformer, build_transformer, generate_thoughts, generate_tokens,
  make_train_step, model_logits, parse_num_length_schedule,
)


def hparams(task, model_type='transformer', cot_steps=0, **extra):
  return dict(model_type=model_type, task=task.name, sequence_length=[2, 1],
              n_heads=1, d_model=4, n_layers=2, hidden_dims=[4], use_bias=False,
              vocab_size=task.vocab_size + (2 if cot_steps else 0),
              normalization='rms', activation='silu', cot_steps=cot_steps,
              max_seq_len=task.episode_length(2, 1) + (cot_steps + 2 if cot_steps else 0),
              **extra)


def initialize(model, task, cot_steps=0):
  sequence = generate_episode(task, 2, 1, 1, jax.random.key(0))[0]
  if cot_steps:
    end = task.answer_start_index(2, 1) + 1
    sequence = jnp.concatenate((sequence[:end], jnp.array([task.think_token]),
                                jnp.zeros((cot_steps,), dtype=jnp.int32),
                                jnp.array([task.answer_token]), sequence[end:]))
  if isinstance(model, MemoryTransformer):
    return model.init(jax.random.key(1), sequence, jnp.zeros((sequence.size, model.d_model)))
  return model.init(jax.random.key(1), sequence)


class TransformerMergeTests(unittest.TestCase):
  def test_digit_significance_follows_answer_after_thoughts(self):
    for task_name in TASKS:
      task = TASKS[task_name]()
      answer_start = task.answer_start_index(2, 1)
      n_answer = task.answer_length(2, 1)
      for reverse in (False, True):
        for cot_steps in (0, 3):
          for causal in (False, True):
            ids = task.significance_ids(2, 1, reverse, cot_steps=cot_steps, causal=causal)
            self.assertEqual(ids[:2], (0, 1) if reverse else (1, 0))
            marker = answer_start + (cot_steps + 2 if cot_steps else 0)
            if cot_steps:
              self.assertEqual(ids[answer_start:marker], (n_answer + 1,) * (cot_steps + 2))
            offset = marker if causal else marker + 1
            self.assertEqual(ids[offset:offset + n_answer], tuple(range(n_answer)))
            expected_len = task.episode_length(2, 1) + (cot_steps + 2 if cot_steps else 0)
            self.assertEqual(len(ids), expected_len if causal else expected_len - 1)

  def test_cot_combines_with_all_answer_losses(self):
    cases = [('add', 'transformer', 'mse', False),
             ('multiply', 'memory', 'ce', True),
             ('add', 'memory', 'mse', True),
             ('multiply', 'transformer', 'ce', False)]
    for task_name, model_type, loss_type, early_eos in cases:
      with self.subTest(task=task_name, model=model_type, loss=loss_type):
        task = TASKS[task_name]()
        outputs = 1 if loss_type == 'mse' else task.max_place_sum(2, 1) * 10 // 9 + 1
        model = build_transformer(hparams(task, model_type, 2, aux_outputs=outputs,
                                          mtp_tokens=3, digit_pos_embed=True, memory_passes=2))
        params = initialize(model, task, 2)
        optimizer = optax.sgd(0.01)
        step = make_train_step(
          model, task, optimizer, 2, 1, 2, early_eos=early_eos, cot_steps=2,
          cot_samples=3, full_width_prob=0.5, mix_full_size_prob=0.5,
          deep_supervision=model_type == 'memory', aux_loss_weight=0.2,
          aux_loss_type=loss_type, mtp_weight=0.3, return_aux_loss=True)
        (updated, _), metrics = jax.jit(lambda p: step(
          (p, optimizer.init(p)), jax.random.key(4), 1, 1, 3))(params)
        self.assertEqual(len(metrics), 3)
        self.assertTrue(np.isfinite(np.asarray(metrics)).all())
        self.assertGreater(float(metrics[2]), 0)
        for name in ('aux_probe', 'mtp_head_2'):
          self.assertGreater(float(jnp.linalg.norm(
            updated['params'][name]['kernel'] - params['params'][name]['kernel'])), 0)
        if not early_eos:
          self.assertGreater(float(jnp.linalg.norm(updated['params']['mtp_head_3']['kernel']
                                                  - params['params']['mtp_head_3']['kernel'])), 0)
        prompts, _, _ = generate_prompts(task, 2, 1, 1, jax.random.key(5))
        thoughts = generate_thoughts(model, updated, prompts, task, 2)
        answer_prefix = jnp.concatenate((thoughts, jnp.full((1, 1), task.answer_token)), axis=-1)
        tokens = generate_tokens(model, updated, answer_prefix, task.answer_length(2, 1) + 1)
        self.assertEqual(tokens.shape, (1, task.answer_length(2, 1) + 1))

  def test_multi_token_loss_masks_padding_even_when_slice_skips_eos(self):
    task = TASKS['add']()
    model = build_transformer(hparams(task, cot_steps=2, mtp_tokens=4))
    params = initialize(model, task, 2)
    optimizer = optax.sgd(0.01)
    step = make_train_step(model, task, optimizer, 2, 1, 2,
                           cot_steps=2, early_eos=True)
    # One-digit operands capped at 1 sum to at most 2: one digit, EOS, padding.
    (updated, _), metrics = jax.jit(lambda p: step(
      (p, optimizer.init(p)), jax.random.key(12), 1, 1, 1))(params)
    self.assertTrue(np.isfinite(np.asarray(metrics)).all())
    for name in ('mtp_head_3', 'mtp_head_4'):
      for leaf in ('kernel', 'bias'):
        np.testing.assert_array_equal(updated['params'][name][leaf], params['params'][name][leaf])
    self.assertGreater(float(jnp.linalg.norm(updated['params']['mtp_head_2']['kernel']
                                            - params['params']['mtp_head_2']['kernel'])), 0)

  def test_cot_policy_gradient_still_changes_updates(self):
    task = TASKS['multiply']()
    model = build_transformer(hparams(task, cot_steps=2, digit_pos_embed=True, mtp_tokens=2))
    params = initialize(model, task, 2)
    optimizer = optax.sgd(0.01)
    updates = []
    for weight in (0.0, 0.1):
      step = make_train_step(model, task, optimizer, 2, 1, 3, cot_steps=2,
                             cot_samples=3, cot_pg_weight=weight)
      (updated, _), _ = jax.jit(lambda p: step(
        (p, optimizer.init(p)), jax.random.key(6), 2, 1))(params)
      updates.append(updated)
    differences = jax.tree_util.tree_map(lambda a, b: a - b, *updates)
    self.assertGreater(float(sum(jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(differences))), 0)

  def test_profiler_api_and_plain_answer_loss_are_preserved(self):
    for model_type in ('transformer', 'memory'):
      task = TASKS['add']()
      model = build_transformer(hparams(task, model_type, memory_passes=2))
      params = initialize(model, task)
      optimizer = optax.sgd(0.01)
      key = jax.random.key(7)
      step = make_train_step(model, task, optimizer, 2, 1, 2)
      _, metrics = jax.jit(lambda p: step((p, optimizer.init(p)), key, 2, 1))(params)
      sequences = generate_episode(task, 2, 1, 2, key, 2, 1)
      start = task.answer_start_index(2, 1)
      expected = jax.vmap(lambda sequence: optax.softmax_cross_entropy_with_integer_labels(
        model_logits(model, params, sequence)[start:-1], sequence[start + 1:]).mean())(sequences).mean()
      self.assertEqual(len(metrics), 2)
      np.testing.assert_allclose(metrics[0], expected, rtol=1e-5)

  def test_onehot_reuses_gather_checkpoint_and_gradients(self):
    task = TASKS['add']()
    gather = build_transformer(hparams(task, embed_lookup='gather'))
    onehot = build_transformer(hparams(task, embed_lookup='onehot'))
    params = initialize(gather, task)
    sequence = generate_episode(task, 2, 1, 1, jax.random.key(8))[0]
    for model in (gather, onehot):
      value, grad = jax.value_and_grad(lambda p: jnp.square(model_logits(model, p, sequence)).sum())(params)
      if model is gather:
        expected, expected_grad = value, grad
      else:
        np.testing.assert_allclose(value, expected, rtol=1e-5)
        for actual, reference in zip(jax.tree_util.tree_leaves(grad), jax.tree_util.tree_leaves(expected_grad)):
          np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)

  def test_diffusion_aux_weight_does_not_add_arithmetic_parameters(self):
    task = TASKS['multiply']()
    model = build_transformer(hparams(task, 'd3pm', 2, diffusion_steps=2,
                                      aux_loss_weight=0.1, digit_pos_embed=True))
    self.assertEqual(model.aux_outputs, 0)
    self.assertFalse(model.causal)
    prompts, _, _ = generate_prompts(task, 2, 1, 1, jax.random.key(9))
    prefix = thought_prefix(prompts, jnp.zeros((1, 2), dtype=jnp.int32), task)
    sequence = jnp.concatenate((prefix[0], jnp.zeros((task.answer_length(2, 1),), dtype=jnp.int32)))
    params = model.init(jax.random.key(10), sequence, timestep=jnp.array(2))
    self.assertNotIn('aux_probe', params['params'])
    thoughts = diffusion_thoughts(model, params, D3PM(2, 'uniform'), prompts, task,
                                  2, jax.random.key(11), 15, sampling_steps=1)
    self.assertEqual(thoughts.shape, (1, 2))
    self.assertTrue((np.asarray(thoughts) < 10).all())

  def test_curriculum_keeps_digit_caps_and_final_stage(self):
    self.assertEqual(parse_num_length_schedule(['1:2', '2:1:3:4'], 2, 1, 6),
                     [(1, 1, 2, 9), (2, 1, 3, 4), (2, 1, 1, 9)])


if __name__ == '__main__':
  unittest.main()
