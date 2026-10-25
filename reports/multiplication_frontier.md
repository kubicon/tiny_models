# Report: the multiplication frontier of tiny Transformers

Study run 2026-10-02 to 2026-10-05. Raw runs, logs and checkpoints are in
`experiments/multiplication_frontier/` (git-ignored; its `README.md` is the day-by-day lab log).
This file is the summary to keep in git.

## TL;DR

- **Best result:** plain 2-layer Transformer (d128, 4 heads, MLP 256, 0.34M params, no CoT, no
  looping) solved **7x7** multiplication (>= 90% exact match on 7-digit x 7-digit), using an adaptive
  size curriculum and a **cosine LR decay restarted at every stage** (f09 s0). It sits at ~1% on 8x8.
- **The biggest single factor is the LR schedule.** At constant LR every model stalls at 5x5 to 6x6
  (noisy 70-90% ceilings). The per-stage cosine is what pushed L2 from 5x5 to 7x7.
- **Seed matters as much as any architectural change.** Two seeds of the same recipe often differ by
  2-3 curriculum stages (f09: 7x7 vs 5x5). Runs sit on flat plateaus and escape at random times.
- **Looping does not replace distinct parameters.** Weight-shared depth (memory / recurrent, 4 or 8
  passes) did not beat plain L2. With one layer, looping helps a lot over plain L1 (3x3 -> up to 6x6)
  but is seed-fragile and on average still behind L2.
- **Every stall looks the same:** 1-3 middle answer digits (where the biggest carry arrives) are at
  chance, the rest ~100%. The bottleneck is the long carry chain in the middle of the product.
- More passes (8), more heads (8), bigger batch, oversampling full-size problems and the Abacus
  digit-position embedding gave **no consistent gain**.

## Goal

How small a Transformer can learn N x N multiplication, and where does a given size stop?
Started from `multiplication_benchmark` (plain L2 solved 5x5 with one of two seeds). The question
evolved into: *what is the smallest model that learns multiplication well* (user, 2026-10-03), so
width and depth were deliberately **not** increased; only optimisation, curriculum and weight-shared
looping were explored.

### Ground rule: no algorithmic supervision

Supervision comes from the data only: operands in, answer digits out. No auxiliary losses or probes
on carries / running sums / partial products, no CoT or scratchpad targets. Those pick the algorithm
for the model and defeat the point of the study. An f15 run with an aux carry probe was launched and
deleted for this reason. Allowed knobs: architecture, optimisation, curriculum / operand sampling.
The Abacus embedding (f16) bends this rule and was run only as a reference.

## Setup (common to all runs)

- Task: `transformer.py --task multiply`, answer tokens only, operands always padded to width 10
  (answer width 20), so token positions never change across the curriculum.
- Base model: d_model 128, 4 heads, MLP 256, SiLU, RMSNorm. L2 = 335,488 params, L1 = 171,392.
- **Adaptive curriculum** (`--advance-at X --advance-patience 2`): stages 2x2, 3x3, ..., 10x10. Each
  stage samples operand sizes 1..N; its validation uses 2000 fresh problems with exactly N-digit
  operands every 500 steps. A stage ends when validation >= X twice in a row. A stage that hits its
  step limit ends the run. The "frontier" of a run = the last stage it passed.
- AdamW, grad clip 1.0, wd 0.01, batch 512, 1k warmup.
- LR schedules: `warmup-constant` (f01-f07), `stage-cosine` (f08 on: every stage restarts with a 1k
  warmup to 3e-3 then cosine to 1e-4 over the stage's max length, 300k steps).
- Model variants:
  - `L1`, `L2`: plain stack of 1 / 2 blocks.
  - `recur_LkxP` (`--recurrent-steps P`): each block applied P times in a row (L2x4 = AAAABBBB).
  - `memory_LkxP` (`--model-type memory --memory-passes P`): the whole stack looped P times with the
    input embedding re-added on every pass (L2x4 = ABABABAB).
- Infrastructure: `common.sh` (base args, `run` with auto-resume), `mk.sh` (job.sh writer),
  `submitter.sh` + `queue.txt` (keeps <= 6 SLURM jobs alive, resubmits after the time limit).

## Experiments and results

"Passed" = last stage solved; "stuck at (max)" = the stage that ran out of steps and the best
validation accuracy reached on it.

### Phase 1 — constant LR (f01-f07)

| job | change | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| f01 L2 | lr 1e-3, adv 95% | 3x3 / 4x4 (29) | 3x3 / 4x4 (95.2, late jump) | | |
| f02 L2 | lr 3e-3, adv 95% | 4x4 / 5x5 (92) | 4x4 / 5x5 (92) | | |
| f03 L2 | lr 1e-3, adv 80% | **6x6** / 7x7 (6) | 5x5 / 6x6 (19) | | |
| f04 L2 | lr 3e-3, adv 80% | 4x4 / 5x5 (84) | 3x3 / 4x4 (42) | | |
| f05 memory_L2x4 | lr 3e-3, adv 80% | 3x3 / 4x4 (12) | 5x5 / 6x6 (76) | | |
| f06 recur_L2x4 | lr 3e-3, adv 80% | 4x4 / 5x5 (29) | 5x5 / 6x6 (23) | | |
| f07 L2 | finer N x N+1 stages, wd 0.04, adv 80% | 5x5 / 5x6 (81) | 5x5 / 5x6 (81) | 4x5 / 5x5 (64) | 4x5 / 5x5 (82) |

Findings:
- With constant LR everything tops out at 5x5-6x6. At 95% the threshold itself stops lr 3e-3 runs:
  accuracy oscillates in an 85-92% band and needs LR decay to reach 99%.
- Looping L2 to effective depth 8 (f05/f06) did not move the frontier.
- Finer stages (f07) make early stages fast and consistent across seeds but all four still stall at
  5x5 / 5x6.

### Phase 2 — per-stage cosine LR (f08-f10), threshold 90%

| job | change | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| f08 memory_L2x4 | stage-cosine | 3x3 / 4x4 (24) | 5x5 / 6x6 (29) | 3x3 / 4x4 (37) | 3x3 / 4x4 (30) |
| f09 L2 | stage-cosine | **7x7** / 8x8 (1.5) | 5x5 / 6x6 (72) | | |
| f10 L2 | f09 s0 + batch 2048 | 4x4 / 5x5 (80) | | | |
| f10 L2 | f09 s0 + `--full-width-prob 0.3` | 6x6 / 7x7 (8) | | | |

Findings:
- Decaying LR inside every stage is the most important change: plain L2 reaches 7x7 (best run of
  the study). 8x8 stays flat at ~1% for the whole 300k-step stage.
- memory_L2x4 got *worse* with the cosine schedule (3 of 4 seeds stuck at 4x4).
- Stall diagnosis: two kinds. (A) long flat plateaus with falling grad norm, escaped suddenly (a
  saddle, not a minimum; e.g. f01 s1 sat at 25% on 4x4 for 340k steps, then reached ~90% in 20k);
  (B) noisy ceilings at high constant LR, fixed by LR decay. In a uniform NxN stage only 1/N^2 of a
  batch has both operands at full size.
- Neither 4x batch nor 30% full-size rows helped: the plateaus are just flat, not starved of
  gradient signal (single seed each, so weak evidence).

### Phase 3 — one layer: can looping replace the second layer? (f11-f14)

f09 recipe with `--n-layers 1` (half the block params).

| job | change | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|
| f11 L1 | plain | 2x2 / 3x3 (11) | 2x2 / 3x3 (13) | | | | |
| f11 recur_L1x4 | 4 recurrent steps | 4x4 / 5x5 (22) | **6x6** / 7x7 (34) | | | | |
| f11 memory_L1x4 | 4 memory passes | 3x3 / 4x4 (29) | 4x4 / 5x5 (5) | | | | |
| f12 recur_L1x4 | + full-width 0.3 | | | 3x3 / 4x4 (7) | 4x4 / 5x5 (14) | | |
| f12 memory_L1x4 | + full-width 0.3 | | | 3x3 / 4x4 (17) | 4x4 / 5x5 (5) | 3x3 / 4x4 (36) | 4x4 / 5x5 (11) |
| f13 recur_L1x8 | 8 recurrent steps | 3x3 / 4x4 (23) | 3x3 / 4x4 (7) | | | | |
| f13 memory_L1x8 | 8 memory passes | 2x2 / 3x3 (82) | **6x6** / 7x7 (13) | | | | |
| f14 recur_L1x4_h8 | 8 heads (head dim 16) | 3x3 / 4x4 (30) | 3x3 / 4x4 (25) | | | | |
| f14 memory_L1x4_h8 | 8 heads | 3x3 / 4x4 (29) | 3x3 / 4x4 (29) | | | | |

Findings:
- Plain L1 cannot even do 3x3 (flat ~10%). Looping one block is a real gain (typically 4x4-5x5,
  best seeds 6x6), so sequential computation matters, not only parameter count.
- But looped L1 is very seed-fragile and on average below plain L2. Memory vs recurrent: no clear
  winner; differences look like seed luck.
- 8 passes or 8 heads did not move the frontier. f14 put all four seeds on the same ~25-30% 4x4
  plateau.

### Diagnostics (`diag_stream.py`, results in `diag/diag_stream.json`)

Checkpoint probes (residual stream per pass, readout per pass, error types, accuracy by carry) on
every f08/f09/f11/f12 run:
- Hypothesis "memory model drowns in re-added input copies" is **refuted**: the copies are only
  0.1-0.25 of the stream; reading out with one copy changes accuracy by <= 1.5 points.
- **Common failure signature for every stalled model:** 1-3 middle answer places — where the carry
  out of the column with the most partial products lands (place 4 at 4x4, 5-7 at 5x5, 5-6 at 6x6) —
  are near chance; all other places ~100%. Accuracy falls as the incoming carry grows.
- Only memory-specific trait: stuck memory runs often answer the stuck place by copying the previous
  digit (43-62% of errors vs ~11% chance).
- Masking padded answer positions out of the loss was considered and dropped (Adam undoes the
  rescaling it would cause).

### Phase 4 — Abacus digit-significance embedding as a reference (f16)

`--digit-pos-embed`: learned embedding of each position's digit significance, shared by a_i, b_i
and answer digit i. Not a loss, but a hand-made alignment hint. f09/f11 recipe, seeds 0/1.

| model | s0 | s1 | without Abacus (f09 / f11) |
|---|---|---|---|
| L1 | 2x2 / 3x3 (13) | 2x2 / 3x3 (11) | 2x2 / 2x2 |
| L2 | 4x4 / 5x5 (26) | 5x5 / 6x6 (74) | 7x7 / 5x5 |
| recur_L1x4 | 3x3 / 4x4 (27) | 3x3 / 4x4 (87) | 4x4 / 6x6 |
| memory_L1x4 | 5x5 / 6x6 (3) | 3x3 / 4x4 (21) | 3x3 / 4x4 |

Finding: Abacus does **not** help here (equal or worse; with width-10 padding, position already
fixes digit significance). Consistent with u02 in `multiplication_unsupervised`.

## Overall conclusions

1. Frontier for the 0.34M-param plain L2 with data-only supervision: **7x7** (one of two seeds),
   reliably 5x5-6x6. Frontier for 0.17M-param L1: 2x2 plain, up to 6x6 when looped (lucky seeds).
2. Optimisation (LR decay per stage) beats every architectural knob tried. Training dynamics
   (plateaus, seed luck) dominate the results more than model capacity does.
3. The limit is always the middle-digit carry chain. Nothing tried so far (passes, heads, batch,
   sampling, Abacus) breaks it reliably.
4. With 2 seeds per arm most comparisons are within seed noise; conclusions about memory vs
   recurrent and the f10/f16 tweaks are weak.

## Open directions (proposed, not yet run)

- Depth / width controls at the frontier (L3/L4, d256 or 8 heads at L2) to test whether two layers
  really are the limit (conflicts with the "smallest model" goal; run only as a reference).
- 2x stage budget (600k steps/stage) to separate "not enough training" from "cannot learn".
- LR re-warm when a stage plateaus (plateaus are escaped at random; a new LR kick may trigger it).
- Digit-value curriculum (cap digit values, e.g. `N:N:ITERS:D`) to make carries rarer early on.
- More seeds per arm before drawing memory-vs-recurrent conclusions.

## Reproducing

All flags live in `transformer.py` (`--advance-at`, `--advance-patience`, `--lr-schedule
stage-cosine|warmup-constant`, `--full-width-prob`, `--digit-pos-embed`, `--recurrent-steps`,
`--model-type memory --memory-passes`). The exact command of every run is in
`experiments/multiplication_frontier/<job>/commands.txt`. Base arguments (from `common.sh`):

```
--task multiply --activation silu --normalization rms --optimizer adamw
--grad-clip-norm 1.0 --weight-decay 0.01 --num-samples 2000 --steps-per-chunk 500
--sequence-length 10 10 --num-length-schedule 2:2:S 3:3:S ... 10:10:S
--advance-at 0.9 --advance-patience 2 --batch-size 512 --warmup-steps 1000
--d-model 128 --n-heads 4 --hidden-dims 256 --n-layers 2
```

Best run (f09 s0) adds `--seed 0 --learning-rate 3e-3 --lr-schedule stage-cosine --lr-min 1e-4
--checkpoint-every 25000 --num-steps 2700000` with S = 300000.
