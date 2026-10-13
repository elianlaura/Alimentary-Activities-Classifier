# PaD-TS W500 runtime estimate

Estimated on 2026-10-03 for one NVIDIA H200 (141 GB), FP32, physical batch
size 512, hidden size 256, one encoder block, three decoder blocks, 500
diffusion steps, and nine sensor axes. These are planning estimates, not measured
W500 runtimes.

## Recommended training budget

Use training steps proportional to neighborhood size:

| Neighborhood | Steps | LR anneal steps |
|---|---:|---:|
| Central 25% | 25,000 | 75,000 |
| Central 50% | 50,000 | 150,000 |
| Central 75% | 75,000 | 225,000 |
| Full | 100,000 | 300,000 |

This keeps data exposure approximately constant instead of training the Central
25% data about four times more heavily than Full. Expected row draws are close
to 8,000 per drinking window and 7,700 per eating window at every tier. Setting
the annealing horizon to three times the training steps preserves the existing
PaD-TS ratio between training and annealing horizons.

## Estimated time for each training job

| Job | Activity | Neighborhood | Windows | Steps | Attention | BiGRU |
|---:|---|---|---:|---:|---:|---:|
| 1-2 | Drink | Central 25% | 1,590 | 25,000 | 34 h (25-45 h) | 36 h (30-42 h) |
| 3-4 | Drink | Central 50% | 3,196 | 50,000 | 68 h (50-90 h) | 71 h (60-83 h) |
| 5-6 | Drink | Central 75% | 4,811 | 75,000 | 102 h (75-135 h) | 107 h (90-125 h) |
| 7-8 | Drink | Full | 6,399 | 100,000 | 136 h (100-181 h) | 143 h (119-167 h) |
| 9-10 | Eat | Central 25% | 1,661 | 25,000 | 34 h (25-45 h) | 36 h (30-42 h) |
| 11-12 | Eat | Central 50% | 3,335 | 50,000 | 68 h (50-90 h) | 71 h (60-83 h) |
| 13-14 | Eat | Central 75% | 5,007 | 75,000 | 102 h (75-135 h) | 107 h (90-125 h) |
| 15-16 | Eat | Full | 6,697 | 100,000 | 136 h (100-181 h) | 143 h (119-167 h) |

Each row represents two jobs: one Attention generator and one BiGRU generator.
Runtime is driven mainly by architecture, sequence length, batch size, and step
count, so drinking and eating have approximately the same training time at the
same tier.

Recommended schedule totals:

- Attention: approximately 680 GPU-hours.
- BiGRU: approximately 714 GPU-hours.
- All 16 jobs: approximately 1,394 GPU-hours, with a conservative aggregate
  range of about 1,098-1,736 GPU-hours.
- With two GPUs continuously occupied: approximately 29 compute-only days
  (range about 23-36 days), excluding queue and resubmission delays.

With the current 48-hour job limit, plan roughly one allocation for Central
25%, two for Central 50%, three for Central 75%, and three to four for Full.
Training is checkpointed and resumable.

## Synthetic sampling after training

The following assumes one synthetic five-second window per real window,
ancestral sampling over all 500 diffusion steps, batch size 4096, and about 25
seconds of job startup. Batch size 4096 at W500 is not yet memory-tested.

| Activity | Neighborhood | Synthetic windows | Attention | BiGRU |
|---|---|---:|---:|---:|
| Drink | Central 25% | 1,590 | 2.5 min | 2.0 min |
| Drink | Central 50% | 3,196 | 4.6 min | 3.5 min |
| Drink | Central 75% | 4,811 | 6.7 min | 5.1 min |
| Drink | Full | 6,399 | 8.8 min | 6.6 min |
| Eat | Central 25% | 1,661 | 2.6 min | 2.0 min |
| Eat | Central 50% | 3,335 | 4.8 min | 3.6 min |
| Eat | Central 75% | 5,007 | 7.0 min | 5.3 min |
| Eat | Full | 6,697 | 9.2 min | 6.9 min |

All 16 one-to-one sampling jobs total approximately 81 GPU-minutes if run
sequentially. For larger generated volumes, loop time scales approximately
linearly; startup time does not. The extrapolated W500 rates are approximately
0.079 seconds/window for Attention and 0.058 seconds/window for BiGRU.

## Fixed-100k alternative

If strict compute parity with the previous generators is preferred, train all
16 models for 100,000 steps. Every Attention job would then take about 136 hours
(100-181 hours), and every BiGRU job about 143 hours (119-167 hours), regardless
of neighborhood size. The total would be about 2,232 GPU-hours, or 46.5
compute-only days with two GPUs. This makes the Central 25% windows receive
approximately four times as many draws as Full, so it is not the recommended
comparison.

## Basis and checks required before launch

Measured H200 W24 baselines were approximately 0.325 seconds/step for Attention
and 0.369 seconds/step for BiGRU. Architecture-operation scaling from W24 to
W500, adjusted for plausible GPU-efficiency changes, gives planning rates of
about 4.9 seconds/step for Attention and 5.1 seconds/step for BiGRU. Conservative
ranges are 3.6-6.5 and 4.3-6.0 seconds/step, respectively.

Before launching all jobs:

1. Run a 100-step W500 smoke test for each architecture to verify that physical
   batch size 512 fits and record peak GPU memory.
2. Run at least a 1,000-step timing pilot per architecture and replace these
   extrapolated estimates with measured seconds/step.
3. Reuse each activity's Full-data min-max scaler for its Central 25%, Central
   50%, Central 75%, and Full generators. The current trainer otherwise fits a
   different scaler to every input array, confounding the comparison.
4. Decide whether generator batches should remain uniformly sampled by row or
   preserve equal participant influence with a participant-balanced sampler.
   This choice should have negligible effect on runtime.

No generator job was launched as part of this estimate.
