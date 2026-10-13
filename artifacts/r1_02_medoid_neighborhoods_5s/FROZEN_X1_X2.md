# Frozen sources for the x5/x10/x20 expansion

The x5, x10, and x20 experiments must reuse the exact data and generators used
for x1 and x2.  No medoid, neighborhood, participant set, scaler, generator
checkpoint, or existing synthetic sample is recomputed.

For each of `central25`, `central50`, `central75`, and `full`:

- the real five-second EAT and DRINK generator inputs and their source-row
  indices are frozen;
- the final BiGRU checkpoint used for x1/x2 is frozen;
- `samples_2x.npy` is frozen and is always the first shard of the expanded
  synthetic pool;
- the existing x1 and x2 corpus files are frozen;
- a new `samples_extra_18x_seed1.npy` shard supplies only the additional
  samples required after x2.

Consequently, the synthetic samples are nested exactly within each activity:

```text
x1 = first 1N samples from the frozen 2N shard
x2 = all 2N samples from the frozen 2N shard
x5 = frozen 2N + first 3N new samples
x10 = frozen 2N + first 8N new samples
x20 = frozen 2N + all 18N new samples
```

Here, `N` is the real train-window count for the corresponding activity and
neighborhood.  A corpus file stores the complete DRINK block followed by the
complete EAT block, so nesting is checked separately inside those two blocks.
The frozen checksums are in `frozen_x1_x2_sources.sha256`.  The expansion
submission script verifies them before creating or submitting any job.
