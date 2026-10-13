# R1-02: real and synthetic activity windows

`r1_02_one_sample_per_activity.pdf` is the paper figure. It shows exactly one
real training window and one synthetic window for each activity. The nine axes
are shown separately, with real and synthetic traces overlaid on matched scales.
The lower-RMSE synthetic source is used for each activity:

- drinking: class-specific generator (train-z-scored RMSE 0.53);
- eating: pooled drink+eat generator (train-z-scored RMSE 0.65).

The matching PNG is a high-resolution preview. The earlier pooled/per-class
comparison files in this directory are retained as diagnostics but are not the
paper figure.

The selection is deterministic (`seed=0`). Real examples are active,
non-extreme feature medoids from the training split. Synthetic examples are the
nearest neighbours to those same real windows among 20,000 candidates, using
per-axis z-score statistics fitted only on the real training split. No signal
smoothing or other post-processing is applied. The pooled generator is
unlabelled, so its drinking/eating association is determined only by proximity
to the corresponding real reference. Exact indices, subject IDs, distances,
corpus manifests, and source choices are recorded in `selection.json`.
The same audit also validates the generator-input metadata: the drink, eat, and
pooled generators each used only the 55 training subjects and no validation or
test subjects.

Suggested caption:

> Representative 5-s smartwatch windows for drinking and eating. One real
> training example and its nearest synthetic neighbour are shown for each
> activity. The nine accelerometer, gyroscope, and magnetometer axes are plotted
> separately; real and synthetic traces use the same scale within each panel.
> Neighbours are selected after normalization with real-training statistics,
> and signals are shown without smoothing.
