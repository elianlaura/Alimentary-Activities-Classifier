# R1-02: 30-second real medoids and class-specific PaD-TS BiGRU matches

Start with [all six nine-axis comparisons](all_medoid_pairs.pdf), one page per
medoid, and [the distributions with medoids marked](distributions_and_medoids.pdf).
All figures also have PNG versions. Class overviews:
[drinking](drink_30s_medoids.pdf), [eating](eat_30s_medoids.pdf).
Nine-axis value distributions: [drinking](drink_axis_distributions.pdf),
[eating](eat_axis_distributions.pdf).

## Selected real excerpts

These are actual recorded training excerpts, selected before accessing the
synthetic data. M1–M3 are ordered by decreasing participant-weighted cluster size.

| Activity | Medoid | Participant | Moment | Cluster weight | Participants in cluster |
|---|---|---|---|---:|---:|
| Drinking | M1 | S1688 | MB | 50.66% | 52 |
| Drinking | M2 | S1061 | MB | 44.53% | 50 |
| Drinking | M3 | S1045 | MB | 4.81% | 18 |
| Eating | M1 | S1359 | MA | 40.94% | 44 |
| Eating | M2 | S1046 | MA | 39.53% | 48 |
| Eating | M3 | S1007 | MB | 19.54% | 28 |

Drinking M3 represents a low-motion part of a drinking-labelled session. If only
two drinking examples fit in the paper, M1 and M2 cover clusters accounting for
95.19% of participant-weighted excerpts. Participants can appear in more than
one cluster; their individual excerpts have different patterns.

## Selection and matching

1. Use `deo_data/deo_dedup`, restricted to its 55 training participants. Assemble
   six consecutive five-second rows within the same subject, activity, wrist,
   dominance and moment. Require every start-time increment to be exactly five
   seconds. A six-second increment starts a new run; no missing interval is
   interpolated. Excerpts are disjoint and contain 3,000 × 9 original values.
   This yields 939 drinking and 994 eating candidates, each class retaining all
   55 training participants. No candidate fails the finite-value/16-g checks.
2. Give every participant equal total weight within the class. Each of their
   excerpts has weight `1 / (55 × number_of_that_participant's_excerpts)`.
3. Describe each excerpt with 45 features: for each of nine axes, compute mean,
   log standard deviation, log first-difference RMS, and Welch power fractions
   in 0.1–1 Hz and 1–5 Hz relative to power in 0.1–20 Hz. Welch uses 1,000-point
   segments and 50% overlap. Axis normalization uses the existing real-training
   statistics. Descriptor means and standard deviations are subsequently
   fitted per class with participant weights. Distances are Euclidean in these
   standardized features, divided by √45.
4. Run weighted PAM k-medoids with `k=3`, six deterministic/seeded starts, and
   medoids from different participants. Optimize the sum of weighted distances
   to the nearest medoid. All improving single-medoid swaps are evaluated until
   none remains. This is a local optimum under the participant constraint, not
   a proof of a global optimum or of three distinct action types. The number
   three follows the requested maximum.
5. Freeze the real choices in `real_selection.json`. Then load the existing
   separate drinking/eating generators from
   `/mnt/beegfs/home/elian.riveros/deo_runs_replica_dedup_perclass/generators`.
   The configuration, training-data metadata, completed sample counts and
   checkpoint paths confirm BiGRU generators using only training subjects.
6. For each class, form 2,200 disjoint 30-second synthetic candidates from
   consecutive groups of 125 original 24-step segments. Preserve segment order
   and all nine channels; do not smooth, align, reorder or regenerate them.
   The final 37 segments in each shard cannot form a complete 30-second
   candidate and are unused.
7. Using the frozen real feature transformation, assign three distinct synthetic
   candidates to the three real medoids to minimize total feature distance.
   Distinctness can make a match differ from its independent nearest neighbour.
   Both the independent nearest ID and the selected match's rank are recorded.

During the real-only scaling check, median/IQR scaling overemphasized a few
near-constant descriptors and isolated one participant as a drinking cluster.
The final method uses participant-weighted mean/standard-deviation scaling,
giving each descriptor equal weighted variance. This decision did not use
synthetic similarity or classifier performance.

## Reading the distributions

`distributions_and_medoids.pdf` shows real excerpts colored by cluster and all
synthetic candidates in gray. Stars M1–M3 mark real medoids; crosses S1–S3 mark
their synthetic matches. PCA is fitted only to weighted real features and is
used for visualization; medoids are calculated in all 45 dimensions. The two
components explain 50.30% of real drinking variance and 44.56% of eating variance.
The right panels show participant-weighted gyroscope-motion distributions with
real-medoid locations marked.

The per-axis CDF figures compare all real excerpts (black, participant weighted),
all synthetic excerpts (gray dashed), each real medoid (colored solid), and its
synthetic match (same color, dashed). Aggregate curves use empirical quantiles;
each selected curve uses every observed value. Display limits show the central
99% of aggregate values; the CDF itself retains tail probability. A medoid is a
whole multivariate sequence, so it appears as a curve, not a single sensor value.

## Interpretation and checks

The 30-second excerpts offer more temporal context, but the class labels do not
mark the start/end of individual gestures. These are representative labelled
excerpts, not verified complete bites or sips. No activity classifier or visual
preference was used for selection. The weighted silhouette values are 0.150
(drinking) and 0.100 (eating); the groups overlap, particularly for eating.

Synthetic matches are existing generated examples, not paired reconstructions
of the real medoids. Each is assembled from 125 independently generated short
segments, rather than jointly generated over 30 seconds. The distribution plots
show substantial feature differences between real and synthetic excerpts; the
matches should not be presented as proof of high temporal fidelity. Gyroscope
units are not independently verified, so plots retain and label source values.

Validation checked training membership, exact timestamp continuity, no overlap,
equal total participant weights, bit-for-bit reconstruction of selected real and
synthetic arrays, within-cluster medoid optimality under the participant
constraint, and absence of any improving allowed single-medoid swap.

Exact row/segment indices, hashes, cluster membership, normalization, provenance
and distances are in `selection.json`, `selected_medoids.csv`, and the candidate
CSV files. `*_selected_sequences.npz` contains the six selected arrays per class
(three real plus three synthetic); `*_feature_space.npz` contains all descriptor
coordinates, cluster labels and selections.

## Reproduce

From the repository root:

```bash
env MPLCONFIGDIR=/tmp/deo-r1-02-mpl OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  /mnt/beegfs/home/elian.riveros/conda/envs/deo-tf/bin/python \
  scripts/plot_r1_02_medoids.py \
  --data-dir /mnt/beegfs/home/elian.riveros/deo_data/deo_dedup \
  --generator-root /mnt/beegfs/home/elian.riveros/deo_runs_replica_dedup_perclass/generators \
  --out-dir artifacts/r1_02_medoids_30s --medoids 3 --seed 42
```
