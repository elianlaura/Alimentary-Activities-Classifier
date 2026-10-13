# Five-second participant-weighted medoid neighborhoods

These datasets were selected independently for **drinking** and **eating** from
the `deo_dedup` training split. No validation/test window was used, and no
synthetic sample has been generated.

Each candidate is one native 5-second `(500, 9)` sensor window. Within each
activity, every participant receives equal total weight. A real global medoid is
the observed window minimizing weighted mean distance in the standardized
45-feature space. Windows are ranked by distance to that medoid.

`central25`, `central50`, and `central75` are the inclusive distance prefixes
that first reach 25%, 50%, and 75% of cumulative participant-balanced weight.
`full` is the unchanged activity-specific training baseline. Therefore, the
percentages are not percentages of participants or raw window counts.

| Activity | Dataset | Windows | Achieved weight | Participants |
|---|---:|---:|---:|---:|
| Drink | Central 25% | 1,590 | 0.250144 | 53 |
| Drink | Central 50% | 3,196 | 0.500131 | 55 |
| Drink | Central 75% | 4,811 | 0.750045 | 55 |
| Drink | Full | 6,399 | 1.000000 | 55 |
| Eat | Central 25% | 1,661 | 0.250003 | 55 |
| Eat | Central 50% | 3,335 | 0.500116 | 55 |
| Eat | Central 75% | 5,007 | 0.750112 | 55 |
| Eat | Full | 6,697 | 1.000000 | 55 |

The drinking medoid is zero-based source dataset row `12743` (S1355, MA). The
eating medoid is row `1626` (S1533, MA).

## Files

- `generator_inputs/*_w500.npy`: directly usable five-second float32 arrays.
- Matching `.csv`: row-aligned source metadata, distance, rank, weight, and
  neighborhood membership.
- Matching `*_indices.npy`: row-aligned indices into the original `X.npy`.
- Matching `.json`: counts, cutoff, subject coverage, paths, and SHA-256 hashes.
- `{drink,eat}_ranking.csv`: every eligible window ordered nearest to farthest.
- `{drink,eat}_feature_space.npz`: features, scaling, distances, weights, and
  exact medoid objective values.
- `selection.json`: complete method and provenance record.
- `plots/medoid_neighborhood_distributions.{png,pdf}`: activity-specific t-SNE
  projections paired with exact weighted distance CDFs and cutoff markers.
- `plots/{drink,eat}_tsne_neighborhoods.{png,pdf}`: individual t-SNE plots.
- `plots/{drink,eat}_tsne_coordinates.csv`: reproducible plotted coordinates.
- `plots/plot_manifest.json`: t-SNE parameters and projection diagnostics.

The eight arrays are nested within each activity: `central25` is a subset of
`central50`, which is a subset of `central75`, which is a subset of `full`.
Every array was verified bit-for-bit against the indicated source rows.

t-SNE is only a visualization of local structure. Neighborhood membership,
cutoffs, and the medoid were computed in the standardized 45-dimensional space,
not from the two-dimensional projection. The distance-CDF panels show the exact
selection calculation.

## Before generator training

Participant balancing currently defines the medoid and neighborhood cutoff.
The existing `train_deo.py` samples array rows uniformly and fits a new min-max
scaler for each input file. For a controlled comparison, generator training
should reuse the full-activity scaler across all four datasets. If equal
participant influence is also required during training, the trainer needs a
participant-balanced row sampler.

Rebuild with:

```bash
python scripts/build_medoid_neighborhoods_5s.py \
  --data-dir /mnt/beegfs/home/elian.riveros/deo_data/deo_dedup \
  --out-dir artifacts/r1_02_medoid_neighborhoods_5s

python scripts/plot_medoid_neighborhoods_5s.py \
  --input-dir artifacts/r1_02_medoid_neighborhoods_5s
```
