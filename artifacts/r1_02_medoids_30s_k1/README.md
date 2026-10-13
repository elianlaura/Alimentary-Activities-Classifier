# One global 30-second medoid per activity

This run allows any training participant to provide the representative excerpt,
while retaining equal total weight per participant. Participant identity is not
a clustering feature or a selection constraint.

The global drinking medoid is candidate 458 from S1078, moment MA, starting at
timestamp 1677422831. The global eating medoid is candidate 192 from S1046,
moment MA, starting at timestamp 1675428213. Each is one observed 30-second
sequence assembled from six consecutive, deduplicated five-second training rows.

With `k=1`, every excerpt belongs to the only cluster by definition. A displayed
coverage of 100% therefore means that the global medoid was selected against all
participant-weighted excerpts. It does not mean that one signal reproduces 100%
of their temporal variation.

Main outputs:

- `all_medoid_pairs.pdf`: drinking and eating comparisons, one page each;
- `distributions_and_medoids.pdf`: global medoid locations in feature and motion distributions;
- `drink_medoid_1.png` and `eat_medoid_1.png`: individual nine-axis comparisons;
- `drink_axis_distributions.pdf` and `eat_axis_distributions.pdf`: value CDFs;
- `selection.json`: exact indices, hashes, distances, provenance and selection method.

The real medoid is fixed before synthetic data are inspected. Its orange match is
the nearest completed class-specific PaD-TS BiGRU candidate in the 45-feature
space. The synthetic sequence is illustrative, not a reconstruction of the real
signal. It consists of 125 consecutive generated 24-step segments. Two drinking
shards without completion records were excluded; only completed shard 0 was used.

The methodology otherwise matches `../r1_02_medoids_30s/README.md`.
