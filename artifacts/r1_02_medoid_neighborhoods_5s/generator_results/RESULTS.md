# Results: W500 BiGRU medoid-neighborhood generators

All eight BiGRU generators and all eight 2x sampling jobs completed on an NVIDIA H200.
Attention generators were not part of these jobs. No classifier pretraining, fine-tuning,
or balanced-accuracy evaluation has been run for these neighborhood corpora yet.

## Job outputs

| Activity | Neighborhood | Real N | Steps | Train h | Synthetic N | Sample min | s/step |
|---|---|---:|---:|---:|---:|---:|---:|
| drink | central25 | 1,590 | 25,000 | 4.20 | 3,180 | 2.78 | 0.605 |
| drink | central50 | 3,196 | 50,000 | 7.77 | 6,392 | 5.68 | 0.559 |
| drink | central75 | 4,811 | 75,000 | 11.19 | 9,622 | 8.39 | 0.537 |
| drink | full | 6,399 | 100,000 | 15.29 | 12,798 | 11.56 | 0.550 |
| eat | central25 | 1,661 | 25,000 | 4.23 | 3,322 | 2.86 | 0.610 |
| eat | central50 | 3,335 | 50,000 | 7.99 | 6,670 | 5.77 | 0.575 |
| eat | central75 | 5,007 | 75,000 | 11.44 | 10,014 | 8.74 | 0.549 |
| eat | full | 6,697 | 100,000 | 14.90 | 13,394 | 11.99 | 0.536 |

## Direct synthetic-data diagnostics

Distances use the original real medoid and the original standardized 45-feature space.
Lower median ratio and lower feature Wasserstein distance mean closer agreement with the
real training neighborhood; these diagnostics do not measure downstream classifier accuracy.

| Activity | Neighborhood | Syn/real median distance | Within C25 | Within C50 | Within C75 | Mean feature W1 |
|---|---|---:|---:|---:|---:|---:|
| drink | central25 | 2.433 | 0.0% | 0.8% | 5.9% | 0.748 |
| drink | central50 | 2.647 | 0.1% | 0.2% | 2.2% | 0.845 |
| drink | central75 | 2.882 | 0.0% | 0.0% | 0.1% | 1.000 |
| drink | full | 2.568 | 0.0% | 0.0% | 0.0% | 0.875 |
| eat | central25 | 2.084 | 0.0% | 0.0% | 0.0% | 0.573 |
| eat | central50 | 1.862 | 0.0% | 0.0% | 0.0% | 0.683 |
| eat | central75 | 1.372 | 0.3% | 3.4% | 34.0% | 0.515 |
| eat | full | 1.434 | 0.1% | 2.8% | 19.0% | 0.485 |

## Interpretation boundary

These completed jobs establish that the generators trained and produced finite W500 signals.
They do not yet establish which neighborhood improves the eating/drinking classifier. That
requires building x1/x2 corpora followed by the same pretraining, fine-tuning, and subject-level
validation/test evaluation for every neighborhood.
