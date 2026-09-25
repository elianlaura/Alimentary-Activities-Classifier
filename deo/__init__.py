"""DEO (drink / eat / other) activity recognition with synthetic-data pretraining.

Modules
-------
io        CSV -> memory-mapped .npy conversion and dataset loading
splits    subject-wise train/val/test splits (legacy split + repeated splits)
norm      per-axis z-score normalisation fitted on training subjects only
models    CABiGRU autoencoder, encoder extraction and classification heads
metrics   balanced accuracy, sensitivity/specificity, F1, mAP, per-subject scores
corpora   pretraining corpora: diffusion, matched-volume controls, real data
utils     seeding, timing and JSON helpers
"""

N_TIMESTEPS = 500
N_FEATURES = 9
CLASS_NAMES = ["DRINK", "EAT", "OTHER"]
AXIS_NAMES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "mag_x", "mag_y", "mag_z"]
