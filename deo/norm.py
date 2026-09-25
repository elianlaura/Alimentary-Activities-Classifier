"""Per-axis z-score normalisation.

Legacy behaviour (kept for reproduction with --norm none):
  * fine-tuning data (eatdrinkanother_94u) was NOT normalised;
  * synthetic pretraining data (de_fake_padts_94u) was z-scored per axis with
    statistics of the whole synthetic file.
The corrected protocol (--norm train) z-scores real and synthetic windows with the
mean/std of the REAL training subjects only, so both stages see the same scale and
no statistic comes from validation or test subjects.
"""
import numpy as np

from . import AXIS_NAMES
from .utils import read_json, write_json


def fit_stats(X, idx=None, chunk=20000):
    """Mean/std per axis over all time steps, streamed over a memmap."""
    idx = np.arange(len(X)) if idx is None else np.sort(np.asarray(idx))
    s = np.zeros(X.shape[-1], dtype=np.float64)
    ss = np.zeros(X.shape[-1], dtype=np.float64)
    n = 0
    for i in range(0, len(idx), chunk):
        block = np.asarray(X[idx[i:i + chunk]], dtype=np.float64)
        s += block.sum(axis=(0, 1))
        ss += (block ** 2).sum(axis=(0, 1))
        n += block.shape[0] * block.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist(), "axes": AXIS_NAMES, "n_values_per_axis": int(n)}


def save_stats(path, stats):
    write_json(path, stats)


def load_stats(path):
    return read_json(path)


def apply(x, stats):
    """Return a float32 normalised copy (or view-safe in-place result)."""
    if stats is None:
        return np.asarray(x, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)
