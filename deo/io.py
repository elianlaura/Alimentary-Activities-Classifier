"""Conversion of the wide window CSVs into memory-mapped arrays, and loading.

Source CSV layout (one row per 5-s window, no header):
    col 0  subject id (e.g. S1001)
    col 1  unix timestamp of the window start
    col 2  activity id
    col 3  recording info, e.g. "R,D,MA" = right wrist, dominant hand, moment A
           (-1 when the file holds a single hand/moment combination)
    col 4  activity name
    col 5: 500 time steps x 9 axes, interleaved per time step
           (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z)

Converted dataset directory (<data_root>/<name>/):
    X.npy       float32 (N, 500, 9), opened with mmap_mode="r"
    meta.csv    subject, timestamp, act_raw, act_name, hand, dominance, moment, label
    info.json   provenance (source file, size, sha256, class counts)
"""
import hashlib
import os
import time

import numpy as np
import pandas as pd

from . import N_FEATURES, N_TIMESTEPS, CLASS_NAMES
from .utils import write_json

N_META = 5

# How the raw activity ids / names become the three DEO classes.
LABELERS = {
    # fullraws3_vivabem012_drink0eat1another2_94u_...: ids are already 0/1/2.
    "deo": lambda act_raw, act_name: act_raw.astype(np.int8),
    # fullraws2_vivabem12_noaer_R_D_MA_MB_..._6act: every activity on the right,
    # dominant wrist. DRINK -> 0, EAT -> 1, DRY/PAPERS/SWEEP/TYPING -> 2.
    "samehand": lambda act_raw, act_name: np.where(
        act_name == "DRINK", 0, np.where(act_name == "EAT", 1, 2)).astype(np.int8),
}


def count_lines(path, block=64 * 1024 * 1024):
    n = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(block)
            if not buf:
                break
            n += buf.count(b"\n")
    return n


def sha256_file(path, block=64 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(block)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _split_info(info):
    """'R,D,MA' -> ('R', 'D', 'MA'); anything else -> ('unknown',) * 3."""
    parts = str(info).split(",")
    if len(parts) == 3:
        return parts
    return ["unknown", "unknown", "unknown"]


def convert_csv(csv_path, out_dir, labeler="deo", chunksize=2000, default_info=None, hash_source=True):
    """Stream the CSV once and write X.npy (float32 memmap) + meta.csv."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    n_rows = count_lines(csv_path)
    print("[convert] %s: %d rows" % (csv_path, n_rows), flush=True)
    X = np.lib.format.open_memmap(os.path.join(out_dir, "X.npy"), mode="w+", dtype=np.float32,
                                  shape=(n_rows, N_TIMESTEPS, N_FEATURES))
    label_fn = LABELERS[labeler]
    metas = []
    pos = 0
    n_nonfinite = 0
    for chunk in pd.read_csv(csv_path, header=None, chunksize=chunksize, low_memory=False):
        data = chunk.iloc[:, N_META:].to_numpy(dtype=np.float32)
        if data.shape[1] != N_TIMESTEPS * N_FEATURES:
            raise ValueError("expected %d data columns, got %d" % (N_TIMESTEPS * N_FEATURES, data.shape[1]))
        bad = ~np.isfinite(data)
        n_nonfinite += int(bad.sum())
        data[bad] = 0.0
        X[pos:pos + len(chunk)] = data.reshape(-1, N_TIMESTEPS, N_FEATURES)
        info = chunk[3].astype(str).to_numpy()
        if default_info is not None:
            info = np.where(np.char.find(info.astype(str), ",") >= 0, info, default_info)
        split = np.array([_split_info(s) for s in info])
        act_raw = chunk[2].to_numpy()
        act_name = chunk[4].astype(str).to_numpy()
        metas.append(pd.DataFrame({
            "subject": chunk[0].astype(str).to_numpy(),
            "timestamp": chunk[1].to_numpy(),
            "act_raw": act_raw,
            "act_name": act_name,
            "hand": split[:, 0], "dominance": split[:, 1], "moment": split[:, 2],
            "label": label_fn(act_raw, act_name),
        }))
        pos += len(chunk)
        if (pos // chunksize) % 20 == 0:
            print("[convert] %d / %d rows (%.0fs)" % (pos, n_rows, time.time() - t0), flush=True)
    if pos != n_rows:
        raise RuntimeError("line count %d != parsed rows %d (trailing newline?)" % (n_rows, pos))
    X.flush()
    del X
    meta = pd.concat(metas, ignore_index=True)
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    st = os.stat(csv_path)
    info = {
        "source_csv": os.path.abspath(csv_path),
        "source_bytes": st.st_size,
        "source_sha256": sha256_file(csv_path) if hash_source else None,
        "labeler": labeler,
        "n_windows": int(n_rows),
        "n_subjects": int(meta["subject"].nunique()),
        "nonfinite_values_set_to_zero": n_nonfinite,
        "class_counts": {CLASS_NAMES[k]: int(v) for k, v in meta["label"].value_counts().sort_index().items()},
        "hand_by_class": {CLASS_NAMES[k]: g.groupby(["hand", "dominance"]).size().rename("n").reset_index()
                          .to_dict("records") for k, g in meta.groupby("label")},
        "seconds": round(time.time() - t0, 1),
    }
    write_json(os.path.join(out_dir, "info.json"), info)
    print("[convert] done: %s" % info["class_counts"], flush=True)
    return info


def load_dataset(data_dir, mmap=True):
    """Return (X, meta). X is a read-only memmap unless mmap=False."""
    X = np.load(os.path.join(data_dir, "X.npy"), mmap_mode="r" if mmap else None)
    meta = pd.read_csv(os.path.join(data_dir, "meta.csv"), keep_default_na=False,
                       dtype={"subject": str, "act_name": str, "hand": str, "dominance": str, "moment": str})
    if len(meta) != len(X):
        raise ValueError("meta.csv (%d) and X.npy (%d) disagree" % (len(meta), len(X)))
    return X, meta


def take(X, idx, chunk=20000):
    """Materialise X[idx] as float32 without loading the whole memmap."""
    idx = np.asarray(idx)
    out = np.empty((len(idx),) + X.shape[1:], dtype=np.float32)
    order = np.argsort(idx)
    for s in range(0, len(idx), chunk):
        sel = order[s:s + chunk]
        out[sel] = X[idx[sel]]
    return out
