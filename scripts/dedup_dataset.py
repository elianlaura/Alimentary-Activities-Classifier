#!/usr/bin/env python
"""Write a copy of a converted dataset without the exact duplicate windows.

    python scripts/dedup_dataset.py --data-root $DATA_ROOT --src deo --dst deo_dedup

36% of the DEO windows are byte-identical copies (same subject, timestamp and values;
whole blocks of the source CSV were appended twice). One window per (subject, timestamp)
is kept (the first); a window is dropped only when its values are identical too.
The split, training-subject normalisation statistics, volume unit R and the dataset
tables are then recomputed as in scripts/prepare_data.py.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deo import io, splits  # noqa: E402
from deo.utils import read_json, write_json  # noqa: E402
from prepare_data import prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--src", default="deo")
    ap.add_argument("--dst", default="deo_dedup")
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()

    src, dst = os.path.join(args.data_root, args.src), os.path.join(args.data_root, args.dst)
    if os.path.exists(os.path.join(dst, "meta.csv")):
        raise SystemExit("[dedup] %s already exists" % dst)
    os.makedirs(dst)
    X, meta = io.load_dataset(src)
    # content hash: a window is dropped only if subject, timestamp AND the 500x9 values repeat
    # (48 TV/STAND windows of S1045 share timestamps with SWEEP windows but differ: kept)
    h = np.empty(len(X), dtype=object)
    for s in range(0, len(X), args.chunk):
        blk = np.asarray(X[s:s + args.chunk])
        for i in range(len(blk)):
            h[s + i] = hashlib.md5(blk[i].tobytes()).hexdigest()
    dup = meta.assign(_h=h).duplicated(["subject", "timestamp", "_h"], keep="first").to_numpy()
    n_conflict = int(meta.duplicated(["subject", "timestamp"]).sum() - dup.sum())
    print("[dedup] %d exact copies; %d windows share subject+timestamp with different values (kept)"
          % (dup.sum(), n_conflict), flush=True)
    keep = np.flatnonzero(~dup)
    out = np.lib.format.open_memmap(os.path.join(dst, "X.npy"), mode="w+", dtype=np.float32, shape=(len(keep),) + X.shape[1:])
    for s in range(0, len(keep), args.chunk):
        out[s:s + args.chunk] = X[keep[s:s + args.chunk]]
    out.flush()
    del out
    m = meta.iloc[keep].reset_index(drop=True)
    m.to_csv(os.path.join(dst, "meta.csv"), index=False)
    info = read_json(os.path.join(src, "info.json"))
    info.update({"deduplicated_from": src, "n_windows_before": int(len(meta)), "n_windows": int(len(m)),
                 "duplicates_removed": int(dup.sum()), "same_timestamp_different_values_kept": n_conflict,
                 "class_counts": {k: int(v) for k, v in zip(["DRINK", "EAT", "OTHER"], np.bincount(m.label, minlength=3))}})
    write_json(os.path.join(dst, "info.json"), info)
    print("[dedup] %d -> %d windows (%d identical copies removed)" % (len(meta), len(m), dup.sum()), flush=True)
    split = splits.load_split(os.path.join(src, "split.json"))
    prepare(args.dst, None, "deo", args.data_root, split, skip_existing=True)


if __name__ == "__main__":
    main()
