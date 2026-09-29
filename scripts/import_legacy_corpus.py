#!/usr/bin/env python
"""Convert the archived synthetic CSV behind the 0.9055 run into a pretraining corpus.

    python scripts/import_legacy_corpus.py \
        --csv $DATA_ROOT/legacy_synthetic/ddpm_fake_drinkeat_4505_20251006-191354.csv \
        --out-dir $RUNS/corpora/legacy_de_fake_padts_94u

Input layout (dataset 'de_fake_padts_94u' of the legacy code): one header row (0..4504),
then one row per window: 5 placeholder metadata columns (S0000, 1111111111, 0, "A,B,C",
Drink) and 500 x 9 values (24-step PaD-TS segments concatenated). The legacy loader
dropped the header row and the 5 metadata columns, as done here. Values are kept in
float32 (no rounding); scripts/pretrain.py --norm corpus then applies the legacy per-axis
z-score with the statistics of this file.

WARNING: the generator behind this file was trained on drink/eat windows of all 94
subjects, test subjects included. Use it only to reproduce the archived result.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import N_FEATURES, N_TIMESTEPS  # noqa: E402
from deo.io import count_lines, sha256_file  # noqa: E402
from deo.utils import write_json  # noqa: E402

N_META = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunksize", type=int, default=2000)
    ap.add_argument("--no-hash", action="store_true")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.out_dir, "manifest.json")):
        print("[legacy-corpus] %s exists, skipping" % args.out_dir)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    n = count_lines(args.csv) - 1  # header row
    print("[legacy-corpus] %s: %d windows" % (args.csv, n), flush=True)
    corpus = np.lib.format.open_memmap(os.path.join(args.out_dir, "corpus.npy"), mode="w+", dtype=np.float32,
                                       shape=(n, N_TIMESTEPS, N_FEATURES))
    pos, n_nonfinite = 0, 0
    for chunk in pd.read_csv(args.csv, header=0, chunksize=args.chunksize, low_memory=False):
        data = chunk.iloc[:, N_META:].to_numpy(dtype=np.float32)
        if data.shape[1] != N_TIMESTEPS * N_FEATURES:
            raise ValueError("expected %d value columns, got %d" % (N_TIMESTEPS * N_FEATURES, data.shape[1]))
        bad = ~np.isfinite(data)
        n_nonfinite += int(bad.sum())
        data[bad] = 0.0  # legacy: np.nan_to_num
        corpus[pos:pos + len(chunk)] = data.reshape(-1, N_TIMESTEPS, N_FEATURES)
        pos += len(chunk)
        if (pos // args.chunksize) % 50 == 0:
            print("[legacy-corpus] %d / %d (%.0fs)" % (pos, n, time.time() - t0), flush=True)
    if pos != n:
        raise RuntimeError("line count %d != parsed rows %d" % (n, pos))
    corpus.flush()
    del corpus
    # no class labels in the legacy file (drink and eat were pooled in one generator)
    np.save(os.path.join(args.out_dir, "labels.npy"), np.full(n, -1, dtype=np.int8))
    st = os.stat(args.csv)
    write_json(os.path.join(args.out_dir, "manifest.json"), {
        "kind": "legacy_csv", "source_csv": os.path.abspath(args.csv), "source_bytes": st.st_size,
        "source_sha256": None if args.no_hash else sha256_file(args.csv), "n_windows": int(n),
        "per_class": {"unlabelled": int(n)}, "nonfinite_values_set_to_zero": n_nonfinite,
        "warning": "generator trained on all 94 subjects (test included): reproduction only",
        "seconds": round(time.time() - t0, 1)})
    print("[legacy-corpus] done: %d windows, %d non-finite values" % (n, n_nonfinite), flush=True)


if __name__ == "__main__":
    main()
