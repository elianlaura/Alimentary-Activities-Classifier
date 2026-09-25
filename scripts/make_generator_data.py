#!/usr/bin/env python
"""Stage 1: training data for the diffusion generator, one file per class.

Corrections with respect to the legacy pipeline:
  * only windows of the TRAINING subjects are used (the legacy file
    drinkeat_94u_train.csv contained all 94 subjects, 28 of the 29 test subjects
    included);
  * one generator per class (drink, eat[, other]) instead of a pooled drink+eat
    model, so the class of every synthetic window is known;
  * segments of length W are cut inside each 5-s window. The legacy loader treated
    the concatenation of all windows as one continuous series, so its training
    segments crossed window and subject boundaries.

    python scripts/make_generator_data.py --data-dir $DATA_ROOT/deo --window 24 --stride 12
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import N_TIMESTEPS, io, splits  # noqa: E402
from deo.corpora import class_train_indices  # noqa: E402
from deo.utils import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="converted dataset dir (contains X.npy, split.json)")
    ap.add_argument("--out-dir", default=None, help="default: <data-dir>/generator")
    ap.add_argument("--classes", nargs="+", default=["drink", "eat", "other"])
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--max-segments", type=int, default=0, help="random cap per class (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "generator")
    os.makedirs(out_dir, exist_ok=True)
    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(os.path.join(args.data_dir, "split.json"))
    idx = class_train_indices(meta, split)
    starts = np.arange(0, N_TIMESTEPS - args.window + 1, args.stride)
    rng = np.random.default_rng(args.seed)
    for c in args.classes:
        wins = io.take(X, idx[c])
        seg = np.stack([wins[:, s:s + args.window] for s in starts], axis=1).reshape(-1, args.window, X.shape[-1])
        if args.max_segments and len(seg) > args.max_segments:
            seg = seg[rng.choice(len(seg), args.max_segments, replace=False)]
        path = os.path.join(out_dir, "%s_w%d.npy" % (c, args.window))
        np.save(path, seg.astype(np.float32))
        write_json(path.replace(".npy", ".json"), {
            "class": c, "window": args.window, "stride": args.stride, "n_segments": int(len(seg)),
            "n_source_windows": int(len(wins)), "train_subjects": len(split["train"]),
            "subjects_used": sorted(meta.loc[idx[c], "subject"].unique().tolist())})
        print("[gen-data] %s: %d windows -> %s segments %s" % (c, len(wins), len(seg), path), flush=True)


if __name__ == "__main__":
    main()
