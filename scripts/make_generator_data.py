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
    segments crossed window and subject boundaries;
  * windows with physically impossible accelerometer values (> 16 g) are left out
    of the generator data (only OTHER windows of S1003/S1029 have them). The
    classifier data are not changed.

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
    ap.add_argument("--max-abs-acc", type=float, default=16 * 9.81,
                    help="drop windows with any |acc| above this value (m/s^2). 16 g is beyond the watch's "
                         "range; 1,043 OTHER training windows of S1003/S1029 hold values of ~±997 that would "
                         "set the generator's min-max range. 0 = keep every window")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "generator")
    os.makedirs(out_dir, exist_ok=True)
    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(os.path.join(args.data_dir, "split.json"))
    idx = class_train_indices(meta, split)
    held_out = set(split["val"]) | set(split["test"])
    starts = np.arange(0, N_TIMESTEPS - args.window + 1, args.stride)
    rng = np.random.default_rng(args.seed)
    for c in args.classes:
        used = set(meta.loc[idx[c], "subject"])
        if used & held_out or not used <= set(split["train"]):
            raise SystemExit("[gen-data] %s: windows of non-training subjects selected: %s"
                             % (c, sorted(used - set(split["train"]))))
        wins = io.take(X, idx[c])
        keep = np.ones(len(wins), dtype=bool)
        if args.max_abs_acc:
            keep = np.abs(wins[:, :, :3]).max(axis=(1, 2)) <= args.max_abs_acc
            wins = wins[keep]
        seg = np.stack([wins[:, s:s + args.window] for s in starts], axis=1).reshape(-1, args.window, X.shape[-1])
        if args.max_segments and len(seg) > args.max_segments:
            seg = seg[rng.choice(len(seg), args.max_segments, replace=False)]
        path = os.path.join(out_dir, "%s_w%d.npy" % (c, args.window))
        np.save(path, seg.astype(np.float32))
        write_json(path.replace(".npy", ".json"), {
            "class": c, "window": args.window, "stride": args.stride, "n_segments": int(len(seg)),
            "n_source_windows": int(len(wins)), "train_subjects": len(split["train"]),
            "subjects_used": sorted(used), "val_or_test_subjects_used": sorted(used & held_out),
            "max_abs_acc": args.max_abs_acc, "windows_dropped_implausible_acc": int((~keep).sum()),
            "subjects_with_dropped_windows": sorted(set(meta.loc[idx[c][~keep], "subject"]))})
        print("[gen-data] %s: %d windows (%d dropped, |acc| > %g) -> %s segments %s"
              % (c, len(wins), int((~keep).sum()), args.max_abs_acc, len(seg), path), flush=True)


if __name__ == "__main__":
    main()
