#!/usr/bin/env python
"""Stage 4: build one pretraining corpus (see deo/corpora.py for the kinds).

    # diffusion, 10 R windows, generators sampled beforehand
    python scripts/build_corpus.py --data-dir $DATA_ROOT/deo --kind diffusion --multiplier 10 \
        --generated drink=$RUNS/generators/gru_drink/samples.npy eat=$RUNS/generators/gru_eat/samples.npy \
        --out-dir $RUNS/corpora/diff_gru_x10
    # matched-volume control
    python scripts/build_corpus.py --data-dir $DATA_ROOT/deo --kind surrogate --multiplier 10 \
        --out-dir $RUNS/corpora/surrogate_x10
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import corpora, io, norm, splits  # noqa: E402
from deo.utils import read_json  # noqa: E402


def check_shard(path):
    """A sampling shard is usable only once sample_deo.py wrote its completion record."""
    record = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(record):
        raise SystemExit("[corpus] %s is incomplete (no %s): sampling did not finish, resubmit stage 3"
                         % (path, os.path.basename(record)))
    n, rows = int(read_json(record)["n"]), len(np.load(path, mmap_mode="r"))
    if n != rows:
        raise SystemExit("[corpus] %s has %d segments but its record says %d" % (path, rows, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--kind", required=True, choices=["diffusion", "augment", "noise", "surrogate", "real"])
    ap.add_argument("--multiplier", type=float, default=1.0, help="corpus size in units of R (ignored for real)")
    ap.add_argument("--scope", default="minority", choices=list(corpora.SCOPES))
    ap.add_argument("--generated", nargs="*", default=[], help="class=path.npy for --kind diffusion")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.out_dir, "manifest.json")):
        print("[corpus] %s exists, skipping" % args.out_dir)
        return
    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(os.path.join(args.data_dir, "split.json"))
    generated = {}
    for item in args.generated:
        c, paths = item.split("=", 1)
        for p in paths.split(","):
            check_shard(p)
        generated[c] = corpora.ConcatArray(paths.split(","))
    if args.kind == "diffusion":
        missing = set(corpora.SCOPES[args.scope]) - set(generated)
        if missing:
            raise SystemExit("[corpus] --generated has no samples for %s" % sorted(missing))
    stats = norm.load_stats(os.path.join(args.data_dir, "norm_train.json"))
    manifest = corpora.build(args.kind, args.out_dir, X, meta, split, args.multiplier, scope=args.scope,
                             seed=args.seed, generated=generated, axis_std=np.asarray(stats["std"]))
    print("[corpus] %s" % manifest, flush=True)


if __name__ == "__main__":
    main()
