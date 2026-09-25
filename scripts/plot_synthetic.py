#!/usr/bin/env python
"""Real vs synthetic windows (R1-02) and simple fidelity diagnostics.

    python scripts/plot_synthetic.py --data-dir $DATA_ROOT/deo --corpus $RUNS/corpora/diff_gru_x10 \
        --out-dir $RUNS/analysis/synthetic_examples

For drink and eat: one real training window, a random synthetic window and the
synthetic window nearest to the real one (Euclidean distance on z-scored data,
searched in a random subsample), plotted per sensor (acc / gyro / mag) on shared
axes. Also per-axis value histograms and average power spectra, real vs synthetic.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import AXIS_NAMES, io, norm, splits  # noqa: E402
from deo.corpora import CLASS_IDS, class_train_indices  # noqa: E402

SENSORS = [("accelerometer (m/s²)", slice(0, 3)), ("gyroscope (rad/s)", slice(3, 6)), ("magnetometer (µT)", slice(6, 9))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--search", type=int, default=20000, help="synthetic windows searched for the nearest neighbour")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(os.path.join(args.data_dir, "split.json"))
    stats = norm.load_stats(os.path.join(args.data_dir, "norm_train.json"))
    corpus = np.load(os.path.join(args.corpus, "corpus.npy"), mmap_mode="r")
    labels = np.load(os.path.join(args.corpus, "labels.npy"))
    idx = class_train_indices(meta, split)
    t = np.arange(500) / 100.0
    for cname in ("drink", "eat"):
        syn_idx = np.flatnonzero(labels == CLASS_IDS[cname])
        if len(syn_idx) == 0:
            continue
        real = io.take(X, [rng.choice(idx[cname])])[0]
        cand = np.sort(rng.choice(syn_idx, min(args.search, len(syn_idx)), replace=False))
        pool = np.asarray(corpus[cand], dtype=np.float32)
        d = ((norm.apply(pool, stats) - norm.apply(real[None], stats)) ** 2).sum((1, 2))
        nn = pool[np.argmin(d)]
        rnd = pool[rng.integers(len(pool))]
        fig, axes = plt.subplots(3, 3, figsize=(11, 6.5), sharex=True)
        for row, (label, sl) in enumerate(SENSORS):
            lo = min(real[:, sl].min(), nn[:, sl].min(), rnd[:, sl].min())
            hi = max(real[:, sl].max(), nn[:, sl].max(), rnd[:, sl].max())
            for col, (title, w) in enumerate((("real", real), ("synthetic, nearest to real", nn),
                                              ("synthetic, random", rnd))):
                ax = axes[row, col]
                for j, ax_name in zip(range(sl.start, sl.stop), AXIS_NAMES[sl]):
                    ax.plot(t, w[:, j], lw=0.8, label=ax_name[-1])
                ax.set_ylim(lo, hi)
                if row == 0:
                    ax.set_title(title, fontsize=10)
                if col == 0:
                    ax.set_ylabel(label, fontsize=8)
                if row == 2:
                    ax.set_xlabel("time (s)")
        axes[0, 0].legend(fontsize=7, ncol=3)
        fig.suptitle("%s: real vs synthetic 5-s windows" % cname.upper())
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out_dir, "examples_%s.%s" % (cname, ext)), dpi=200)
        plt.close(fig)

        real_pool = io.take(X, np.sort(rng.choice(idx[cname], min(3000, len(idx[cname])), replace=False)))
        fig, axes = plt.subplots(2, 9, figsize=(18, 4.2))
        for j in range(9):
            a, b = real_pool[:, :, j].ravel(), pool[:3000, :, j].ravel()
            rng_ = (np.percentile(np.r_[a, b], 0.5), np.percentile(np.r_[a, b], 99.5))
            axes[0, j].hist(a, bins=60, range=rng_, density=True, alpha=0.5, label="real")
            axes[0, j].hist(b, bins=60, range=rng_, density=True, alpha=0.5, label="synthetic")
            axes[0, j].set_title(AXIS_NAMES[j], fontsize=8)
            f = np.fft.rfftfreq(500, d=0.01)
            pa = (np.abs(np.fft.rfft(real_pool[:, :, j] - real_pool[:, :, j].mean(1, keepdims=True), axis=1)) ** 2).mean(0)
            pb = (np.abs(np.fft.rfft(pool[:3000, :, j] - pool[:3000, :, j].mean(1, keepdims=True), axis=1)) ** 2).mean(0)
            axes[1, j].semilogy(f, pa, lw=0.8, label="real")
            axes[1, j].semilogy(f, pb, lw=0.8, label="synthetic")
            axes[1, j].set_xlabel("Hz", fontsize=7)
        axes[0, 0].legend(fontsize=7)
        axes[1, 0].set_ylabel("mean power")
        fig.suptitle("%s: value distributions (top) and power spectra (bottom)" % cname.upper())
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out_dir, "distributions_%s.%s" % (cname, ext)), dpi=150)
        plt.close(fig)
    print("[plot] figures in %s" % args.out_dir)


if __name__ == "__main__":
    main()
