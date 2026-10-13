#!/usr/bin/env python
"""Plot real and synthetic DEO windows for reviewer item R1-02.

The figure puts the class-specific and pooled drink+eat GRU generators beside
the same real training window for each activity::

    python scripts/plot_r1_02.py \
        --data-dir $DATA_ROOT/deo \
        --per-class-corpus $RUNS/corpora/diff_gru_x10 \
        --pooled-corpus $RUNS/corpora/diff_gru_pooled_x2 \
        --out-dir artifacts/r1_02

The real example is selected only from the training split. To avoid showing an
inactive or extreme window, it is the feature medoid among candidates whose
motion energy lies between the 60th and 80th percentiles for that class. Each
synthetic example is the nearest neighbour to that same real window in
train-statistics-z-scored Euclidean distance. Candidate counts, selected
indices, distances, and split membership are written to ``selection.json``.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import AXIS_NAMES, N_TIMESTEPS, io, norm, splits  # noqa: E402
from deo.corpora import CLASS_IDS, class_train_indices  # noqa: E402
from deo.utils import write_json  # noqa: E402

SENSORS = [
    ("Accelerometer (m/s\N{SUPERSCRIPT TWO})", slice(0, 3)),
    ("Gyroscope", slice(3, 6)),
    ("Magnetometer (\N{MICRO SIGN}T)", slice(6, 9)),
]
COLORS = ("#0072B2", "#E69F00", "#009E73")


def load_corpus(path):
    corpus = np.load(os.path.join(path, "corpus.npy"), mmap_mode="r")
    labels = np.load(os.path.join(path, "labels.npy"), mmap_mode="r")
    if len(corpus) != len(labels):
        raise ValueError("corpus.npy and labels.npy disagree in %s" % path)
    if corpus.shape[1:] != (N_TIMESTEPS, len(AXIS_NAMES)):
        raise ValueError("expected corpus shape (N, %d, %d), got %s in %s"
                         % (N_TIMESTEPS, len(AXIS_NAMES), corpus.shape, path))
    with open(os.path.join(path, "manifest.json")) as fh:
        manifest = json.load(fh)
    return corpus, labels, manifest


def window_features(windows, stats):
    """Compact features used only to select a representative real window."""
    z = norm.apply(windows, stats)
    dz = np.diff(z, axis=1)
    return np.concatenate([
        z.mean(axis=1),
        z.std(axis=1),
        np.sqrt(np.mean(dz * dz, axis=1)),
    ], axis=1)


def representative_real(X, candidate_indices, stats, rng, search):
    """Return an active, non-extreme feature medoid from real training data."""
    chosen = np.sort(rng.choice(candidate_indices, min(search, len(candidate_indices)), replace=False))
    windows = io.take(X, chosen)
    features = window_features(windows, stats)
    # Normalised first-difference energy captures motion without favouring axes
    # merely because their physical units have a larger numerical range.
    motion = np.mean(features[:, 18:] ** 2, axis=1)
    lo, hi = np.percentile(motion, [60, 80])
    active = (motion >= lo) & (motion <= hi)
    if not active.any():
        active = np.ones(len(chosen), dtype=bool)
    f = features[active]
    center = np.median(f, axis=0)
    scale = np.median(np.abs(f - center), axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    distance = np.mean(((f - center) / scale) ** 2, axis=1)
    local = np.flatnonzero(active)[np.argmin(distance)]
    return int(chosen[local]), windows[local], {
        "n_candidates": int(len(chosen)),
        "motion_percentile_interval": [60, 80],
        "candidate_motion_interval": [float(lo), float(hi)],
        "selected_motion": float(motion[local]),
        "feature_medoid_distance": float(distance.min()),
    }


def nearest_synthetic(corpus, candidate_indices, real, stats, rng, search, chunk=1000):
    """Find the exact nearest candidate without materialising the full pool."""
    chosen = np.sort(rng.choice(candidate_indices, min(search, len(candidate_indices)), replace=False))
    target = norm.apply(real[None], stats)[0]
    best_distance = np.inf
    best_index = None
    best_window = None
    for start in range(0, len(chosen), chunk):
        sub_idx = chosen[start:start + chunk]
        block = np.asarray(corpus[sub_idx], dtype=np.float32)
        z = norm.apply(block, stats)
        mse = np.mean((z - target) ** 2, axis=(1, 2))
        pos = int(np.argmin(mse))
        if mse[pos] < best_distance:
            best_distance = float(mse[pos])
            best_index = int(sub_idx[pos])
            best_window = block[pos].copy()
    return best_index, best_window, {
        "n_candidates": int(len(chosen)),
        "z_scored_rmse": float(np.sqrt(best_distance)),
    }


def signal_limits(windows, sensor_slice):
    values = np.concatenate([w[:, sensor_slice].ravel() for w in windows])
    lo, hi = float(values.min()), float(values.max())
    pad = 0.04 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def generator_training_provenance(data_dir, window, train_subjects):
    """Validate that every generator input used only training subjects."""
    provenance = {}
    generator_dir = os.path.join(data_dir, "generator")
    for source in ("drink", "eat", "drinkeat"):
        path = os.path.join(generator_dir, "%s_w%d.json" % (source, window))
        if not os.path.exists(path):
            raise FileNotFoundError("generator training metadata not found: %s" % path)
        with open(path) as fh:
            payload = json.load(fh)
        used = set(payload["subjects_used"])
        unexpected = sorted(used - train_subjects)
        declared_held_out = sorted(payload.get("val_or_test_subjects_used", []))
        if unexpected or declared_held_out:
            raise ValueError("%s generator includes held-out subjects: %s"
                             % (source, sorted(set(unexpected) | set(declared_held_out))))
        provenance[source] = {
            "metadata_file": os.path.abspath(path),
            "window": int(payload["window"]),
            "n_source_windows": int(payload["n_source_windows"]),
            "subjects_used": len(used),
            "all_subjects_in_train_split": True,
            "val_or_test_subjects_used": [],
        }
    return provenance


def plot_comparison(selected, out_dir, dpi):
    import matplotlib.pyplot as plt

    t = np.arange(N_TIMESTEPS) / 100.0
    fig, axes = plt.subplots(6, 3, figsize=(11.2, 10.2), sharex=True)
    column_titles = ("Real (training split)", "Synthetic: per-class generator",
                     "Synthetic: pooled drink+eat generator")
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=11, pad=8)

    for class_row, cname in enumerate(("drink", "eat")):
        windows = selected[cname]["windows"]
        for sensor_row, (sensor_name, sensor_slice) in enumerate(SENSORS):
            row = class_row * len(SENSORS) + sensor_row
            ylim = signal_limits(windows, sensor_slice)
            for col, window in enumerate(windows):
                ax = axes[row, col]
                for color, j, axis_name in zip(COLORS, range(sensor_slice.start, sensor_slice.stop),
                                               AXIS_NAMES[sensor_slice]):
                    ax.plot(t, window[:, j], lw=0.75, color=color, label=axis_name[-1])
                ax.set_ylim(*ylim)
                ax.grid(True, color="0.9", linewidth=0.45)
                if row in (2, 5):
                    ax.set_xlabel("Time (s)")
                if col == 0:
                    prefix = cname.upper() + "\n" if sensor_row == 0 else ""
                    ax.set_ylabel(prefix + sensor_name, fontsize=9)
            if sensor_row == 0:
                for col, key in ((1, "per_class"), (2, "pooled")):
                    axes[row, col].text(
                        0.98, 0.96, "RMSE %.2f" % selected[cname][key]["z_scored_rmse"],
                        transform=axes[row, col].transAxes, ha="right", va="top", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.8, "pad": 2},
                    )

    axes[0, 0].legend(title="Axis", fontsize=8, title_fontsize=8, ncol=3,
                      loc="lower right", framealpha=0.9)
    # A visible separator keeps the two activities legible in the composite.
    for ax in axes[3]:
        ax.spines["top"].set_linewidth(1.8)
        ax.spines["top"].set_color("0.25")
    fig.suptitle("Real and synthetic 5-s drinking and eating windows", fontsize=14, y=0.995)
    fig.text(0.5, 0.004,
             "Synthetic traces are nearest neighbours to the same real window; distance uses "
             "real-training-set z-score statistics.",
             ha="center", va="bottom", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0.02, 0.025, 1, 0.985), h_pad=0.65, w_pad=0.8)
    stem = os.path.join(out_dir, "r1_02_real_vs_synthetic")
    fig.savefig(stem + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_selected_examples(selected, out_dir, dpi):
    """Compact paper figure using the lower-RMSE source for each activity."""
    import matplotlib.pyplot as plt

    t = np.arange(N_TIMESTEPS) / 100.0
    fig, axes = plt.subplots(3, 4, figsize=(12.6, 6.1), sharex=True)
    choices = {}
    for cname in ("drink", "eat"):
        choices[cname] = ("per_class" if selected[cname]["per_class"]["z_scored_rmse"]
                          <= selected[cname]["pooled"]["z_scored_rmse"] else "pooled")
    source_titles = {"per_class": "per-class", "pooled": "pooled drink+eat"}
    columns = []
    for cname in ("drink", "eat"):
        key = choices[cname]
        synthetic_col = 1 if key == "per_class" else 2
        columns.extend([
            (cname, "Real (training split)", selected[cname]["windows"][0], None),
            (cname, "Synthetic (%s)" % source_titles[key], selected[cname]["windows"][synthetic_col], key),
        ])

    for col, (cname, source, window, _) in enumerate(columns):
        axes[0, col].set_title("%s \u2014 %s" % (cname.upper(), source), fontsize=10, pad=8)
    for activity_start in (0, 2):
        pair_windows = [columns[activity_start][2], columns[activity_start + 1][2]]
        for row, (sensor_name, sensor_slice) in enumerate(SENSORS):
            ylim = signal_limits(pair_windows, sensor_slice)
            for col in (activity_start, activity_start + 1):
                ax = axes[row, col]
                window = columns[col][2]
                for color, j, axis_name in zip(COLORS, range(sensor_slice.start, sensor_slice.stop),
                                               AXIS_NAMES[sensor_slice]):
                    ax.plot(t, window[:, j], lw=0.75, color=color, label=axis_name[-1])
                ax.set_ylim(*ylim)
                ax.grid(True, color="0.9", linewidth=0.45)
                if row == 2:
                    ax.set_xlabel("Time (s)")
                if col == activity_start:
                    ax.set_ylabel(sensor_name, fontsize=9)
        synthetic_col = activity_start + 1
        cname = columns[activity_start][0]
        key = columns[synthetic_col][3]
        axes[0, synthetic_col].text(
            0.98, 0.96, "RMSE %.2f" % selected[cname][key]["z_scored_rmse"],
            transform=axes[0, synthetic_col].transAxes, ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.8, "pad": 2},
        )

    axes[0, 0].legend(title="Axis", fontsize=8, title_fontsize=8, ncol=3,
                      loc="lower right", framealpha=0.9)
    for ax in axes[:, 2]:
        ax.spines["left"].set_linewidth(1.8)
        ax.spines["left"].set_color("0.25")
    fig.suptitle("Real and selected synthetic 5-s activity windows", fontsize=14, y=0.995)
    fig.text(0.5, 0.005,
             "For each activity, the synthetic source with lower train-z-scored nearest-neighbour "
             "RMSE is shown; signals are not smoothed.",
             ha="center", va="bottom", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0.015, 0.035, 1, 0.98), h_pad=0.7, w_pad=0.8)
    stem = os.path.join(out_dir, "r1_02_selected_examples")
    fig.savefig(stem + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return stem, choices


def plot_one_sample_per_activity(selected, out_dir, dpi):
    """One matched real/synthetic pair per activity, with all nine axes."""
    import matplotlib.pyplot as plt

    t = np.arange(N_TIMESTEPS) / 100.0
    fig, axes = plt.subplots(6, 3, figsize=(10.8, 11.4), sharex=True)
    choices = {}
    source_titles = {"per_class": "per-class generator", "pooled": "pooled drink+eat generator"}
    for col, axis_name in enumerate(("x axis", "y axis", "z axis")):
        axes[0, col].set_title(axis_name, fontsize=11, pad=7)

    for class_row, cname in enumerate(("drink", "eat")):
        key = ("per_class" if selected[cname]["per_class"]["z_scored_rmse"]
               <= selected[cname]["pooled"]["z_scored_rmse"] else "pooled")
        choices[cname] = key
        synthetic_col = 1 if key == "per_class" else 2
        real = selected[cname]["windows"][0]
        synthetic = selected[cname]["windows"][synthetic_col]
        for sensor_row, (sensor_name, sensor_slice) in enumerate(SENSORS):
            row = class_row * len(SENSORS) + sensor_row
            for col, j in enumerate(range(sensor_slice.start, sensor_slice.stop)):
                ax = axes[row, col]
                values = np.concatenate([real[:, j], synthetic[:, j]])
                lo, hi = float(values.min()), float(values.max())
                pad = 0.05 * max(hi - lo, 1e-6)
                ax.plot(t, real[:, j], color="#0072B2", lw=0.9, label="Real")
                ax.plot(t, synthetic[:, j], color="#D55E00", lw=0.9, label="Synthetic", alpha=0.9)
                ax.set_ylim(lo - pad, hi + pad)
                ax.grid(True, color="0.9", linewidth=0.45)
                if row in (2, 5):
                    ax.set_xlabel("Time (s)")
                if col == 0:
                    prefix = cname.upper() + "\n" if sensor_row == 0 else ""
                    ax.set_ylabel(prefix + sensor_name, fontsize=9)
        axes[class_row * 3, 2].text(
            0.98, 0.96,
            "%s; RMSE %.2f" % (source_titles[key], selected[cname][key]["z_scored_rmse"]),
            transform=axes[class_row * 3, 2].transAxes, ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 2},
        )

    axes[0, 0].legend(fontsize=8, loc="lower right", framealpha=0.9)
    for ax in axes[3]:
        ax.spines["top"].set_linewidth(1.8)
        ax.spines["top"].set_color("0.25")
    fig.suptitle("One real and synthetic training example per activity", fontsize=14, y=0.995)
    fig.text(0.5, 0.004,
             "Each panel shows one sensor axis; real and synthetic traces use the same scale. "
             "Signals are not smoothed.",
             ha="center", va="bottom", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0.02, 0.025, 1, 0.985), h_pad=0.65, w_pad=0.8)
    stem = os.path.join(out_dir, "r1_02_one_sample_per_activity")
    fig.savefig(stem + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return stem, choices


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--per-class-corpus", required=True, help="labelled drink/eat synthetic corpus")
    ap.add_argument("--pooled-corpus", required=True, help="unlabelled pooled drink+eat synthetic corpus")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--search", type=int, default=20000,
                    help="synthetic candidates searched per source and activity (default: 20000)")
    ap.add_argument("--real-search", type=int, default=2000,
                    help="real training candidates for representative-window selection (default: 2000)")
    ap.add_argument("--search-chunk", type=int, default=1000,
                    help="synthetic search chunk size (default: 1000)")
    ap.add_argument("--generator-window", type=int, default=24,
                    help="segment length used by the generators, for provenance validation (default: 24)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(os.path.join(args.data_dir, "split.json"))
    stats = norm.load_stats(os.path.join(args.data_dir, "norm_train.json"))
    idx = class_train_indices(meta, split)
    per_corpus, per_labels, per_manifest = load_corpus(args.per_class_corpus)
    pooled_corpus, pooled_labels, pooled_manifest = load_corpus(args.pooled_corpus)
    if set(np.unique(pooled_labels)) != {-1}:
        raise ValueError("pooled corpus must use unlabelled (-1) labels")

    train_subjects = set(split["train"])
    generator_provenance = generator_training_provenance(
        args.data_dir, args.generator_window, train_subjects)
    selected = {}
    audit = {
        "selection_method": {
            "real": "feature medoid among candidates in the 60th-80th percentile of motion energy",
            "synthetic": "nearest to the same real window by RMSE after real-train z-scoring",
        },
        "seed": args.seed,
        "data_dir": os.path.abspath(args.data_dir),
        "per_class_corpus": os.path.abspath(args.per_class_corpus),
        "pooled_corpus": os.path.abspath(args.pooled_corpus),
        "per_class_manifest": per_manifest,
        "pooled_manifest": pooled_manifest,
        "train_subject_count": len(train_subjects),
        "generator_training_provenance": generator_provenance,
        "classes": {},
    }
    pooled_idx = np.arange(len(pooled_labels))
    for cname in ("drink", "eat"):
        real_index, real, real_info = representative_real(
            X, idx[cname], stats, rng, args.real_search)
        per_idx = np.flatnonzero(per_labels == CLASS_IDS[cname])
        if len(per_idx) == 0:
            raise ValueError("no %s windows in per-class corpus" % cname)
        per_index, per_window, per_info = nearest_synthetic(
            per_corpus, per_idx, real, stats, rng, args.search, args.search_chunk)
        pooled_index, pooled_window, pooled_info = nearest_synthetic(
            pooled_corpus, pooled_idx, real, stats, rng, args.search, args.search_chunk)
        subject = str(meta.iloc[real_index]["subject"])
        if subject not in train_subjects:
            raise RuntimeError("selected real window is not in the training split: %s" % subject)
        selected[cname] = {
            "windows": (real, per_window, pooled_window),
            "per_class": per_info,
            "pooled": pooled_info,
        }
        audit["classes"][cname] = {
            "real": {
                "dataset_index": real_index,
                "subject": subject,
                "timestamp": meta.iloc[real_index]["timestamp"],
                "activity": str(meta.iloc[real_index]["act_name"]),
                "split": "train",
                **real_info,
            },
            "per_class_synthetic": {"corpus_index": per_index, **per_info},
            "pooled_synthetic": {"corpus_index": pooled_index, **pooled_info},
            "lower_rmse_source": ("per_class" if per_info["z_scored_rmse"] <= pooled_info["z_scored_rmse"]
                                  else "pooled"),
        }

    one_sample_stem, choices = plot_one_sample_per_activity(selected, args.out_dir, args.dpi)
    audit["selected_sources"] = choices
    audit["outputs"] = [
        os.path.abspath(one_sample_stem + ".png"),
        os.path.abspath(one_sample_stem + ".pdf"),
    ]
    write_json(os.path.join(args.out_dir, "selection.json"), audit)
    print("[plot] one sample per activity: %s.{png,pdf}" % one_sample_stem)
    print("[plot] audit: %s" % os.path.join(args.out_dir, "selection.json"))


if __name__ == "__main__":
    main()
