#!/usr/bin/env python
"""Plot complete train recordings and their first/last 30 seconds.

A recording is defined by (activity, subject, moment). The deduplicated DEO
dataset contains consecutive, non-overlapping 5-second windows; this script
orders them by their start timestamp and concatenates their 500 x 9 signals.
Only DRINK and EAT recordings belonging to training subjects are eligible.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deo import AXIS_NAMES, io, splits  # noqa: E402


ACTIVITIES = (("DRINK", "Drinking"), ("EAT", "Eating"))
SENSOR_COLORS = ("#0072B2",) * 3 + ("#D55E00",) * 3 + ("#009E73",) * 3
FS = 100


def recording_candidates(meta, split):
    train = set(split["train"])
    eligible = meta[
        meta["subject"].isin(train) & meta["act_name"].isin([a for a, _ in ACTIVITIES])
    ]
    candidates = {}
    for activity, _ in ACTIVITIES:
        rows = []
        for (subject, moment), group in eligible[eligible.act_name == activity].groupby(
            ["subject", "moment"], sort=True
        ):
            ordered = group.sort_values(["timestamp"], kind="stable")
            timestamp = ordered.timestamp.to_numpy(dtype=np.int64)
            gaps = np.diff(timestamp)
            if len(ordered) >= 2 and (gaps <= 6).all() and (gaps >= 5).all():
                rows.append((str(subject), str(moment), ordered.index.to_numpy(dtype=np.int64)))
        candidates[activity] = rows
    return candidates


def select_recordings(candidates, seed):
    """Select two recordings per activity, using four distinct random subjects."""
    rng = np.random.default_rng(seed)
    selected = {}
    used_subjects = set()
    for activity, _ in ACTIVITIES:
        by_subject = {}
        for subject, moment, indices in candidates[activity]:
            by_subject.setdefault(subject, []).append((moment, indices))
        subjects = np.array(sorted(set(by_subject) - used_subjects), dtype=object)
        chosen_subjects = rng.choice(subjects, size=2, replace=False)
        records = []
        for subject in chosen_subjects:
            options = by_subject[str(subject)]
            moment, indices = options[int(rng.integers(len(options)))]
            records.append({
                "activity": activity,
                "subject": str(subject),
                "moment": moment,
                "dataset_indices": indices,
            })
            used_subjects.add(str(subject))
        selected[activity] = records
    return selected


def load_recording(X, meta, record):
    indices = record["dataset_indices"]
    rows = meta.loc[indices].sort_values("timestamp", kind="stable")
    indices = rows.index.to_numpy(dtype=np.int64)
    signal = io.take(X, indices).reshape(-1, X.shape[-1])
    time = np.arange(len(signal), dtype=np.float64) / FS
    timestamp = rows.timestamp.to_numpy(dtype=np.int64)
    gaps = np.diff(timestamp)
    output = dict(record)
    output.update({
        "dataset_indices": indices,
        "signal": signal,
        "time": time,
        "n_windows": int(len(indices)),
        "duration_seconds": float(len(signal) / FS),
        "first_timestamp": int(timestamp[0]),
        "last_timestamp": int(timestamp[-1]),
        "timestamp_gap_min_seconds": int(gaps.min()),
        "timestamp_gap_max_seconds": int(gaps.max()),
    })
    return output


def axis_label(name):
    return name.replace("_", " ")


def plot_complete(activity_display, recordings, output_stem, boundary_seconds):
    fig, axes = plt.subplots(
        len(AXIS_NAMES), 2, figsize=(16, 19), sharex="col", sharey="row",
        constrained_layout=True,
    )
    for col, record in enumerate(recordings):
        t = record["time"]
        signal = record["signal"]
        duration = record["duration_seconds"]
        for row, name in enumerate(AXIS_NAMES):
            ax = axes[row, col]
            ax.plot(t, signal[:, row], color=SENSOR_COLORS[row], lw=0.45, rasterized=True)
            ax.axvspan(0, min(boundary_seconds, duration), color="#56B4E9", alpha=0.14)
            ax.axvspan(max(0, duration - boundary_seconds), duration, color="#E69F00", alpha=0.14)
            ax.grid(alpha=0.15, lw=0.5)
            if col == 0:
                ax.set_ylabel(axis_label(name))
            if row == 0:
                ax.set_title(
                    f"{record['subject']} · {record['moment']} · "
                    f"{record['n_windows']} windows · {duration:.0f} s"
                )
        axes[-1, col].set_xlabel("Time from recording start (s)")
    fig.suptitle(
        f"{activity_display}: two random complete recordings from the deduplicated train split\n"
        f"blue shading = first {boundary_seconds:g} s; orange shading = last {boundary_seconds:g} s",
        fontsize=15,
    )
    fig.savefig(output_stem.with_name(output_stem.name + "_complete.png"), dpi=180)
    fig.savefig(output_stem.with_name(output_stem.name + "_complete.pdf"))
    plt.close(fig)


def plot_boundaries(activity_display, recordings, output_stem, boundary_seconds):
    fig, axes = plt.subplots(
        len(AXIS_NAMES), 4, figsize=(20, 19), sharex="col", sharey="row",
        constrained_layout=True,
    )
    for rec_i, record in enumerate(recordings):
        signal = record["signal"]
        n = min(int(boundary_seconds * FS), len(signal))
        views = (
            (signal[:n], np.arange(n) / FS, "start"),
            (signal[-n:], np.arange(n) / FS, "end"),
        )
        for view_i, (part, t, boundary) in enumerate(views):
            col = 2 * rec_i + view_i
            for row, name in enumerate(AXIS_NAMES):
                ax = axes[row, col]
                ax.plot(t, part[:, row], color=SENSOR_COLORS[row], lw=0.65, rasterized=True)
                for mark in np.arange(0, boundary_seconds + 0.1, 5):
                    ax.axvline(mark, color="0.78", lw=0.45, zorder=0)
                ax.grid(axis="y", alpha=0.15, lw=0.5)
                if col == 0:
                    ax.set_ylabel(axis_label(name))
                if row == 0:
                    ax.set_title(f"{record['subject']} · {record['moment']} · {boundary} {boundary_seconds:g} s")
            axes[-1, col].set_xlabel(f"Seconds from {boundary}")
    fig.suptitle(
        f"{activity_display}: recording boundaries by sensor axis · deduplicated train split",
        fontsize=15,
    )
    fig.savefig(output_stem.with_name(output_stem.name + "_boundaries.png"), dpi=180)
    fig.savefig(output_stem.with_name(output_stem.name + "_boundaries.pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--boundary-seconds", type=float, default=30.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(args.data_dir / "split.json")
    candidates = recording_candidates(meta, split)
    selected = select_recordings(candidates, args.seed)
    records_for_csv = []
    manifest = {
        "data_dir": str(args.data_dir.resolve()),
        "split": "train",
        "dataset": "deo_dedup",
        "seed": args.seed,
        "boundary_seconds": args.boundary_seconds,
        "recording_definition": "activity + subject + moment, windows sorted by start timestamp",
        "activities": {},
    }

    for activity, display in ACTIVITIES:
        recordings = [load_recording(X, meta, r) for r in selected[activity]]
        stem = args.out_dir / activity.lower()
        plot_complete(display, recordings, stem, args.boundary_seconds)
        plot_boundaries(display, recordings, stem, args.boundary_seconds)
        manifest["activities"][activity] = []
        for record in recordings:
            public = {k: v for k, v in record.items() if k not in {"signal", "time"}}
            public["dataset_indices"] = public["dataset_indices"].tolist()
            manifest["activities"][activity].append(public)
            records_for_csv.append({k: v for k, v in public.items() if k != "dataset_indices"})

    (args.out_dir / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pd.DataFrame(records_for_csv).to_csv(args.out_dir / "selection.csv", index=False)
    print(pd.DataFrame(records_for_csv).to_string(index=False))
    print(f"[done] plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
