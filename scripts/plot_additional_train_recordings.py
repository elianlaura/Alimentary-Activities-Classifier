#!/usr/bin/env python
"""Plot additional random EAT/DRINK train recordings, one figure per recording."""
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
from scripts.plot_train_recording_boundaries import (  # noqa: E402
    ACTIVITIES,
    FS,
    SENSOR_COLORS,
    axis_label,
    load_recording,
    recording_candidates,
)


def excluded_subjects(path):
    if path is None:
        return set()
    manifest = json.loads(path.read_text())
    return {
        str(row["subject"])
        for records in manifest.get("activities", {}).values()
        for row in records
    }


def select(candidates, seed, n_per_activity, excluded):
    """Choose distinct subjects across all activities and prior selections."""
    rng = np.random.default_rng(seed)
    used = set(excluded)
    selected = {}
    for activity, _ in ACTIVITIES:
        by_subject = {}
        for subject, moment, indices in candidates[activity]:
            by_subject.setdefault(subject, []).append((moment, indices))
        available = np.array(sorted(set(by_subject) - used), dtype=object)
        if len(available) < n_per_activity:
            raise ValueError(f"Only {len(available)} eligible subjects remain for {activity}")
        chosen = rng.choice(available, size=n_per_activity, replace=False)
        records = []
        for subject in chosen:
            options = by_subject[str(subject)]
            moment, indices = options[int(rng.integers(len(options)))]
            records.append({
                "activity": activity,
                "subject": str(subject),
                "moment": str(moment),
                "dataset_indices": indices,
            })
            used.add(str(subject))
        selected[activity] = records
    return selected


def plot_recording(display, record, path, boundary_seconds):
    signal = record["signal"]
    time = record["time"]
    duration = record["duration_seconds"]
    n = min(int(boundary_seconds * FS), len(signal))
    start = signal[:n]
    end = signal[-n:]
    boundary_time = np.arange(n, dtype=np.float64) / FS

    fig, axes = plt.subplots(
        len(AXIS_NAMES), 3, figsize=(18, 19), sharey="row", constrained_layout=True
    )
    for row, name in enumerate(AXIS_NAMES):
        color = SENSOR_COLORS[row]
        axes[row, 0].plot(time, signal[:, row], color=color, lw=0.45, rasterized=True)
        axes[row, 0].axvspan(0, min(boundary_seconds, duration), color="#56B4E9", alpha=0.14)
        axes[row, 0].axvspan(
            max(0, duration - boundary_seconds), duration, color="#E69F00", alpha=0.14
        )
        axes[row, 1].plot(boundary_time, start[:, row], color=color, lw=0.65, rasterized=True)
        axes[row, 2].plot(boundary_time, end[:, row], color=color, lw=0.65, rasterized=True)
        for col in (1, 2):
            for mark in np.arange(0, boundary_seconds + 0.1, 5):
                axes[row, col].axvline(mark, color="0.78", lw=0.45, zorder=0)
        for col in range(3):
            axes[row, col].grid(axis="y", alpha=0.15, lw=0.5)
        axes[row, 0].set_ylabel(axis_label(name))

    axes[0, 0].set_title(f"Complete recording · {duration:.0f} s")
    axes[0, 1].set_title(f"First {boundary_seconds:g} s")
    axes[0, 2].set_title(f"Last {boundary_seconds:g} s")
    axes[-1, 0].set_xlabel("Time from recording start (s)")
    axes[-1, 1].set_xlabel("Seconds from start")
    axes[-1, 2].set_xlabel("Seconds from end-window start")
    fig.suptitle(
        f"{display} · {record['subject']} · {record['moment']} · "
        f"{record['n_windows']} windows · deduplicated train split",
        fontsize=15,
    )
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--n-per-activity", type=int, default=5)
    parser.add_argument("--boundary-seconds", type=float, default=30.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(args.data_dir / "split.json")
    candidates = recording_candidates(meta, split)
    prior = excluded_subjects(args.exclude_selection)
    chosen = select(candidates, args.seed, args.n_per_activity, prior)

    manifest = {
        "data_dir": str(args.data_dir.resolve()),
        "dataset": "deo_dedup",
        "split": "train",
        "seed": args.seed,
        "n_per_activity": args.n_per_activity,
        "boundary_seconds": args.boundary_seconds,
        "excluded_prior_subjects": sorted(prior),
        "recording_definition": "activity + subject + moment, windows sorted by start timestamp",
        "activities": {},
    }
    csv_rows = []
    for activity, display in ACTIVITIES:
        manifest["activities"][activity] = []
        for number, raw in enumerate(chosen[activity], start=1):
            record = load_recording(X, meta, raw)
            filename = f"{activity.lower()}_additional_{number:02d}_{record['subject']}_{record['moment']}"
            plot_recording(display, record, args.out_dir / filename, args.boundary_seconds)
            public = {k: v for k, v in record.items() if k not in {"signal", "time"}}
            public["dataset_indices"] = public["dataset_indices"].tolist()
            public["plot_stem"] = filename
            manifest["activities"][activity].append(public)
            csv_rows.append({k: v for k, v in public.items() if k != "dataset_indices"})

    (args.out_dir / "selection_additional_5.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    pd.DataFrame(csv_rows).to_csv(args.out_dir / "selection_additional_5.csv", index=False)
    print(pd.DataFrame(csv_rows).to_string(index=False))
    print(f"[done] plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
