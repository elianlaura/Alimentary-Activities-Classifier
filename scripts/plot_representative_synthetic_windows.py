#!/usr/bin/env python
"""Plot representative PaD-TS BiGRU five-second EAT/DRINK windows.

Representativeness is measured in the fixed real 45-feature medoid space.  The
script uses the five synthetic windows with the smallest distance to the real
activity medoid; it does not assign synthetic users or recording identities.
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
from deo import AXIS_NAMES


ACTIVITIES = (("drink", "Drinking"), ("eat", "Eating"))
SENSOR_COLORS = ("#0072B2",) * 3 + ("#D55E00",) * 3 + ("#009E73",) * 3
UNITS = ("m/s²",) * 3 + ("deg/s",) * 3 + ("µT",) * 3
FS = 100


def plot_window(signal, activity, display, rank, sample_index, distance, output_stem):
    time = np.arange(len(signal), dtype=np.float64) / FS
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, constrained_layout=True)
    for axis_i, ax in enumerate(axes.flat):
        ax.plot(time, signal[:, axis_i], color=SENSOR_COLORS[axis_i], lw=1.0)
        ax.set_title(str(AXIS_NAMES[axis_i]).replace("_", " "))
        ax.set_ylabel(UNITS[axis_i])
        ax.set_xlim(0, len(signal) / FS)
        ax.grid(alpha=0.25, lw=0.6)
        if axis_i >= 6:
            ax.set_xlabel("Time (s)")
    fig.suptitle(
        f"Synthetic {display} #{rank} · full-neighborhood PaD-TS BiGRU\n"
        f"sample index {sample_index} · medoid distance {distance:.3f}",
        fontsize=14,
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-per-activity", type=int, default=5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generator": "PaD-TS BiGRU",
        "neighborhood": "full",
        "window_seconds": 5,
        "sampling_rate_hz": FS,
        "n_axes": len(AXIS_NAMES),
        "n_per_activity": args.n_per_activity,
        "selection": "smallest distance to the fixed real activity medoid in the standardized 45-feature space",
        "interpretation": "Independent synthetic windows; no participant or recording identity and no one-to-one match to real recordings",
        "activities": {},
    }
    rows = []
    for activity, display in ACTIVITIES:
        generator_dir = args.runs_dir / "generators" / f"gru_{activity}_full_w500"
        sample_path = generator_dir / "samples_2x.npy"
        feature_path = args.feature_dir / f"{activity}_full_synthetic_features.npz"
        samples = np.load(sample_path, mmap_mode="r")
        feature_data = np.load(feature_path)
        distances = feature_data["synthetic_distance_to_real_medoid"].astype(np.float64)
        if len(samples) != len(distances):
            raise ValueError(f"{activity}: {len(samples)} samples but {len(distances)} distances")
        order = np.argsort(distances, kind="stable")[: args.n_per_activity]
        metadata = json.loads((generator_dir / "samples_2x.json").read_text())
        manifest["activities"][activity] = {
            "samples_path": str(sample_path.resolve()),
            "checkpoint": metadata["checkpoint"],
            "selected": [],
        }
        for rank, sample_index in enumerate(order, start=1):
            sample_index = int(sample_index)
            distance = float(distances[sample_index])
            stem_name = f"synthetic_{activity}_representative_{rank:02d}_full_idx{sample_index:05d}"
            plot_window(
                np.asarray(samples[sample_index], dtype=np.float32),
                activity,
                display,
                rank,
                sample_index,
                distance,
                args.out_dir / stem_name,
            )
            item = {
                "activity": activity,
                "representative_rank": rank,
                "synthetic_index": sample_index,
                "distance_to_real_medoid": distance,
                "plot_stem": stem_name,
            }
            manifest["activities"][activity]["selected"].append(item)
            rows.append(item)

    (args.out_dir / "synthetic_representative_selection.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    pd.DataFrame(rows).to_csv(
        args.out_dir / "synthetic_representative_selection.csv", index=False
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[done] plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
