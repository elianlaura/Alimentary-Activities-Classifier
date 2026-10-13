#!/usr/bin/env python
"""Analyze completed W500 BiGRU generators trained on medoid neighborhoods.

The analysis keeps the original real medoid and the original standardized
45-feature space fixed. Synthetic samples never redefine the medoid, feature
scaling, or neighborhood cutoffs.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deo import AXIS_NAMES, norm  # noqa: E402
from scripts.build_medoid_neighborhoods_5s import signal_features  # noqa: E402


ACTIVITIES = (("drink", "Drinking"), ("eat", "Eating"))
SUBSETS = (("central25", "Central 25%"), ("central50", "Central 50%"),
           ("central75", "Central 75%"), ("full", "Full"))
COLORS = {"central25": "#0072B2", "central50": "#56B4E9",
          "central75": "#E69F00", "full": "#666666"}


def boolean_column(series):
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().eq("true").to_numpy()


def weighted_cdf(values, weights):
    order = np.argsort(values, kind="stable")
    x = np.asarray(values)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    return x, np.cumsum(w) / w.sum()


def q(values, probabilities=(0.1, 0.5, 0.9)):
    return np.quantile(values, probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads((args.input_dir / "selection.json").read_text())
    stats = norm.load_stats(Path(selection["normalization_path"]))
    metric_rows = []
    runtime_rows = []
    cached = {}

    for activity, display in ACTIVITIES:
        feature_data = np.load(args.input_dir / f"{activity}_feature_space.npz")
        real_feature_space = feature_data["standardized_features"].astype(np.float64)
        feature_center = feature_data["feature_center"].astype(np.float64)
        feature_scale = feature_data["feature_scale"].astype(np.float64)
        medoid_candidate = int(feature_data["medoid_candidate"])
        medoid_feature = real_feature_space[medoid_candidate]
        real_z_features = real_feature_space * np.sqrt(real_feature_space.shape[1])

        table = pd.read_csv(args.input_dir / f"{activity}_ranking.csv")
        table = table.sort_values("candidate_id", kind="stable").reset_index(drop=True)
        assert len(table) == len(real_feature_space)
        full_real = np.load(
            args.input_dir / "generator_inputs" / f"{activity}_full_w500.npy",
            mmap_mode="r",
        )
        full_meta = pd.read_csv(
            args.input_dir / "generator_inputs" / f"{activity}_full_w500.csv"
        )
        medoid_rows = np.flatnonzero(boolean_column(full_meta["is_medoid"]))
        assert len(medoid_rows) == 1
        real_medoid_signal = np.asarray(full_real[medoid_rows[0]], dtype=np.float32)

        cutoffs = {
            name: float(selection["classes"][activity]["subsets"][name]["distance_cutoff"])
            for name, _ in SUBSETS
        }
        cached[activity] = {
            "display": display,
            "table": table,
            "cutoffs": cutoffs,
            "real_medoid_signal": real_medoid_signal,
            "subsets": {},
        }

        for subset, subset_display in SUBSETS:
            run_dir = args.runs_dir / "generators" / f"gru_{activity}_{subset}_w500"
            sample_path = run_dir / "samples_2x.npy"
            samples = np.load(sample_path, mmap_mode="r")
            assert samples.shape[1:] == (500, 9)
            assert np.isfinite(samples).all()

            print(f"[{activity}/{subset}] features for {len(samples)} synthetic windows", flush=True)
            synthetic_raw = signal_features(samples, stats)
            synthetic_z = (synthetic_raw - feature_center) / feature_scale
            synthetic_feature_space = synthetic_z / np.sqrt(synthetic_z.shape[1])
            synthetic_distance = np.linalg.norm(
                synthetic_feature_space - medoid_feature[None, :], axis=1
            )

            mask = boolean_column(table[f"in_{subset}"])
            real_distance = table.loc[mask, "distance_to_medoid"].to_numpy(dtype=np.float64)
            real_weights = table.loc[mask, "participant_weight"].to_numpy(dtype=np.float64)
            real_z = real_z_features[mask]
            real_q = q(real_distance)
            synthetic_q = q(synthetic_distance)
            nearest_index = int(np.argmin(synthetic_distance))
            nearest_signal = np.asarray(samples[nearest_index], dtype=np.float32)
            one_d_wasserstein = np.array([
                wasserstein_distance(real_z[:, j], synthetic_z[:, j])
                for j in range(real_z.shape[1])
            ])

            train_done = json.loads((run_dir / "train_done.json").read_text())
            sample_done = json.loads((run_dir / "samples_2x.json").read_text())
            loss = pd.read_csv(run_dir / "loss.csv")
            tail = loss.tail(max(10, len(loss) // 10))
            runtime_rows.append({
                "activity": activity,
                "neighborhood": subset,
                "train_windows": int(mask.sum()),
                "steps": int(train_done["steps"]),
                "train_hours": float(train_done["seconds_this_run"]) / 3600,
                "seconds_per_step": float(train_done["seconds_this_run"]) / int(train_done["steps"]),
                "synthetic_windows": int(sample_done["n"]),
                "sample_minutes": float(sample_done["seconds_this_run"]) / 60,
                "seconds_per_synthetic_window": float(sample_done["seconds_this_run"]) / int(sample_done["n"]),
                "final_total_loss": float(loss.iloc[-1]["total"]),
                "last_10pct_total_loss_mean": float(tail["total"].mean()),
                "last_10pct_total_loss_std": float(tail["total"].std()),
            })
            metric_rows.append({
                "activity": activity,
                "neighborhood": subset,
                "real_windows": int(mask.sum()),
                "synthetic_windows": int(len(samples)),
                "real_distance_p10": float(real_q[0]),
                "real_distance_median": float(real_q[1]),
                "real_distance_p90": float(real_q[2]),
                "synthetic_distance_p10": float(synthetic_q[0]),
                "synthetic_distance_median": float(synthetic_q[1]),
                "synthetic_distance_p90": float(synthetic_q[2]),
                "synthetic_to_real_median_ratio": float(synthetic_q[1] / max(real_q[1], 1e-12)),
                "synthetic_within_central25_pct": float(100 * np.mean(synthetic_distance <= cutoffs["central25"])),
                "synthetic_within_central50_pct": float(100 * np.mean(synthetic_distance <= cutoffs["central50"])),
                "synthetic_within_central75_pct": float(100 * np.mean(synthetic_distance <= cutoffs["central75"])),
                "mean_feature_shift_z_rmse": float(np.sqrt(np.mean((synthetic_z.mean(0) - real_z.mean(0)) ** 2))),
                "mean_feature_wasserstein_z": float(one_d_wasserstein.mean()),
                "nearest_synthetic_index": nearest_index,
                "nearest_synthetic_distance": float(synthetic_distance[nearest_index]),
                "finite_fraction": 1.0,
            })
            cached[activity]["subsets"][subset] = {
                "display": subset_display,
                "real_distance": real_distance,
                "real_weights": real_weights,
                "synthetic_distance": synthetic_distance,
                "nearest_signal": nearest_signal,
                "nearest_index": nearest_index,
            }
            np.savez_compressed(
                args.out_dir / f"{activity}_{subset}_synthetic_features.npz",
                synthetic_distance_to_real_medoid=synthetic_distance.astype(np.float32),
                nearest_synthetic_index=np.array(nearest_index),
                nearest_synthetic_distance=np.array(synthetic_distance[nearest_index]),
            )

    metrics = pd.DataFrame(metric_rows)
    runtimes = pd.DataFrame(runtime_rows)
    metrics.to_csv(args.out_dir / "synthetic_medoid_metrics.csv", index=False)
    runtimes.to_csv(args.out_dir / "job_results.csv", index=False)

    # Real-versus-synthetic medoid-distance distributions.
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True, sharey=True)
    for row, (activity, display) in enumerate(ACTIVITIES):
        for col, (subset, subset_display) in enumerate(SUBSETS):
            ax = axes[row, col]
            item = cached[activity]["subsets"][subset]
            rx, ry = weighted_cdf(item["real_distance"], item["real_weights"])
            sx, sy = weighted_cdf(item["synthetic_distance"], np.ones(len(item["synthetic_distance"])))
            ax.plot(np.log1p(rx), ry, color="#0072B2", lw=2, label="Real training")
            ax.plot(np.log1p(sx), sy, color="#D55E00", lw=2, label="Synthetic 2x")
            if subset != "full":
                ax.axvline(np.log1p(cached[activity]["cutoffs"][subset]), color="black", ls="--", lw=1)
            row_metric = metrics[(metrics.activity == activity) & (metrics.neighborhood == subset)].iloc[0]
            ax.set_title(
                f"{display} - {subset_display}\n"
                f"synthetic median / real median = {row_metric.synthetic_to_real_median_ratio:.2f}"
            )
            ax.set_xlabel("log(1 + distance to real medoid)")
            if col == 0:
                ax.set_ylabel("Cumulative proportion")
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(loc="lower right")
    fig.suptitle("BiGRU synthetic windows versus their real medoid neighborhoods", fontsize=16)
    fig.savefig(args.out_dir / "synthetic_medoid_distance_cdfs.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.out_dir / "synthetic_medoid_distance_cdfs.pdf", bbox_inches="tight")
    plt.close(fig)

    # Training losses; raw points are noisy, so display a 20-record rolling mean.
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    for row, (activity, display) in enumerate(ACTIVITIES):
        for col, (subset, subset_display) in enumerate(SUBSETS):
            ax = axes[row, col]
            loss_path = args.runs_dir / "generators" / f"gru_{activity}_{subset}_w500" / "loss.csv"
            loss = pd.read_csv(loss_path)
            ax.plot(loss.step, loss.total, color=COLORS[subset], alpha=0.18, lw=0.7)
            ax.plot(loss.step, loss.total.rolling(20, min_periods=1).mean(), color=COLORS[subset], lw=1.7)
            ax.set_yscale("log")
            ax.set_title(f"{display} - {subset_display}")
            ax.set_xlabel("Training step")
            if col == 0:
                ax.set_ylabel("MSE + weighted MMD loss")
            ax.grid(alpha=0.25)
    fig.suptitle("PaD-TS BiGRU training losses", fontsize=16)
    fig.savefig(args.out_dir / "training_losses.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.out_dir / "training_losses.pdf", bbox_inches="tight")
    plt.close(fig)

    # Nine-axis comparisons of the fixed real medoid and nearest generated sample.
    time = np.arange(500) / 100.0
    for activity, display in ACTIVITIES:
        real = cached[activity]["real_medoid_signal"]
        for subset, subset_display in SUBSETS:
            item = cached[activity]["subsets"][subset]
            synthetic = item["nearest_signal"]
            fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True, sharex=True)
            for axis_i, ax in enumerate(axes.flat):
                ax.plot(time, real[:, axis_i], color="#0072B2", lw=1.2, label="Real medoid")
                ax.plot(time, synthetic[:, axis_i], color="#D55E00", lw=1.1, label="Nearest synthetic")
                ax.set_title(str(AXIS_NAMES[axis_i]))
                ax.grid(alpha=0.25)
                if axis_i >= 6:
                    ax.set_xlabel("Time (s)")
                if axis_i % 3 == 0:
                    ax.set_ylabel("Sensor value")
            axes[0, 0].legend(loc="best")
            distance = metrics[
                (metrics.activity == activity) & (metrics.neighborhood == subset)
            ].iloc[0].nearest_synthetic_distance
            fig.suptitle(
                f"{display} - {subset_display}: real medoid and nearest BiGRU sample\n"
                f"synthetic row {item['nearest_index']}, 45-feature distance = {distance:.3f}",
                fontsize=15,
            )
            stem = args.out_dir / f"{activity}_{subset}_nearest_synthetic"
            fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)

    # Compact human-readable report.
    lines = [
        "# Results: W500 BiGRU medoid-neighborhood generators",
        "",
        "All eight BiGRU generators and all eight 2x sampling jobs completed on an NVIDIA H200.",
        "Attention generators were not part of these jobs. No classifier pretraining, fine-tuning,",
        "or balanced-accuracy evaluation has been run for these neighborhood corpora yet.",
        "",
        "## Job outputs",
        "",
        "| Activity | Neighborhood | Real N | Steps | Train h | Synthetic N | Sample min | s/step |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in runtimes.itertuples(index=False):
        lines.append(
            f"| {row.activity} | {row.neighborhood} | {row.train_windows:,} | {row.steps:,} | "
            f"{row.train_hours:.2f} | {row.synthetic_windows:,} | {row.sample_minutes:.2f} | "
            f"{row.seconds_per_step:.3f} |"
        )
    lines += [
        "",
        "## Direct synthetic-data diagnostics",
        "",
        "Distances use the original real medoid and the original standardized 45-feature space.",
        "Lower median ratio and lower feature Wasserstein distance mean closer agreement with the",
        "real training neighborhood; these diagnostics do not measure downstream classifier accuracy.",
        "",
        "| Activity | Neighborhood | Syn/real median distance | Within C25 | Within C50 | Within C75 | Mean feature W1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.activity} | {row.neighborhood} | {row.synthetic_to_real_median_ratio:.3f} | "
            f"{row.synthetic_within_central25_pct:.1f}% | {row.synthetic_within_central50_pct:.1f}% | "
            f"{row.synthetic_within_central75_pct:.1f}% | {row.mean_feature_wasserstein_z:.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "These completed jobs establish that the generators trained and produced finite W500 signals.",
        "They do not yet establish which neighborhood improves the eating/drinking classifier. That",
        "requires building x1/x2 corpora followed by the same pretraining, fine-tuning, and subject-level",
        "validation/test evaluation for every neighborhood.",
        "",
    ]
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines))
    print(f"[done] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
