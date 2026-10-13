#!/usr/bin/env python
"""Plot five-second medoid neighborhoods with t-SNE and exact distance CDFs.

t-SNE is used only to visualize the standardized 45-feature space. Neighborhood
membership is always the previously saved participant-weighted distance to the
real medoid, never a distance measured in the t-SNE projection.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE


ACTIVITIES = (("drink", "Drinking"), ("eat", "Eating"))
LAYERS = (
    ("outer", "Outer 25%", "#B8B8B8", 8, 0.42),
    ("shell_50_75", "50–75% shell", "#E69F00", 10, 0.58),
    ("shell_25_50", "25–50% shell", "#56B4E9", 11, 0.62),
    ("central25", "Central 25%", "#0072B2", 12, 0.72),
)
CUTOFFS = (
    ("central25", "25%", "#0072B2"),
    ("central50", "50%", "#56B4E9"),
    ("central75", "75%", "#E69F00"),
)


def layer_masks(table):
    c25 = table.in_central25.to_numpy(dtype=bool)
    c50 = table.in_central50.to_numpy(dtype=bool)
    c75 = table.in_central75.to_numpy(dtype=bool)
    assert np.all(c25 <= c50) and np.all(c50 <= c75)
    return {
        "central25": c25,
        "shell_25_50": c50 & ~c25,
        "shell_50_75": c75 & ~c50,
        "outer": ~c75,
    }


def draw_embedding(ax, xy, table, title, show_legend=False):
    masks = layer_masks(table)
    for key, label, color, size, alpha in LAYERS:
        mask = masks[key]
        ax.scatter(
            xy[mask, 0], xy[mask, 1], s=size, c=color, alpha=alpha,
            linewidths=0, label=label, rasterized=True,
        )
    medoid = int(np.flatnonzero(table.is_medoid.to_numpy(dtype=bool))[0])
    ax.scatter(
        xy[medoid, 0], xy[medoid, 1], marker="*", s=360,
        c="#D55E00", edgecolors="black", linewidths=1.2,
        label="Real medoid", zorder=20,
    )
    ax.set_title(f"{title}: t-SNE of 45-feature space")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
                   markeredgewidth=0, markersize=7, label=label)
            for _, label, color, _, _ in reversed(LAYERS)
        ]
        handles.append(
            Line2D([0], [0], marker="*", linestyle="", markerfacecolor="#D55E00",
                   markeredgecolor="black", markersize=14, label="Real medoid")
        )
        ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)


def draw_distance_cdf(ax, table, class_record, title):
    ordered = table.sort_values(["distance_to_medoid", "dataset_index"], kind="stable")
    distance = ordered.distance_to_medoid.to_numpy()
    cumulative = np.cumsum(ordered.participant_weight.to_numpy())
    x = np.log1p(distance)
    ax.plot(x, cumulative, color="black", lw=1.7, label="Weighted empirical CDF")
    for subset, label, color in CUTOFFS:
        record = class_record["subsets"][subset]
        cutoff_x = np.log1p(record["distance_cutoff"])
        achieved = record["achieved_participant_weight"]
        ax.axvline(cutoff_x, color=color, lw=1.4, ls="--")
        ax.scatter(cutoff_x, achieved, color=color, s=32, zorder=5)
        ax.annotate(
            label, (cutoff_x, achieved), xytext=(4, -12),
            textcoords="offset points", color=color, fontsize=9,
        )
    first = ordered.iloc[0]
    assert bool(first.is_medoid) and np.isclose(first.distance_to_medoid, 0.0)
    ax.scatter(0, first.participant_weight, marker="*", s=150, c="#D55E00",
               edgecolors="black", linewidths=0.8, zorder=8)
    ax.set_title(f"{title}: exact distance distribution")
    ax.set_xlabel("log(1 + standardized 45-feature distance to medoid)")
    ax.set_ylabel("Cumulative participant-balanced weight")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.grid(True, alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=50.0)
    parser.add_argument("--iterations", type=int, default=1500)
    args = parser.parse_args()
    out_dir = args.out_dir or args.input_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads((args.input_dir / "selection.json").read_text())
    results = {}
    for activity, display in ACTIVITIES:
        feature_data = np.load(args.input_dir / f"{activity}_feature_space.npz")
        features = feature_data["standardized_features"]
        dataset_indices = feature_data["dataset_indices"]
        table = pd.read_csv(args.input_dir / f"{activity}_ranking.csv").sort_values(
            "candidate_id", kind="stable"
        ).reset_index(drop=True)
        assert np.array_equal(table.dataset_index.to_numpy(), dataset_indices)
        assert len(features) == len(table)
        print(f"[{activity}] fitting t-SNE to {len(features)} windows", flush=True)
        model = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            learning_rate="auto",
            max_iter=args.iterations,
            init="pca",
            metric="euclidean",
            method="barnes_hut",
            angle=0.5,
            random_state=args.seed,
            n_jobs=1,
            verbose=1,
        )
        xy = model.fit_transform(features)
        assert np.isfinite(xy).all()
        coordinate_table = table[[
            "candidate_id", "dataset_index", "subject", "timestamp",
            "distance_to_medoid", "is_medoid", "in_central25",
            "in_central50", "in_central75", "in_full",
        ]].copy()
        coordinate_table["tsne_1"] = xy[:, 0]
        coordinate_table["tsne_2"] = xy[:, 1]
        coordinate_path = out_dir / f"{activity}_tsne_coordinates.csv"
        coordinate_table.to_csv(coordinate_path, index=False)
        results[activity] = {
            "display": display,
            "xy": xy,
            "table": table,
            "coordinates": str(coordinate_path.resolve()),
            "kl_divergence": float(model.kl_divergence_),
            "iterations_completed": int(model.n_iter_),
        }

        figure, axis = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
        draw_embedding(axis, xy, table, display, show_legend=True)
        figure.text(
            0.5, 0.005,
            "Visualization only: neighborhood membership and medoid were fixed in the standardized 45D space, not in t-SNE.",
            ha="center", fontsize=8.5,
        )
        figure.savefig(out_dir / f"{activity}_tsne_neighborhoods.png", dpi=220, bbox_inches="tight")
        figure.savefig(out_dir / f"{activity}_tsne_neighborhoods.pdf", bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 11.0), constrained_layout=True)
    for row, (activity, display) in enumerate(ACTIVITIES):
        result = results[activity]
        draw_embedding(axes[row, 0], result["xy"], result["table"], display, show_legend=(row == 0))
        draw_distance_cdf(axes[row, 1], result["table"], selection["classes"][activity], display)
    figure.suptitle("Five-second real training neighborhoods around each activity medoid", fontsize=16)
    figure.text(
        0.5, -0.005,
        "t-SNE shows local feature-space structure; the right panels show the exact participant-weighted selection distances.",
        ha="center", fontsize=9,
    )
    figure.savefig(out_dir / "medoid_neighborhood_distributions.png", dpi=220, bbox_inches="tight")
    figure.savefig(out_dir / "medoid_neighborhood_distributions.pdf", bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "input_dir": str(args.input_dir.resolve()),
        "method": "t-SNE applied separately per activity to the saved standardized 45-feature vectors",
        "selection_warning": "t-SNE is visualization only; medoids and neighborhoods were selected before projection in 45D",
        "parameters": {
            "random_state": args.seed,
            "perplexity": args.perplexity,
            "max_iter": args.iterations,
            "learning_rate": "auto",
            "init": "pca",
            "metric": "euclidean",
            "method": "barnes_hut",
            "angle": 0.5,
        },
        "activities": {
            activity: {
                "n_windows": int(len(result["table"])),
                "kl_divergence": result["kl_divergence"],
                "iterations_completed": result["iterations_completed"],
                "coordinates": result["coordinates"],
            }
            for activity, result in results.items()
        },
    }
    with open(out_dir / "plot_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
