#!/usr/bin/env python
"""Build train-only 5-s medoid neighborhoods separately for DRINK and EAT.

For each activity, this script:
  * gives every training participant equal total weight;
  * computes 45 signal descriptors for every native 5-s (500 x 9) window;
  * selects the exact weighted global medoid in the standardized feature space;
  * ranks windows by distance to that medoid; and
  * materializes nested central-25%, central-50%, central-75%, and full arrays.

The central percentages refer to cumulative participant-balanced weight, not to
percentages of windows or participants.  Synthetic samples are not generated.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deo import AXIS_NAMES, N_FEATURES, N_TIMESTEPS, io, norm, splits  # noqa: E402
from deo.corpora import class_train_indices  # noqa: E402
from deo.utils import write_json  # noqa: E402


FS = 100
ACTIVITIES = {"drink": 0, "eat": 1}
FEATURE_GROUPS = ["mean", "log_std", "log_diff_rms", "power_0.1_1Hz", "power_1_5Hz"]
FEATURE_NAMES = [f"{group}:{axis}" for group in FEATURE_GROUPS for axis in AXIS_NAMES]
SUBSETS = (
    ("central25", 0.25),
    ("central50", 0.50),
    ("central75", 0.75),
    ("full", 1.00),
)


def sha256_file(path, block_size=64 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def signal_features(windows, stats, chunk_size=128):
    """Return 9 axes x 5 descriptors for native five-second windows."""
    output = []
    for start in range(0, len(windows), chunk_size):
        z = norm.apply(windows[start:start + chunk_size], stats).astype(np.float64)
        level = z.mean(axis=1)
        variation = np.log(np.maximum(z.std(axis=1), 1e-6))
        roughness = np.log(
            np.maximum(np.sqrt(np.mean(np.diff(z, axis=1) ** 2, axis=1)), 1e-6)
        )
        # A 500-point segment gives 0.2-Hz resolution. With one segment this is
        # equivalent to a windowed periodogram; noverlap remains valid (<500).
        frequency, power = welch(
            z, fs=FS, nperseg=N_TIMESTEPS, noverlap=N_TIMESTEPS // 2,
            axis=1, detrend="constant"
        )
        total = power[:, (frequency >= 0.1) & (frequency <= 20)].sum(axis=1)
        bands = [
            power[:, (frequency >= low) & (frequency < high)].sum(axis=1)
            / np.maximum(total, 1e-12)
            for low, high in ((0.1, 1.0), (1.0, 5.0))
        ]
        output.append(np.concatenate([level, variation, roughness, *bands], axis=1))
    result = np.concatenate(output)
    assert result.shape == (len(windows), len(FEATURE_NAMES))
    assert np.isfinite(result).all()
    return result


def participant_weights(subjects):
    subjects = pd.Series(subjects, dtype=str)
    counts = subjects.value_counts()
    n_participants = len(counts)
    weights = subjects.map(lambda subject: 1.0 / (n_participants * counts[subject])).to_numpy()
    assert np.isclose(weights.sum(), 1.0)
    totals = pd.Series(weights).groupby(subjects).sum()
    assert np.allclose(totals.to_numpy(), 1.0 / n_participants)
    return weights


def standardize_features(raw_features, weights):
    center = weights @ raw_features
    scale = np.sqrt(weights @ ((raw_features - center) ** 2))
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (raw_features - center) / scale / np.sqrt(raw_features.shape[1])
    assert np.isfinite(standardized).all()
    return standardized, center, scale


def exact_weighted_medoid(features, weights, block_size=256):
    """Find argmin_j sum_i weights_i * distance(i,j), without a dense NxN matrix."""
    costs = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), block_size):
        stop = min(start + block_size, len(features))
        costs[start:stop] = weights @ cdist(features, features[start:stop], metric="euclidean")
        if start == 0 or stop == len(features) or stop % 1024 == 0:
            print(f"    medoid costs {stop}/{len(features)}", flush=True)
    medoid = int(np.argmin(costs))
    distance = cdist(features, features[[medoid]], metric="euclidean")[:, 0]
    return medoid, distance, costs


def neighborhood(distance, weights, dataset_indices, target):
    """Inclusive distance cutoff at the first rank reaching target weight."""
    order = np.lexsort((dataset_indices, distance))
    cumulative = np.cumsum(weights[order])
    if target == 1.0:
        cutoff = float(distance[order[-1]])
        selected = np.ones(len(distance), dtype=bool)
    else:
        crossing = int(np.searchsorted(cumulative, target, side="left"))
        cutoff = float(distance[order[crossing]])
        selected = distance <= cutoff  # include exact boundary ties
    achieved = float(weights[selected].sum())
    strictly_below = float(weights[distance < cutoff].sum())
    assert achieved >= target - 1e-12
    if target < 1.0:
        assert strictly_below < target + 1e-12
    return selected, cutoff, achieved, strictly_below, order, cumulative


def write_windows(path, source, dataset_indices, chunk_size=512):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32,
        shape=(len(dataset_indices), N_TIMESTEPS, N_FEATURES)
    )
    for start in range(0, len(dataset_indices), chunk_size):
        stop = min(start + chunk_size, len(dataset_indices))
        block = np.asarray(source[dataset_indices[start:stop]], dtype=np.float32)
        output[start:stop] = block
        assert np.array_equal(output[start:stop], block)
    output.flush()
    del output
    saved = np.load(path, mmap_mode="r")
    assert saved.shape == (len(dataset_indices), N_TIMESTEPS, N_FEATURES)
    if len(saved):
        assert np.array_equal(saved[0], source[dataset_indices[0]])
        assert np.array_equal(saved[-1], source[dataset_indices[-1]])
    return sha256_file(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--medoid-block-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (args.out_dir / "selection.json").exists() and not args.overwrite:
        raise SystemExit(f"Output already complete: {args.out_dir}; use --overwrite to rebuild")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generator_dir = args.out_dir / "generator_inputs"
    generator_dir.mkdir(parents=True, exist_ok=True)

    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(args.data_dir / "split.json")
    stats = norm.load_stats(args.data_dir / "norm_train.json")
    class_indices = class_train_indices(meta, split)
    train_subjects = set(split["train"])
    held_out = set(split["val"]) | set(split["test"])

    selection = {
        "purpose": "Train-only native five-second activity neighborhoods selected before synthetic generation",
        "data_dir": str(args.data_dir.resolve()),
        "source_shape": list(X.shape),
        "sampling_rate_hz": FS,
        "window_shape": [N_TIMESTEPS, N_FEATURES],
        "split_path": str((args.data_dir / "split.json").resolve()),
        "split_sha256": sha256_file(args.data_dir / "split.json"),
        "normalization_path": str((args.data_dir / "norm_train.json").resolve()),
        "normalization_sha256": sha256_file(args.data_dir / "norm_train.json"),
        "selection_scope": "Each activity is processed independently",
        "participant_weighting": "Within each activity, every train participant has equal total weight; each of that participant's windows shares it equally",
        "distance": "Euclidean distance after participant-weighted feature mean/std scaling, divided by sqrt(45)",
        "cutoff_rule": "Sort by (distance, dataset_index); choose the first distance reaching the target cumulative full-pool weight and include all exact boundary ties",
        "features": FEATURE_NAMES,
        "feature_details": {
            "mean": "per-axis mean after full-train z-normalization",
            "log_std": "log per-axis standard deviation",
            "log_diff_rms": "log RMS of consecutive-sample differences",
            "power_0.1_1Hz": "Welch power fraction in [0.1,1) Hz relative to [0.1,20] Hz",
            "power_1_5Hz": "Welch power fraction in [1,5) Hz relative to [0.1,20] Hz",
            "welch": {"nperseg": N_TIMESTEPS, "noverlap": N_TIMESTEPS // 2, "resolution_hz": FS / N_TIMESTEPS},
        },
        "quality_filter": {"finite": True, "max_abs_acc_m_s2": 16 * 9.81},
        "synthetic_generation_started": False,
        "classes": {},
    }

    for activity, label in ACTIVITIES.items():
        print(f"[{activity}] loading eligible five-second training windows", flush=True)
        original_indices = class_indices[activity]
        windows = io.take(X, original_indices)
        quality = np.isfinite(windows).all(axis=(1, 2)) & (
            np.abs(windows[:, :, :3]).max(axis=(1, 2)) <= 16 * 9.81
        )
        dataset_indices = original_indices[quality]
        windows = windows[quality]
        activity_meta = meta.iloc[dataset_indices].copy().reset_index(drop=True)
        assert len(activity_meta) == len(windows)
        assert (activity_meta.label.to_numpy() == label).all()
        assert set(activity_meta.subject) <= train_subjects
        assert not (set(activity_meta.subject) & held_out)
        assert len(np.unique(dataset_indices)) == len(dataset_indices)

        weights = participant_weights(activity_meta.subject.to_numpy())
        print(f"[{activity}] computing 45 features for {len(windows)} windows", flush=True)
        raw_features = signal_features(windows, stats)
        feature_space, feature_center, feature_scale = standardize_features(raw_features, weights)
        print(f"[{activity}] finding exact participant-weighted medoid", flush=True)
        medoid, distance, medoid_costs = exact_weighted_medoid(
            feature_space, weights, block_size=args.medoid_block_size
        )
        assert np.isclose(medoid_costs[medoid], np.min(medoid_costs))

        table = activity_meta.copy()
        table.insert(0, "candidate_id", np.arange(len(table), dtype=np.int64))
        table.insert(1, "dataset_index", dataset_indices)
        table["participant_weight"] = weights
        table["distance_to_medoid"] = distance
        table["is_medoid"] = np.arange(len(table)) == medoid

        subset_records = {}
        masks = {}
        ranking_order = None
        ranking_cumulative = None
        for subset_name, target in SUBSETS:
            selected, cutoff, achieved, below, order, cumulative = neighborhood(
                distance, weights, dataset_indices, target
            )
            masks[subset_name] = selected
            table[f"in_{subset_name}"] = selected
            ranking_order, ranking_cumulative = order, cumulative

            selected_indices = np.sort(dataset_indices[selected])
            selected_table = table.loc[selected].sort_values("dataset_index", kind="stable")
            assert np.array_equal(selected_table.dataset_index.to_numpy(), selected_indices)
            assert selected[medoid]
            if (~selected).any():
                assert distance[selected].max() <= distance[~selected].min()

            stem = generator_dir / f"{activity}_{subset_name}_w500"
            index_path = stem.with_name(stem.name + "_indices.npy")
            np.save(index_path, selected_indices.astype(np.int64))
            index_hash = sha256_file(index_path)
            selected_table.to_csv(stem.with_suffix(".csv"), index=False)
            print(
                f"[{activity}] writing {subset_name}: {len(selected_indices)} windows, "
                f"weight={achieved:.6f}, subjects={selected_table.subject.nunique()}",
                flush=True,
            )
            array_hash = write_windows(stem.with_suffix(".npy"), X, selected_indices)
            subject_counts = selected_table.subject.value_counts().sort_index()
            record = {
                "class": activity,
                "label": label,
                "selection": subset_name,
                "target_participant_weight": target,
                "achieved_participant_weight": achieved,
                "weight_strictly_below_cutoff": below,
                "distance_cutoff": cutoff,
                "window": N_TIMESTEPS,
                "sampling_rate_hz": FS,
                "n_segments": int(len(selected_indices)),
                "n_source_windows": int(len(selected_indices)),
                "n_full_activity_windows": int(len(dataset_indices)),
                "train_subjects": len(split["train"]),
                "subjects_used": sorted(selected_table.subject.unique()),
                "n_subjects_used": int(selected_table.subject.nunique()),
                "windows_per_subject": {str(k): int(v) for k, v in subject_counts.items()},
                "val_or_test_subjects_used": sorted(set(selected_table.subject) & held_out),
                "max_abs_acc": 16 * 9.81,
                "windows_dropped_implausible_acc": int((~quality).sum()),
                "source_dataset_indices": str(index_path.resolve()),
                "source_dataset_indices_sha256": index_hash,
                "array_path": str(stem.with_suffix(".npy").resolve()),
                "array_sha256": array_hash,
                "row_metadata_path": str(stem.with_suffix(".csv").resolve()),
                "row_order": "ascending source dataset_index",
                "participant_balancing_applied_to": "medoid, feature scaling, and neighborhood cutoff; generator training sampler is not defined here",
            }
            write_json(stem.with_suffix(".json"), record)
            subset_records[subset_name] = record

        assert np.all(masks["central25"] <= masks["central50"])
        assert np.all(masks["central50"] <= masks["central75"])
        assert np.all(masks["central75"] <= masks["full"])
        assert masks["full"].all()
        rank = np.empty(len(table), dtype=np.int64)
        rank[ranking_order] = np.arange(1, len(table) + 1)
        cumulative_by_candidate = np.empty(len(table), dtype=np.float64)
        cumulative_by_candidate[ranking_order] = ranking_cumulative
        table["distance_rank"] = rank
        table["cumulative_participant_weight_at_rank"] = cumulative_by_candidate
        ranking_path = args.out_dir / f"{activity}_ranking.csv"
        table.sort_values(["distance_rank"], kind="stable").to_csv(ranking_path, index=False)
        # Rewrite aligned subset metadata after all membership and ranking columns
        # exist (the arrays themselves remain in ascending source-row order).
        for subset_name, _ in SUBSETS:
            selected_table = table.loc[masks[subset_name]].sort_values("dataset_index", kind="stable")
            stem = generator_dir / f"{activity}_{subset_name}_w500"
            selected_table.to_csv(stem.with_suffix(".csv"), index=False)

        feature_path = args.out_dir / f"{activity}_feature_space.npz"
        np.savez_compressed(
            feature_path,
            dataset_indices=dataset_indices,
            raw_features=raw_features,
            standardized_features=feature_space,
            participant_weights=weights,
            distance_to_medoid=distance,
            medoid_costs=medoid_costs,
            medoid_candidate=np.array(medoid),
            feature_center=feature_center,
            feature_scale=feature_scale,
            feature_names=np.array(FEATURE_NAMES),
        )
        medoid_row = table.iloc[medoid]
        selection["classes"][activity] = {
            "label": label,
            "eligible_windows": int(len(dataset_indices)),
            "quality_rejected": int((~quality).sum()),
            "participants": int(activity_meta.subject.nunique()),
            "medoid": {
                "candidate_id": medoid,
                "dataset_index": int(dataset_indices[medoid]),
                "subject": str(medoid_row.subject),
                "timestamp": int(medoid_row.timestamp),
                "hand": str(medoid_row.hand),
                "dominance": str(medoid_row.dominance),
                "moment": str(medoid_row.moment),
                "weighted_mean_distance": float(medoid_costs[medoid]),
            },
            "ranking_path": str(ranking_path.resolve()),
            "feature_space_path": str(feature_path.resolve()),
            "subsets": subset_records,
        }
        del windows, raw_features, feature_space, medoid_costs

    write_json(args.out_dir / "selection.json", selection)
    print(f"[done] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
