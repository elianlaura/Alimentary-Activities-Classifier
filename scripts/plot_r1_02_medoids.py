#!/usr/bin/env python
"""Select real 30-s training medoids first, then match class-specific BiGRU samples.

Outputs: nine-axis traces, real-feature distributions with medoids marked,
sensor-value CDFs, selected arrays, candidate tables, and a provenance manifest.
All modelling/selection statistics are fitted on real training excerpts only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import welch
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deo import AXIS_NAMES, io, norm, splits
from deo.utils import write_json

FS = 100
N = 30 * FS
COLORS = ['#0072B2', '#D55E00', '#009E73']
DISPLAY = {'drink': 'Drinking', 'eat': 'Eating'}
UNITS = ['m/s²'] * 3 + ['source units'] * 3 + ['µT'] * 3
FEATURE_GROUPS = ['mean', 'log_std', 'log_diff_rms', 'power_0.1_1Hz', 'power_1_5Hz']
FEATURE_NAMES = [f'{g}:{a}' for g in FEATURE_GROUPS for a in AXIS_NAMES]


def weighted_quantile(x, weights, quantiles):
    order = np.argsort(x)
    v, w = x[order], weights[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(quantiles, cumulative, v)


def real_excerpts(X, meta, split, label):
    """Disjoint six-window blocks, never crossing a non-five-second gap."""
    subset = meta[meta.subject.isin(split['train']) & (meta.label == label)]
    keys = ['subject', 'label', 'hand', 'dominance', 'moment']
    blocks, records = [], []
    n_gaps, rejected = 0, 0
    for key, group in subset.groupby(keys, sort=True):
        group = group.sort_values('timestamp', kind='stable')
        times = group.timestamp.to_numpy()
        assert len(np.unique(times)) == len(times), 'Duplicate timestamps remain within a session'
        breaks = np.flatnonzero(np.diff(times) != 5) + 1
        n_gaps += len(breaks)
        bounds = np.r_[0, breaks, len(group)]
        for start, stop in zip(bounds[:-1], bounds[1:]):
            for pos in range(start, stop - 5, 6):
                rows = group.iloc[pos:pos + 6]
                assert np.all(np.diff(rows.timestamp) == 5)
                window = np.asarray(X[rows.index.to_numpy()], dtype=np.float32).reshape(N, 9)
                if not np.isfinite(window).all() or np.abs(window[:, :3]).max() > 16 * 9.81:
                    rejected += 1
                    continue
                blocks.append(window)
                records.append({
                    'candidate_id': len(records), 'subject': str(key[0]), 'label': label,
                    'hand': key[2], 'dominance': key[3], 'moment': key[4],
                    'timestamp_start': int(rows.timestamp.iloc[0]),
                    'timestamp_end_exclusive': int(rows.timestamp.iloc[-1]) + 5,
                    'dataset_indices': rows.index.to_list(),
                })
    result = np.stack(blocks)
    table = pd.DataFrame(records)
    counts = table.subject.value_counts()
    weights = table.subject.map(1.0 / counts).to_numpy()
    weights /= weights.sum()
    assert set(table.subject) <= set(split['train'])
    all_indices = np.concatenate(table.dataset_indices.to_numpy())
    assert len(np.unique(all_indices)) == len(all_indices), 'Overlapping excerpts'
    return result, table, weights, {'gaps_not_crossed': n_gaps, 'quality_rejected': rejected}


def features(windows, stats, chunk=64):
    """45 descriptors: nine axes x (level, variation, roughness, two PSD fractions)."""
    out = []
    for start in range(0, len(windows), chunk):
        z = norm.apply(windows[start:start + chunk], stats).astype(np.float64)
        level = z.mean(1)
        variation = np.log(np.maximum(z.std(1), 1e-6))
        roughness = np.log(np.maximum(np.sqrt(np.mean(np.diff(z, axis=1) ** 2, axis=1)), 1e-6))
        f, p = welch(z, fs=FS, nperseg=1000, noverlap=500, axis=1, detrend='constant')
        total = p[:, (f >= 0.1) & (f <= 20)].sum(1)
        bands = [p[:, (f >= lo) & (f < hi)].sum(1) / np.maximum(total, 1e-12)
                 for lo, hi in [(0.1, 1), (1, 5)]]
        out.append(np.concatenate([level, variation, roughness, *bands], axis=1))
    return np.concatenate(out)


def fit_feature_scale(F, weights):
    # Equal weighted variance per descriptor prevents near-zero IQR features
    # (e.g. band fractions concentrated near one) from dominating distances.
    center = weights @ F
    scale = np.sqrt(weights @ ((F - center) ** 2))
    scale = np.where(scale > 1e-8, scale, 1.0)
    return center, scale


def pam(D, weights, subjects, k, seed, restarts=6):
    """Weighted PAM with exhaustive improving swaps and distinct-subject medoids.

    One deterministic BUILD start plus seeded random starts; choose lowest cost.
    The result is a local optimum under one-medoid swaps, not a global guarantee.
    """
    n = len(D)
    rng = np.random.default_rng(seed)
    first = int(np.argmin(weights @ D))
    greedy = [first]
    while len(greedy) < k:
        nearest = D[:, greedy].min(1)
        costs = weights @ np.minimum(nearest[:, None], D)
        costs[np.isin(subjects, subjects[greedy])] = np.inf
        greedy.append(int(np.argmin(costs)))
    starts = [greedy]
    for _ in range(restarts - 1):
        chosen_subjects = rng.choice(np.unique(subjects), k, replace=False)
        starts.append([int(rng.choice(np.flatnonzero(subjects == s))) for s in chosen_subjects])
    best = None
    for initial in starts:
        medoids = np.array(initial)
        cost = float(weights @ D[:, medoids].min(1))
        for iteration in range(100):
            next_cost, swap = cost, None
            for position in range(k):
                others = np.delete(medoids, position)
                base = D[:, others].min(1) if len(others) else np.full(n, np.inf)
                costs = weights @ np.minimum(base[:, None], D)
                costs[np.isin(subjects, subjects[others])] = np.inf
                candidate = int(np.argmin(costs))
                if costs[candidate] < next_cost - 1e-10:
                    next_cost, swap = float(costs[candidate]), (position, candidate)
            if swap is None:
                break
            medoids[swap[0]] = swap[1]
            cost = next_cost
        else:
            raise RuntimeError('PAM did not converge')
        if best is None or cost < best[0]:
            best = (cost, medoids.copy(), iteration)
    cost, medoids, iterations = best
    labels = D[:, medoids].argmin(1)
    shares = np.bincount(labels, weights=weights, minlength=k)
    order = np.argsort(-shares, kind='stable')
    medoids = medoids[order]
    labels = D[:, medoids].argmin(1)
    assert len(set(subjects[medoids])) == k
    assert np.array_equal(labels[medoids], np.arange(k))
    return medoids, labels, {'weighted_distance_objective': cost, 'swap_iterations': iterations,
                            'restarts': restarts, 'distinct_participant_constraint': True}


def synthetic_excerpts(generator_dir, cname, data_dir, split):
    cfg = json.loads((generator_dir / 'config.json').read_text())
    assert cfg['arch'] == 'gru' and cfg['n_features'] == 9
    training_path = Path(cfg['data'])
    assert training_path.resolve().parent.parent == data_dir.resolve()
    training = json.loads(training_path.with_suffix('.json').read_text())
    assert training['class'] == cname
    assert set(training['subjects_used']) <= set(split['train'])
    assert not training['val_or_test_subjects_used']
    segment_len = cfg['window']
    assert N % segment_len == 0
    per_excerpt = N // segment_len
    arrays, records, shards, skipped_incomplete = [], [], [], []
    for path in sorted(generator_dir.glob('samples_shard*.npy')):
        completion_path = path.with_suffix('.json')
        if not completion_path.exists():
            skipped_incomplete.append(str(path.resolve()))
            continue
        completion = json.loads(completion_path.read_text())
        progress = json.loads(Path(str(path) + '.progress.json').read_text())
        arr = np.load(path, mmap_mode='r')
        assert len(arr) == completion['n'] == progress['done']
        assert progress['checkpoint'] == completion['checkpoint']
        assert completion['arch'] == 'gru' and completion['window'] == segment_len
        assert Path(completion['checkpoint']).parent.resolve() == generator_dir.resolve()
        assert arr.shape[1:] == (segment_len, 9)
        n = len(arr) // per_excerpt
        block = np.asarray(arr[:n * per_excerpt]).reshape(n, N, 9)
        assert np.isfinite(block).all()
        arrays.append(block)
        for i in range(n):
            records.append({'candidate_id': len(records), 'shard': str(path.resolve()),
                            'segment_start': i * per_excerpt,
                            'segment_stop_exclusive': (i + 1) * per_excerpt})
        shards.append({'path': str(path.resolve()), 'completion': completion,
                       'unused_tail_segments': len(arr) % per_excerpt})
    if not arrays:
        raise ValueError('No completed synthetic shards')
    return np.concatenate(arrays), pd.DataFrame(records), {
        'generator_dir': str(generator_dir.resolve()), 'config': cfg,
        'training_data_metadata': training, 'shards': shards,
        'skipped_shards_without_completion_record': skipped_incomplete,
        'segments_per_excerpt': per_excerpt,
        'assembly': 'Consecutive original segments, disjoint, within one shard, without smoothing or reordering',
    }


def weighted_pca(F, weights):
    center = weights @ F
    cov = (F - center).T @ ((F - center) * weights[:, None])
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return center, vectors[:, order[:2]], values[order[:2]] / values.sum()


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix('.png'), dpi=200, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')


def trace_figures(result, out_dir, pdf):
    import matplotlib.pyplot as plt
    name, k = result['name'], len(result['medoids'])
    real = result['real'][result['medoids']]
    syn = result['synthetic'][result['matches']]
    t = np.arange(N) / FS
    # Class overview: columns are examples, rows are the nine individual axes.
    fig, axes = plt.subplots(9, k, figsize=(5 * k, 15), sharex=True, squeeze=False)
    for j in range(9):
        for col in range(k):
            ax = axes[j, col]
            ax.plot(t, real[col, :, j], c='#0072B2', lw=0.65, label='Real medoid')
            ax.plot(t, syn[col, :, j], c='#D55E00', lw=0.6, alpha=0.85, label='Synthetic match')
            ax.set_xlim(0, 30)
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel(f'{AXIS_NAMES[j]}\n({UNITS[j]})')
            if j == 8:
                ax.set_xlabel('Time (s)')
            if j == 0:
                rec = result['examples'][col]
                scope = ('global medoid' if k == 1 else f"cluster {100*rec['cluster_weight']:.1f}%")
                ax.set_title(f"M{col+1} · {rec['real']['subject']} · {scope}")
    axes[0, 0].legend(fontsize=8, loc='best')
    fig.suptitle(f'{DISPLAY[name]}: 30-second real training medoids and class-specific BiGRU matches', y=0.995)
    fig.text(0.5, 0.004, 'Synthetic: 125 consecutive 0.24-s generated segments; joins retained. Gyroscope values retain source units.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    save_figure(fig, out_dir / f'{name}_30s_medoids')
    plt.close(fig)
    # Legible separate page per pair, nine-axis grid.
    for col in range(k):
        fig, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.plot(t, real[col, :, j], c='#0072B2', lw=0.8, label='Real medoid')
            ax.plot(t, syn[col, :, j], c='#D55E00', lw=0.7, alpha=0.85, label='Synthetic match')
            ax.set_title(f'{AXIS_NAMES[j]} ({UNITS[j]})')
            ax.set_xlim(0, 30)
            ax.grid(alpha=0.2)
            if j >= 6:
                ax.set_xlabel('Time (s)')
        axes[0, 0].legend(fontsize=9)
        rec = result['examples'][col]
        scope = ('global medoid of all excerpts' if k == 1
                 else f"cluster {100*rec['cluster_weight']:.1f}%")
        fig.suptitle(f"{DISPLAY[name]} M{col+1}: {rec['real']['subject']}, {rec['real']['moment']} · 30 s · {scope}")
        fig.text(0.5, 0.004, 'Real medoids are fixed first; distinct synthetic matches minimize total distance in the 45-feature space.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.025, 1, 0.97))
        pdf.savefig(fig, bbox_inches='tight')
        fig.savefig(out_dir / f'{name}_medoid_{col+1}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)


def distribution_figures(results, out_dir):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row, result in enumerate(results):
        F, w = result['F'], result['weights']
        center, vectors, explained = weighted_pca(F, w)
        real_xy = (F - center) @ vectors
        syn_xy = (result['G'] - center) @ vectors
        result['pca_explained_variance'] = explained.tolist()
        ax = axes[row, 0]
        ax.scatter(syn_xy[:, 0], syn_xy[:, 1], c='0.65', s=7, alpha=0.22, label='All synthetic candidates', rasterized=True)
        for i, m in enumerate(result['medoids']):
            mask = result['labels'] == i
            ax.scatter(real_xy[mask, 0], real_xy[mask, 1], c=COLORS[i], s=12, alpha=0.5,
                       label=f'Real cluster {i+1}', rasterized=True)
            ax.scatter(*real_xy[m], c=COLORS[i], marker='*', s=230, edgecolors='black', zorder=10)
            match = result['matches'][i]
            ax.scatter(*syn_xy[match], c=COLORS[i], marker='X', s=100, edgecolors='black', zorder=9)
            ax.annotate(f'M{i+1}', real_xy[m], xytext=(7, 7), textcoords='offset points', weight='bold')
            offset = [(14, 2), (-20, 22), (-20, -25)][i] if result['name'] == 'eat' else (7, -12)
            ax.annotate(f'S{i+1}', syn_xy[match], xytext=offset, textcoords='offset points',
                        arrowprops={'arrowstyle': '-', 'lw': 0.6, 'color': '0.25'})
        ax.set_xlabel(f'PC1 ({100*explained[0]:.1f}% real variance)')
        ax.set_ylabel(f'PC2 ({100*explained[1]:.1f}% real variance)')
        ax.set_title(f"{DISPLAY[result['name']]}: feature distribution")
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.2)
        # Motion distribution: std of the three gyroscope axes in train-z units.
        motion = result['motion_real']
        synthetic_motion = result['motion_synthetic']
        lo, hi = min(motion.min(), synthetic_motion.min()), max(motion.max(), synthetic_motion.max())
        bins = np.geomspace(max(lo, 1e-7), hi * 1.01, 40)
        ax = axes[row, 1]
        ax.hist(motion, bins=bins, weights=w, color='#0072B2', alpha=0.5, label='Real (participant weighted)')
        ax.hist(synthetic_motion, bins=bins, weights=np.ones(len(synthetic_motion))/len(synthetic_motion),
                histtype='step', color='0.3', lw=1.5, label='Synthetic')
        for i, m in enumerate(result['medoids']):
            ax.axvline(motion[m], c=COLORS[i], lw=1.4, label=f'M{i+1}')
            ax.text(motion[m], 0.97 - i*0.10, f'M{i+1}', transform=ax.get_xaxis_transform(),
                    color=COLORS[i], ha='left', va='top', weight='bold')
        ax.set_xscale('log')
        ax.set_xlabel('Gyroscope motion RMS (real-train z-score units; log scale)')
        ax.set_ylabel('Probability mass per log-spaced bin')
        ax.set_title(f"{DISPLAY[result['name']]}: motion distribution and real medoids")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle('Thirty-second training distributions · stars = real medoids M · crosses = synthetic matches S')
    fig.text(0.5, 0.003, 'Medoids use all 45 features. PCA is a two-dimensional view, fitted to real training excerpts only.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    save_figure(fig, out_dir / 'distributions_and_medoids')
    plt.close(fig)

    for result in results:
        fig, axes = plt.subplots(3, 3, figsize=(13, 9))
        for j, ax in enumerate(axes.flat):
            # Exact empirical quantiles, subject-weighted for real excerpts.
            a = result['real'][:, :, j].ravel()
            b = result['synthetic'][:, :, j].ravel()
            w = np.repeat(result['weights'] / N, N)
            probabilities = np.unique(np.r_[0, np.geomspace(1e-5, 0.01, 25),
                                            np.linspace(.01, .99, 200), 1-np.geomspace(1e-5, .01, 25), 1])
            aq = weighted_quantile(a, w, probabilities)
            bq = np.quantile(b, probabilities)
            ax.plot(aq, probabilities, c='black', lw=2, label='All real (weighted)')
            ax.plot(bq, probabilities, c='0.55', lw=2, ls='--', label='All synthetic')
            for i, medoid in enumerate(result['medoids']):
                ax.plot(np.sort(result['real'][medoid, :, j]), np.arange(1, N+1)/N,
                        c=COLORS[i], lw=1, label=f'M{i+1} real')
                ax.plot(np.sort(result['synthetic'][result['matches'][i], :, j]), np.arange(1, N+1)/N,
                        c=COLORS[i], lw=1, ls='--', label=f'S{i+1} synthetic')
            lo = min(np.interp(.005, probabilities, aq), np.interp(.005, probabilities, bq))
            hi = max(np.interp(.995, probabilities, aq), np.interp(.995, probabilities, bq))
            pad = (hi-lo)*.04
            ax.set_xlim(lo-pad, hi+pad)
            ax.set_ylim(0, 1)
            ax.set_title(f'{AXIS_NAMES[j]} ({UNITS[j]})')
            ax.grid(alpha=.2)
            if j % 3 == 0:
                ax.set_ylabel('Cumulative probability')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
        fig.suptitle(f"{DISPLAY[result['name']]}: sensor-value distributions and selected medoids")
        fig.text(.5, .075, 'Solid colored curves: real medoids. Dashed colored curves: their synthetic matches. Display spans central 99% of aggregate values.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .10, 1, .96))
        save_figure(fig, out_dir / f"{result['name']}_axis_distributions")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--generator-root', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--medoids', type=int, default=3, choices=[1, 2, 3])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    args.out_dir.mkdir(parents=True, exist_ok=True)
    X, meta = io.load_dataset(str(args.data_dir))
    split = splits.load_split(str(args.data_dir / 'split.json'))
    stats = norm.load_stats(str(args.data_dir / 'norm_train.json'))
    data_info = json.loads((args.data_dir / 'info.json').read_text())
    assert data_info.get('deduplicated_from'), 'Use the deduplicated dataset'
    results = []
    # Entire real-selection phase precedes access to synthetic data.
    for label, cname in enumerate(['drink', 'eat']):
        real, table, weights, reconstruction = real_excerpts(X, meta, split, label)
        raw_features = features(real, stats)
        center, scale = fit_feature_scale(raw_features, weights)
        F = (raw_features - center) / scale / np.sqrt(len(FEATURE_NAMES))
        D = cdist(F, F)
        k = min(args.medoids, table.subject.nunique())
        medoids, labels, optimization = pam(D, weights, table.subject.to_numpy(), k, args.seed)
        silhouette = float(weights @ silhouette_samples(D, labels, metric='precomputed')) if k > 1 else None
        table['participant_weight'] = weights
        table['cluster'] = labels + 1
        table['is_medoid'] = np.isin(np.arange(len(table)), medoids)
        table['distance_to_medoid'] = D[np.arange(len(D)), medoids[labels]]
        table.to_csv(args.out_dir / f'{cname}_real_candidates.csv', index=False)
        realz = norm.apply(real, stats)
        motion = np.sqrt(np.mean(np.var(realz[:, :, 3:6], axis=1), axis=1))
        results.append(dict(name=cname, real=real, real_table=table, weights=weights,
                            raw_features=raw_features, center=center, scale=scale, F=F,
                            medoids=medoids, labels=labels, optimization=optimization,
                            silhouette=silhouette, reconstruction=reconstruction, motion_real=motion))
        print(f'[real] {cname}: {len(real)} disjoint 30-s excerpts, {table.subject.nunique()} subjects; '
              f'medoids {medoids.tolist()}, weighted silhouette {silhouette}', flush=True)
    # Freeze real choices in a file before matching.
    write_json(str(args.out_dir / 'real_selection.json'), {
        r['name']: {'candidate_ids': r['medoids'].tolist(),
                    'records': r['real_table'].iloc[r['medoids']].to_dict('records')}
        for r in results})
    for result in results:
        cname = result['name']
        syn, stable, provenance = synthetic_excerpts(args.generator_root / f'gru_{cname}_w24',
                                                       cname, args.data_dir, split)
        raw = features(syn, stats)
        G = (raw - result['center']) / result['scale'] / np.sqrt(len(FEATURE_NAMES))
        distances = cdist(result['F'][result['medoids']], G)
        rows, cols = linear_sum_assignment(distances)
        matches = np.empty(len(rows), dtype=int)
        matches[rows] = cols
        stable['is_selected'] = np.isin(np.arange(len(stable)), matches)
        stable.to_csv(args.out_dir / f'{cname}_synthetic_candidates.csv', index=False)
        result.update(synthetic=syn, G=G, matches=matches, synthetic_provenance=provenance)
        synz = norm.apply(syn, stats)
        result['motion_synthetic'] = np.sqrt(np.mean(np.var(synz[:, :, 3:6], axis=1), axis=1))
        examples = []
        for i, (m, match) in enumerate(zip(result['medoids'], matches)):
            member = result['labels'] == i
            selected = syn[match]
            real = result['real'][m]
            rmse = np.sqrt(np.mean(((real-selected) / np.asarray(stats['std']))**2))
            examples.append({
                'medoid_id': f'M{i+1}', 'real': result['real_table'].iloc[m].to_dict(),
                'synthetic': stable.iloc[match].to_dict(),
                'cluster_weight': float(result['weights'][member].sum()),
                'cluster_excerpts': int(member.sum()),
                'cluster_participants': int(result['real_table'][member].subject.nunique()),
                'match_feature_distance': float(distances[i, match]),
                'unconstrained_nearest_synthetic_id': int(distances[i].argmin()),
                'synthetic_match_distance_rank': int(1 + (distances[i] < distances[i, match]).sum()),
                'pointwise_train_z_rmse': float(rmse),
                'real_sha256': hashlib.sha256(real.tobytes()).hexdigest(),
                'synthetic_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
            })
        result['examples'] = examples
        np.savez_compressed(args.out_dir / f'{cname}_selected_sequences.npz',
                            real=result['real'][result['medoids']], synthetic=syn[matches],
                            time_seconds=np.arange(N)/FS, axis_names=np.array(AXIS_NAMES))
        np.savez_compressed(args.out_dir / f'{cname}_feature_space.npz',
                            real=result['F'], synthetic=G, weights=result['weights'],
                            labels=result['labels'], medoid_indices=result['medoids'], synthetic_indices=matches,
                            feature_names=np.array(FEATURE_NAMES), real_center=result['center'], real_feature_std=result['scale'])
        print(f'[synthetic] {cname}: {len(syn)} candidates; matches {matches.tolist()}', flush=True)
        del synz
    summary_rows = []
    for result in results:
        for e in result['examples']:
            summary_rows.append({
                'activity': result['name'], 'medoid': e['medoid_id'],
                'participant': e['real']['subject'], 'moment': e['real']['moment'],
                'real_candidate': e['real']['candidate_id'],
                'timestamp_start': e['real']['timestamp_start'], 'seconds': 30,
                'cluster_percent': 100 * e['cluster_weight'],
                'cluster_participants': e['cluster_participants'],
                'synthetic_candidate': e['synthetic']['candidate_id'],
                'match_feature_distance': e['match_feature_distance'],
                'synthetic_match_rank': e['synthetic_match_distance_rank'],
            })
    pd.DataFrame(summary_rows).to_csv(args.out_dir / 'selected_medoids.csv', index=False)
    with PdfPages(args.out_dir / 'all_medoid_pairs.pdf') as pdf:
        for result in results:
            trace_figures(result, args.out_dir, pdf)
    distribution_figures(results, args.out_dir)
    audit = {
        'data_dir': str(args.data_dir.resolve()), 'data_info': data_info,
        'seed': args.seed, 'n_train_subjects': len(split['train']), 'sample_rate_hz': FS,
        'duration_seconds': 30, 'real_first': True,
        'real_reconstruction': 'Six consecutive unique five-second windows; exact 5-s timestamp increments; disjoint excerpts',
        'features': FEATURE_NAMES,
        'feature_scaling': 'Weighted mean / standard deviation per feature and class fitted only to real excerpts; Euclidean distance / sqrt(45)',
        'clustering': 'Weighted PAM, distinct-subject medoids, six starts; fixed k requested, no claim of optimal k or global minimum',
        'requested_medoids_per_class': args.medoids,
        'participant_weighting': 'Each real participant contributes equal total weight within their class',
        'matching': 'Minimum-total-feature-distance unique assignment to existing class-specific BiGRU sequences',
        'caveats': ['Thirty-second labelled excerpts have no gesture-level boundaries and do not prove a complete action.',
                    'Synthetic excerpts concatenate 125 independent 24-step segments, not a jointly generated 30-second action.',
                    'Matches are illustrative nearest neighbours, not paired reconstructions or estimates of typical fidelity.',
                    'Gyroscope physical unit is unverified; figures preserve source values.'],
        'classes': {r['name']: {
            'n_real_candidates': len(r['real']), 'n_synthetic_candidates': len(r['synthetic']),
            'real_subjects': r['real_table'].subject.nunique(), 'reconstruction': r['reconstruction'],
            'optimization': r['optimization'], 'weighted_silhouette': r['silhouette'],
            'pca_explained_variance': r['pca_explained_variance'],
            'generator': r['synthetic_provenance'], 'examples': r['examples'],
        } for r in results},
    }
    write_json(str(args.out_dir / 'selection.json'), audit)
    print(f'[done] {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
