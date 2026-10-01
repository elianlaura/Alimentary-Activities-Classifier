#!/usr/bin/env python
"""Causal temporal smoothing of saved predictions, with subject-bootstrap 95% CIs.

    python scripts/smoothing.py --runs $RUNS --data-root $DATA_ROOT [--horizons 0 5 10 15]

No retraining: uses predictions_{val,test}.npz of scripts/finetune.py. For each window, the
class probabilities are averaged with those of the previous 1, 2, 3 ... windows of the same
subject (horizon 5, 10, 15 ... s of past, the current window included), only while the
recording is contiguous (consecutive windows <= --max-gap s apart). Causal: no future window.

36% of the DEO windows are exact copies (same subject, timestamp and values). They are
dropped first (one window per subject and timestamp), otherwise the "previous" window of a
copy would be the copy itself; --keep-duplicates evaluates on all windows instead.

Outputs (<runs>/analysis/smoothing/): smoothing.csv, smoothing.md
  per condition and horizon: seed-mean BA on validation and test with a subject-bootstrap 95%
  CI, and the paired difference with 'scratch' at the same horizon (same subject draws).
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import io, splits  # noqa: E402


def timeline(meta, split, part):
    idx = splits.indices(meta, split, part)  # same order as the saved predictions
    m = meta.iloc[idx]
    return m.subject.to_numpy().astype(str), m.timestamp.to_numpy().astype(np.int64)


def smooth(probs, subj, ts, n_past, max_gap):
    """Causal moving average over the current and up to n_past previous contiguous windows."""
    order = np.lexsort((ts, subj))
    p, s, t = probs[order], subj[order], ts[order]
    total, count = p.copy(), np.ones(len(p))
    n = len(p)
    ok = np.ones(n, dtype=bool)
    for d in range(1, n_past + 1):
        # window i may use window i-d if the d-th link (i-d -> i-d+1) is contiguous and ok holds for links < d
        newer, older = t[1:n - d + 1], t[0:n - d]
        prev_ok = np.zeros(n, dtype=bool)
        prev_ok[d:] = (s[d:] == s[:-d]) & (newer - older <= max_gap) & (newer > older)
        ok &= prev_ok  # the chain stops at the first gap
        total[d:] += np.where(ok[d:, None], p[:-d], 0.0)
        count[d:] += ok[d:]
    out = np.empty_like(p)
    out[order] = total / count[:, None]
    return out


def subject_cms(y, pred, subj, subjects):
    pos = {s: i for i, s in enumerate(subjects)}
    cm = np.zeros((len(subjects), 3, 3))
    np.add.at(cm, (np.array([pos[s] for s in subj]), y, pred), 1)
    return cm


def ba(cm):
    """cm (..., 3, 3) -> balanced accuracy over the classes with support."""
    support = cm.sum(-1)
    rec = np.diagonal(cm, axis1=-2, axis2=-1) / np.maximum(support, 1)
    has = support > 0
    return (rec * has).sum(-1) / np.maximum(has.sum(-1), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="default: scratch and the per-class BiGRU volume sweep diff_gru_x*")
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 5, 10, 15], help="seconds of past (multiples of 5)")
    ap.add_argument("--max-gap", type=int, default=6, help="max seconds between consecutive windows of one recording")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20251028)
    args = ap.parse_args()

    ft = os.path.join(args.runs, "finetune", "deo")
    conds = args.conditions or ["scratch"] + sorted(
        (c for c in os.listdir(ft) if re.match(r"^diff_gru_x[\dp]+$", c)),
        key=lambda c: float(c[len("diff_gru_x"):].replace("p", ".")))
    X, meta = io.load_dataset(os.path.join(args.data_root, "deo"))
    split = splits.load_split(os.path.join(args.data_root, "deo", "split.json"))
    out_dir = os.path.join(args.runs, "analysis", "smoothing")
    os.makedirs(out_dir, exist_ok=True)

    parts = {}
    for part in ("val", "test"):
        subj, ts = timeline(meta, split, part)
        keep = np.ones(len(subj), dtype=bool) if args.keep_duplicates else \
            ~pd.DataFrame({"s": subj, "t": ts}).duplicated(keep="first").to_numpy()
        subjects = np.unique(subj)
        rng = np.random.default_rng(args.seed)
        picks = rng.integers(0, len(subjects), (args.n_boot, len(subjects)))
        W = np.zeros((args.n_boot, len(subjects)))
        np.add.at(W, (np.arange(args.n_boot)[:, None], picks), 1)
        parts[part] = dict(subj=subj[keep], ts=ts[keep], keep=keep, subjects=subjects, W=W)
        print("[smoothing] %s: %d windows, %d after removing copies" % (part, len(keep), keep.sum()), flush=True)

    # CM[cond][h][part]: (seeds, subjects, 3, 3)
    CM = {}
    for c in conds:
        CM[c] = {}
        for h in args.horizons:
            if h % 5:
                raise SystemExit("horizons must be multiples of 5 s")
            CM[c][h] = {}
            for part, P in parts.items():
                cms = []
                for s in args.seeds:
                    z = np.load(os.path.join(ft, c, "seed%d" % s, "predictions_%s.npz" % part))
                    y, probs = z["y_true"].astype(int)[P["keep"]], z["probs"][P["keep"]].astype(np.float64)
                    if h:
                        probs = smooth(probs, P["subj"], P["ts"], h // 5, args.max_gap)
                    cms.append(subject_cms(y, probs.argmax(1), P["subj"], P["subjects"]))
                CM[c][h][part] = np.stack(cms)

    rows = []
    pct = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    for c in conds:
        for h in args.horizons:
            row = {"condition": c, "past_s": h, "windows_used": int(1 + h // 5)}
            for part, P in parts.items():
                C = CM[c][h][part]
                boot = ba(np.einsum("bs,ksij->bkij", P["W"], C)).mean(1)
                row["%s_ba" % part] = float(ba(C.sum(1)).mean())
                row["%s_ci_low" % part], row["%s_ci_high" % part] = pct(boot)
                if c != "scratch" and "scratch" in CM:
                    B = CM["scratch"][h][part]
                    d = boot - ba(np.einsum("bs,ksij->bkij", P["W"], B)).mean(1)
                    row["%s_minus_scratch" % part] = row["%s_ba" % part] - float(ba(B.sum(1)).mean())
                    row["%s_minus_scratch_ci_low" % part], row["%s_minus_scratch_ci_high" % part] = pct(d)
                    per_a = ba(C).mean(0)
                    per_b = ba(B).mean(0)
                    row["%s_subjects_better" % part] = "%d/%d" % ((per_a > per_b).sum(), len(per_a))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "smoothing.csv"), index=False)

    lines = ["# Causal smoothing (seed mean of %d models, copies removed: %s)\n" % (len(args.seeds), not args.keep_duplicates)]
    for h in args.horizons:
        lines += ["\n## %s\n" % ("No smoothing" if h == 0 else "Current window + %d previous (%d s of past)" % (h // 5, h)),
                  "| condition | BA val [95% CI] | BA test [95% CI] | test - scratch [95% CI] | subjects better |",
                  "|---|---|---|---|---|"]
        for _, r in df[df.past_s == h].iterrows():
            diff = "-" if r.condition == "scratch" else "%+.4f [%+.4f, %+.4f]" % (
                r.test_minus_scratch, r.test_minus_scratch_ci_low, r.test_minus_scratch_ci_high)
            better = "-" if r.condition == "scratch" else r.test_subjects_better
            lines.append("| %s | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] | %s | %s |" % (
                r.condition, r.val_ba, r.val_ci_low, r.val_ci_high, r.test_ba, r.test_ci_low, r.test_ci_high, diff, better))
    lines.append("\nSubject-cluster bootstrap, %d replicates (same draws for every condition and horizon); the "
                 "difference with scratch is paired. Choose the horizon on validation, not on test." % args.n_boot)
    open(os.path.join(out_dir, "smoothing.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
