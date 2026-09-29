#!/usr/bin/env python
"""Stage 7: aggregate all runs into tables, statistics and figures.

    python scripts/analyze.py --runs $RUNS --data-root $DATA_ROOT

Outputs (<runs>/analysis/):
  summary.csv / summary.md    mean +- std over seeds of every metric, per condition   (R3-01, R2-07)
  comparisons.md              paired tests between conditions:                         (R3-01)
                                - Wilcoxon signed-rank + paired t-test over seeds
                                - Wilcoxon over the 29 test subjects (seed-averaged per-subject BA)
                                - subject-cluster bootstrap 95% CI of the BA difference
  bootstrap_ci.csv / .md      per condition, validation and test: seed-mean balanced accuracy and
                              per-class recall with a subject-cluster bootstrap 95% percentile CI
                              (10,000 replicates, models fixed)
  volume_curve.png/.pdf       balanced accuracy vs synthetic volume, mean +- std      (R1-05, R3-03)
  learning_curves.png/.pdf    Fig. 3 redrawn: loss and accuracy on separate axes      (R1-08, R1-04)
  class_balance.csv           windows per class, real training set vs each corpus     (S-19)
  cost.csv                    wall-clock time per stage and model sizes               (R3-05)
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import CLASS_NAMES  # noqa: E402
from deo.utils import read_json  # noqa: E402

METRICS = ["balanced_accuracy", "f1_macro", "f1_weighted", "mAP_score", "mAP_hard", "kappa"]
LEGACY_NOTE = ("\n**legacy_recipe_pretrained** is fine-tuned from the archived backbone of the 0.9055 run, whose "
               "pretraining data came from a generator trained on all 94 subjects (test included): it measures "
               "the reproducibility of that recipe and is not a valid performance estimate.\n")


def collect(runs):
    rows = []
    for done in glob.glob(os.path.join(runs, "finetune", "*", "*", "seed*", "done.json")):
        d = os.path.dirname(done)
        seed = int(re.search(r"seed(\d+)$", d).group(1))
        cond = os.path.basename(os.path.dirname(d))
        dataset = os.path.basename(os.path.dirname(os.path.dirname(d)))
        te = read_json(os.path.join(d, "metrics_test.json"))
        va = read_json(os.path.join(d, "metrics_val.json"))
        info = read_json(done)
        row = {"dataset": dataset, "condition": cond, "seed": seed, "dir": d,
               "val_balanced_accuracy": va["balanced_accuracy"], "epochs_run": info.get("epochs_run"),
               "best_epoch": info.get("best_epoch"), "fit_seconds": info.get("timing_s", {}).get("fit")}
        for m in METRICS:
            row["test_" + m] = te[m]
        for k, c in enumerate(CLASS_NAMES):
            row["test_sens_" + c] = te["sensitivity"][k]
            row["test_spec_" + c] = te["specificity"][k]
            row["test_auc_" + c] = te["auc_score"][k]
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df):
    num = [c for c in df.columns if c.startswith("test_") or c.startswith("val_")] + ["fit_seconds", "best_epoch"]
    g = df.groupby(["dataset", "condition"])
    out = g[num].agg(["mean", "std"])
    out.columns = ["%s_%s" % c for c in out.columns]
    out.insert(0, "n_seeds", g.size())
    return out.reset_index().sort_values(["dataset", "test_balanced_accuracy_mean"], ascending=[True, False])


def fmt_pm(mean, std):
    return "%.4f ± %.4f" % (mean, 0.0 if np.isnan(std) else std)


def subject_confusions(run_dir, part="test"):
    z = np.load(os.path.join(run_dir, "predictions_%s.npz" % part))
    y, p, s = z["y_true"].astype(int), z["probs"].argmax(1), z["subject"]
    subs = np.unique(s)
    cm = np.zeros((len(subs), 3, 3))
    for i, sub in enumerate(subs):
        m = s == sub
        np.add.at(cm[i], (y[m], p[m]), 1)
    return subs, cm


def ba_from_cm(cm):
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return rec[cm.sum(1) > 0].mean()


def recalls_ba(cm):
    """cm (..., 3, 3) -> recalls (..., 3), BA (...) over the classes with support, full support flag (...)."""
    support = cm.sum(-1)
    rec = np.diagonal(cm, axis1=-2, axis2=-1) / np.maximum(support, 1)
    has = support > 0
    return rec, (rec * has).sum(-1) / np.maximum(has.sum(-1), 1), has.all(-1)


def bootstrap_intervals(df, n_boot=10000, seed=20251028):
    """Subject-cluster bootstrap of the seed-mean BA and recalls, validation and test.

    Each replicate draws the subjects of the partition with replacement (the same draw for
    every seed of the condition), recomputes each seed's confusion matrix from the drawn
    subjects and averages BA over seeds. Percentile 95% interval; models are fixed, so only
    the variability due to the sample of subjects is covered.
    """
    rows = []
    for (dataset, cond), g in df.groupby(["dataset", "condition"]):
        g = g.sort_values("seed")
        for part in ("val", "test"):
            cms = [subject_confusions(d, part) for d in g["dir"]]
            subs = cms[0][0]
            if any(len(s) != len(subs) or np.any(s != subs) for s, _ in cms):
                raise ValueError("%s/%s: seeds disagree on the %s subjects" % (dataset, cond, part))
            C = np.stack([cm for _, cm in cms])  # (seeds, subjects, 3, 3)
            rng = np.random.default_rng(seed)
            picks = rng.integers(0, len(subs), (n_boot, len(subs)))
            W = np.zeros((n_boot, len(subs)))
            np.add.at(W, (np.arange(n_boot)[:, None], picks), 1)
            rec_b, ba_b, full_b = recalls_ba(np.einsum("bs,ksij->bkij", W, C))
            rec_o, ba_o, _ = recalls_ba(C.sum(1))
            ba_b, rec_b = ba_b.mean(1), rec_b.mean(1)
            row = {"dataset": dataset, "condition": cond, "partition": part, "n_seeds": len(g),
                   "n_subjects": len(subs), "n_windows": int(C[0].sum()), "balanced_accuracy": float(ba_o.mean()),
                   "ci95_low": float(np.percentile(ba_b, 2.5)), "ci95_high": float(np.percentile(ba_b, 97.5)),
                   "boot_sd": float(ba_b.std(ddof=1)), "replicates": n_boot,
                   "replicates_missing_a_class": int((~full_b.all(1)).sum())}
            for k, c in enumerate(CLASS_NAMES):
                row["recall_" + c] = float(rec_o[:, k].mean())
                row["recall_%s_ci95_low" % c] = float(np.percentile(rec_b[:, k], 2.5))
                row["recall_%s_ci95_high" % c] = float(np.percentile(rec_b[:, k], 97.5))
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_markdown(ci):
    lines = ["# Balanced accuracy with subject-cluster bootstrap 95% CI (percentile, seed mean)\n",
             "| dataset | condition | seeds | validation BA [95% CI] | test BA [95% CI] | "
             "test recall drink / eat / other |", "|---|---|---|---|---|---|"]
    for (dataset, cond), g in ci.groupby(["dataset", "condition"], sort=False):
        v = g[g.partition == "val"].iloc[0]
        t = g[g.partition == "test"].iloc[0]
        lines.append("| %s | %s | %d | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] | %s |" % (
            dataset, cond, t.n_seeds, v.balanced_accuracy, v.ci95_low, v.ci95_high,
            t.balanced_accuracy, t.ci95_low, t.ci95_high,
            " / ".join("%.3f" % t["recall_" + c] for c in CLASS_NAMES)))
    n_val, n_test = ci[ci.partition == "val"].n_subjects.max(), ci[ci.partition == "test"].n_subjects.max()
    lines.append("\nSubjects are resampled with replacement (%d validation, %d test subjects), %d replicates; "
                 "the interval covers subject sampling only, not training variability (see the seed std "
                 "in summary.md). Replicates in which a class has no windows: see bootstrap_ci.csv."
                 % (n_val, n_test, ci.replicates.max()))
    return "\n".join(lines) + "\n"


def compare(df, a, b, dataset, n_boot=2000, seed=20251028):
    from scipy import stats
    A = df[(df.dataset == dataset) & (df.condition == a)].set_index("seed")
    B = df[(df.dataset == dataset) & (df.condition == b)].set_index("seed")
    common = sorted(set(A.index) & set(B.index))
    if len(common) < 2:
        return None
    da = A.loc[common, "test_balanced_accuracy"].to_numpy()
    db = B.loc[common, "test_balanced_accuracy"].to_numpy()
    res = {"A": a, "B": b, "dataset": dataset, "n_seeds": len(common), "mean_diff": float(np.mean(da - db))}
    res["wilcoxon_seeds_p"] = float(stats.wilcoxon(da, db).pvalue) if np.any(da != db) else 1.0
    res["ttest_seeds_p"] = float(stats.ttest_rel(da, db).pvalue)
    # subject level: confusion matrices per subject and seed
    cms_a = [subject_confusions(A.loc[s, "dir"]) for s in common]
    cms_b = [subject_confusions(B.loc[s, "dir"]) for s in common]
    subs = cms_a[0][0]
    per_sub_a = np.mean([[ba_from_cm(c) for c in cm] for _, cm in cms_a], axis=0)
    per_sub_b = np.mean([[ba_from_cm(c) for c in cm] for _, cm in cms_b], axis=0)
    res["n_subjects"] = len(subs)
    res["wilcoxon_subjects_p"] = float(stats.wilcoxon(per_sub_a, per_sub_b).pvalue)
    res["subjects_A_better"] = int(np.sum(per_sub_a > per_sub_b))
    rng = np.random.default_rng(seed)
    CA = np.stack([cm for _, cm in cms_a])  # (seeds, subjects, 3, 3)
    CB = np.stack([cm for _, cm in cms_b])
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(subs), len(subs))
        ba_a = np.mean([ba_from_cm(CA[k, pick].sum(0)) for k in range(len(common))])
        ba_b = np.mean([ba_from_cm(CB[k, pick].sum(0)) for k in range(len(common))])
        diffs.append(ba_a - ba_b)
    res["boot_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    return res


def volume_curve(summary, out_dir, dataset="deo"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[(summary.dataset == dataset) & summary.condition.str.match(r"^diff_gru_x[\dp]+$")].copy()
    if s.empty:
        return
    s["mult"] = s.condition.str.replace("diff_gru_x", "").str.replace("p", ".").astype(float)
    s = s.sort_values("mult")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for col, label, mk in (("test_balanced_accuracy", "test", "o"), ("val_balanced_accuracy", "validation", "s")):
        ax.errorbar(s["mult"], s[col + "_mean"], yerr=s[col + "_std"], marker=mk, capsize=3, label=label)
    base = summary[(summary.dataset == dataset) & (summary.condition == "scratch")]
    if not base.empty:
        m, sd = base["test_balanced_accuracy_mean"].iloc[0], base["test_balanced_accuracy_std"].iloc[0]
        ax.axhline(m, color="gray", ls="--", label="no pretraining (test)")
        ax.axhspan(m - sd, m + sd, color="gray", alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(s["mult"])
    ax.set_xticklabels(["%g" % v for v in s["mult"]])
    ax.set_xlabel("synthetic pretraining windows (multiples of R real minority windows)")
    ax.set_ylabel("balanced accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, "volume_curve." + ext), dpi=200)
    plt.close(fig)


def learning_curves(runs, out_dir, conds, seed=0, dataset="deo"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs = {}
    for c in conds:
        p = os.path.join(runs, "finetune", dataset, c, "seed%d" % seed, "history.csv")
        if os.path.exists(p):
            hs[c] = pd.read_csv(p)
    if not hs:
        return
    fig, axes = plt.subplots(2, len(hs), figsize=(4.2 * len(hs), 5.2), sharex=True, squeeze=False)
    xmax = max(len(h) for h in hs.values())
    for j, (c, h) in enumerate(hs.items()):
        ep = np.arange(1, len(h) + 1)
        axes[0, j].plot(ep, h["loss"], label="train")
        axes[0, j].plot(ep, h["val_loss"], label="validation")
        axes[0, j].set_title(c, fontsize=10)
        axes[0, j].set_ylabel("focal loss")
        axes[1, j].plot(ep, h["accuracy"], label="train accuracy")
        axes[1, j].plot(ep, h["val_accuracy"], label="validation accuracy")
        if "val_balanced_accuracy" in h:
            axes[1, j].plot(ep, h["val_balanced_accuracy"], label="validation balanced acc.")
        axes[1, j].set_xlabel("epoch")
        axes[1, j].set_ylabel("accuracy")
        best = int(np.argmax(h["val_balanced_accuracy"])) + 1 if "val_balanced_accuracy" in h else None
        for ax in axes[:, j]:
            ax.set_xlim(1, xmax)
            ax.grid(alpha=0.3)
            if best:
                ax.axvline(best, color="k", ls=":", lw=1)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, "learning_curves." + ext), dpi=200)
    plt.close(fig)


def class_balance(runs, data_root):
    rows = []
    tab = os.path.join(data_root, "deo", "dataset_table.csv")
    if os.path.exists(tab):
        t = pd.read_csv(tab)
        tr = t[t.partition == "train"].iloc[0]
        rows.append({"set": "real training windows", **{c: int(tr[c]) for c in CLASS_NAMES}})
    for man in sorted(glob.glob(os.path.join(runs, "corpora", "*", "manifest.json"))):
        m = read_json(man)
        pc = m["per_class"]
        rows.append({"set": "corpus " + os.path.basename(os.path.dirname(man)),
                     "DRINK": pc.get("drink", 0), "EAT": pc.get("eat", 0), "OTHER": pc.get("other", 0)})
    df = pd.DataFrame(rows)
    if not df.empty:
        tot = df[CLASS_NAMES].sum(1)
        for c in CLASS_NAMES:
            df[c + "_%"] = (100 * df[c] / tot).round(2)
    return df


def cost_table(runs):
    rows = []
    for f in glob.glob(os.path.join(runs, "generators", "*", "train_done.json")):
        d = read_json(f)
        rows.append({"stage": "generator training", "name": os.path.basename(os.path.dirname(f)),
                     "seconds": d.get("seconds_this_run"), "params": d.get("n_params"), "hardware": d.get("gpu")})
    for f in glob.glob(os.path.join(runs, "generators", "*", "samples_shard*.json")):
        if f.endswith(".progress.json"):
            continue
        d = read_json(f)
        rows.append({"stage": "sampling", "name": os.path.relpath(f, runs), "seconds": d.get("seconds_this_run"),
                     "segments": d.get("n"), "hardware": d.get("gpu")})
    for f in glob.glob(os.path.join(runs, "pretrain", "*", "seed*", "done.json")):
        d = read_json(f)
        rows.append({"stage": "pretraining", "name": os.path.relpath(os.path.dirname(f), runs),
                     "seconds": d["timing_s"].get("pretrain_fit"), "windows": d.get("n_windows"),
                     "params": d.get("n_params"), "hardware": ",".join(d["environment"].get("gpu_names", []) or [])})
    for f in glob.glob(os.path.join(runs, "finetune", "*", "*", "seed*", "done.json")):
        d = read_json(f)
        rows.append({"stage": "fine-tuning", "name": os.path.relpath(os.path.dirname(f), runs),
                     "seconds": d["timing_s"].get("fit"), "epochs": d.get("epochs_run"), "params": d.get("n_params"),
                     "hardware": ",".join(d["environment"].get("gpu_names", []) or [])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--main", default="diff_gru_x10", help="main diffusion condition")
    args = ap.parse_args()
    out = os.path.join(args.runs, "analysis")
    os.makedirs(out, exist_ok=True)

    df = collect(args.runs)
    if df.empty:
        print("[analyze] no finished fine-tuning runs found")
        return
    df.to_csv(os.path.join(out, "all_runs.csv"), index=False)
    summary = summarise(df)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)

    lines = ["# Results (mean ± std over seeds)\n",
             "| dataset | condition | seeds | test BA | val BA | macro F1 | weighted F1 | mAP (prob.) | "
             "sens drink / eat / other |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        sens = " / ".join("%.3f" % r["test_sens_%s_mean" % c] for c in CLASS_NAMES)
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            r.dataset, r.condition, r.n_seeds,
            fmt_pm(r.test_balanced_accuracy_mean, r.test_balanced_accuracy_std),
            fmt_pm(r.val_balanced_accuracy_mean, r.val_balanced_accuracy_std),
            fmt_pm(r.test_f1_macro_mean, r.test_f1_macro_std), fmt_pm(r.test_f1_weighted_mean, r.test_f1_weighted_std),
            fmt_pm(r.test_mAP_score_mean, r.test_mAP_score_std), sens))
    note = LEGACY_NOTE if (df.condition == "legacy_recipe_pretrained").any() else ""
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n" + note)

    pairs = []
    conds = set(df[df.dataset == "deo"].condition)
    main = args.main if args.main in conds else None
    recipe_re = re.compile(r"^(diff_gru_x[\dp]+)_(ep\d+|enclr[\dp]+|ep\d+_enclr[\dp]+)$")
    for c in sorted(conds):
        if c != "scratch" and "scratch" in conds:
            pairs.append(("deo", c, "scratch"))
        r = recipe_re.match(c)
        if r and r.group(1) in conds:
            pairs.append(("deo", c, r.group(1)))  # recipe ablation vs the same volume with the base recipe
        elif main and c not in (main, "scratch") and not c.startswith("legacy"):
            pairs.append(("deo", main, c))
    for c in sorted(set(df[df.dataset == "deo_samehand"].condition) - {"scratch"}):
        pairs.append(("deo_samehand", c, "scratch"))
    comp = [r for r in (compare(df, a, b, ds) for ds, a, b in pairs) if r]
    pd.DataFrame(comp).to_csv(os.path.join(out, "comparisons.csv"), index=False)
    lines = ["# Paired comparisons (A - B, test balanced accuracy)\n",
             "| dataset | A | B | seeds | mean diff | p Wilcoxon (seeds) | p t-test (seeds) | "
             "p Wilcoxon (subjects) | subjects A>B | bootstrap 95% CI |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in comp:
        lines.append("| %s | %s | %s | %d | %+.4f | %.4f | %.4f | %.4f | %d/%d | [%+.4f, %+.4f] |" % (
            r["dataset"], r["A"], r["B"], r["n_seeds"], r["mean_diff"], r["wilcoxon_seeds_p"], r["ttest_seeds_p"],
            r["wilcoxon_subjects_p"], r["subjects_A_better"], r["n_subjects"], *r["boot_ci95"]))
    lines.append("\nWith n seeds the smallest two-sided Wilcoxon p-value is 2/2^n (0.0625 for 5 seeds); "
                 "the subject-level test and the bootstrap interval are the primary evidence.")
    open(os.path.join(out, "comparisons.md"), "w").write("\n".join(lines) + "\n" + note)

    rank = {k: i for i, k in enumerate(zip(summary.dataset, summary.condition))}
    ci = bootstrap_intervals(df)
    ci["_rank"] = [rank[k] for k in zip(ci.dataset, ci.condition)]
    ci = ci.sort_values(["_rank", "partition"], ascending=[True, False]).drop(columns="_rank")
    ci.to_csv(os.path.join(out, "bootstrap_ci.csv"), index=False)
    open(os.path.join(out, "bootstrap_ci.md"), "w").write(bootstrap_markdown(ci) + note)

    volume_curve(summary, out)
    learning_curves(args.runs, out, ["scratch", args.main])
    class_balance(args.runs, args.data_root).to_csv(os.path.join(out, "class_balance.csv"), index=False)
    cost_table(args.runs).to_csv(os.path.join(out, "cost.csv"), index=False)
    print(open(os.path.join(out, "summary.md")).read())
    print(open(os.path.join(out, "bootstrap_ci.md")).read())
    print(open(os.path.join(out, "comparisons.md")).read())


if __name__ == "__main__":
    main()
