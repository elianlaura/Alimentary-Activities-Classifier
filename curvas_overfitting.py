"""Fine-tuning (stage 2) curves: train/validation loss, accuracy and balanced accuracy per epoch.

Thin lines = each seed; thick line = mean over the seeds still training at that epoch
(drawn while >= 3 seeds remain). Dots on val loss = each seed's val-loss minimum; dots on
val BA = each seed's saved epoch (best val BA). Train BA is only known for the saved epoch
(metrics_train.json): per-epoch train BA was not logged, so the accuracy row shows plain
accuracy for train and validation (dominated by OTHER, ~86 % of the windows) plus val BA dashed.
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/mnt/beegfs/home/elian.riveros/deo_runs/finetune/deo"
OUT = "/home/elian.riveros/deo-classifier/curvas_overfitting"
EXPS = [("0", "scratch", "Sin pre-entrenamiento"),
        ("5h", "diff_gru_x20", "×20 · 20 ép."),
        ("5i", "diff_gru_x40", "×40 · 20 ép."),
        ("5k", "diff_gru_x20_ep40", "×20 · 40 ép."),
        ("5j", "diff_gru_x10_ep40", "×10 · 40 ép."),
        ("7", "diff_gru_all_x10", "×10 beber+comer+otro")]
ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="+", default=None, help="experiment tags to plot, e.g. --only 0")
ap.add_argument("--out", default=OUT)
args = ap.parse_args()
if args.only:
    EXPS = [e for e in EXPS if e[0] in args.only]
OUT = args.out
TRAIN, VAL = "#2a78d6", "#eb6834"          # categorical slots 1-2 of the reference palette
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "font.family": "DejaVu Sans"})


def mean_curve(curves, min_seeds=3):
    n = max(len(c) for c in curves)
    m = np.full(n, np.nan)
    for e in range(n):
        vals = [c[e] for c in curves if len(c) > e]
        if len(vals) >= min_seeds:
            m[e] = np.mean(vals)
    return m


data = {}
for tag, cond, _ in EXPS:
    seeds = []
    for s in range(5):
        h = pd.read_csv(f"{R}/{cond}/seed{s}/history.csv")
        d = json.load(open(f"{R}/{cond}/seed{s}/done.json"))
        tr = json.load(open(f"{R}/{cond}/seed{s}/metrics_train.json"))
        seeds.append(dict(loss=h.loss.values, val_loss=h.val_loss.values, val_ba=h.val_balanced_accuracy.values,
                          acc=h.accuracy.values, val_acc=h.val_accuracy.values,
                          best=d["best_epoch"], train_ba_best=tr["balanced_accuracy"]))
    data[tag] = seeds

loss_max = max(s["val_loss"].max() for v in data.values() for s in v) * 1.05
ba_min = min(min(s["val_ba"].min(), s["acc"].min(), s["val_acc"].min()) for v in data.values() for s in v)
fig, axes = plt.subplots(2, len(EXPS), figsize=(max(19 / 6 * len(EXPS), 6.5), 6.6), sharey="row", sharex=True, squeeze=False)
fig.patch.set_facecolor("white")
for j, (tag, cond, label) in enumerate(EXPS):
    seeds = data[tag]
    a_loss, a_ba = axes[0, j], axes[1, j]
    for s in seeds:
        x = np.arange(1, len(s["loss"]) + 1)
        a_loss.plot(x, s["loss"], color=TRAIN, lw=0.7, alpha=0.3)
        a_loss.plot(x, s["val_loss"], color=VAL, lw=0.7, alpha=0.3)
        m = int(np.argmin(s["val_loss"]))
        a_loss.plot(m + 1, s["val_loss"][m], "o", ms=4.5, color=VAL, mec="white", mew=1, zorder=4)
        a_ba.plot(x, s["acc"], color=TRAIN, lw=0.7, alpha=0.25)
        a_ba.plot(x, s["val_acc"], color=VAL, lw=0.7, alpha=0.25)
        b = s["best"] - 1
        a_ba.plot(b + 1, s["val_ba"][b], "o", ms=4.5, color=VAL, mec="white", mew=1, zorder=4)
        a_ba.plot(b + 1, s["train_ba_best"], "s", ms=4.5, color=TRAIN, mec="white", mew=1, zorder=4)
    for key, color, ax, ls in (("loss", TRAIN, a_loss, "-"), ("val_loss", VAL, a_loss, "-"),
                               ("acc", TRAIN, a_ba, "-"), ("val_acc", VAL, a_ba, "-"), ("val_ba", VAL, a_ba, "--")):
        mc = mean_curve([s[key] for s in seeds])
        ax.plot(np.arange(1, len(mc) + 1), mc, color=color, lw=2 if ls == "-" else 1.6, ls=ls)
    a_loss.set_title(f"{tag} · {cond}\n{label}", fontsize=9.5, loc="left")
    for ax in (a_loss, a_ba):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    a_loss.set_ylim(0, loss_max)
    a_ba.set_ylim(ba_min - 0.01, 1.0)
    a_ba.set_xlabel("Época")
    if j == 0:
        a_loss.set_ylabel("Loss (focal)")
        a_ba.set_ylabel("Accuracy / balanced accuracy")
        # direct labels on the first column
        a_loss.annotate("train", xy=(30, data[tag][0]["loss"][29]), xytext=(31, 0.07), color=INK2, fontsize=8.5)
        a_loss.annotate("validación", xy=(30, 0.3), xytext=(25, 0.42), color=INK2, fontsize=8.5)

handles = [plt.Line2D([], [], color=TRAIN, lw=2, label="Train: loss / accuracy (media; finas = cada semilla)"),
           plt.Line2D([], [], color=VAL, lw=2, label="Validación: loss / accuracy (media; finas = cada semilla)"),
           plt.Line2D([], [], color=VAL, lw=1.6, ls="--", label="Validación: balanced accuracy (media)"),
           plt.Line2D([], [], ls="", marker="o", ms=6, color=VAL, mec="white", label="Fila 1: mínimo de val loss de cada semilla"),
           plt.Line2D([], [], ls="", marker="o", ms=6, color=VAL, mec="white", label="Fila 2: época guardada (mejor val BA), val BA"),
           plt.Line2D([], [], ls="", marker="s", ms=6, color=TRAIN, mec="white", label="Fila 2: train BA en la época guardada")]
fig.legend(handles=handles, loc="lower center", ncol=3 if len(EXPS) > 2 else 1, frameon=False, fontsize=8.5, labelcolor=INK2,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Fine-tuning (etapa 2): loss, accuracy y balanced accuracy por época" + (" " if len(EXPS) > 2 else "\n") + "(5 semillas, datos con duplicados)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.09 if len(EXPS) > 2 else 0.15, 1, 0.95))
fig.savefig(OUT + ".png", dpi=170, bbox_inches="tight")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
print(OUT + ".png")
