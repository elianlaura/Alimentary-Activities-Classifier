"""Paso 2 del protocolo: bootstrap por sujetos sobre la balanced accuracy de prueba.

Unidad de remuestreo: sujeto de prueba completo (cluster bootstrap).
Replicas: 10.000. Semilla: 20251028. Intervalo: percentil bilateral del 95 %.
El modelo permanece fijo: solo se remuestrean las matrices de confusion por sujeto.

No se remuestrean ventanas individuales; ver ClusterBootstrap
https://pmc.ncbi.nlm.nih.gov/articles/PMC7148287/
"""
import argparse
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--replicates", type=int, default=C.BOOTSTRAP_REPLICATES)
    p.add_argument("--seed", type=int, default=C.BOOTSTRAP_SEED)
    p.add_argument("--alpha", type=float, default=C.BOOTSTRAP_ALPHA)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir
    cm_path = os.path.join(out_dir, "subject_confusion_matrices.csv")
    long_cm = pd.read_csv(cm_path)

    # Orden de sujetos: el del archivo de particion historico (fija la semilla de forma reproducible).
    subjects = C.parse_split_file()["uuid_test"]
    present = set(long_cm["subject_id"].unique())
    if present != set(subjects):
        raise SystemExit("Los sujetos de %s no coinciden con la particion guardada." % cm_path)

    n_subjects = len(subjects)
    n_classes = len(C.CLASSES)
    cls_index = {c: i for i, c in enumerate(C.CLASSES)}

    C_mat = np.zeros((n_subjects, n_classes, n_classes), dtype=np.int64)
    subj_index = {s: i for i, s in enumerate(subjects)}
    for row in long_cm.itertuples(index=False):
        C_mat[subj_index[row.subject_id], cls_index[int(row.y_true)], cls_index[int(row.y_pred)]] = row.count

    C_total = C_mat.sum(axis=0)
    support_total = C_total.sum(axis=1)
    recalls_obs = C_total.diagonal() / support_total
    ba_obs = float(recalls_obs.mean())

    print("[obs] ventanas = %d | sujetos = %d" % (int(C_total.sum()), n_subjects))
    print("[obs] soporte por clase = %s" % support_total.tolist())
    print("[obs] recalls = %s" % np.round(recalls_obs, 6).tolist())
    print("[obs] balanced accuracy observada = %.12f (%.4f)" % (ba_obs, ba_obs))

    if int(C_total.sum()) != C.HISTORIC["n_test_windows"]:
        raise SystemExit("Total de ventanas %d != %d historico"
                         % (int(C_total.sum()), C.HISTORIC["n_test_windows"]))
    if "%.4f" % ba_obs != C.HISTORIC["balanced_accuracy_4dp"]:
        raise SystemExit("BA observada %.4f != %s historica"
                         % (ba_obs, C.HISTORIC["balanced_accuracy_4dp"]))

    # Cobertura de clases por sujeto (diagnostico de replicas potencialmente invalidas).
    subject_support = C_mat.sum(axis=2)
    subjects_missing_class = [subjects[i] for i in range(n_subjects) if (subject_support[i] == 0).any()]
    print("[obs] sujetos sin alguna clase real: %s" % (subjects_missing_class or "ninguno"))

    rng = np.random.default_rng(args.seed)
    R = args.replicates
    t0 = time.time()
    idx = rng.integers(0, n_subjects, size=(R, n_subjects))
    # Multiplicidades por replica: cada aparicion suma la matriz del sujeto otra vez.
    mult = np.zeros((R, n_subjects), dtype=np.int64)
    np.add.at(mult, (np.repeat(np.arange(R), n_subjects), idx.ravel()), 1)
    if not np.array_equal(mult.sum(axis=1), np.full(R, n_subjects)):
        raise SystemExit("Las multiplicidades no suman 29 por replica.")

    C_b = (mult @ C_mat.reshape(n_subjects, n_classes * n_classes)).reshape(R, n_classes, n_classes)
    support_b = C_b.sum(axis=2)
    valid = (support_b > 0).all(axis=1)
    n_invalid = int((~valid).sum())

    recalls_b = np.full((R, n_classes), np.nan)
    np.divide(np.diagonal(C_b, axis1=1, axis2=2), support_b,
              out=recalls_b, where=support_b > 0)
    ba_b = np.where(valid, np.nanmean(recalls_b, axis=1), np.nan)
    print("[boot] %d replicas en %.1fs | invalidas = %d" % (R, time.time() - t0, n_invalid))

    reps = pd.DataFrame({
        "replicate": np.arange(1, R + 1, dtype=np.int64),
        "balanced_accuracy": ba_b,
        "recall_drink": recalls_b[:, 0],
        "recall_eat": recalls_b[:, 1],
        "recall_another": recalls_b[:, 2],
        "valid": valid,
        "n_windows": C_b.sum(axis=(1, 2)),
    })
    reps_path = os.path.join(out_dir, "bootstrap_replicates.csv")
    reps.to_csv(reps_path, index=False, float_format="%.10f")
    print("[out] %s" % reps_path)

    lo_q, hi_q = 100 * args.alpha / 2.0, 100 * (1 - args.alpha / 2.0)
    if n_invalid == 0:
        ci = [float(np.percentile(ba_b, lo_q)), float(np.percentile(ba_b, hi_q))]
        ci_status = "publicable"
        recall_ci = {C.CLASS_NAMES[j].lower(): [float(np.percentile(recalls_b[:, j], lo_q)),
                                                float(np.percentile(recalls_b[:, j], hi_q))]
                     for j in range(n_classes)}
    else:
        ci = None
        ci_status = ("retenido: %d replicas sin alguna clase real. El protocolo prohibe promediar "
                     "solo las clases presentes, asignar recall cero o reemplazar la replica en "
                     "silencio. Revisar la cobertura de clases por sujeto antes de publicar." % n_invalid)
        recall_ci = None
        print("[ATENCION] %s" % ci_status)

    summary = {
        "protocol": "SnsEnc/docs/protocolo_bootstrap_deo_09055.md",
        "stage": "paso_2_bootstrap",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join([sys.executable] + sys.argv),
        "method": {
            "name": "bootstrap percentil bilateral del 95 % con remuestreo de sujetos completos",
            "resampling_unit": "sujeto de prueba completo (cluster bootstrap)",
            "reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7148287/",
            "replicates": R,
            "seed": args.seed,
            "rng": "numpy.random.default_rng (PCG64), integers(0, %d) con reemplazo" % n_subjects,
            "alpha": args.alpha,
            "percentiles": [lo_q, hi_q],
            "percentile_interpolation": "numpy.percentile por defecto (lineal)",
            "subject_order": subjects,
            "model": "checkpoint historico fijo en todas las replicas",
            "classes": C.CLASSES,
        },
        "observed": {
            "balanced_accuracy": ba_obs,
            "balanced_accuracy_4dp": "%.4f" % ba_obs,
            "recalls": {C.CLASS_NAMES[j].lower(): float(recalls_obs[j]) for j in range(n_classes)},
            "confusion_matrix": C_total.tolist(),
            "class_support": {C.CLASS_NAMES[j].lower(): int(support_total[j]) for j in range(n_classes)},
            "n_subjects": n_subjects,
            "n_windows": int(C_total.sum()),
        },
        "interval": {
            "ci95_balanced_accuracy": ci,
            "ci95_recalls": recall_ci,
            "status": ci_status,
        },
        "diagnostics": {
            "invalid_replicates": n_invalid,
            "subjects_missing_a_true_class": subjects_missing_class,
            "bootstrap_mean": float(np.nanmean(ba_b)),
            "bootstrap_sd": float(np.nanstd(ba_b, ddof=1)),
            "bootstrap_median": float(np.nanmedian(ba_b)),
            "bootstrap_min": float(np.nanmin(ba_b)),
            "bootstrap_max": float(np.nanmax(ba_b)),
            "bias_mean_minus_observed": float(np.nanmean(ba_b) - ba_obs),
            "note": ("La estimacion puntual publicada es la BA observada; la media de las replicas "
                     "es solo diagnostico. La informacion independiente proviene de 29 sujetos."),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "node": platform.node(),
        },
        "inputs": {"subject_confusion_matrices": C.file_stat(cm_path)},
        "outputs": {"bootstrap_replicates": reps_path},
    }
    C.write_json(os.path.join(out_dir, "bootstrap_summary.json"), summary)

    if ci is not None:
        print("\n[OK] BA = %.4f | IC95 percentil = [%.4f, %.4f]" % (ba_obs, ci[0], ci[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
