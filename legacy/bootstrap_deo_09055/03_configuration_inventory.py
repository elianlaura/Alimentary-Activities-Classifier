"""Paso 3 del protocolo: inventario de las configuraciones comparadas en el barrido.

Recorre las subcarpetas de recurrent_models_20251026-024834/ y documenta, por ejecucion,
los hiperparametros, el modelo preentrenado y la balanced accuracy de train/val/test.
No reselecciona el maximo por replica: la ejecucion historica queda fija en el bootstrap.
"""
import argparse
import hashlib
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HP_FIELDS = ["modeltype", "gpu", "test_type", "k_folds", "overlap_shift", "norm_method",
             "dataset", "n_dense", "learning_rate", "dropout_rate", "n_batch", "n_epochs",
             "LSTM_layers", "sensors", "activation", "head_mode"]


def parse_hyperparams(path):
    """Invierte el formato de recurrent_models_main.py:

    {modeltype}_{gpu}_{test_type}_{k_folds}_{overlap_shift}_{norm_method}_{dataset}
    _{n_dense}_{learning_rate}_{dropout_rate}_{n_batch}_{n_epochs}_{LSTM_layers}
    _{sensors}_{activation}_{mode}

    Se parsea desde el final (9 campos fijos) y se retira el nombre conocido del
    dataset, que contiene guiones bajos.
    """
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    out = {"hyperparams_file": os.path.basename(path)}
    m = re.match(r"hyperparams_(\d+)\.txt", out["hyperparams_file"])
    out["config_index"] = int(m.group(1)) if m else None
    if not lines:
        return out
    raw = lines[0]
    out["hyperparams_raw"] = raw
    out["pretrained_model"] = lines[1] if len(lines) > 1 else ""

    tokens = raw.split("_")
    tail_names = ["n_dense", "learning_rate", "dropout_rate", "n_batch", "n_epochs",
                  "LSTM_layers", "sensors", "activation", "head_mode"]
    if len(tokens) <= len(tail_names) + 5:
        return out
    for name, value in zip(tail_names, tokens[-len(tail_names):]):
        out[name] = value

    head = "_".join(tokens[:-len(tail_names)])
    suffix = "_" + C.DATASET
    if head.endswith(suffix):
        out["dataset"] = C.DATASET
        head = head[:-len(suffix)]
    head_tokens = head.split("_")
    if len(head_tokens) >= 5:
        out["norm_method"] = head_tokens[-1]
        out["overlap_shift"] = head_tokens[-2]
        out["k_folds"] = head_tokens[-3]
        out["test_type"] = head_tokens[-4]
        out["gpu"] = head_tokens[-5]
        out["modeltype"] = "_".join(head_tokens[:-5])
    return out


def parse_metrics(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        text = fh.read()
    out = {}
    for key in ("mAP score", "precision", "recall", "f1_score", "balanced_accuracy_score"):
        m = re.search(re.escape(key) + r":\s*([\d.]+)", text)
        if m:
            out[key.replace(" ", "_")] = float(m.group(1))
    m = re.search(r"Sensitivity per class:,\s*\[(.*?)\]", text, flags=re.S)
    if m:
        # El archivo historico guarda "[np.float64(0.8849), ...]".
        out["sensitivity_per_class"] = [float(v) for v in re.findall(r"\d+\.\d+|\d+", m.group(1))]
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sweep-dir", default=C.SWEEP_DIR)
    args = p.parse_args()

    rows = []
    subdirs = sorted(d for d in os.listdir(args.sweep_dir)
                     if os.path.isdir(os.path.join(args.sweep_dir, d)))
    for name in subdirs:
        d = os.path.join(args.sweep_dir, name)
        files = os.listdir(d)
        run_id = name.rsplit("_", 1)[-1]
        row = {"run_folder": name, "run_id": run_id, "selected_run": name == os.path.basename(C.RUN_DIR)}

        hp = [f for f in files if f.startswith("hyperparams_") and f.endswith(".txt")]
        if hp:
            row.update(parse_hyperparams(os.path.join(d, sorted(hp)[0])))

        for split in ("train", "val", "test"):
            met = os.path.join(d, "metrics_%s_%s_%s.txt" % (split, C.DATASET, run_id))
            m = parse_metrics(met)
            row["has_%s_eval" % split] = bool(m)
            if "balanced_accuracy_score" in m:
                row["balanced_accuracy_%s" % split] = m["balanced_accuracy_score"]
            if split == "test" and "sensitivity_per_class" in m:
                row["test_sensitivity_per_class"] = m["sensitivity_per_class"]

        split_file = os.path.join(d, "%s_1.txt" % C.DATASET)
        row["split_file_md5"] = md5(split_file) if os.path.exists(split_file) else ""
        ckpt = [f for f in files if f.startswith("best_model_") and f.endswith(".keras")]
        row["best_checkpoint"] = ckpt[0] if ckpt else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ranking descendente por balanced accuracy de prueba (criterio de seleccion historico).
    if "balanced_accuracy_test" in df:
        df["rank_by_test_balacc"] = df["balanced_accuracy_test"].rank(ascending=False, method="min")
    cols = ([c for c in ["run_folder", "run_id", "config_index", "selected_run",
                         "rank_by_test_balacc", "balanced_accuracy_train",
                         "balanced_accuracy_val", "balanced_accuracy_test"] if c in df]
            + [c for c in HP_FIELDS if c in df]
            + [c for c in ["pretrained_model", "hyperparams_raw", "hyperparams_file",
                           "has_train_eval", "has_val_eval", "has_test_eval",
                           "test_sensitivity_per_class", "best_checkpoint", "split_file_md5"] if c in df])
    df = df[cols].sort_values("run_id").reset_index(drop=True)

    out_csv = os.path.join(args.out_dir, "configuration_inventory.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[out] %s" % out_csv)

    n_with_test = int(df["has_test_eval"].sum())
    md5s = df["split_file_md5"].replace("", pd.NA).dropna().unique().tolist()
    best = df.loc[df["balanced_accuracy_test"].idxmax()] if "balanced_accuracy_test" in df else None
    sel = df[df["selected_run"]].iloc[0] if df["selected_run"].any() else None

    summary = {
        "protocol": "SnsEnc/docs/protocolo_bootstrap_deo_09055.md",
        "stage": "paso_3_inventario",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sweep_dir": args.sweep_dir,
        "n_runs": int(len(df)),
        "n_runs_with_test_eval": n_with_test,
        "identical_test_split_across_runs": len(md5s) == 1,
        "split_file_md5": md5s,
        "sweep_grid": {
            "pretrained_models": sorted(df["pretrained_model"].dropna().unique().tolist())
            if "pretrained_model" in df else [],
            "learning_rate": sorted(df["learning_rate"].dropna().unique().tolist())
            if "learning_rate" in df else [],
            "activation": sorted(df["activation"].dropna().unique().tolist())
            if "activation" in df else [],
            "head_mode": sorted(df["head_mode"].dropna().unique().tolist())
            if "head_mode" in df else [],
        },
        "selected_run": {
            "run_folder": None if sel is None else sel["run_folder"],
            "config_index": None if sel is None else sel.get("config_index"),
            "balanced_accuracy_test": None if sel is None else sel.get("balanced_accuracy_test"),
            "balanced_accuracy_val": None if sel is None else sel.get("balanced_accuracy_val"),
            "rank_by_test_balacc": None if sel is None else sel.get("rank_by_test_balacc"),
        },
        "best_test_run": {
            "run_folder": None if best is None else best["run_folder"],
            "balanced_accuracy_test": None if best is None else best["balanced_accuracy_test"],
        },
        "selection_criterion": ("La ejecucion historica se eligio por su balanced accuracy de "
                                "PRUEBA, la mas alta entre las configuraciones evaluadas del "
                                "barrido. El test intervino en la seleccion de la configuracion, "
                                "no en el ajuste de pesos."),
        "caveat": ("Mantener fija la ejecucion historica en el bootstrap principal. Reseleccionar "
                   "el maximo en cada replica estima otra cantidad y no corrige el sesgo de "
                   "seleccion. Si hubo comparaciones fuera de esta carpeta, incorporarlas al "
                   "proceso de seleccion documentado."),
    }
    C.write_json(os.path.join(args.out_dir, "configuration_inventory_summary.json"), summary)

    print("[inv] %d ejecuciones | %d con evaluacion de prueba" % (len(df), n_with_test))
    print("[inv] particion de prueba identica en todas: %s" % (len(md5s) == 1))
    if sel is not None:
        print("[inv] seleccionada: %s  BA_test=%.4f  rank=%s"
              % (sel["run_folder"], sel["balanced_accuracy_test"], sel.get("rank_by_test_balacc")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
