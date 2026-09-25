"""Paso 1 del protocolo: recuperar las predicciones del experimento historico.

Reconstruye la particion de prueba original a partir de los sujetos GUARDADOS en
eatdrinkanother_94u_1.txt (no se vuelve a sortear la particion), ejecuta inferencia
una vez con el checkpoint historico y exporta predicciones por ventana con su sujeto.

Replica el flujo archivado utils.get_processed_fold(..., overlap=False) con seg5=True:
  - x_test = columnas 5: del csv, float32, reshape (-1, 500, 9)
  - y_test = columna 2 remapeada con dict2 = {valor: indice} sobre np.unique(df[2]) global
  - nan_to_num(nan=0, posinf=0, neginf=0)
  - clase = argmax(model.predict(x_test))
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
    p.add_argument("--csv", default=C.SOURCE_CSV)
    p.add_argument("--checkpoint", default=C.CHECKPOINT)
    p.add_argument("--split-file", default=C.SPLIT_FILE)
    p.add_argument("--chunksize", type=int, default=2000,
                   help="filas por bloque al leer el csv de 10 GB")
    p.add_argument("--batch-size", type=int, default=32,
                   help="batch de model.predict; 32 es el valor por defecto de Keras usado por el evaluador original")
    p.add_argument("--hash-csv", dest="hash_csv", action="store_true", default=True)
    p.add_argument("--no-hash-csv", dest="hash_csv", action="store_false")
    return p.parse_args()


def read_test_partition(csv_path, test_users, chunksize):
    """Un solo barrido del csv: uniques globales de la columna 2 + filas de los sujetos de prueba."""
    test_set = set(test_users)
    global_act_values = set()
    seen_users = set()
    x_blocks, meta_blocks = [], []
    rows_seen = 0
    t0 = time.time()

    reader = pd.read_csv(csv_path, header=None, chunksize=chunksize, low_memory=False)
    for k, chunk in enumerate(reader):
        n = len(chunk)
        global_act_values.update(np.asarray(pd.unique(chunk[2])).tolist())
        seen_users.update(np.asarray(pd.unique(chunk[0])).tolist())
        mask = chunk[0].isin(test_set).to_numpy()
        if mask.any():
            sub = chunk.loc[mask]
            block = sub.iloc[:, C.N_META_COLS:].to_numpy(dtype=np.float32)
            if block.shape[1] != C.N_TIMESTEPS * C.N_FEATURES:
                raise ValueError("Se esperaban %d columnas de datos, se leyeron %d"
                                 % (C.N_TIMESTEPS * C.N_FEATURES, block.shape[1]))
            x_blocks.append(block)
            meta_blocks.append(pd.DataFrame({
                "window_id": (rows_seen + np.flatnonzero(mask)).astype(np.int64),
                "subject_id": sub[0].to_numpy().astype(str),
                "timestamp": sub[1].to_numpy(),
                "act_id_raw": sub[2].to_numpy(),
                "method_id": sub[3].to_numpy().astype(str),
                "act_name": sub[4].to_numpy().astype(str),
            }))
        rows_seen += n
        if (k + 1) % 20 == 0:
            print("[csv] %d filas leidas (%d de prueba) %.1fs"
                  % (rows_seen, sum(len(m) for m in meta_blocks), time.time() - t0), flush=True)

    if not x_blocks:
        raise RuntimeError("Ninguna fila del csv pertenece a los sujetos de prueba guardados.")

    x_flat = np.concatenate(x_blocks, axis=0)
    del x_blocks
    meta = pd.concat(meta_blocks, ignore_index=True)
    print("[csv] total filas %d | filas de prueba %d | %.1fs"
          % (rows_seen, len(meta), time.time() - t0), flush=True)
    return x_flat, meta, sorted(global_act_values), rows_seen, sorted(seen_users)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        print("[check] %-42s %s %s" % (name, "OK" if ok else "FALLA", detail), flush=True)
        return ok

    splits = C.parse_split_file(args.split_file)
    test_users = splits["uuid_test"]
    print("[split] train=%d val=%d test=%d"
          % (len(splits["uuid_train"]), len(splits["uuid_val"]), len(test_users)))
    check("sujetos_train_historicos", len(splits["uuid_train"]) == C.HISTORIC["n_train_subjects"],
          len(splits["uuid_train"]))
    check("sujetos_val_historicos", len(splits["uuid_val"]) == C.HISTORIC["n_val_subjects"],
          len(splits["uuid_val"]))
    check("sujetos_test_historicos", len(test_users) == C.HISTORIC["n_test_subjects"], len(test_users))
    check("sujetos_test_unicos", len(set(test_users)) == len(test_users), len(set(test_users)))

    x_flat, meta, act_values, total_rows, all_users = read_test_partition(
        args.csv, test_users, args.chunksize)

    # dict2 del flujo original: se construye sobre np.unique(df[2]) de TODO el dataframe.
    dict2 = {v: i for i, v in enumerate(act_values)}
    print("[labels] valores globales de la columna 2: %s -> %s" % (act_values, dict2))
    check("mapeo_clases_identidad", list(dict2.keys()) == list(dict2.values()), str(dict2))
    check("n_clases_global", len(act_values) == len(C.CLASSES), len(act_values))

    y_true = meta["act_id_raw"].map(dict2).to_numpy().astype(np.float32)
    x_test = x_flat.reshape(-1, C.N_TIMESTEPS, C.N_FEATURES)
    del x_flat
    n_nonfinite = int(np.sum(~np.isfinite(x_test)))
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    print("[data] x_test %s | valores no finitos saneados: %d" % (str(x_test.shape), n_nonfinite))

    check("ventanas_historicas", len(meta) == C.HISTORIC["n_test_windows"], len(meta))
    check("alineacion_x_y_meta",
          x_test.shape[0] == len(y_true) == len(meta), "%d/%d/%d" % (x_test.shape[0], len(y_true), len(meta)))
    check("sujetos_presentes_en_csv", set(meta["subject_id"]) == set(test_users),
          "faltan=%s" % sorted(set(test_users) - set(meta["subject_id"])))
    check("x_test_finito", bool(np.isfinite(x_test).all()))

    counts = {int(c): int((y_true == c).sum()) for c in C.CLASSES}
    check("conteos_por_clase_historicos", counts == C.HISTORIC["class_counts"], counts)

    import tensorflow as tf
    import keras
    print("[tf] tensorflow %s | keras %s | GPUs %s"
          % (tf.__version__, keras.__version__, tf.config.list_physical_devices("GPU")))
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:  # pragma: no cover
            print("[tf] set_memory_growth fallo: %s" % exc)

    model = tf.keras.models.load_model(args.checkpoint, compile=False)
    print("[model] %s -> input %s output %s"
          % (os.path.basename(args.checkpoint), model.input_shape, model.output_shape))
    check("input_shape_modelo", tuple(model.input_shape[1:]) == (C.N_TIMESTEPS, C.N_FEATURES),
          str(model.input_shape))

    t0 = time.time()
    probs = model.predict(x_test, batch_size=args.batch_size, verbose=2)
    print("[infer] %d ventanas en %.1fs" % (len(probs), time.time() - t0))
    probs = np.asarray(probs, dtype=np.float32)
    y_pred = probs.argmax(1).astype(np.int64)
    check("probabilidades_finitas", bool(np.isfinite(probs).all()))
    check("n_clases_salida", probs.shape[1] == len(C.CLASSES), probs.shape)

    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    y_true_i = y_true.astype(np.int64)
    ba = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true_i, y_pred, labels=C.CLASSES)
    recalls = cm.diagonal() / cm.sum(axis=1)
    print("[metric] balanced accuracy = %.12f" % ba)
    print("[metric] recalls = %s" % np.round(recalls, 4).tolist())
    print("[metric] matriz de confusion (filas=real, columnas=predicha):\n%s" % cm)

    check("balanced_accuracy_4dp", "%.4f" % ba == C.HISTORIC["balanced_accuracy_4dp"], "%.4f" % ba)
    check("recalls_4dp_historicos",
          np.allclose(np.round(recalls, 4), C.HISTORIC["sensitivity_per_class_4dp"]),
          np.round(recalls, 4).tolist())
    check("ba_igual_media_recalls", abs(ba - float(recalls.mean())) < 1e-12,
          "%.12f" % abs(ba - float(recalls.mean())))

    preds = pd.DataFrame({
        "window_id": meta["window_id"].to_numpy(),
        "test_index": np.arange(len(meta), dtype=np.int64),
        "subject_id": meta["subject_id"].to_numpy(),
        "timestamp": meta["timestamp"].to_numpy(),
        "method_id": meta["method_id"].to_numpy(),
        "act_name": meta["act_name"].to_numpy(),
        "y_true": y_true_i.astype(np.int8),
        "y_pred": y_pred.astype(np.int8),
        "p_drink": probs[:, 0],
        "p_eat": probs[:, 1],
        "p_another": probs[:, 2],
    })
    check("window_id_unico", preds["window_id"].is_unique, preds["window_id"].nunique())

    pred_path = os.path.join(args.out_dir, "test_predictions.parquet")
    try:
        preds.to_parquet(pred_path, index=False)
        print("[out] %s" % pred_path)
    except Exception as exc:
        print("[warn] parquet no disponible (%s); se escribe csv.gz" % exc)
        pred_path = os.path.join(args.out_dir, "test_predictions.csv.gz")
        preds.to_csv(pred_path, index=False)
        print("[out] %s" % pred_path)

    # Se agrupa sobre int64: un MultiIndex int64 no casa con claves int8 al reindexar.
    pairs = pd.DataFrame({
        "subject_id": preds["subject_id"].to_numpy(),
        "y_true": preds["y_true"].to_numpy().astype(np.int64),
        "y_pred": preds["y_pred"].to_numpy().astype(np.int64),
    })
    long_cm = (pairs.groupby(["subject_id", "y_true", "y_pred"]).size()
               .rename("count").reset_index())
    full_index = pd.MultiIndex.from_product(
        [test_users, C.CLASSES, C.CLASSES], names=["subject_id", "y_true", "y_pred"])
    long_cm = (long_cm.set_index(["subject_id", "y_true", "y_pred"])
               .reindex(full_index, fill_value=0).reset_index())
    long_cm["subject_id"] = pd.Categorical(long_cm["subject_id"], categories=test_users, ordered=True)
    long_cm = long_cm.sort_values(["subject_id", "y_true", "y_pred"]).reset_index(drop=True)
    cm_path = os.path.join(args.out_dir, "subject_confusion_matrices.csv")
    long_cm.to_csv(cm_path, index=False)
    print("[out] %s" % cm_path)

    check("suma_matrices_sujeto", int(long_cm["count"].sum()) == len(preds), int(long_cm["count"].sum()))
    support = long_cm.groupby(["subject_id", "y_true"], observed=True)["count"].sum().unstack()
    zero_support = support[(support == 0).any(axis=1)]
    check("cobertura_3_clases_por_sujeto", len(zero_support) == 0,
          "sujetos sin alguna clase: %s" % list(zero_support.index))

    manifest = {
        "protocol": "SnsEnc/docs/protocolo_bootstrap_deo_09055.md",
        "stage": "paso_1_predicciones",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join([sys.executable] + sys.argv),
        "working_directory": os.getcwd(),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "node": platform.node(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        },
        "inputs": {
            "source_csv": C.file_stat(args.csv, with_hash=args.hash_csv),
            "checkpoint": C.file_stat(args.checkpoint),
            "split_file": C.file_stat(args.split_file),
            "hyperparams_file": C.file_stat(C.HYPERPARAMS_FILE),
            "metrics_test_file": C.file_stat(C.METRICS_TEST_FILE),
            "analysis_code": {os.path.basename(f): C.file_stat(f)
                              for f in sorted(_analysis_sources())},
        },
        "reconstruction": {
            "source": "sujetos guardados en el archivo de particion (sin re-sortear)",
            "overlap": False,
            "overlap_shift_declared_in_hyperparams": 0.5,
            "seg5": True,
            "normalize_branch_applied": False,
            "window_shape": [C.N_TIMESTEPS, C.N_FEATURES],
            "label_map_dict2": {str(k): int(v) for k, v in dict2.items()},
            "predict_batch_size": args.batch_size,
            "csv_total_rows": int(total_rows),
            "csv_total_subjects": len(all_users),
            "nonfinite_values_sanitised": n_nonfinite,
        },
        "results": {
            "n_test_subjects": int(preds["subject_id"].nunique()),
            "n_test_windows": int(len(preds)),
            "class_counts": counts,
            "balanced_accuracy": ba,
            "balanced_accuracy_4dp": "%.4f" % ba,
            "recalls": [float(r) for r in recalls],
            "confusion_matrix": cm.astype(int).tolist(),
            "historic_balanced_accuracy_4dp": C.HISTORIC["balanced_accuracy_4dp"],
        },
        "outputs": {"test_predictions": pred_path, "subject_confusion_matrices": cm_path},
        "checks": checks,
        "all_checks_passed": all(c["passed"] for c in checks),
    }
    C.write_json(os.path.join(args.out_dir, "reproduction_manifest.json"), manifest)

    failed = [c["check"] for c in checks if not c["passed"]]
    if failed:
        print("\n[ERROR] Verificaciones fallidas: %s" % failed)
        print("[ERROR] El protocolo prohibe ajustar predicciones, etiquetas o preprocesamiento")
        print("[ERROR] para forzar la coincidencia. Resolver la discrepancia antes del Paso 2.")
        return 3
    print("\n[OK] Paso 1 completo: reproduccion verificada (BA = %.4f)." % ba)
    return 0


def _analysis_sources():
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(here, f) for f in sorted(os.listdir(here)) if f.endswith(".py")]


if __name__ == "__main__":
    raise SystemExit(main())
