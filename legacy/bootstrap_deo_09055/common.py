"""Rutas y utilidades compartidas del analisis bootstrap del experimento DEO 0.9055.

Protocolo: SnsEnc/docs/protocolo_bootstrap_deo_09055.md
"""
import hashlib
import json
import os
import re

PROJECT_ROOT = "/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main"

SWEEP_DIR = os.path.join(
    PROJECT_ROOT, "saved_models_prev/202508-202512/recurrent_models_20251026-024834")
RUN_ID = "20251028-174115"
RUN_DIR = os.path.join(SWEEP_DIR, "eatdrinkanother_94u_1f_autoencoder_s_" + RUN_ID)

CHECKPOINT = os.path.join(RUN_DIR, "best_model_eatdrinkanother_94u_%s.keras" % RUN_ID)
SPLIT_FILE = os.path.join(RUN_DIR, "eatdrinkanother_94u_1.txt")
HYPERPARAMS_FILE = os.path.join(RUN_DIR, "hyperparams_19.txt")
METRICS_TEST_FILE = os.path.join(RUN_DIR, "metrics_test_eatdrinkanother_94u_%s.txt" % RUN_ID)

DATASET = "eatdrinkanother_94u"
# utils.get_raw_datasets(DATASET) -> file_full_raws
SOURCE_CSV = ("/home/elian.riveros/dl-13-elian/notebooks/workspaces/files/"
              "fullraws3_vivabem012_drink0eat1another2_94u_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv")

# Columnas de metadatos del csv: uuid(0), timestamp(1), act id(2), met id(3), met name(4), datos(5:)
N_META_COLS = 5
N_TIMESTEPS = 500
N_FEATURES = 9
CLASSES = [0, 1, 2]
CLASS_NAMES = ["DRINK", "EAT", "ANOTHER"]

# Valores historicos a reproducir (Paso 1.7 del protocolo).
HISTORIC = {
    "n_train_subjects": 55,
    "n_val_subjects": 10,
    "n_test_subjects": 29,
    "n_test_windows": 71591,
    "class_counts": {0: 4933, 1: 5243, 2: 61415},
    "balanced_accuracy_4dp": "0.9055",
    "sensitivity_per_class_4dp": [0.8849, 0.9014, 0.9304],
    "specificity_per_class_4dp": [0.9486, 0.9779, 0.9537],
}

# Paso 2: parametros fijados antes del calculo.
BOOTSTRAP_SEED = 20251028
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_ALPHA = 0.05


def parse_split_file(path=SPLIT_FILE):
    """Devuelve los sujetos guardados de train/val/test respetando el orden del archivo."""
    with open(path) as fh:
        text = fh.read()
    out = {}
    for key in ("uuid_train", "uuid_val", "uuid_test"):
        m = re.search(key + r":\s*\[(.*?)\]", text, flags=re.S)
        if m is None:
            raise ValueError("No se encontro %s en %s" % (key, path))
        out[key] = re.findall(r"'([^']+)'", m.group(1))
    return out


def sha256_file(path, chunk=32 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_stat(path, with_hash=True):
    st = os.stat(path)
    info = {"path": path, "bytes": st.st_size, "mtime_epoch": int(st.st_mtime)}
    if with_hash:
        info["sha256"] = sha256_file(path)
    return info


def write_json(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")
    print("[out] %s" % path)
