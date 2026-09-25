#!/usr/bin/env python
"""Re-evaluate the archived checkpoints behind the paper's numbers.

    python scripts/reproduce_legacy.py --data-dir $DATA_ROOT/deo --out-dir $RUNS/legacy_eval

Uses the legacy preprocessing (raw sensor units, non-finite values set to 0, the
saved 55/10/29 subject split) and compares the recomputed test metrics with the
metrics_test_*.txt archived next to each checkpoint (Table I and Sec. IV-E).
Also writes per-window test predictions, which scripts/analyze.py-style subject
bootstraps can reuse.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import io, metrics, splits  # noqa: E402
from deo.utils import environment_info, write_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = ["table1_diffusion_09055", "table1_noSynthetic_08655", "table1_pcfgan_08802", "sec4e_2N_08338"]


def archived_ba(run_dir):
    txt = glob.glob(os.path.join(run_dir, "metrics_test_*.txt"))[0]
    return float(re.search(r"balanced_accuracy_score:\s*([\d.]+)", open(txt).read()).group(1))


def load_any(path):
    """Keras 3 checkpoints load with tf.keras; the 2N run was saved with Keras 2.15 -> tf_keras."""
    import tensorflow as tf
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        print("[legacy] Keras 3 could not load %s (%s); trying tf_keras" % (os.path.basename(path), str(exc)[:80]))
        import tf_keras
        return tf_keras.models.load_model(path, compile=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--runs", nargs="+", default=RUNS)
    args = ap.parse_args()
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(os.path.join(REPO, "splits", "deo_legacy.json"))
    idx = splits.indices(meta, split, "test")
    x = io.take(X, idx)
    y = meta["label"].to_numpy()[idx]
    subj = meta["subject"].to_numpy()[idx]
    print("[legacy] test windows %d, class counts %s" % (len(y), np.bincount(y).tolist()), flush=True)
    report = {"environment": environment_info(), "n_test_windows": int(len(y)),
              "class_counts": np.bincount(y).tolist(), "runs": {}}
    for run in args.runs:
        run_dir = os.path.join(REPO, "artifacts", "legacy", run)
        ckpt = glob.glob(os.path.join(run_dir, "best_model_*.keras"))[0]
        model = load_any(ckpt)
        probs = model.predict(x, batch_size=32, verbose=0)
        m = metrics.classification_metrics(y, probs)
        m["per_subject_balanced_accuracy"] = metrics.per_subject_balanced_accuracy(y, probs.argmax(1), subj)
        out = os.path.join(args.out_dir, run)
        os.makedirs(out, exist_ok=True)
        write_json(os.path.join(out, "metrics_test.json"), m)
        np.savez_compressed(os.path.join(out, "predictions_test.npz"), y_true=y.astype(np.int8),
                            probs=probs.astype(np.float32), subject=subj.astype(str))
        ref = archived_ba(run_dir)
        ok = "%.4f" % m["balanced_accuracy"] == "%.4f" % ref
        report["runs"][run] = {"checkpoint": os.path.relpath(ckpt, REPO), "balanced_accuracy": m["balanced_accuracy"],
                               "archived_balanced_accuracy": ref, "match_4dp": ok,
                               "mAP_hard": m["mAP_hard"], "mAP_score": m["mAP_score"], "auc_score": m["auc_score"]}
        print("[legacy] %-26s BA %.4f (archived %.4f) %s" % (run, m["balanced_accuracy"], ref,
                                                              "OK" if ok else "MISMATCH"), flush=True)
    write_json(os.path.join(args.out_dir, "legacy_eval.json"), report)


if __name__ == "__main__":
    main()
