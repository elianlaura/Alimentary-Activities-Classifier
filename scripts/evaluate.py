#!/usr/bin/env python
"""Inference with a saved classifier: metrics and subject-bootstrap 95% CIs per partition.

    python scripts/evaluate.py --model $RUNS/finetune/deo/<condition>/seed<k>/best_model.keras \
        --data-dir $DATA_ROOT/deo --norm none --out-dir $RUNS/eval/<name> [--reference-ba 0.9055]

The input scale must match the one used in fine-tuning (--norm none for the legacy recipe).
Inference runs in full fp32 (TF32 off), as scripts/reproduce_legacy.py.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import CLASS_NAMES, io, metrics, norm, splits  # noqa: E402
from deo.utils import environment_info, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default=None, help="split json (default: <data-dir>/split.json)")
    ap.add_argument("--norm", choices=["train", "none"], required=True)
    ap.add_argument("--parts", nargs="+", default=["val", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--reference-ba", type=float, default=None, help="test BA to compare with (e.g. 0.9055)")
    args = ap.parse_args()

    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    os.makedirs(args.out_dir, exist_ok=True)
    model = tf.keras.models.load_model(args.model, compile=False)
    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(args.split or os.path.join(args.data_dir, "split.json"))
    stats = norm.load_stats(os.path.join(args.data_dir, "norm_train.json")) if args.norm == "train" else None

    report = {"model": os.path.abspath(args.model), "norm": args.norm, "environment": environment_info(), "parts": {}}
    lines = ["# Inference: %s\n" % args.model, "| partition | subjects | windows | BA [95% CI] | recall drink / eat / other |",
             "|---|---|---|---|---|"]
    for part in args.parts:
        idx = splits.indices(meta, split, part)
        x = norm.apply(io.take(X, idx), stats)
        y = meta["label"].to_numpy()[idx].astype(int)
        subj = meta["subject"].to_numpy()[idx].astype(str)
        probs = model.predict(x, batch_size=1024, verbose=0)
        m = metrics.classification_metrics(y, probs)
        m["per_subject_balanced_accuracy"] = metrics.per_subject_balanced_accuracy(y, probs.argmax(1), subj)
        m["bootstrap"] = metrics.subject_bootstrap(y, probs.argmax(1), subj, n_boot=args.n_boot)
        write_json(os.path.join(args.out_dir, "metrics_%s.json" % part), m)
        np.savez_compressed(os.path.join(args.out_dir, "predictions_%s.npz" % part), y_true=y.astype(np.int8),
                            probs=probs.astype(np.float32), subject=subj)
        ci = m["bootstrap"]["balanced_accuracy_ci95"]
        report["parts"][part] = {"balanced_accuracy": m["balanced_accuracy"], "ci95": ci, "n_windows": m["n"],
                                 "n_subjects": m["bootstrap"]["n_subjects"], "sensitivity": m["sensitivity"]}
        lines.append("| %s | %d | %d | %.4f [%.4f, %.4f] | %s |" % (
            part, m["bootstrap"]["n_subjects"], m["n"], m["balanced_accuracy"], ci[0], ci[1],
            " / ".join("%.4f" % v for v in m["sensitivity"])))
        print("[evaluate] %-5s BA %.4f  95%% CI [%.4f, %.4f]" % (part, m["balanced_accuracy"], ci[0], ci[1]), flush=True)
    if args.reference_ba is not None and "test" in report["parts"]:
        t = report["parts"]["test"]
        inside = t["ci95"][0] <= args.reference_ba <= t["ci95"][1]
        report["reference_test_ba"] = args.reference_ba
        report["reference_inside_test_ci95"] = inside
        lines.append("\nReference test BA %.4f: difference %+.4f; %s the 95%% CI of this run." % (
            args.reference_ba, t["balanced_accuracy"] - args.reference_ba, "inside" if inside else "outside"))
    write_json(os.path.join(args.out_dir, "report.json"), report)
    open(os.path.join(args.out_dir, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
