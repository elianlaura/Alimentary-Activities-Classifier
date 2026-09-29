#!/usr/bin/env python
"""Stage 6: supervised training on the real DEO windows.

    # fine-tune a pretrained autoencoder
    python scripts/finetune.py --data-dir $DATA_ROOT/deo --pretrained $RUNS/pretrain/diff_gru_x10/seed0/autoencoder.keras \
        --seed 0 --out-dir $RUNS/finetune/diff_gru_x10/seed0
    # same architecture from random initialisation (no pretraining)
    python scripts/finetune.py --data-dir $DATA_ROOT/deo --pretrained none --seed 0 --out-dir $RUNS/finetune/scratch/seed0

Training (legacy settings unless stated): encoder + head ('deep' by default),
categorical focal loss (gamma 3, alpha = inverse class frequency), Adam lr 3e-4,
batch 128, <= 50 epochs, ReduceLROnPlateau(val_loss, 0.5, patience 15).
The legacy lr 1e-4 was tuned for raw sensor units; with z-scored inputs it under-trains in
50 epochs. 3e-4 was chosen on VALIDATION balanced accuracy of the from-scratch model (seed 0)
among 1e-4 / 3e-4 / 1e-3 (0.845 / 0.901 / 0.891; raw units with 1e-4: 0.862), 2026-09-26.
Model selection uses ONLY the validation subjects: the checkpoint with the best
validation balanced accuracy is kept and training stops after --patience epochs
without improvement. The test split is evaluated once, after training.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import io, metrics, norm, splits  # noqa: E402
from deo.utils import Timer, environment_info, set_seed, write_json  # noqa: E402


def load_split_arrays(data_dir, split, stats):
    X, meta = io.load_dataset(data_dir)
    out = {}
    for part in ("train", "val", "test"):
        idx = splits.indices(meta, split, part)
        out[part] = (norm.apply(io.take(X, idx), stats), meta["label"].to_numpy()[idx].astype(int),
                     meta["subject"].to_numpy()[idx])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default=None, help="split json (default: <data-dir>/split.json)")
    ap.add_argument("--pretrained", required=True, help="autoencoder .keras, or 'none' to train from scratch")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", default="deep", choices=["balanced", "light", "deep", "classic"])
    ap.add_argument("--activation", default="gelu")
    ap.add_argument("--n-dense", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4, help="legacy recipe: 1e-4 with --norm none")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--encoder-lr-mult", type=float, default=1.0,
                    help="the encoder's Adam step is multiplied by this factor (discriminative fine-tuning, "
                         "e.g. 0.33: encoder ~1e-4, head 3e-4); 1 = one learning rate for all layers")
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--early-stop-monitor", default="val_balanced_accuracy",
                    choices=["val_balanced_accuracy", "val_accuracy"], help="legacy: val_accuracy")
    ap.add_argument("--norm", choices=["train", "none"], default="train",
                    help="train: z-score with training-subject stats; none: raw units (legacy)")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    done_flag = os.path.join(args.out_dir, "done.json")
    if os.path.exists(done_flag):
        print("[finetune] %s already done" % args.out_dir)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    import tensorflow as tf
    import keras
    from deo.models import (ScaledAdam, build_autoencoder_s, build_classifier, encoder_from_autoencoder,
                            load_autoencoder)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    set_seed(args.seed, args.deterministic)
    timer = Timer()

    split = splits.load_split(args.split or os.path.join(args.data_dir, "split.json"))
    stats = norm.load_stats(os.path.join(args.data_dir, "norm_train.json")) if args.norm == "train" else None
    with timer("load_data"):
        data = load_split_arrays(args.data_dir, split, stats)
    (x_tr, y_tr, _), (x_va, y_va, s_va), (x_te, y_te, s_te) = data["train"], data["val"], data["test"]
    n_classes = 3
    counts = Counter(y_tr.tolist())
    alpha = [len(y_tr) / (n_classes * counts[c]) for c in range(n_classes)]

    head_kw = dict(head=args.head, activation=args.activation, n_dense=args.n_dense, n_classes=n_classes)
    if args.pretrained.lower() == "none":  # same calls as build_scratch_classifier
        encoder = encoder_from_autoencoder(build_autoencoder_s(x_tr.shape[1:]))
    else:
        encoder = encoder_from_autoencoder(load_autoencoder(args.pretrained))
    model = build_classifier(encoder, **head_kw)
    if args.encoder_lr_mult != 1.0:
        optimizer = ScaledAdam(scaled_variables=encoder.trainable_variables, multiplier=args.encoder_lr_mult,
                               learning_rate=args.lr)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(loss=keras.losses.CategoricalFocalCrossentropy(gamma=args.gamma, alpha=alpha),
                  optimizer=optimizer, metrics=["accuracy"])

    class ValBalancedAccuracy(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            pred = self.model.predict(x_va, batch_size=1024, verbose=0).argmax(1)
            logs["val_balanced_accuracy"] = float(metrics.classification_metrics(
                y_va, np.eye(n_classes)[pred])["balanced_accuracy"])
            print("epoch %d val_balanced_accuracy %.4f" % (epoch + 1, logs["val_balanced_accuracy"]), flush=True)

    best_path = os.path.join(args.out_dir, "best_model.keras")
    callbacks = [
        ValBalancedAccuracy(),
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_balanced_accuracy", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor=args.early_stop_monitor, mode="max", patience=args.patience),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6),
        keras.callbacks.CSVLogger(os.path.join(args.out_dir, "history.csv")),
    ]
    with timer("fit"):
        hist = model.fit(x_tr, np.eye(n_classes)[y_tr], validation_data=(x_va, np.eye(n_classes)[y_va]),
                         batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks, verbose=2)
    best = keras.models.load_model(best_path, compile=False)
    if args.encoder_lr_mult != 1.0:
        best.save(best_path)  # drop the ScaledAdam compile config so any tool can load the model
    results = {}
    for part, (x, y, subj) in data.items():
        with timer("predict_" + part):
            probs = best.predict(x, batch_size=1024, verbose=0)
        m = metrics.classification_metrics(y, probs, n_classes)
        m["per_subject_balanced_accuracy"] = metrics.per_subject_balanced_accuracy(y, probs.argmax(1), subj)
        results[part] = m
        write_json(os.path.join(args.out_dir, "metrics_%s.json" % part), m)
        if part in ("val", "test"):
            np.savez_compressed(os.path.join(args.out_dir, "predictions_%s.npz" % part),
                                y_true=y.astype(np.int8), probs=probs.astype(np.float32), subject=subj.astype(str))
    with open(os.path.join(args.out_dir, "metrics_test.txt"), "w") as fh:
        fh.write(metrics.format_legacy(results["test"]) + "\n")
    bal_hist = hist.history.get("val_balanced_accuracy", [])
    summary = {
        "config": vars(args), "class_alpha": alpha,
        "n_windows": {k: int(len(v[1])) for k, v in data.items()},
        "n_subjects": {k: int(len(np.unique(v[2]))) for k, v in data.items()},
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmax(bal_hist)) + 1 if bal_hist else None,
        "val_balanced_accuracy": results["val"]["balanced_accuracy"],
        "test_balanced_accuracy": results["test"]["balanced_accuracy"],
        "n_params": int(best.count_params()),
        "timing_s": timer.stages, "environment": environment_info(),
    }
    write_json(done_flag, summary)
    print("[finetune] val BA %.4f | test BA %.4f" % (summary["val_balanced_accuracy"],
                                                    summary["test_balanced_accuracy"]), flush=True)


if __name__ == "__main__":
    main()
