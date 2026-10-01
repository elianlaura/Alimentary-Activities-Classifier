#!/usr/bin/env python
"""Stage 5: self-supervised pretraining of the CABiGRU autoencoder on a corpus.

Objective: reconstruct each normalised input window (MSE), RMSprop lr 1e-4,
batch 256, 20 epochs (legacy settings). The labels of the corpus are not used.

    python scripts/pretrain.py --data-dir $DATA_ROOT/deo --corpus $RUNS/corpora/diff_gru_x10 \
        --seed 0 --out-dir $RUNS/pretrain/diff_gru_x10/seed0

Normalisation (--norm):
  train   z-score with the real TRAINING-subject statistics (corrected protocol)
  corpus  z-score with the corpus' own statistics (legacy behaviour)
  none    raw sensor units (same scale as finetune.py --norm none)

Early stopping (--early-stop-patience N, off by default): --epochs becomes the maximum; training
stops after N epochs without a lower reconstruction MSE on the real validation windows, and the
weights of the best epoch are saved. With --norm corpus a few implausible validation windows
(|acc| ~ 1e3 m/s^2) dominate that MSE; --val-max-abs-acc drops them from the monitor.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import io, norm, splits  # noqa: E402
from deo.utils import Timer, environment_info, set_seed, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--corpus", required=True, help="corpus directory (corpus.npy)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--norm", choices=["train", "corpus", "none"], default="train")
    ap.add_argument("--output", choices=["linear", "sigmoid"], default="linear")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="stop after N epochs without improvement of the real-validation MSE and keep the "
                         "best epoch (0 = off: train exactly --epochs and keep the last epoch)")
    ap.add_argument("--val-max-abs-acc", type=float, default=0,
                    help="drop validation windows with any |acc| above this value (m/s^2) from the "
                         "reconstruction monitor (0 = keep all; 156.96 = 16 g, as make_generator_data.py)")
    ap.add_argument("--max-windows", type=int, default=0, help="debug: use only the first N corpus windows")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    done_flag = os.path.join(args.out_dir, "done.json")
    if os.path.exists(done_flag):
        print("[pretrain] %s already done" % args.out_dir)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    import tensorflow as tf
    import keras
    from deo.models import build_autoencoder_legacy, build_autoencoder_s
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    set_seed(args.seed, args.deterministic)
    timer = Timer()

    corpus = np.load(os.path.join(args.corpus, "corpus.npy"), mmap_mode="r")
    if args.max_windows:
        corpus = corpus[: args.max_windows]
    if args.norm == "train":
        stats = norm.load_stats(os.path.join(args.data_dir, "norm_train.json"))
    elif args.norm == "corpus":
        stats = norm.fit_stats(corpus)
    else:
        stats = None  # norm.apply(x, None) returns the raw float32 values
    write_json(os.path.join(args.out_dir, "norm_used.json"), stats or {"norm": "none"})

    # Diagnostic: reconstruction error on real windows of the VALIDATION subjects.
    X, meta = io.load_dataset(args.data_dir)
    split = splits.load_split(os.path.join(args.data_dir, "split.json"))
    val_idx = splits.indices(meta, split, "val")
    n_val_dropped = 0
    if args.val_max_abs_acc:
        ok = np.abs(io.take(X, val_idx)[:, :, :3]).max(axis=(1, 2)) <= args.val_max_abs_acc
        n_val_dropped = int((~ok).sum())
        val_idx = val_idx[ok]
        print("[pretrain] %d validation windows with |acc| > %g dropped from the monitor"
              % (n_val_dropped, args.val_max_abs_acc), flush=True)
    rng = np.random.default_rng(args.seed)
    val_idx = np.sort(rng.choice(val_idx, min(5000, len(val_idx)), replace=False))
    x_val = norm.apply(io.take(X, val_idx), stats)

    class CorpusSequence(keras.utils.PyDataset):
        def __init__(self, arr, batch, seed):
            super().__init__()
            self.arr, self.batch = arr, batch
            self.rng = np.random.default_rng(seed)
            self.order = self.rng.permutation(len(arr))

        def __len__(self):
            return int(np.ceil(len(self.arr) / self.batch))

        def __getitem__(self, i):
            sel = np.sort(self.order[i * self.batch:(i + 1) * self.batch])
            x = norm.apply(self.arr[sel], stats)
            return x, x

        def on_epoch_end(self):
            self.order = self.rng.permutation(len(self.arr))

    ae = build_autoencoder_s() if args.output == "linear" else build_autoencoder_legacy()
    ae.compile(optimizer=keras.optimizers.RMSprop(learning_rate=args.lr), loss="mse")
    callbacks = [keras.callbacks.CSVLogger(os.path.join(args.out_dir, "history.csv"))]
    early = None
    if args.early_stop_patience:
        # Keras 3 restores the best weights at the end of fit, also when --epochs is reached first
        early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stop_patience,
                                              restore_best_weights=True, verbose=1)
        callbacks.append(early)
    with timer("pretrain_fit"):
        hist = ae.fit(CorpusSequence(corpus, args.batch_size, args.seed), epochs=args.epochs,
                      validation_data=(x_val, x_val), callbacks=callbacks, verbose=2)
    path = os.path.join(args.out_dir, "autoencoder.keras")
    ae.save(path)
    epochs_run = len(hist.history["loss"])
    sel = early.best_epoch if early is not None else epochs_run - 1  # 0-based epoch of the saved weights
    write_json(done_flag, {
        "autoencoder": path, "corpus": os.path.abspath(args.corpus), "n_windows": int(len(corpus)),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "norm": args.norm,
        "output": args.output, "seed": args.seed, "n_params": int(ae.count_params()),
        "early_stop_patience": args.early_stop_patience, "epochs_run": epochs_run,
        "selected_epoch": sel + 1, "val_max_abs_acc": args.val_max_abs_acc,
        "val_windows_dropped": n_val_dropped, "n_val_monitor_windows": int(len(val_idx)),
        "final_train_mse": float(hist.history["loss"][-1]),
        "final_real_val_mse": float(hist.history["val_loss"][-1]),
        "selected_train_mse": float(hist.history["loss"][sel]),
        "selected_real_val_mse": float(hist.history["val_loss"][sel]),
        "timing_s": timer.stages, "environment": environment_info()})
    print("[pretrain] saved %s" % path, flush=True)


if __name__ == "__main__":
    main()
