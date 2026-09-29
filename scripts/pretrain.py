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
    csv_log = keras.callbacks.CSVLogger(os.path.join(args.out_dir, "history.csv"))
    with timer("pretrain_fit"):
        hist = ae.fit(CorpusSequence(corpus, args.batch_size, args.seed), epochs=args.epochs,
                      validation_data=(x_val, x_val), callbacks=[csv_log], verbose=2)
    path = os.path.join(args.out_dir, "autoencoder.keras")
    ae.save(path)
    write_json(done_flag, {
        "autoencoder": path, "corpus": os.path.abspath(args.corpus), "n_windows": int(len(corpus)),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "norm": args.norm,
        "output": args.output, "seed": args.seed, "n_params": int(ae.count_params()),
        "final_train_mse": float(hist.history["loss"][-1]),
        "final_real_val_mse": float(hist.history["val_loss"][-1]),
        "timing_s": timer.stages, "environment": environment_info()})
    print("[pretrain] saved %s" % path, flush=True)


if __name__ == "__main__":
    main()
