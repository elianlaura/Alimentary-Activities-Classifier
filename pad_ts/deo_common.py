"""Shared helpers for train_deo.py / sample_deo.py (PaD-TS on DEO windows).

Two generator architectures (R3-02 ablation):
  attn  original PaD-TS: dual-channel transformer encoder + DiT decoder blocks.
        This is the architecture that generated the data behind the 0.9055 result
        (log of 2025-09-22: TransformerEncoderBlock + DiTBlock, 8.96 M parameters).
  gru   the modification described in the paper: self-attention replaced by
        bidirectional GRU blocks in the encoder and the AdaLN-conditioned decoder.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Hyper-parameters of the legacy run (./OUTPUT/drinkeat_20250922-111522, log header).
LEGACY_DEFAULTS = dict(hidden_size=256, num_heads=4, n_encoder=1, n_decoder=3, mlp_ratio=4.0,
                       diffusion_steps=500, noise_schedule="cosine", loss="MSE_MMD", predict_xstart=True,
                       batch_size=512, lr=1e-4, lr_anneal_steps=300000, mmd_alpha=0.0005, window=24)


class MinMaxScaler:
    """Per-axis min-max scaling to [-1, 1] (legacy CustomDataset with neg_one_to_one)."""

    def __init__(self, lo=None, hi=None):
        self.lo = None if lo is None else np.asarray(lo, dtype=np.float32)
        self.hi = None if hi is None else np.asarray(hi, dtype=np.float32)

    def fit(self, x):
        flat = x.reshape(-1, x.shape[-1])
        self.lo = flat.min(0).astype(np.float32)
        self.hi = flat.max(0).astype(np.float32)
        return self

    def transform(self, x):
        return ((x - self.lo) / np.maximum(self.hi - self.lo, 1e-8)) * 2.0 - 1.0

    def inverse(self, x):
        return (x + 1.0) * 0.5 * (self.hi - self.lo) + self.lo

    def to_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}


def build_model(cfg, use_gru):
    from Model import PaD_TS
    return PaD_TS(hidden_size=cfg["hidden_size"], num_heads=cfg["num_heads"], n_encoder=cfg["n_encoder"],
                  n_decoder=cfg["n_decoder"], feature_last=True, mlp_ratio=cfg["mlp_ratio"],
                  input_shape=(cfg["window"], cfg["n_features"]), use_gru=use_gru)


def build_diffusion(cfg, respacing=""):
    from diffmodel_init import create_gaussian_diffusion
    return create_gaussian_diffusion(predict_xstart=cfg["predict_xstart"], diffusion_steps=cfg["diffusion_steps"],
                                     noise_schedule=cfg["noise_schedule"], loss=cfg["loss"],
                                     rescale_timesteps=False, timestep_respacing=respacing)


def latest_checkpoint(out_dir):
    ckpts = sorted(f for f in os.listdir(out_dir) if f.startswith("model_") and f.endswith(".pt"))
    return os.path.join(out_dir, ckpts[-1]) if ckpts else None


def save_json(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
