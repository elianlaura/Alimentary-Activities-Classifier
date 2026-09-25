#!/usr/bin/env python
"""Train one PaD-TS generator on segments of one DEO class (resumable).

    python pad_ts/train_deo.py --data $DATA_ROOT/deo/generator/drink_w24.npy \
        --arch gru --out-dir $RUNS/generators/gru_drink --steps 100000

Objective (Li et al., AAAI 2025): the denoiser predicts x_0 from x_t
(predict_xstart), trained with MSE plus mmd_alpha * MMD between the distributions
of cross-correlation matrices of real and denoised batches (population-level
term). Cosine noise schedule, T = --diffusion-steps. Timesteps are shared across
the batch (Batch_Same_Sampler), AdamW with linear learning-rate annealing to 0 at
--anneal-steps.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from deo_common import LEGACY_DEFAULTS, MinMaxScaler, build_diffusion, build_model, latest_checkpoint, save_json


def main():
    d = LEGACY_DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="(n, W, 9) float32 .npy from make_generator_data.py")
    ap.add_argument("--arch", choices=["gru", "attn"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=100000, help="training steps to run (legacy checkpoint: 100000)")
    ap.add_argument("--anneal-steps", type=int, default=d["lr_anneal_steps"],
                    help="lr decays linearly to 0 at this step (legacy: 300000)")
    ap.add_argument("--batch-size", type=int, default=d["batch_size"])
    ap.add_argument("--lr", type=float, default=d["lr"])
    ap.add_argument("--hidden-size", type=int, default=d["hidden_size"])
    ap.add_argument("--num-heads", type=int, default=d["num_heads"])
    ap.add_argument("--n-encoder", type=int, default=d["n_encoder"])
    ap.add_argument("--n-decoder", type=int, default=d["n_decoder"])
    ap.add_argument("--diffusion-steps", type=int, default=d["diffusion_steps"])
    ap.add_argument("--mmd-alpha", type=float, default=d["mmd_alpha"])
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = np.load(args.data).astype(np.float32)
    cfg_path = os.path.join(args.out_dir, "config.json")
    if os.path.exists(cfg_path):  # resume: keep the scaler fitted in the first run
        import json
        cfg = json.load(open(cfg_path))
        scaler = MinMaxScaler(cfg["scaler"]["lo"], cfg["scaler"]["hi"])
    else:
        scaler = MinMaxScaler().fit(x)
        cfg = dict(d, window=int(x.shape[1]), n_features=int(x.shape[2]), arch=args.arch, data=os.path.abspath(args.data),
                   n_train_segments=int(len(x)), hidden_size=args.hidden_size, num_heads=args.num_heads,
                   n_encoder=args.n_encoder, n_decoder=args.n_decoder, diffusion_steps=args.diffusion_steps,
                   batch_size=args.batch_size, lr=args.lr, lr_anneal_steps=args.anneal_steps,
                   mmd_alpha=args.mmd_alpha, seed=args.seed, scaler=scaler.to_dict())
        save_json(cfg_path, cfg)
    data = torch.from_numpy(scaler.transform(x).astype(np.float32))

    model = build_model(cfg, use_gru=(args.arch == "gru")).to(device)
    diffusion = build_diffusion(cfg)
    from resample import Batch_Same_Sampler
    sampler = Batch_Same_Sampler(diffusion)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())

    step = 0
    last = latest_checkpoint(args.out_dir)
    if last:
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["opt_state_dict"])
        step = int(ck["step"])
        print("[train] resumed from %s (step %d)" % (last, step), flush=True)
    print("[train] arch=%s params=%d segments=%d window=%d device=%s" % (
        args.arch, n_params, len(x), x.shape[1], device), flush=True)

    loss_log = open(os.path.join(args.out_dir, "loss.csv"), "a", newline="")
    writer = csv.writer(loss_log)
    if step == 0:
        writer.writerow(["step", "mse", "mmd", "total", "lr", "seconds"])
    gen = torch.Generator().manual_seed(args.seed + step)
    t0 = time.time()
    t_start = t0
    model.train()
    while step < args.steps:
        idx = torch.randint(0, len(data), (args.batch_size,), generator=gen)
        batch = data[idx].to(device, non_blocking=True)
        t, weights = sampler.sample(args.batch_size, device)
        losses = diffusion.training_losses(model, batch, t)
        loss = (losses["mse"] * weights).mean()
        mmd = args.mmd_alpha * losses["mmd"] if "mmd" in losses else torch.zeros((), device=device)
        total = loss + mmd
        opt.zero_grad(set_to_none=True)
        total.backward()
        lr = args.lr * (1 - step / args.anneal_steps) if args.anneal_steps else args.lr
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        step += 1
        if step % 100 == 0:
            writer.writerow([step, float(loss), float(mmd), float(total), lr, round(time.time() - t0, 1)])
            loss_log.flush()
        if step % 1000 == 0:
            print("[train] step %d mse %.5f mmd %.5f (%.2f s/step)" % (
                step, float(loss), float(mmd), (time.time() - t_start) / 1000), flush=True)
            t_start = time.time()
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"step": step, "model_state_dict": model.state_dict(), "opt_state_dict": opt.state_dict()},
                       os.path.join(args.out_dir, "model_%06d.pt" % step))
    save_json(os.path.join(args.out_dir, "train_done.json"), {
        "steps": step, "n_params": n_params, "seconds_this_run": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__})
    print("[train] done", flush=True)


if __name__ == "__main__":
    main()
