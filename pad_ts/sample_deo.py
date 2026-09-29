#!/usr/bin/env python
"""Sample N segments from a trained generator (resumable, writes physical units).

    python pad_ts/sample_deo.py --gen-dir $RUNS/generators/gru_drink --n 500000 \
        --out $RUNS/generators/gru_drink/samples.npy

Ancestral sampling (p_sample_loop) over all diffusion steps, x_0 clipped to
[-1, 1], then mapped back to sensor units with the generator's min-max scaler.
--respacing K samples with K evenly spaced steps instead (faster, not legacy).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from deo_common import MinMaxScaler, build_diffusion, build_model, latest_checkpoint, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--checkpoint", default=None, help="default: latest model_*.pt in --gen-dir")
    ap.add_argument("--n", type=int, required=True, help="number of segments to generate")
    ap.add_argument("--out", required=True)
    # Samples are independent of the batch (no batch-dependent layers); on an H200 4096 is
    # ~1.9x faster per segment than the legacy 512 (3.8 vs 7.1 ms, BiGRU variant).
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--respacing", default="", help="e.g. 100 (empty = all steps)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.gen_dir, "config.json")))
    scaler = MinMaxScaler(cfg["scaler"]["lo"], cfg["scaler"]["hi"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.checkpoint or latest_checkpoint(args.gen_dir)
    if ckpt is None:
        raise SystemExit("[sample] no model_*.pt in %s" % args.gen_dir)
    ck = torch.load(ckpt, map_location=device)
    if args.checkpoint is None:
        # Never sample from an interrupted training run (e.g. a job that hit its time limit).
        done_path = os.path.join(args.gen_dir, "train_done.json")
        if not os.path.exists(done_path):
            raise SystemExit("[sample] %s has no train_done.json: generator training did not finish "
                             "(resubmit stage 2) or pass --checkpoint" % args.gen_dir)
        steps = int(json.load(open(done_path))["steps"])
        if int(ck["step"]) != steps:
            raise SystemExit("[sample] latest checkpoint %s is at step %d, training finished at step %d"
                             % (ckpt, int(ck["step"]), steps))
    state = ck["model_state_dict"]
    use_gru = any(".gru." in k for k in state)
    model = build_model(cfg, use_gru=use_gru).to(device)
    model.load_state_dict(state)
    model.eval()
    diffusion = build_diffusion(cfg, respacing=args.respacing)

    shape = (args.n, cfg["window"], cfg["n_features"])
    progress_path = args.out + ".progress.json"
    if os.path.exists(args.out) and os.path.exists(progress_path):
        out = np.load(args.out, mmap_mode="r+")
        prog = json.load(open(progress_path))
        done, elapsed = prog["done"], prog.get("seconds", 0.0)
        if out.shape != shape:
            raise ValueError("existing %s has shape %s, requested %s" % (args.out, out.shape, shape))
        print("[sample] resuming at %d / %d" % (done, args.n), flush=True)
    else:
        out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float32, shape=shape)
        done, elapsed = 0, 0.0
    t0 = time.time()
    with torch.no_grad():
        while done < args.n:
            m = min(args.batch_size, args.n - done)
            torch.manual_seed(args.seed * 1_000_003 + done)
            x = diffusion.p_sample_loop(model, (m, cfg["window"], cfg["n_features"]), clip_denoised=True)
            out[done:done + m] = scaler.inverse(x.cpu().numpy())
            done += m
            out.flush()
            save_json(progress_path, {"done": done, "checkpoint": ckpt, "seconds": elapsed + time.time() - t0})
            rate = (time.time() - t0) / max(done, 1)
            print("[sample] %d / %d (%.4f s/segment)" % (done, args.n, rate), flush=True)
    # Completion record: build_corpus.py refuses shards without it.
    save_json(os.path.splitext(args.out)[0] + ".json", {
        "checkpoint": ckpt, "arch": "gru" if use_gru else "attn", "n": args.n, "window": cfg["window"],
        "respacing": args.respacing or cfg["diffusion_steps"], "seconds_this_run": round(elapsed + time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})


if __name__ == "__main__":
    main()
