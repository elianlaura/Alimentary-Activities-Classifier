#!/usr/bin/env python
"""Write the task lists (one shell command per line) for every pipeline stage.

Each file <runs>/plan/<stage>.tsv has lines "ENV<TAB>command" where ENV is 'tf'
(TensorFlow environment) or 'torch' (PaD-TS environment). slurm/array_task.sbatch
runs line $SLURM_ARRAY_TASK_ID of a file. Inspect the files before submitting.

Experiments (ids refer to docs/REVIEW_RESPONSE.md):
  main        scratch vs real-only SSL vs diffusion (BiGRU) vs diffusion (attention),
              --seeds seeds each                                           R3-01, R1-06
  controls    augment / noise / surrogate corpora at the main volume        R1-06
  volume      diffusion (BiGRU) at every --multipliers value                R1-05, R3-03, S-18
  scope       diffusion for all three classes instead of minority only      R3-02
  heads       the four classification heads on the main diffusion corpus    R3-02 (optional)
  samehand    scratch / real SSL / diffusion on the same-wrist dataset      R2-06, R1-03
  legacy      the exact legacy recipe (raw units, legacy pretrained AE)     R3-01 (opt-in)
  recipe      training-recipe ablation on diff_gru_x{--recipe-multipliers} (opt-in): pretraining
              for --recipe-pretrain-epochs instead of 20 (_ep40) and/or discriminative fine-tuning
              with the encoder's step x --recipe-encoder-lr-mult (_enclr0p33), 2x2 with the base
              condition. Its tasks are appended at the end of the task files.

'legacy' is not in the default list: its pretrained backbone (artifacts/legacy/
pretrained_ae_20251012) was trained on synthetic data from a generator that saw 28 of
the 29 test subjects, so its test numbers are contaminated. It only measures the
seed-to-seed variance of the archived recipe.
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo.utils import read_json, write_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def fmt_mult(m):
    return ("%g" % m).replace(".", "p")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--multipliers", type=float, nargs="+", default=[1, 2, 5, 10, 20, 40],
                    help="volume sweep in units of R (legacy corpus ~ 40 R)")
    ap.add_argument("--main-multiplier", type=float, default=10)
    ap.add_argument("--gen-window", type=int, default=24, help="generator segment length (legacy 24; 500 = whole windows)")
    ap.add_argument("--gen-stride", type=int, default=12)
    ap.add_argument("--gen-steps", type=int, default=100000)
    ap.add_argument("--sample-shards", type=int, default=4, help="parallel sampling jobs per generator")
    ap.add_argument("--gen-train-args", default="", help="extra arguments for pad_ts/train_deo.py")
    ap.add_argument("--sample-args", default="", help="extra arguments for pad_ts/sample_deo.py")
    ap.add_argument("--pretrain-args", default="", help="extra arguments for scripts/pretrain.py, e.g. '--epochs 5'")
    ap.add_argument("--finetune-args", default="", help="extra arguments for scripts/finetune.py")
    ap.add_argument("--norm", choices=["train", "none", "mixed"], default="train",
                    help="input scale of both stages: train = z-score with training-subject stats in "
                         "pretraining and fine-tuning; none = raw sensor units in both; mixed = z-scored "
                         "pretraining, raw fine-tuning (the arrangement of the archived runs). "
                         "The 'legacy' experiment always fine-tunes in raw units.")
    ap.add_argument("--experiments", nargs="+",
                    default=["main", "controls", "volume", "scope", "samehand"],
                    choices=["main", "controls", "volume", "scope", "heads", "samehand", "legacy", "recipe"],
                    help="add 'heads' for the head ablation; 'legacy' uses the leaked archived backbone")
    ap.add_argument("--recipe-multipliers", type=float, nargs="+", default=[10, 20, 40])
    ap.add_argument("--recipe-pretrain-epochs", type=int, default=40)
    ap.add_argument("--recipe-encoder-lr-mult", type=float, default=0.33)
    args = ap.parse_args()

    deo_dir = os.path.join(args.data_root, "deo")
    sh_dir = os.path.join(args.data_root, "deo_samehand")
    R = read_json(os.path.join(deo_dir, "volume_unit.json"))["R_minority_train_windows"]
    exps = set(args.experiments)
    if "legacy" in exps:
        print("[plan] WARNING: 'legacy' fine-tunes the archived backbone, pretrained on synthetic data "
              "whose generator saw test subjects; do not report its test metrics as results")
    if "samehand" in exps and not os.path.exists(os.path.join(sh_dir, "meta.csv")):
        print("[plan] %s not prepared: dropping the samehand experiment" % sh_dir)
        exps.discard("samehand")
    runs = os.path.abspath(args.runs)
    pre_norm = {"train": "train", "none": "none", "mixed": "train"}[args.norm]
    ft_norm = {"train": "train", "none": "none", "mixed": "none"}[args.norm]
    signature = {"pretrain": [pre_norm, args.pretrain_args.strip()], "finetune": [ft_norm, args.finetune_args.strip()]}
    previous = os.path.join(runs, "plan", "plan.json")
    if os.path.exists(previous):
        # Finished tasks are skipped (done.json), so changing the settings of a stage that already
        # has results would silently mix runs with different settings. Stages 1-4 do not depend on them.
        old = read_json(previous).get("signature", {})
        for stage in ("pretrain", "finetune"):
            done = glob.glob(os.path.join(runs, stage, "*", "seed*", "done.json")) + \
                glob.glob(os.path.join(runs, stage, "*", "*", "seed*", "done.json"))
            if done and stage in old and old[stage] != signature[stage]:
                raise SystemExit("[plan] %s/%s has results made with %s, now requested %s: use another --runs "
                                 "directory (or move the old results away)" % (runs, stage, old[stage], signature[stage]))
    seeds = list(range(args.seeds))
    M = args.main_multiplier
    k = math.ceil(500 / args.gen_window)
    py = "python -u"

    # ---------------------------------------------------------------- corpora
    corpora = {}  # name -> (kind, multiplier, scope, generator arch or None)
    if exps & {"main", "controls", "legacy", "heads", "samehand", "volume", "scope"}:
        corpora["diff_gru_x" + fmt_mult(M)] = ("diffusion", M, "minority", "gru")
    if "main" in exps:
        corpora["diff_attn_x" + fmt_mult(M)] = ("diffusion", M, "minority", "attn")
        corpora["real_all"] = ("real", 0, "all", None)
    if "controls" in exps:
        for kind in ("augment", "noise", "surrogate"):
            corpora["%s_x%s" % (kind, fmt_mult(M))] = (kind, M, "minority", None)
    if "volume" in exps:
        for m in args.multipliers:
            corpora["diff_gru_x" + fmt_mult(m)] = ("diffusion", m, "minority", "gru")
    if "scope" in exps:
        corpora["diff_gru_all_x" + fmt_mult(M)] = ("diffusion", M, "all", "gru")
    if "samehand" in exps:
        corpora.setdefault("real_all", ("real", 0, "all", None))
    if "recipe" in exps:
        for m in args.recipe_multipliers:
            corpora.setdefault("diff_gru_x" + fmt_mult(m), ("diffusion", m, "minority", "gru"))
    ep_tag = "_ep%d" % args.recipe_pretrain_epochs
    lr_tag = "_enclr" + fmt_mult(args.recipe_encoder_lr_mult)
    recipe_bases = ["diff_gru_x" + fmt_mult(m) for m in args.recipe_multipliers] if "recipe" in exps else []

    # ------------------------------------------------ generators and sampling
    need = {}  # (arch, class) -> segments
    for name, (kind, m, scope, arch) in corpora.items():
        if kind != "diffusion":
            continue
        classes = ["drink", "eat"] if scope == "minority" else ["drink", "eat", "other"]
        per_class = math.ceil(m * R / len(classes)) + 1
        for c in classes:
            need[(arch, c)] = max(need.get((arch, c), 0), per_class * k)
    gen_data, gen_train, gen_sample = [], [], []
    classes_needed = sorted({c for (_, c) in need})
    if need:
        gen_data.append("tf\t%s %s/scripts/make_generator_data.py --data-dir %s --window %d --stride %d --classes %s"
                        % (py, REPO, deo_dir, args.gen_window, args.gen_stride, " ".join(classes_needed)))
    samples = {}
    for (arch, c), n in sorted(need.items()):
        gdir = os.path.join(runs, "generators", "%s_%s_w%d" % (arch, c, args.gen_window))
        data = os.path.join(deo_dir, "generator", "%s_w%d.npy" % (c, args.gen_window))
        gen_train.append("torch\t%s %s/pad_ts/train_deo.py --data %s --arch %s --out-dir %s --steps %d %s"
                         % (py, REPO, data, arch, gdir, args.gen_steps, args.gen_train_args))
        shard_n = math.ceil(n / args.sample_shards)
        paths = []
        for s in range(args.sample_shards):
            out = os.path.join(gdir, "samples_shard%d.npy" % s)
            paths.append(out)
            gen_sample.append("torch\t%s %s/pad_ts/sample_deo.py --gen-dir %s --n %d --out %s --seed %d %s"
                              % (py, REPO, gdir, shard_n, out, 1000 + s, args.sample_args))
        samples[(arch, c)] = ",".join(paths)

    corpus_tasks = []
    for name, (kind, m, scope, arch) in sorted(corpora.items()):
        cmd = "%s %s/scripts/build_corpus.py --data-dir %s --kind %s --scope %s --out-dir %s" % (
            py, REPO, deo_dir, kind, scope, os.path.join(runs, "corpora", name))
        if kind != "real":
            cmd += " --multiplier %g" % m
        if kind == "diffusion":
            classes = ["drink", "eat"] if scope == "minority" else ["drink", "eat", "other"]
            cmd += " --generated " + " ".join("%s=%s" % (c, samples[(arch, c)]) for c in classes)
        corpus_tasks.append("tf\t" + cmd)

    # ------------------------------------------------------------ pretraining
    pre_tasks = []
    for name in sorted(corpora):
        for s in seeds:
            pre_tasks.append("tf\t%s %s/scripts/pretrain.py --data-dir %s --corpus %s --seed %d --out-dir %s --norm %s %s" % (
                py, REPO, deo_dir, os.path.join(runs, "corpora", name), s,
                os.path.join(runs, "pretrain", name, "seed%d" % s), pre_norm, args.pretrain_args))
    for base in recipe_bases:  # longer pretraining; --epochs last so it wins over --pretrain-args
        for s in seeds:
            pre_tasks.append("tf\t%s %s/scripts/pretrain.py --data-dir %s --corpus %s --seed %d --out-dir %s --norm %s %s --epochs %d" % (
                py, REPO, deo_dir, os.path.join(runs, "corpora", base), s,
                os.path.join(runs, "pretrain", base + ep_tag, "seed%d" % s), pre_norm, args.pretrain_args,
                args.recipe_pretrain_epochs))

    # ------------------------------------------------------------ fine-tuning
    ft_tasks = []

    def ft(dataset_dir, dataset, cond, pretrained, seed, extra=""):
        out = os.path.join(runs, "finetune", dataset, cond, "seed%d" % seed)
        # argparse keeps the last value: 'extra' (per-condition settings such as the fixed legacy
        # recipe or --head) overrides both --norm and --finetune-args
        ft_tasks.append("tf\t%s %s/scripts/finetune.py --data-dir %s --pretrained %s --seed %d --out-dir %s --norm %s %s %s" % (
            py, REPO, dataset_dir, pretrained, seed, out, ft_norm, args.finetune_args, extra))

    ae = lambda name, s: os.path.join(runs, "pretrain", name, "seed%d" % s, "autoencoder.keras")
    for s in seeds:
        ft(deo_dir, "deo", "scratch", "none", s)
        for name in sorted(corpora):
            ft(deo_dir, "deo", name, ae(name, s), s)
        if "heads" in exps:
            main = "diff_gru_x" + fmt_mult(M)
            for head in ("light", "balanced", "classic"):
                ft(deo_dir, "deo", "%s_head-%s" % (main, head), ae(main, s), s, "--head %s" % head)
        if "samehand" in exps:
            ft(sh_dir, "deo_samehand", "scratch", "none", s)
            ft(sh_dir, "deo_samehand", "real_all", ae("real_all", s), s)
            ft(sh_dir, "deo_samehand", "diff_gru_x" + fmt_mult(M), ae("diff_gru_x" + fmt_mult(M), s), s)
        if "legacy" in exps:
            legacy_ae = os.path.join(REPO, "artifacts", "legacy", "pretrained_ae_20251012",
                                     "best_model_de_fake_padts_94u_20251012-101512.keras")
            legacy_kw = "--norm none --lr 1e-4 --early-stop-monitor val_accuracy"
            ft(deo_dir, "deo", "legacy_recipe_pretrained", legacy_ae, s, legacy_kw)
            ft(deo_dir, "deo", "legacy_recipe_scratch", "none", s, legacy_kw)
    enc_kw = "--encoder-lr-mult %g" % args.recipe_encoder_lr_mult
    for s in seeds:  # recipe ablation, appended so earlier task indices do not move
        for base in recipe_bases:
            ft(deo_dir, "deo", base + ep_tag, ae(base + ep_tag, s), s)
            ft(deo_dir, "deo", base + lr_tag, ae(base, s), s, enc_kw)
            ft(deo_dir, "deo", base + ep_tag + lr_tag, ae(base + ep_tag, s), s, enc_kw)

    plan_dir = os.path.join(runs, "plan")
    os.makedirs(plan_dir, exist_ok=True)
    stages = [("01_gen_data", gen_data), ("02_gen_train", gen_train), ("03_gen_sample", gen_sample),
              ("04_corpora", corpus_tasks), ("05_pretrain", pre_tasks), ("06_finetune", ft_tasks)]
    for stage, tasks in stages:
        # atomic replace: array tasks that are already queued read their line from this file
        path = os.path.join(plan_dir, stage + ".tsv")
        with open(path + ".tmp", "w") as fh:
            fh.write("\n".join(tasks) + ("\n" if tasks else ""))
        os.replace(path + ".tmp", path)
        print("[plan] %-14s %4d tasks" % (stage, len(tasks)))
    write_json(os.path.join(plan_dir, "plan.json"), {
        "R": R, "seeds": seeds, "norm": args.norm, "signature": signature,
        "main_multiplier": M, "multipliers": args.multipliers,
        "gen_window": args.gen_window, "segments_per_window": k, "experiments": sorted(exps),
        "recipe": {"bases": recipe_bases, "pretrain_epochs": args.recipe_pretrain_epochs,
                   "encoder_lr_mult": args.recipe_encoder_lr_mult} if recipe_bases else None,
        "corpora": {n: {"kind": v[0], "multiplier": v[1], "scope": v[2], "arch": v[3],
                        "windows": (v[1] * R if v[0] != "real" else None)} for n, v in corpora.items()},
        "segments_to_sample": {"%s_%s" % a: n for a, n in need.items()}})


if __name__ == "__main__":
    main()
