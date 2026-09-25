# Legacy code (read-only provenance)

Snapshots of the code that produced the archived results. They contain absolute paths of the
original cluster and are **not** meant to be run; the maintained pipeline is in `deo/`, `scripts/`
and `pad_ts/`. See `docs/REPRODUCIBILITY.md` for the full audit.

| Folder | What it is |
|---|---|
| `finetune_sweep_20251026/` | Code copied by the fine-tuning sweep `recurrent_models_20251026-024834` (96 runs; the 0.9055 run is config 19). `recurrent_models_main.py` + `recurrent_models_reversioned_100625_ori.py` (training loop, `clfOnlyTowers`), `models.py` (`build_autoencoder_s`, `build_finetune_model`), `utils.py` (data loading, split, metrics), `parse_args.py`, `finetune_models.py` (id → pretrained checkpoint). |
| `pretrain_20251012/` | Autoencoder pretraining script copied by `recurrent_models_20251012-100533` (backbone of the 0.9055 run). It imported `utils.py` / `models.py` from `Recurrence/` at that date. |
| `padts_20250922/` | Head of the PaD-TS training log (architecture summary: attention blocks, 8,959,009 parameters) and the generation log of 2025-09-25. |
| `bootstrap_deo_09055/` | Subject bootstrap of the 0.9055 run (2026-09-22): scripts, README and the Slurm file. Results in `artifacts/legacy/table1_diffusion_09055/`. |
| `iscmi_src/` | Scripts of the ISCMI 2025 CABiGRU model trained from scratch (previous content of this repository's `src/`). |
