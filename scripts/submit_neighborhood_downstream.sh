#!/bin/bash
# R1-02 downstream comparison for the completed W500 BiGRU neighborhood generators.
#
# Eight synthetic corpora are built: central25/50/75/full x x1/x2, where x1 and
# x2 are defined relative to the exact drink/eat training-window count of each
# neighborhood. Each condition is pretrained and fine-tuned with seeds 30-34,
# using the same early-stopping recipe as the deduplicated per-class x1/x2 runs.
# Evaluation and aggregation are chained so the completed experiment has
# subject-level confidence intervals.
#
#   bash scripts/submit_neighborhood_downstream.sh [--from N] [--to N]
#       [--seeds "30 31 32 33 34"] [--parallel N] [--dry-run]
#
# Stages: 3 corpora, 4 pretrain, 5 finetune, 6 evaluate, 7 analyze.
set -euo pipefail
source /mnt/beegfs/home/elian.riveros/deo-classifier/slurm/common.sh

OUT=/mnt/beegfs/home/elian.riveros/deo_runs_r1_02_neighborhoods
DATA="$DATA_ROOT/deo_dedup"
INPUT="$REPO_DIR/artifacts/r1_02_medoid_neighborhoods_5s/generator_inputs"
PLAN="$REPO_DIR/artifacts/r1_02_medoid_neighborhoods_5s/downstream_plan"
PY="python -u $REPO_DIR"
SEEDS="30 31 32 33 34"
PARALLEL="$SLURM_MAX_PARALLEL"
MAX_PRETRAIN_EPOCHS=1000

from=3; to=7; dry=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) from="$2"; shift 2 ;;
    --to) to="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --dry-run) dry=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

if [ "$from" -lt 3 ] || [ "$to" -gt 7 ] || [ "$from" -gt "$to" ]; then
  echo "stages must satisfy 3 <= from <= to <= 7" >&2
  exit 2
fi

mkdir -p "$PLAN/submitted"
P=$(mktemp -d "$PLAN/.new.XXXX")
trap 'rm -rf "$P"' EXIT
for stage in 3_corpora 4_pretrain 5_finetune 6_evaluate 7_analyze; do
  : > "$P/$stage.tsv"
done

for tier in central25 central50 central75 full; do
  n_drink=$(python3 -c "import json; print(json.load(open('$INPUT/drink_${tier}_w500.json'))['n_segments'])")
  n_eat=$(python3 -c "import json; print(json.load(open('$INPUT/eat_${tier}_w500.json'))['n_segments'])")
  drink_samples="$OUT/generators/gru_drink_${tier}_w500/samples_2x.npy"
  eat_samples="$OUT/generators/gru_eat_${tier}_w500/samples_2x.npy"
  for mult in 1 2; do
    count_drink=$((mult * n_drink))
    count_eat=$((mult * n_eat))
    corpus="$OUT/corpora/nbh_${tier}_x${mult}"
    condition="diff_gru_nbh_${tier}_x${mult}_es"
    printf 'tf\t%s/scripts/build_corpus.py --data-dir %s --kind diffusion --scope minority --out-dir %s --generated drink=%s eat=%s --class-counts drink=%d eat=%d\n' \
      "$PY" "$DATA" "$corpus" "$drink_samples" "$eat_samples" "$count_drink" "$count_eat" \
      >> "$P/3_corpora.tsv"
    for seed in $SEEDS; do
      pre="$OUT/pretrain/$condition/seed$seed"
      ft="$OUT/finetune/deo_dedup/$condition/seed$seed"
      printf 'tf\t%s/scripts/pretrain.py --data-dir %s --corpus %s --norm corpus --output sigmoid --epochs %d --early-stop-patience 20 --val-max-abs-acc 156.96 --batch-size 256 --lr 1e-4 --seed %d --out-dir %s\n' \
        "$PY" "$DATA" "$corpus" "$MAX_PRETRAIN_EPOCHS" "$seed" "$pre" >> "$P/4_pretrain.tsv"
      printf 'tf\t%s/scripts/finetune.py --data-dir %s --pretrained %s/autoencoder.keras --norm none --lr 1e-4 --early-stop-monitor val_accuracy --head deep --activation gelu --n-dense 200 --epochs 50 --batch-size 128 --seed %d --out-dir %s\n' \
        "$PY" "$DATA" "$pre" "$seed" "$ft" >> "$P/5_finetune.tsv"
      printf 'tf\t%s/scripts/evaluate.py --model %s/best_model.keras --data-dir %s --norm none --reference-ba 0.9055 --out-dir %s/inference/%s/seed%d\n' \
        "$PY" "$ft" "$DATA" "$OUT" "$condition" "$seed" >> "$P/6_evaluate.tsv"
    done
  done
done

printf 'tf\t%s/scripts/analyze.py --runs %s --data-root %s --data-name deo_dedup --main diff_gru_nbh_full_x1_es\n' \
  "$PY" "$OUT" "$DATA_ROOT" >> "$P/7_analyze.tsv"

for f in "$P"/*.tsv; do
  target="$PLAN/$(basename "$f")"
  cmp -s "$f" "$target" || mv "$f" "$target"
done

files=(x x x 3_corpora 4_pretrain 5_finetune 6_evaluate 7_analyze)
names=(x x x corpora pretrain finetune evaluate lysi)
kind=(x x x cpu gpu gpu gpu cpu)
time=(x x x 02:00:00 48:00:00 12:00:00 01:00:00 01:00:00)
mem=(x x x 64G 128G 96G 64G 32G)

dep=""
for stage_number in $(seq "$from" "$to"); do
  tasks="$PLAN/${files[$stage_number]}.tsv"
  n=$(grep -c . "$tasks")
  if [ "$dry" -eq 0 ]; then
    snapshot="$PLAN/submitted/${files[$stage_number]}_$(date +%Y%m%d-%H%M%S).tsv"
    cp "$tasks" "$snapshot"
    tasks="$snapshot"
  fi
  # shellcheck disable=SC2207
  resources=($(sbatch_args "${kind[$stage_number]}"))
  command=(sbatch --parsable "${resources[@]}" --time="${time[$stage_number]}"
           --mem="${mem[$stage_number]}" --job-name="nbh2-${names[$stage_number]}"
           --array="0-$((n - 1))%$PARALLEL"
           --output="$OUT/logs/%x_%A_%a.out"
           --export=ALL,REPO_DIR="$REPO_DIR",TASKS="$tasks")
  if [ -n "$dep" ]; then
    # Pretraining has 40 tasks but corpus construction has 8, so it waits for
    # the complete corpus array. Fine-tuning and evaluation have matching
    # 40-task indices and can advance independently per seed/condition.
    if [ "$stage_number" -eq 5 ] || [ "$stage_number" -eq 6 ]; then
      command+=(--dependency="aftercorr:$dep" --kill-on-invalid-dep=yes)
    else
      command+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
    fi
  fi
  command+=("$REPO_DIR/slurm/array_task.sbatch")
  if [ "$dry" -eq 1 ]; then
    echo "[dry-run] ${command[*]}"
    dep="DRY$stage_number"
  else
    dep=$("${command[@]}")
    echo "[submit] stage $stage_number (${names[$stage_number]}, $n tasks): job $dep"
  fi
done
