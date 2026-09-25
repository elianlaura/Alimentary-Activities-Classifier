#!/bin/bash
#SBATCH --job-name=boot_deo
#SBATCH --output=bootstrap_deo_09055/logs/boot_deo_%j.out
#SBATCH --error=bootstrap_deo_09055/logs/boot_deo_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# Varias particiones: slurm toma la primera con GPU libre. l40s suele estar saturada y
# esto es solo inferencia + lectura de un csv de 10 GB, no entrenamiento.
#SBATCH --partition=l40s,rtx8000,a5000,rtx5000

# Protocolo: SnsEnc/docs/protocolo_bootstrap_deo_09055.md
# Intervalo bootstrap del 95 % por sujetos para la balanced accuracy 0.9055 del
# experimento DEO (CaBiGRU preentrenado con datos sinteticos + fine-tuning).
#
# Etapas (se pueden limitar con STAGES="1 2 3 4"):
#   1  reconstruye la particion de prueba guardada, ejecuta inferencia y exporta predicciones
#   2  bootstrap por sujetos, 10.000 replicas, semilla 20251028
#   3  inventario de las 96 configuraciones del barrido
#   4  report.md
#
# Nota: a diferencia de submit_job_rec_main.sh NO se activa TF_FORCE_RUN_EAGERLY.
# Ese modo desactiva los kernels fusionados de cuDNN y volveria la inferencia sobre
# 71.591 ventanas de 500x9 impracticable. Solo se hace inferencia, no entrenamiento.

export PYTHONUNBUFFERED=1
source ~/miniconda3/bin/activate

echo "activating a2-project environment (TF 2.20 / Keras 3, requerido por el checkpoint)"
conda activate a2-project

## CUDA/cuDNN setup
# Build a CUDA 12 runtime library path from pip nvidia packages.
cuda12_libs=$(python - <<'PY'
import glob
import os
import site

# A diferencia de submit_job_rec_main.sh, que lista cinco directorios a mano, aqui se
# exportan TODOS los nvidia/*/lib del entorno. TF 2.20 tambien necesita cufft, cusolver,
# cusparse, curand y nvjitlink: si falta alguno registra "Cannot dlopen some GPU
# libraries" y cae silenciosamente a CPU.
lib_chunks = []
for sp in site.getsitepackages():
  nvidia_root = os.path.join(sp, "nvidia")
  if not os.path.isdir(nvidia_root):
    continue
  found = sorted(p for p in glob.glob(os.path.join(nvidia_root, "*", "lib"))
                 if os.path.isdir(p))
  if found:
    lib_chunks = found
    break

print(":".join(lib_chunks))
PY
)

# ptxas/nvcc empaquetados por pip, como respaldo del toolkit del sistema.
nvcc_bin=$(python - <<'PY'
import glob
import os
import site

for sp in site.getsitepackages():
  for cand in sorted(glob.glob(os.path.join(sp, "nvidia", "cuda_nvcc", "bin"))):
    if os.path.isdir(cand):
      print(cand)
      raise SystemExit
print("")
PY
)
if [ -n "$nvcc_bin" ]; then
  export PATH="${nvcc_bin}:$PATH"
fi

if [ -z "${CUDA_HOME:-}" ]; then
  for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12 /usr/local/cuda; do
    if [ -d "$cuda_dir" ]; then
      export CUDA_HOME="$cuda_dir"
      break
    fi
  done
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

cuda_lib_path="$cuda12_libs"
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/lib64" ]; then
  if [ -n "$cuda_lib_path" ]; then
    cuda_lib_path="$cuda_lib_path:$CUDA_HOME/lib64"
  else
    cuda_lib_path="$CUDA_HOME/lib64"
  fi
  if [ -d "$CUDA_HOME/extras/CUPTI/lib64" ]; then
    cuda_lib_path="$cuda_lib_path:$CUDA_HOME/extras/CUPTI/lib64"
  fi
fi
if [ -n "$cuda_lib_path" ]; then
  export LD_LIBRARY_PATH="$cuda_lib_path:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# Safe CUDA defaults for heterogeneous cluster nodes.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=LAZY
export TF_FORCE_GPU_ALLOW_GROWTH=true
unset XLA_FLAGS
export TF_XLA_FLAGS="--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
if [ -n "${CUDA_HOME:-}" ]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
fi

PROJECT=/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main
ANALYSIS="${PROJECT}/bootstrap_deo_09055"
cd "$PROJECT"

# pyarrow instalado en un directorio local del analisis (no modifica el entorno conda).
export PYTHONPATH="${ANALYSIS}/pylibs:${PYTHONPATH:-}"

mkdir -p "${ANALYSIS}/logs"

ts=$(date +%Y%m%d-%H%M%S)
RUN_TAG="${RUN_TAG:-${ts}_${SLURM_JOB_ID:-local}}"
OUT_DIR="${ANALYSIS}/results/bootstrap_${RUN_TAG}"
mkdir -p "$OUT_DIR"

STAGES="${STAGES:-1 2 3 4}"

node_name="${SLURMD_NODENAME:-$(hostname)}"
echo "=================================================================="
echo "Bootstrap DEO 0.9055 - protocolo_bootstrap_deo_09055.md"
echo "Job ID   : ${SLURM_JOB_ID:-<sin slurm>}"
echo "Nodo     : ${node_name}"
echo "Inicio   : ${ts}"
echo "Salidas  : ${OUT_DIR}"
echo "Etapas   : ${STAGES}"
echo "=================================================================="

echo "[Preflight] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "[Preflight] CUDA_HOME=${CUDA_HOME:-<unset>}"
echo "[Preflight] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
echo "[Preflight] PYTHONPATH=${PYTHONPATH:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
else
  echo "[Preflight][WARN] nvidia-smi no esta en PATH."
fi

python - <<'PY'
import sys
print("[Preflight] Python:", sys.executable)
import numpy, pandas, sklearn
print("[Preflight] numpy", numpy.__version__, "| pandas", pandas.__version__, "| sklearn", sklearn.__version__)
try:
    import pyarrow
    print("[Preflight] pyarrow", pyarrow.__version__)
except Exception as exc:
    print("[Preflight][WARN] pyarrow no disponible (%s); se escribira csv.gz" % exc)
import tensorflow as tf, keras
print("[Preflight] tensorflow", tf.__version__, "| keras", keras.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("[Preflight] GPUs visibles:", gpus)
if not gpus:
    print("[Preflight][WARN] TensorFlow no ve GPU; la inferencia correra en CPU.")
PY

if [ "$?" -ne 0 ]; then
  echo "[Preflight][ERROR] Fallo la verificacion del entorno. Se aborta."
  exit 2
fi

echo "[Preflight] Checks completados."
echo

run_stage () {
  local n="$1"; shift
  case " $STAGES " in
    *" $n "*) ;;
    *) echo "--- Etapa $n omitida ---"; return 0 ;;
  esac
  echo "------------------------------------------------------------------"
  echo "--- Etapa $n: $* "
  echo "------------------------------------------------------------------"
}

# --- Etapa 1: predicciones por ventana con el sujeto correspondiente ---------
run_stage 1 "Paso 1 - recuperar las predicciones del experimento"
case " $STAGES " in *" 1 "*)
  python -u "${ANALYSIS}/scripts/01_export_test_predictions.py" --out-dir "$OUT_DIR" || {
    echo "[ERROR] Etapa 1 fallida. El protocolo exige resolver la discrepancia antes"
    echo "[ERROR] de atribuir un intervalo al resultado historico. Se detiene aqui."
    exit 3
  }
;; esac

# --- Etapa 2: bootstrap por sujetos -----------------------------------------
run_stage 2 "Paso 2 - bootstrap por sujetos (10.000 replicas, semilla 20251028)"
case " $STAGES " in *" 2 "*)
  python -u "${ANALYSIS}/scripts/02_run_bootstrap.py" --out-dir "$OUT_DIR" || exit 4
;; esac

# --- Etapa 3: inventario del barrido ----------------------------------------
run_stage 3 "Paso 3 - inventario de las configuraciones comparadas"
case " $STAGES " in *" 3 "*)
  python -u "${ANALYSIS}/scripts/03_configuration_inventory.py" --out-dir "$OUT_DIR" || exit 5
;; esac

# --- Etapa 4: informe --------------------------------------------------------
run_stage 4 "Entregable - report.md"
case " $STAGES " in *" 4 "*)
  python -u "${ANALYSIS}/scripts/04_make_report.py" --out-dir "$OUT_DIR" || exit 6
;; esac

echo
echo "=================================================================="
echo "Entregables en ${OUT_DIR}:"
ls -la "$OUT_DIR"
ts2=$(date +%Y%m%d-%H%M%S)
echo "Fin del proceso ${ts2}"
echo "=================================================================="
