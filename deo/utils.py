"""Seeding, timing and small I/O helpers shared by all scripts."""
import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager

import numpy as np


def set_seed(seed, deterministic=False):
    """Seed python, numpy and (if imported) TensorFlow / PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "tensorflow" in sys.modules:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
        fh.write("\n")
    return path


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def environment_info():
    """Versions and hardware, stored next to every result (R3-05)."""
    info = {
        "python": platform.python_version(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "numpy": np.__version__,
    }
    if "tensorflow" in sys.modules:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        try:
            import keras
            info["keras"] = keras.__version__
        except ImportError:
            pass
        gpus = tf.config.list_physical_devices("GPU")
        info["gpus"] = [g.name for g in gpus]
        try:
            info["gpu_names"] = [tf.config.experimental.get_device_details(g).get("device_name") for g in gpus]
        except Exception:
            pass
    if "torch" in sys.modules:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


class Timer:
    """Collects wall-clock durations of named stages."""

    def __init__(self):
        self.stages = {}

    @contextmanager
    def __call__(self, name):
        t0 = time.time()
        try:
            yield
        finally:
            self.stages[name] = round(time.time() - t0, 2)
            print("[time] %-24s %.1fs" % (name, self.stages[name]), flush=True)
