#!/usr/bin/env python
"""Model complexity and on-device feasibility of the CABiGRU classifier (R1-07, R3-05).

    python scripts/complexity.py --model $RUNS/finetune/deo/diff_gru_x10/seed0/best_model.keras \
        --out $RUNS/analysis/complexity.json

Reports parameters (encoder / head), FLOPs for one 500x9 window, size of the Keras
file, TFLite size in fp32 and with dynamic-range int8 quantisation, single-window
CPU latency (Keras and TFLite, 1 thread) and the process peak RSS. Run it on a CPU
node to get CPU numbers; smartwatch latency still has to be measured on the device.
The Bidirectional GRUs convert to TFLite only with Flex (Select TF) ops; the stock Python
interpreter cannot run them, so the TFLite sizes are reported without TFLite latency.
"""
import argparse
import os
import resource
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo.utils import environment_info, write_json  # noqa: E402


def flops_of(model):
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    fn = tf.function(lambda x: model(x, training=False))
    concrete = fn.get_concrete_function(tf.TensorSpec([1] + list(model.input_shape[1:]), tf.float32))
    frozen = convert_variables_to_constants_v2(concrete)
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    info = tf.compat.v1.profiler.profile(graph=frozen.graph, run_meta=tf.compat.v1.RunMetadata(), cmd="op", options=opts)
    return int(info.total_float_ops)


def latency(fn, x, repeats=200, warmup=20):
    for _ in range(warmup):
        fn(x)
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        t.append(time.perf_counter() - t0)
    return {"median_ms": 1000 * float(np.median(t)), "p90_ms": 1000 * float(np.percentile(t, 90))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="trained classifier; default: untrained deep-head classifier")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from deo.models import build_autoencoder_s, build_scratch_classifier
    model = tf.keras.models.load_model(args.model, compile=False) if args.model else build_scratch_classifier()
    encoder = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    res = {"model": args.model or "untrained deep-head classifier",
           "params_total": int(model.count_params()),
           "params_encoder": int(encoder[0].count_params()) if encoder else None,
           "params_autoencoder_stage1": int(build_autoencoder_s().count_params())}
    res["params_head"] = res["params_total"] - (res["params_encoder"] or 0)
    try:
        res["flops_per_window"] = flops_of(model)
    except Exception as exc:  # profiler availability differs between TF builds
        res["flops_per_window"] = "failed: %s" % exc
    tmp = tempfile.mkdtemp()
    kpath = os.path.join(tmp, "m.keras")
    model.save(kpath)
    res["keras_file_mb"] = os.path.getsize(kpath) / 2 ** 20
    res["fp32_weights_mb"] = res["params_total"] * 4 / 2 ** 20
    x = np.random.default_rng(0).normal(size=(1,) + tuple(model.input_shape[1:])).astype(np.float32)
    compiled = tf.function(lambda a: model(a, training=False), jit_compile=False)
    res["keras_cpu_latency_1thread"] = latency(compiled, tf.constant(x))

    def convert(quant):
        """Builtin ops only (runs in the stock TFLite runtime); Flex ops as fallback (size only)."""
        for builtins_only in (True, False):
            conv = tf.lite.TFLiteConverter.from_keras_model(model)
            if builtins_only:
                conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            else:
                conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                conv._experimental_lower_tensor_list_ops = False
            if quant:
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
            try:
                return conv.convert(), builtins_only
            except Exception as exc:
                if not builtins_only:
                    raise
                print("[complexity] builtin-only conversion failed (%s); retrying with Flex ops" % str(exc)[:120])

    for name, quant in (("tflite_fp32", False), ("tflite_int8_dynamic", True)):
        try:
            blob, builtins_only = convert(quant)
            res[name + "_builtin_ops_only"] = builtins_only
            res[name + "_mb"] = len(blob) / 2 ** 20
            interp = tf.lite.Interpreter(model_content=blob, num_threads=1)
            interp.allocate_tensors()
            i_in = interp.get_input_details()[0]["index"]
            i_out = interp.get_output_details()[0]["index"]

            def run(a):
                interp.set_tensor(i_in, a)
                interp.invoke()
                return interp.get_tensor(i_out)
            res[name + "_cpu_latency_1thread"] = latency(run, x)
        except Exception as exc:
            res[name] = "failed: %s" % str(exc)[:300]
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    res["environment"] = environment_info()
    write_json(args.out, res)
    for k, v in res.items():
        if k != "environment":
            print("%-34s %s" % (k, v))


if __name__ == "__main__":
    main()
