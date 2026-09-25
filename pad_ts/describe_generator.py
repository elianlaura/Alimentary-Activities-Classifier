#!/usr/bin/env python
"""Parameter count per block of the two PaD-TS variants (R2-02, S-15), Markdown.

    python pad_ts/describe_generator.py [--window 24]
"""
import argparse

from deo_common import LEGACY_DEFAULTS, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=24)
    args = ap.parse_args()
    cfg = dict(LEGACY_DEFAULTS, window=args.window, n_features=9)
    print("# PaD-TS generator variants (window %d x 9, hidden %d, %d encoder / %d decoder blocks per channel)\n"
          % (args.window, cfg["hidden_size"], cfg["n_encoder"], cfg["n_decoder"]))
    for arch, use_gru in (("attn (original PaD-TS)", False), ("gru (BiGRU blocks)", True)):
        model = build_model(cfg, use_gru)
        total = sum(p.numel() for p in model.parameters())
        print("### %s: %s parameters\n" % (arch, format(total, ",")))
        print("| block | type | parameters |\n|---|---|---|")
        for name, module in model.named_children():
            n = sum(p.numel() for p in module.parameters())
            inner = type(module).__name__
            if hasattr(module, "encoder_blocks"):
                inner += " [" + ", ".join(sorted({type(b).__name__ for b in module.encoder_blocks})) + "]"
            print("| %s | %s | %s |" % (name, inner, format(n, ",")))
        print()


if __name__ == "__main__":
    main()
