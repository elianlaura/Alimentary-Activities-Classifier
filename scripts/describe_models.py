#!/usr/bin/env python
"""Layer tables of the CABiGRU autoencoder / classifier (R2-02), in Markdown.

    python scripts/describe_models.py > docs/ARCHITECTURES_cabigru.md
    python pad_ts/describe_generator.py >> docs/ARCHITECTURES_padts.md   (PyTorch env)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def table(model, title):
    rows = ["### %s (%s parameters)\n" % (title, format(model.count_params(), ",")),
            "| # | layer | type | output shape | parameters |", "|---|---|---|---|---|"]
    for i, layer in enumerate(model.layers):
        shape = getattr(layer, "output", None)
        shape = tuple(shape.shape) if shape is not None and hasattr(shape, "shape") else "-"
        kind = type(layer).__name__
        if kind == "Bidirectional":
            kind = "Bidirectional(GRU, %d units)" % layer.forward_layer.units
        elif kind == "Dense":
            kind = "Dense(%d, %s)" % (layer.units, layer.activation.__name__)
        elif kind == "Conv1D":
            kind = "Conv1D(%d filters, kernel %d)" % (layer.filters, layer.kernel_size[0])
        elif kind == "MultiHeadAttention":
            kind = "MultiHeadAttention(%d heads, key_dim %d)" % (layer.num_heads, layer.key_dim)
        elif kind == "Dropout":
            kind = "Dropout(%.2f)" % layer.rate
        rows.append("| %d | %s | %s | %s | %s |" % (i, layer.name, kind, str(shape).replace("None", "B"),
                                                  format(layer.count_params(), ",")))
    return "\n".join(rows) + "\n"


def main():
    from deo.models import build_autoencoder_s, build_classifier, encoder_from_autoencoder
    ae = build_autoencoder_s()
    enc = encoder_from_autoencoder(ae)
    print("# CABiGRU architectures\n")
    print("Input: one 5-s window, 500 time steps x 9 axes (acc, gyro, mag; x/y/z). B = batch.\n")
    print(table(ae, "Stage 1: autoencoder (reconstruction / MSE)"))
    print(table(enc, "Encoder reused in stage 2 (autoencoder truncated at the last encoder BiGRU)"))
    for head in ("deep", "balanced", "light", "classic"):
        clf = build_classifier(enc, head=head)
        print(table(clf, "Stage 2: classifier, '%s' head" % head))


if __name__ == "__main__":
    main()
