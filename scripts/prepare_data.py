#!/usr/bin/env python
"""Stage 0: convert the source CSVs, write splits, normalisation stats and the
dataset description table (participants, windows per class and hand; S-07, S-19,
R1-03/R2-06).

    python scripts/prepare_data.py --data-root $DATA_ROOT \
        --deo-csv $DATA_ROOT/raw/fullraws3_vivabem012_drink0eat1another2_94u_..._5sec_100hz.csv \
        [--samehand-csv $DATA_ROOT/raw/fullraws2_vivabem12_noaer_R_D_MA_MB_..._6act.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deo import CLASS_NAMES, io, norm, splits  # noqa: E402
from deo.corpora import volume_unit  # noqa: E402
from deo.utils import write_json  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def describe(meta, split, name):
    rows = []
    for part in ("train", "val", "test", "all"):
        m = meta if part == "all" else meta[meta["subject"].isin(split[part])]
        row = {"dataset": name, "partition": part, "subjects": m["subject"].nunique(), "windows": len(m)}
        for k, c in enumerate(CLASS_NAMES):
            row[c] = int((m["label"] == k).sum())
        row["minority_share_%"] = round(100.0 * (row["DRINK"] + row["EAT"]) / max(len(m), 1), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def prepare(name, csv, labeler, data_root, split, default_info=None, skip_existing=True, hash_source=True):
    out = os.path.join(data_root, name)
    if skip_existing and os.path.exists(os.path.join(out, "meta.csv")):
        print("[prepare] %s already converted, skipping (use --force to redo)" % out)
    else:
        io.convert_csv(csv, out, labeler=labeler, default_info=default_info, hash_source=hash_source)
    X, meta = io.load_dataset(out)
    missing = set(split["train"] + split["val"] + split["test"]) - set(meta["subject"])
    if missing:
        raise ValueError("split subjects missing from %s: %s" % (name, sorted(missing)))
    splits.save_split(os.path.join(out, "split.json"), split)
    stats = norm.fit_stats(X, splits.indices(meta, split, "train"))
    norm.save_stats(os.path.join(out, "norm_train.json"), stats)
    table = describe(meta, split, name)
    hand = meta.groupby(["label", "hand", "dominance", "moment"]).size().rename("windows").reset_index()
    hand["class"] = hand["label"].map(dict(enumerate(CLASS_NAMES)))
    table.to_csv(os.path.join(out, "dataset_table.csv"), index=False)
    hand.to_csv(os.path.join(out, "hand_protocol_table.csv"), index=False)
    write_json(os.path.join(out, "volume_unit.json"), {"R_minority_train_windows": volume_unit(meta, split)})
    print(table.to_string(index=False))
    print(hand.to_string(index=False))
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--deo-csv", required=True)
    ap.add_argument("--samehand-csv", default=None)
    ap.add_argument("--split", default=os.path.join(REPO, "splits", "deo_legacy.json"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 of the source CSVs")
    args = ap.parse_args()

    split = splits.load_split(args.split)
    tables = [prepare("deo", args.deo_csv, "deo", args.data_root, split,
                      skip_existing=not args.force, hash_source=not args.no_hash)]
    if args.samehand_csv:
        out =os.path.join(args.data_root, "deo_samehand")
        if args.force or not os.path.exists(os.path.join(out, "meta.csv")):
            io.convert_csv(args.samehand_csv, out, labeler="samehand", default_info="R,D,MA|MB",
                           hash_source=not args.no_hash)
        _, meta_sh = io.load_dataset(out)
        split_sh = splits.extend_split(split, meta_sh["subject"].unique())
        tables.append(prepare("deo_samehand", args.samehand_csv, "samehand", args.data_root, split_sh,
                              default_info="R,D,MA|MB", skip_existing=True, hash_source=not args.no_hash))
    pd.concat(tables).to_csv(os.path.join(args.data_root, "dataset_table_all.csv"), index=False)


if __name__ == "__main__":
    main()
