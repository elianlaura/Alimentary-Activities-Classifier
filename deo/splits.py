"""Subject-wise splits.

The legacy experiments (Table I of the paper) used a single split drawn with
sklearn.train_test_split(test_size=0.3) and then train_size=0.86 / test_size=0.14
on the remaining subjects, under numpy seed 30. The drawn subjects were saved in
eatdrinkanother_94u_1.txt and are stored here as splits/deo_legacy.json so the
split is never re-drawn (55 train / 10 val / 29 test subjects).

make_split() reproduces the same procedure for repeated random splits.
"""
import re

import numpy as np

from .utils import read_json, write_json


def parse_legacy_split_txt(path):
    with open(path) as fh:
        text = fh.read()
    out = {}
    for key, name in (("uuid_train", "train"), ("uuid_val", "val"), ("uuid_test", "test")):
        m = re.search(key + r":\s*\[(.*?)\]", text, flags=re.S)
        out[name] = re.findall(r"'([^']+)'", m.group(1))
    return out


def make_split(subjects, seed, test_size=0.3, val_size=0.14):
    from sklearn.model_selection import train_test_split
    subjects = np.unique(np.asarray(subjects, dtype=str))
    train, test = train_test_split(subjects, test_size=test_size, random_state=seed)
    train, val = train_test_split(train, test_size=val_size, random_state=seed)
    return {"train": sorted(train.tolist()), "val": sorted(val.tolist()), "test": sorted(test.tolist()),
            "procedure": "train_test_split(test=%.2f) then (val=%.2f), random_state=%d" % (test_size, val_size, seed)}


def extend_split(split, subjects):
    """Keep an existing split and send subjects it does not know to train.

    Used for the same-hand control dataset (97 subjects), so its test subjects are
    exactly the DEO test subjects and the DEO-trained generator never saw them.
    """
    known = set(split["train"]) | set(split["val"]) | set(split["test"])
    extra = sorted(set(map(str, subjects)) - known)
    present = set(map(str, subjects))
    out = {k: [s for s in split[k] if s in present] for k in ("train", "val", "test")}
    out["train"] = sorted(out["train"] + extra)
    out["procedure"] = "extension of %s; %d unseen subjects added to train" % (
        split.get("name", "base split"), len(extra))
    return out


def load_split(path):
    split = read_json(path)
    for k in ("train", "val", "test"):
        split[k] = [str(s) for s in split[k]]
    overlap = (set(split["train"]) & set(split["val"])) | (set(split["train"]) & set(split["test"])) \
        | (set(split["val"]) & set(split["test"]))
    if overlap:
        raise ValueError("subjects in more than one partition: %s" % sorted(overlap))
    return split


def save_split(path, split):
    write_json(path, split)


def indices(meta, split, part):
    return np.flatnonzero(meta["subject"].isin(split[part]).to_numpy())
