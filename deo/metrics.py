"""Evaluation metrics.

Balanced accuracy is the mean of per-class recalls (sensitivities). Specificity is
computed one-vs-rest from the confusion matrix. Two versions of mAP / ROC-AUC are
reported:
  * *_hard   - computed from the arg-max predictions, as in the legacy code that
               produced Table I (mAP 0.7272, AUC 0.92-0.94). With hard labels the
               ROC curve has a single operating point, so this AUC is not a
               threshold-free measure.
  * *_score  - computed from the predicted class probabilities (standard definition).
"""
import numpy as np
from sklearn import metrics as skm

from . import CLASS_NAMES


def one_hot(y, n):
    return np.eye(n)[np.asarray(y, dtype=int)]


def classification_metrics(y_true, probs, n_classes=3):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = probs.argmax(1)
    labels = list(range(n_classes))
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    sens = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    spec = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    Y = one_hot(y_true, n_classes)
    P = one_hot(y_pred, n_classes)
    out = {
        "n": int(len(y_true)),
        "support": cm.sum(1).tolist(),
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": sens.tolist(),
        "specificity": spec.tolist(),
        "f1_per_class": skm.f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "f1_macro": float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(skm.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(skm.precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(skm.cohen_kappa_score(y_true, y_pred)),
        "mAP_hard": float(np.mean([skm.average_precision_score(Y[:, c], P[:, c]) for c in labels])),
        "mAP_score": float(np.mean([skm.average_precision_score(Y[:, c], probs[:, c]) for c in labels])),
        "auc_hard": [float(skm.roc_auc_score(Y[:, c], P[:, c])) for c in labels],
        "auc_score": [float(skm.roc_auc_score(Y[:, c], probs[:, c])) for c in labels],
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES[:n_classes],
    }
    return out


def per_subject_balanced_accuracy(y_true, y_pred, subjects):
    """Balanced accuracy of each subject (classes absent for a subject are skipped)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    subjects = np.asarray(subjects)
    out = {}
    for s in np.unique(subjects):
        m = subjects == s
        out[str(s)] = float(skm.balanced_accuracy_score(y_true[m], y_pred[m]))
    return out


def subject_bootstrap(y_true, y_pred, subjects, n_boot=10000, seed=20251028, n_classes=3):
    """Percentile 95% CI of balanced accuracy and recalls, resampling whole subjects (model fixed)."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    subs, inv = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
    cm = np.zeros((len(subs), n_classes, n_classes))
    np.add.at(cm, (inv, y_true, y_pred), 1)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(subs), (n_boot, len(subs)))
    weights = np.zeros((n_boot, len(subs)))
    np.add.at(weights, (np.arange(n_boot)[:, None], picks), 1)
    boot = np.einsum("bs,sij->bij", weights, cm)
    support = boot.sum(-1)
    rec = np.diagonal(boot, axis1=-2, axis2=-1) / np.maximum(support, 1)
    has = support > 0
    ba = (rec * has).sum(-1) / np.maximum(has.sum(-1), 1)
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"n_subjects": int(len(subs)), "replicates": int(n_boot), "seed": seed,
            "balanced_accuracy_ci95": pct(ba), "balanced_accuracy_boot_sd": float(ba.std(ddof=1)),
            "recall_ci95": {CLASS_NAMES[k]: pct(rec[:, k]) for k in range(n_classes)},
            "replicates_missing_a_class": int((~has.all(-1)).sum())}


def format_legacy(m):
    """Text block in the layout of the legacy metrics_test_*.txt files."""
    r4 = lambda v: [round(float(x), 4) for x in v]
    return "\n".join([
        "mAP score (hard): %.4f" % m["mAP_hard"],
        "mAP score (probabilities): %.4f" % m["mAP_score"],
        "balanced_accuracy_score: %.4f" % m["balanced_accuracy"],
        "F1-score (weighted): %.4f" % m["f1_weighted"],
        "F1-score (macro): %.4f" % m["f1_macro"],
        "Sensitivity per class: %s" % r4(m["sensitivity"]),
        "Specificity per class: %s" % r4(m["specificity"]),
        "AUC per class (probabilities): %s" % r4(m["auc_score"]),
        "Confusion matrix (rows=true): %s" % m["confusion_matrix"],
    ])
