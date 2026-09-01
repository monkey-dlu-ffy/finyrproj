"""Evaluation for the transaction agent.

Three views, because any one alone is misleading at 0.6% fraud prevalence:

  * threshold-free ranking quality (PR-AUC, ROC-AUC)
  * recall at a fixed operating point, chosen on val by precision target
    (an auditor works a queue -- the relevant number is "how much of the
    fraud gets surfaced at a precision the review team can tolerate", not
    accuracy)
  * recall broken out per typology (typ_*), so a model that is strong in
    aggregate but blind to one fraud pattern doesn't hide behind the mean
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from agents.transaction.features import TYPOLOGY_COLUMNS


def rank_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def threshold_at_precision(y_true: pd.Series, y_score: np.ndarray, target_precision: float) -> float:
    """Lowest score threshold on (y_true, y_score) that reaches target_precision.
    Fit on val, applied to test -- the operating point must not be chosen by
    looking at test performance."""
    precision, _, thresholds = precision_recall_curve(y_true, y_score)
    precision = precision[:-1]  # precision_recall_curve appends a trailing (1, 0) point with no threshold
    ok = np.where(precision >= target_precision)[0]
    if len(ok) == 0:
        return float(thresholds.max())
    return float(thresholds[ok[0]])


def recall_at_threshold(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def recall_by_typology(df: pd.DataFrame, y_score: np.ndarray, threshold: float) -> dict:
    pred = (y_score >= threshold).astype(int)
    fraud = df.loc[df["is_fraud"] == 1].copy()
    fraud_pred = pred[df["is_fraud"] == 1]
    out = {}
    for col in TYPOLOGY_COLUMNS:
        mask = fraud[col].to_numpy()
        n = int(mask.sum())
        if n == 0:
            out[col] = {"n": 0, "recall": float("nan")}
            continue
        out[col] = {"n": n, "recall": float(fraud_pred[mask].mean())}
    return out


def stability_check(y_score_a: np.ndarray, y_score_b: np.ndarray) -> dict:
    """E5: two runs of train_model with the same seed on the same input must
    produce identical (or near-identical, allowing hist-method float
    nondeterminism) scores. Reports max absolute score difference so drift
    is measured, not assumed away."""
    diff = np.abs(y_score_a - y_score_b)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "identical": bool(np.array_equal(y_score_a, y_score_b)),
    }
