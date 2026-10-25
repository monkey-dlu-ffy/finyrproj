"""Recompute every headline metric from the raw score files, two independent
ways, and check each one against what the pipeline saved.

    py -3.13 scripts\\verify_metrics.py

Nothing is retrained and nothing is written. The inputs are the per-entry
scores and labels (experiments/fusion_scores.parquet, one row per imaged
entry), the ledger splits, and the saved result JSONs the numbers are checked
against:

  - confusion counts are tallied by hand with numpy, then again by sklearn;
  - ROC-AUC comes from sklearn and from the Mann-Whitney rank formula;
  - PR-AUC comes from sklearn and from a hand-rolled average precision;
  - every recomputed number is printed beside the saved one, OK or DIFF.

Every metric on the imaged entries is on a sample built to hold ALL frauds
(scripts/generate_invoices.py), so it is ~60% fraud.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EXPERIMENTS_DIR, LINKED_DIR

SCORES = ("transaction_score", "document_score", "reconciliation_score", "fused_score")
TYPOLOGIES = ("duplicate_invoice", "ghost_vendor", "threshold_split", "desc_amount_mismatch", "altered_amount")
TOL = 1e-9

n_checks = {"OK": 0, "DIFF": 0}


def check(label: str, ours: float, stored: float) -> None:
    mark = "OK" if abs(float(ours) - float(stored)) <= TOL else "DIFF"
    n_checks[mark] += 1
    print(f"    {mark:<4}  {label:<44} recomputed {float(ours):.6f}   saved {float(stored):.6f}")


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def confusion_by_hand(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(y),
    }


def auc_mann_whitney(y: np.ndarray, score: np.ndarray) -> float:
    """P(a random fraud outscores a random genuine entry), ties counting half
    -- the rank-sum form of ROC-AUC, independent of sklearn's curve code."""
    ranks = rankdata(score)  # ties get their average rank
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision_by_hand(y: np.ndarray, score: np.ndarray) -> float:
    """Sum over distinct score thresholds, highest first, of
    precision x (gain in recall). Same definition as sklearn's, own code."""
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], y[order]
    tp_cum = np.cumsum(t)
    last_of_each_threshold = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = tp_cum[last_of_each_threshold]
    flagged = last_of_each_threshold + 1
    precision = tp / flagged
    recall = tp / t.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def main() -> int:
    scores = pd.read_parquet(EXPERIMENTS_DIR / "fusion_scores.parquet")
    fusion = json.loads((EXPERIMENTS_DIR / "fusion_results.json").read_text())
    experiments = json.loads((EXPERIMENTS_DIR / "experiment_results.json").read_text())
    thresholds = {"F1 threshold": fusion["f1_threshold"], "recall-90 threshold": fusion["recall_90_threshold"]}

    heading("1. What is being scored")
    ledger_counts = []
    for split in ("train", "val", "test"):
        ledger = pd.read_parquet(LINKED_DIR / f"{split}.parquet", columns=["is_fraud"])
        imaged = scores.loc[scores["split"] == split]
        ledger_counts.append({
            "split": split,
            "ledger entries": len(ledger), "ledger fraud": int(ledger["is_fraud"].sum()),
            "ledger fraud rate": round(ledger["is_fraud"].mean(), 4),
            "imaged entries": len(imaged), "imaged fraud": int(imaged["is_fraud"].sum()),
            "imaged fraud rate": round(imaged["is_fraud"].mean(), 4),
        })
    print(pd.DataFrame(ledger_counts).set_index("split").to_string())
    print("\nEvery fraud is imaged; the rest of the image budget is genuine. Metrics below are")
    print("on the imaged entries unless a section says otherwise.")

    heading("2. Operating points (fused_score >= threshold means flagged)")
    for split in ("val", "test"):
        part = scores.loc[scores["split"] == split]
        y, s = part["is_fraud"].to_numpy(), part["fused_score"].to_numpy()
        for name, threshold in thresholds.items():
            m = confusion_by_hand(y, s, threshold)
            # second, independent tally: sklearn
            tn, fp, fn, tp = confusion_matrix(y, s >= threshold, labels=[0, 1]).ravel()
            assert (tp, fp, fn, tn) == (m["tp"], m["fp"], m["fn"], m["tn"]), "numpy and sklearn counts disagree"
            assert abs(precision_score(y, s >= threshold) - m["precision"]) <= TOL
            assert abs(recall_score(y, s >= threshold) - m["recall"]) <= TOL
            print(f"\n  {split}, {name} {threshold}:")
            print(f"    TP {m['tp']:,}   FP {m['fp']:,}   FN {m['fn']:,}   TN {m['tn']:,}   (numpy and sklearn agree)")
            print(
                f"    precision {m['precision']:.4f}   recall {m['recall']:.4f}   F1 {m['f1']:.4f}   "
                f"false-positive rate {m['false_positive_rate']:.4f}   accuracy {m['accuracy']:.4f}"
            )
            key = {("val", "F1 threshold"): "val_at_f1_threshold",
                   ("test", "F1 threshold"): "test_at_f1_threshold",
                   ("test", "recall-90 threshold"): "test_at_recall_90_threshold"}.get((split, name))
            if key:
                for metric in ("tp", "fp", "fn", "precision", "recall"):
                    check(f"{key}.{metric}", m[metric], fusion[key][metric])

    heading("3. Ranking quality (threshold-free)")
    stored_points = {
        ("test", "transaction_score", "pr_auc"): experiments["E1_transaction_only"]["pr_auc"]["point"],
        ("test", "transaction_score", "roc_auc"): experiments["E1_transaction_only"]["roc_auc"]["point"],
        ("test", "document_score", "pr_auc"): experiments["E2_document_only"]["pr_auc"]["point"],
        ("test", "document_score", "roc_auc"): experiments["E2_document_only"]["roc_auc"]["point"],
        ("test", "fused_score", "pr_auc"): experiments["E3_full_framework"]["pr_auc"]["point"],
        ("test", "fused_score", "roc_auc"): experiments["E3_full_framework"]["roc_auc"]["point"],
        ("val", "fused_score", "pr_auc"): fusion["val_at_f1_threshold"]["pr_auc"],
        ("val", "fused_score", "roc_auc"): fusion["val_at_f1_threshold"]["roc_auc"],
    }
    rows = []
    for split in ("val", "test"):
        part = scores.loc[scores["split"] == split]
        y = part["is_fraud"].to_numpy()
        for col in SCORES:
            s = part[col].to_numpy()
            roc_sk, roc_mw = roc_auc_score(y, s), auc_mann_whitney(y, s)
            ap_sk, ap_hand = average_precision_score(y, s), average_precision_by_hand(y, s)
            assert abs(roc_sk - roc_mw) <= 1e-9 and abs(ap_sk - ap_hand) <= 1e-9, f"{split} {col}: methods disagree"
            rows.append({"split": split, "score": col, "ROC-AUC": round(roc_sk, 4), "PR-AUC": round(ap_sk, 4),
                         "random PR-AUC (base rate)": round(y.mean(), 4)})
            for metric, value in (("roc_auc", roc_sk), ("pr_auc", ap_sk)):
                if (split, col, metric) in stored_points:
                    check(f"{split} {col} {metric}", value, stored_points[(split, col, metric)])
    print("\n  Both methods agree on every value below (sklearn vs Mann-Whitney / hand-rolled AP):")
    print(pd.DataFrame(rows).set_index(["split", "score"]).to_string())

    heading("4. Recall by fraud typology (test, F1 threshold)")
    test = scores.loc[scores["split"] == "test"]
    flagged = test["fused_score"] >= fusion["f1_threshold"]
    typ_rows = []
    for typ in TYPOLOGIES:
        mask = test["typology"] == typ
        caught, n = int((flagged & mask).sum()), int(mask.sum())
        typ_rows.append({"typology": typ, "frauds": n, "caught": caught, "missed": n - caught, "recall": round(caught / n, 4)})
        check(f"E7 {typ} recall", caught / n, experiments["E7_recall_by_typology_full_framework"][typ]["recall"])
    print(pd.DataFrame(typ_rows).set_index("typology").to_string())

    heading("5. Risk tiers as stored in fusion_scores.parquet")
    expected = np.where(scores["fused_score"] >= fusion["f1_threshold"], "high", "low")
    check("entries whose tier != (score >= F1 threshold)", int((scores["risk_tier"] != expected).sum()), 0)
    tiers = (
        scores.groupby(["split", "risk_tier"])
        .agg(entries=("is_fraud", "size"), fraud=("is_fraud", "sum"),
             min_score=("fused_score", "min"), max_score=("fused_score", "max"))
    )
    tiers["fraud rate"] = (tiers["fraud"] / tiers["entries"]).round(4)
    print(tiers.round({"min_score": 4, "max_score": 4}).to_string())

    heading("6. Transaction agent alone, on every test-ledger entry")
    txn = pd.read_parquet(EXPERIMENTS_DIR / "transaction_erp_scores.parquet")
    txn_metrics = json.loads((EXPERIMENTS_DIR / "transaction_erp_metrics.json").read_text())
    y, s = txn["is_fraud"].to_numpy(), txn["score"].to_numpy()
    roc, ap = roc_auc_score(y, s), average_precision_score(y, s)
    assert abs(roc - auc_mann_whitney(y, s)) <= 1e-9 and abs(ap - average_precision_by_hand(y, s)) <= 1e-9
    print(f"  {len(y):,} entries, fraud rate {y.mean():.4f} (= a random scorer's PR-AUC)")
    check("full-ledger transaction ROC-AUC", roc, txn_metrics["test"]["roc_auc"])
    check("full-ledger transaction PR-AUC", ap, txn_metrics["test"]["pr_auc"])

    heading("Summary")
    print(f"  {n_checks['OK']} saved numbers reproduced exactly, {n_checks['DIFF']} differ.")
    return 0 if n_checks["DIFF"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
