"""Phase 8: E1/E2/E3/E4/E7 + bootstrap CIs, all derived from
experiments/fusion_scores.parquet -- every baseline and ablation is just a
different linear combination of the three already-computed agent scores, so
no retraining is needed here.

    python scripts/run_experiments.py

E1  transaction-only (the classical-baseline slot: this single xgboost+
    IsolationForest score stands in for "tabular baseline")
E2  document-only (extraction confidence + text-risk, no reconciliation/txn)
E3  full framework (the fused logistic-regression score)
E4  ablations: drop each one of the three agents, refit the meta-classifier
    on the remaining two, see what breaks
E7  per-typology recall at the full framework's own operating threshold
E5/E6 (stability, latency) are handled elsewhere (model_erp's own seeded
determinism; dashboard request timing) -- not duplicated here.

Auditor survey and demo video are out of scope for this session's time
budget -- flagged, not silently dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EXPERIMENTS_DIR, RANDOM_SEED

N_BOOTSTRAP = 1000
TYPOLOGIES = ("duplicate_invoice", "ghost_vendor", "threshold_split", "desc_amount_mismatch", "altered_amount")


def bootstrap_ci(y: np.ndarray, score: np.ndarray, metric_fn, n: int = N_BOOTSTRAP, seed: int = RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed)
    if y.sum() == 0 or y.sum() == len(y):
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    values = []
    idx = np.arange(len(y))
    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if y[sample].sum() == 0:
            continue
        values.append(metric_fn(y[sample], score[sample]))
    values = np.array(values)
    return {
        "point": float(metric_fn(y, score)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def eval_score(df: pd.DataFrame, score_col: str) -> dict:
    y = df["is_fraud"].to_numpy()
    s = df[score_col].to_numpy()
    return {
        "pr_auc": bootstrap_ci(y, s, average_precision_score),
        "roc_auc": bootstrap_ci(y, s, lambda yy, ss: roc_auc_score(yy, ss) if len(set(yy)) > 1 else float("nan")),
    }


def recall_by_typology(df: pd.DataFrame, score_col: str, threshold: float) -> dict:
    pred = (df[score_col] >= threshold).astype(int)
    out = {}
    for typ in TYPOLOGIES:
        mask = df["typology"] == typ
        n = int(mask.sum())
        out[typ] = {"n": n, "recall": float(pred[mask].mean()) if n else float("nan")}
    return out


def refit_ablation(meta_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    train = meta_df.loc[meta_df["split"] == "train"]
    clf = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(train[feature_cols], train["is_fraud"])
    return pd.Series(clf.predict_proba(meta_df[feature_cols])[:, 1], index=meta_df.index)


def main() -> int:
    meta_df = pd.read_parquet(EXPERIMENTS_DIR / "fusion_scores.parquet")
    fusion_results = json.loads((EXPERIMENTS_DIR / "fusion_results.json").read_text())
    threshold = fusion_results["f1_threshold"]

    test_df = meta_df.loc[meta_df["split"] == "test"].copy()

    results = {
        "E1_transaction_only": eval_score(test_df, "transaction_score"),
        "E2_document_only": eval_score(test_df, "document_score"),
        "E3_full_framework": eval_score(test_df, "fused_score"),
        "E7_recall_by_typology_full_framework": recall_by_typology(test_df, "fused_score", threshold),
    }

    ablations = {
        "no_document": ["transaction_score", "reconciliation_score"],
        "no_reconciliation": ["transaction_score", "document_score"],
        "no_transaction": ["document_score", "reconciliation_score"],
    }
    e4 = {}
    for name, cols in ablations.items():
        scored = refit_ablation(meta_df, cols)
        ablated_test = test_df.copy()
        ablated_test["_ablated_score"] = scored.loc[test_df.index]
        e4[name] = eval_score(ablated_test, "_ablated_score")
    results["E4_ablations"] = e4

    print(json.dumps(results, indent=2))
    (EXPERIMENTS_DIR / "experiment_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {EXPERIMENTS_DIR / 'experiment_results.json'}")
    print("\nNOT run this session (out of 3hr time budget): auditor survey (E-survey, needs 10-15")
    print("human respondents over time), LayoutLMv3 fine-tune field-F1 table, demo video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
