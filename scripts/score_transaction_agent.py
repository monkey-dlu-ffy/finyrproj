"""Load a transaction model trained elsewhere (scripts/kaggle_train_transaction.py
on Kaggle) and score data/processed/test.parquet locally.

Local test.parquet already has the right features: its amt/geo/velocity stats
and typology thresholds were fit with the same code + seed as the Kaggle
script uses on the same source rows, so they agree deterministically -- no
need to re-derive anything from Kaggle's output, just the model file.

    python scripts/score_transaction_agent.py

Expects models/transaction_xgb.json (dropped in after a Kaggle run). Writes:
  experiments/transaction_scores.parquet   -- score for every test row
  experiments/transaction_evidence.json    -- top-k SHAP evidence for every
                                               row flagged at the val-chosen
                                               operating threshold
Cross-checks its own PR-AUC/ROC-AUC/recall-by-typology against
experiments/transaction_metrics.json (written by the Kaggle script) and
prints the diff -- this is the local proof that the model that got trained
is the model that gets used.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.transaction.evaluate import (
    rank_metrics,
    recall_at_threshold,
    recall_by_typology,
)
from agents.transaction.explain import explain_rows, make_explainer
from agents.transaction.features import build_xy
from config import EXPERIMENTS_DIR, MODELS_DIR, PROCESSED_DIR

MODEL_PATH = MODELS_DIR / "transaction_xgb.json"
KAGGLE_METRICS_PATH = EXPERIMENTS_DIR / "transaction_metrics.json"


def main() -> int:
    if not MODEL_PATH.exists():
        print(
            f"missing {MODEL_PATH} -- run scripts/kaggle_train_transaction.py on "
            "Kaggle and drop transaction_xgb.json + transaction_metrics.json in "
            "(models/ and experiments/ respectively) first."
        )
        return 1
    if not KAGGLE_METRICS_PATH.exists():
        print(f"missing {KAGGLE_METRICS_PATH} -- drop transaction_metrics.json into experiments/ too.")
        return 1

    kaggle_metrics = json.loads(KAGGLE_METRICS_PATH.read_text())
    threshold = kaggle_metrics["val"]["operating_point"]["threshold"]

    t0 = time.time()
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    print(f"loaded model ({time.time() - t0:.1f}s), best_iteration={model.best_iteration}")

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_test, y_test = build_xy(test_df)

    t0 = time.time()
    score = model.predict_proba(X_test)[:, 1]
    print(f"scored {len(score):,} rows ({time.time() - t0:.1f}s)")

    local_rank = rank_metrics(y_test, score)
    local_op = recall_at_threshold(y_test, score, threshold)
    local_typ = recall_by_typology(test_df, score, threshold)

    print("--- cross-check vs Kaggle-reported test metrics ---")
    for key in ("pr_auc", "roc_auc"):
        kaggle_v = kaggle_metrics["test"][key]
        local_v = local_rank[key]
        print(f"{key}: kaggle={kaggle_v:.4f} local={local_v:.4f} diff={abs(kaggle_v - local_v):.2e}")
    kaggle_recall = kaggle_metrics["test"]["operating_point"]["recall"]
    print(f"recall@threshold: kaggle={kaggle_recall:.4f} local={local_op['recall']:.4f}")

    scores_df = test_df[["trans_num"]].copy() if "trans_num" in test_df.columns else pd.DataFrame(index=test_df.index)
    scores_df["score"] = score
    scores_df["flagged"] = score >= threshold
    scores_out = EXPERIMENTS_DIR / "transaction_scores.parquet"
    scores_df.to_parquet(scores_out, index=False)
    print(f"wrote {scores_out} ({len(scores_df):,} rows, {int(scores_df['flagged'].sum())} flagged)")

    flagged_idx = scores_df.index[scores_df["flagged"]]
    print(f"computing SHAP evidence for {len(flagged_idx):,} flagged rows...")
    t0 = time.time()
    explainer = make_explainer(model)
    evidence = explain_rows(explainer, X_test.loc[flagged_idx])
    print(f"done ({time.time() - t0:.1f}s)")

    evidence_records = [
        {
            "row": int(idx),
            "trans_num": str(test_df.loc[idx, "trans_num"]) if "trans_num" in test_df.columns else None,
            "score": float(score[test_df.index.get_loc(idx)]),
            "is_fraud": int(test_df.loc[idx, "is_fraud"]),
            "evidence": ev,
        }
        for idx, ev in zip(flagged_idx, evidence)
    ]
    evidence_out = EXPERIMENTS_DIR / "transaction_evidence.json"
    evidence_out.write_text(json.dumps(evidence_records, indent=2))
    print(f"wrote {evidence_out}")

    print("--- recall by typology (local) ---")
    for k, v in local_typ.items():
        print(f"  {k:<16} n={v['n']:<5} recall={v['recall']:.2%}" if v["n"] else f"  {k:<16} n=0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
