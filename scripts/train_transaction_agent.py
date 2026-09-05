"""Trains the real (ERP) transaction agent -- small corpus (2,999 train
rows), so unlike Phase 2's Sparkov xgboost this runs locally in seconds, no
Kaggle needed.

    python scripts/train_transaction_agent.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.transaction.evaluate_erp import rank_metrics, recall_at_threshold, recall_by_typology, threshold_at_precision
from agents.transaction.features_erp import build_xy, fit_category_levels, fit_vendor_account_stats, add_features
from agents.transaction.model_erp import TransactionAgent
from config import EXPERIMENTS_DIR, LINKED_DIR, MODELS_DIR, RANDOM_SEED, ensure_dirs, seed_everything


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(LINKED_DIR / f"{name}.parquet")


def main() -> int:
    seed_everything()
    ensure_dirs()

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    t0 = time.time()
    stats = fit_vendor_account_stats(train_df)
    train_df = add_features(train_df, stats, seed=RANDOM_SEED)
    val_df = add_features(val_df, stats, seed=RANDOM_SEED)
    test_df = add_features(test_df, stats, seed=RANDOM_SEED)
    print(f"engineered features ({time.time() - t0:.1f}s)")

    category_levels = fit_category_levels(train_df, val_df, test_df)
    X_train, y_train = build_xy(train_df, category_levels)
    X_val, y_val = build_xy(val_df, category_levels)
    X_test, y_test = build_xy(test_df, category_levels)

    t0 = time.time()
    agent = TransactionAgent(seed=RANDOM_SEED, category_levels=category_levels)
    agent.fit(X_train, y_train, X_val, y_val)
    print(f"trained ({time.time() - t0:.1f}s)")

    val_score = agent.score(val_df)
    test_score = agent.score(test_df)

    threshold = threshold_at_precision(y_val, val_score, target_precision=0.5)
    metrics = {
        "seed": RANDOM_SEED,
        "operating_threshold": threshold,
        "val": {**rank_metrics(y_val, val_score), "operating_point": recall_at_threshold(y_val, val_score, threshold)},
        "test": {
            **rank_metrics(y_test, test_score),
            "operating_point": recall_at_threshold(y_test, test_score, threshold),
            "recall_by_typology": recall_by_typology(test_df, test_score, threshold),
        },
    }
    print(json.dumps(metrics, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent.xgb_model.save_model(str(MODELS_DIR / "transaction_erp_xgb.json"))
    import joblib

    joblib.dump(agent.iforest, MODELS_DIR / "transaction_erp_iforest.joblib")
    joblib.dump(agent.calibrator, MODELS_DIR / "transaction_erp_calibrator.joblib")
    (MODELS_DIR / "transaction_erp_stats.json").write_text(
        json.dumps(stats, indent=2, default=str)
    )

    metrics_out = EXPERIMENTS_DIR / "transaction_erp_metrics.json"
    metrics_out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {metrics_out}")

    scores_out = EXPERIMENTS_DIR / "transaction_erp_scores.parquet"
    pd.DataFrame(
        {
            "row_id": test_df.index,
            "score": test_score,
            "is_fraud": test_df["is_fraud"].to_numpy(),
            "typology": test_df["typology"].to_numpy(),
        }
    ).to_parquet(scores_out, index=False)
    print(f"wrote {scores_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
