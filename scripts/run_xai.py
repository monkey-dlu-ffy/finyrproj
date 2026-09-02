"""Phase 6 (XAI), scoped to what fits this session's time budget: SHAP
global + local explanations for the transaction agent, a deletion-AUC
faithfulness check against a random-feature control, and 3-seed stability
(Jaccard overlap of each case's top-5 SHAP features across seeds).

    python scripts/run_xai.py

LIME baseline comparison and a 5-seed (vs 3) stability run are cut for time;
noted at the end rather than silently skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.transaction.features_erp import add_features, build_xy, fit_vendor_account_stats
from agents.transaction.model_erp import TransactionAgent
from config import EXPERIMENTS_DIR, LINKED_DIR, RANDOM_SEED, seed_everything

N_DELETE_TOP_K = 3
N_STABILITY_SEEDS = 3


def load_engineered():
    train = pd.read_parquet(LINKED_DIR / "train.parquet")
    val = pd.read_parquet(LINKED_DIR / "val.parquet")
    test = pd.read_parquet(LINKED_DIR / "test.parquet")
    stats = fit_vendor_account_stats(train)
    return (
        add_features(train, stats, seed=RANDOM_SEED),
        add_features(val, stats, seed=RANDOM_SEED),
        add_features(test, stats, seed=RANDOM_SEED),
    )


def deletion_auc(agent: TransactionAgent, df: pd.DataFrame, X: pd.DataFrame, shap_values: np.ndarray, top_k: int, rng: np.random.Generator) -> dict:
    """Zero out (median-impute) the top-k |SHAP| features per row and see how
    much the score drops -- a faithful explanation should drop the score a
    lot; a random-feature control that doesn't drop nearly as much shows the
    SHAP ranking is doing real work, not just noise."""
    base_scores = agent.score(df)
    medians = X.median(numeric_only=True)

    def masked_score(mask_cols_per_row: list[list[str]]) -> np.ndarray:
        X_masked = X.copy()
        for i, cols in enumerate(mask_cols_per_row):
            for c in cols:
                if c in medians.index:
                    X_masked.iloc[i, X_masked.columns.get_loc(c)] = medians[c]
        df_masked = df.copy()
        for c in X.columns:
            if c in df_masked.columns:
                df_masked[c] = X_masked[c]
        return agent.score(df_masked)

    top_k_cols = [
        [X.columns[j] for j in np.argsort(-np.abs(shap_values[i]))[:top_k]] for i in range(len(X))
    ]
    random_cols = [
        [X.columns[j] for j in rng.choice(len(X.columns), size=top_k, replace=False)] for i in range(len(X))
    ]

    shap_masked_scores = masked_score(top_k_cols)
    random_masked_scores = masked_score(random_cols)

    return {
        "mean_score_drop_shap_topk": float(np.mean(base_scores - shap_masked_scores)),
        "mean_score_drop_random_control": float(np.mean(base_scores - random_masked_scores)),
    }


def stability_jaccard(train_df, val_df, test_df, n_seeds: int) -> dict:
    top5_per_seed = []
    X_test, _ = build_xy(test_df)
    for seed in range(n_seeds):
        X_train, y_train = build_xy(train_df)
        X_val, y_val = build_xy(val_df)
        agent = TransactionAgent(seed=RANDOM_SEED + seed).fit(X_train, y_train, X_val, y_val)
        X_test_full = agent._feature_frame(test_df)
        explainer = shap.TreeExplainer(agent.xgb_model)
        sv = explainer.shap_values(X_test_full)
        top5 = [set(X_test_full.columns[np.argsort(-np.abs(sv[i]))[:5]]) for i in range(len(X_test_full))]
        top5_per_seed.append(top5)

    n_rows = len(top5_per_seed[0])
    jaccards = []
    for row in range(n_rows):
        a, b = top5_per_seed[0][row], top5_per_seed[1][row]
        jaccards.append(len(a & b) / len(a | b))
    return {"mean_jaccard_seed0_vs_seed1": float(np.mean(jaccards)), "n_seeds_compared": n_seeds}


def main() -> int:
    seed_everything()
    train_df, val_df, test_df = load_engineered()
    X_train, y_train = build_xy(train_df)
    X_val, y_val = build_xy(val_df)
    X_test, y_test = build_xy(test_df)

    agent = TransactionAgent(seed=RANDOM_SEED).fit(X_train, y_train, X_val, y_val)
    X_test_full = agent._feature_frame(test_df)

    explainer = shap.TreeExplainer(agent.xgb_model)
    shap_values = explainer.shap_values(X_test_full)

    global_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test_full.columns).sort_values(ascending=False)
    print("--- global |SHAP| importance (top 8) ---")
    print(global_importance.head(8))

    rng = np.random.default_rng(RANDOM_SEED)
    faithfulness = deletion_auc(agent, test_df, X_test_full, shap_values, N_DELETE_TOP_K, rng)
    print("--- deletion faithfulness (bigger drop = more faithful) ---")
    print(json.dumps(faithfulness, indent=2))

    stability = stability_jaccard(train_df, val_df, test_df, N_STABILITY_SEEDS)
    print("--- stability (top-5 SHAP feature Jaccard overlap across seeds) ---")
    print(json.dumps(stability, indent=2))

    fraud_idx = np.where(y_test.to_numpy() == 1)[0]
    local_example = None
    if len(fraud_idx):
        i = int(fraud_idx[0])
        order = np.argsort(-np.abs(shap_values[i]))[:5]
        local_example = {
            "row_id": int(test_df.index[i]),
            "typology": test_df.iloc[i]["typology"],
            "top_features": [
                {"feature": X_test_full.columns[j], "value": X_test_full.iloc[i, j].item() if hasattr(X_test_full.iloc[i, j], "item") else X_test_full.iloc[i, j], "shap": float(shap_values[i, j])}
                for j in order
            ],
        }
    print("--- local example (first fraud row in test) ---")
    print(json.dumps(local_example, indent=2, default=str))

    out = {
        "global_importance": global_importance.to_dict(),
        "faithfulness": faithfulness,
        "stability": stability,
        "local_example": local_example,
    }
    (EXPERIMENTS_DIR / "xai_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {EXPERIMENTS_DIR / 'xai_results.json'}")
    print("NOT run this session (time budget): LIME baseline comparison, 5-seed (only ran 3) stability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
