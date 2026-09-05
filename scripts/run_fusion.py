"""Phase 5: fuses transaction + document + reconciliation agent scores with
a logistic regression meta-classifier.

    python scripts/run_fusion.py

Fusion runs over the document-linked subset (data/linked/images/manifest.parquet,
~16% of the full corpus) because document/reconciliation signal only exists
where an invoice image (real or ground-truth stand-in) exists -- the full
5,000-row ledger table only ever fed the transaction agent's own vendor-level
stats.

Out-of-fold: the transaction agent is retrained 5x, each time on the full
train split with that fold's manifest-train row_ids held out, so the
meta-classifier's training features for those rows never came from a model
that saw their label. Training the stacker on in-sample transaction-agent
predictions was flagged in the spec as the single most common stacking
mistake -- this is the guard against it. document_score/reconciliation_score
need no such guard: both are pure rule/heuristic functions of the row, not
fit on labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.document.extract import DocumentAgent
from agents.reconciliation.rules import ReconciliationAgent
from agents.transaction.features_erp import add_features, build_xy, fit_category_levels, fit_vendor_account_stats
from agents.transaction.model_erp import TransactionAgent
from config import EXPERIMENTS_DIR, LINKED_DIR, RANDOM_SEED, seed_everything
from linked_pipeline.invoices import resolve_vendor_name

N_FOLDS = 5
# At full ledger scale (533K rows), 5-fold OOF retraining is a Kaggle job --
# see scripts/kaggle_train_transaction_erp.py. When its output parquet is
# present, use it instead of retraining in-process here.
EXTERNAL_TXN_SCORES = EXPERIMENTS_DIR / "transaction_scores.parquet"


def load_engineered_splits() -> dict[str, pd.DataFrame]:
    train = pd.read_parquet(LINKED_DIR / "train.parquet")
    val = pd.read_parquet(LINKED_DIR / "val.parquet")
    test = pd.read_parquet(LINKED_DIR / "test.parquet")
    stats = fit_vendor_account_stats(train)
    return {
        "train": add_features(train, stats, seed=RANDOM_SEED),
        "val": add_features(val, stats, seed=RANDOM_SEED),
        "test": add_features(test, stats, seed=RANDOM_SEED),
    }


def load_manifest_joined() -> pd.DataFrame:
    manifest = pd.read_parquet(LINKED_DIR / "images" / "manifest.parquet").set_index("row_id")
    return manifest


def transaction_oof_scores(
    train_df: pd.DataFrame, val_df: pd.DataFrame, manifest_train_ids: np.ndarray, category_levels: dict[str, list]
) -> pd.Series:
    y_manifest_train = train_df.loc[manifest_train_ids, "is_fraud"]
    X_val, y_val = build_xy(val_df, category_levels)

    oof = pd.Series(index=manifest_train_ids, dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold, (_, held_pos) in enumerate(skf.split(manifest_train_ids, y_manifest_train)):
        held_ids = manifest_train_ids[held_pos]
        fit_pool = train_df.drop(index=held_ids)
        X_fit, y_fit = build_xy(fit_pool, category_levels)

        agent = TransactionAgent(seed=RANDOM_SEED + fold, category_levels=category_levels)
        agent.fit(X_fit, y_fit, X_val, y_val)
        oof.loc[held_ids] = agent.score(train_df.loc[held_ids])
    return oof


def main() -> int:
    seed_everything()
    splits = load_engineered_splits()
    manifest = load_manifest_joined()

    vendor_master = pd.read_json(LINKED_DIR / "vendor_master.json")
    vendor_name_by_id = dict(zip(vendor_master["vendor_id"], vendor_master["vendor_name"]))
    for df in splits.values():
        df["vendor_name"] = df["vendor_id"].map(lambda v: resolve_vendor_name(v, vendor_name_by_id))

    category_levels = fit_category_levels(splits["train"], splits["val"], splits["test"])

    genuine_train = splits["train"].loc[splits["train"]["is_fraud"] == 0]
    high_amount_threshold = float(genuine_train["DMBTR"].quantile(0.9))
    recon_agent = ReconciliationAgent(
        vendor_master_ids=set(vendor_master["vendor_id"]), high_amount_threshold=high_amount_threshold
    )
    doc_agent = DocumentAgent()

    external_txn_scores = None
    full_agent = None
    if EXTERNAL_TXN_SCORES.exists():
        print(f"using precomputed transaction scores from {EXTERNAL_TXN_SCORES} (Kaggle run)")
        external_txn_scores = pd.read_parquet(EXTERNAL_TXN_SCORES).set_index(["split", "row_id"])["transaction_score"]
    else:
        # full-train transaction agent, used for val/test manifest rows (no OOF needed there)
        X_train, y_train = build_xy(splits["train"], category_levels)
        X_val, y_val = build_xy(splits["val"], category_levels)
        full_agent = TransactionAgent(seed=RANDOM_SEED, category_levels=category_levels).fit(X_train, y_train, X_val, y_val)

    # scan the FULL ledger for duplicate invoice#s, not just the imaged subset -- see
    # ReconciliationAgent.analyze_batch docstring: duplicate_invoice only marks one of
    # its two rows fraud, and the other (genuine) copy is usually outside the ~16% subset.
    # row_id/split columns let the rule cite exactly where that other copy lives.
    full_ledger = pd.concat(
        [df.reset_index(names="row_id").assign(split=name) for name, df in splits.items()]
    )

    rows = []
    for split_name, df in splits.items():
        # row_id is only unique WITHIN a split (each split.parquet was written with
        # reset_index(drop=True)), so the manifest must be filtered to this split
        # before indexing by row_id, or .loc matches same-numbered rows from other
        # splits too and raises on the resulting duplicate labels.
        split_manifest = manifest.loc[manifest["split"] == split_name]
        sub_ids = split_manifest.index.intersection(df.index)
        joined = df.loc[sub_ids].copy()
        joined["extracted_amount"] = split_manifest.loc[sub_ids, "invoice_amount"]
        joined["extracted_vendor_name"] = split_manifest.loc[sub_ids, "vendor_name"]

        recon_scores = {e.row_id: e.score for e in recon_agent.analyze_batch(joined, ledger=full_ledger)}
        doc_scores = {e.row_id: e.score for e in doc_agent.analyze_batch(joined)}

        if external_txn_scores is not None:
            txn_scores = external_txn_scores.loc[split_name].to_dict()
        elif split_name == "train":
            txn_scores = transaction_oof_scores(splits["train"], splits["val"], sub_ids.to_numpy(), category_levels).to_dict()
        else:
            txn_scores = dict(zip(sub_ids, full_agent.score(joined)))

        for row_id in sub_ids:
            rows.append(
                {
                    "row_id": int(row_id),
                    "split": split_name,
                    "is_fraud": int(df.at[row_id, "is_fraud"]),
                    "typology": df.at[row_id, "typology"],
                    "transaction_score": float(txn_scores[row_id]),
                    "document_score": float(doc_scores[row_id]),
                    "reconciliation_score": float(recon_scores[row_id]),
                }
            )

    meta_df = pd.DataFrame(rows)
    feature_cols = ["transaction_score", "document_score", "reconciliation_score"]

    meta_train = meta_df.loc[meta_df["split"] == "train"]
    meta_val = meta_df.loc[meta_df["split"] == "val"]
    meta_test = meta_df.loc[meta_df["split"] == "test"]

    meta_clf = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
    meta_clf.fit(meta_train[feature_cols], meta_train["is_fraud"])

    meta_df["fused_score"] = meta_clf.predict_proba(meta_df[feature_cols])[:, 1]
    meta_val = meta_df.loc[meta_df["split"] == "val"]
    meta_test = meta_df.loc[meta_df["split"] == "test"]

    # threshold selection on val only
    val_scores = meta_val["fused_score"].to_numpy()
    val_y = meta_val["is_fraud"].to_numpy()
    candidate_thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(val_y, val_scores >= t, zero_division=0) for t in candidate_thresholds]
    f1_threshold = float(candidate_thresholds[int(np.argmax(f1s))])

    recall_90_threshold = 1.0
    for t in sorted(candidate_thresholds, reverse=True):
        preds = val_scores >= t
        tp = int(((preds == 1) & (val_y == 1)).sum())
        fn = int(((preds == 0) & (val_y == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if recall >= 0.9:
            recall_90_threshold = float(t)
            break

    def tier(score: float) -> str:
        if score < 0.3:
            return "low"
        if score < 0.7:
            return "medium"
        return "high"

    meta_df["risk_tier"] = meta_df["fused_score"].map(tier)

    def summarize(df: pd.DataFrame, threshold: float) -> dict:
        y = df["is_fraud"].to_numpy()
        s = df["fused_score"].to_numpy()
        preds = s >= threshold
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        return {
            "pr_auc": float(average_precision_score(y, s)) if y.sum() else float("nan"),
            "roc_auc": float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
        }

    results = {
        "coefficients": dict(zip(feature_cols, meta_clf.coef_[0].tolist())),
        "intercept": float(meta_clf.intercept_[0]),
        "f1_threshold": f1_threshold,
        "recall_90_threshold": recall_90_threshold,
        "val_at_f1_threshold": summarize(meta_val, f1_threshold),
        "test_at_f1_threshold": summarize(meta_test, f1_threshold),
        "test_at_recall_90_threshold": summarize(meta_test, recall_90_threshold),
    }
    print(json.dumps(results, indent=2))

    meta_df.to_parquet(EXPERIMENTS_DIR / "fusion_scores.parquet", index=False)
    (EXPERIMENTS_DIR / "fusion_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {EXPERIMENTS_DIR / 'fusion_scores.parquet'}, fusion_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
