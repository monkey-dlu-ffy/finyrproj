"""Self-contained transaction-agent training script for a Kaggle notebook --
the ERP-corpus counterpart to scripts/kaggle_train_transaction.py (which
targets the old Sparkov dataset). Mirrors, line for line, the logic in
agents/transaction/{features_erp,model_erp}.py and scripts/run_fusion.py's
transaction_oof_scores()/full_agent scoring, so scores produced here are the
same a local run would produce, just on a machine with real RAM -- needed
now that the ledger is the full 533,009-row corpus (5-fold OOF retrains
xgboost+IsolationForest 5x on a ~310K-row train split, plus once more for
val/test scoring; this machine is RAM-constrained and has no GPU).

Not imported by anything in this repo -- it is a paste target.

How to use
----------
1. Run scripts/build_linked_corpus.py and scripts/generate_invoices.py
   locally first (produces data/linked/{train,val,test}.parquet and
   data/linked/images/manifest.parquet).
2. On kaggle.com, New Notebook -> Add data -> Upload -> the 4 files above.
3. Settings -> Accelerator: CPU is fine (xgboost hist method, no GPU needed).
4. Paste this whole file into one cell, Run All.
5. From the notebook's Output/Data pane, download transaction_scores.parquet
   (columns: row_id, split, transaction_score) and drop it into
   experiments/ locally, then tell Claude -- scripts/run_fusion.py picks it
   up in place of computing OOF scores in-process.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_FOLDS = 5
OUT_DIR = Path("/kaggle/working")

# ---------------------------------------------------------------------------
# Locate Kaggle inputs
# ---------------------------------------------------------------------------


def _find(name: str) -> Path:
    hits = list(Path("/kaggle/input").rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under /kaggle/input -- upload it as a notebook input first.")
    return hits[0]


TRAIN_PARQUET = _find("train.parquet")
VAL_PARQUET = _find("val.parquet")
TEST_PARQUET = _find("test.parquet")
MANIFEST_PARQUET = _find("manifest.parquet")

# ---------------------------------------------------------------------------
# Feature engineering (agents/transaction/features_erp.py, verbatim)
# ---------------------------------------------------------------------------

_BENFORD_EXPECTED = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
MIN_VENDOR_TXNS_FOR_BENFORD = 5


def _leading_digit(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    x = np.where(x < 1, x * (10 ** np.ceil(-np.log10(np.where(x == 0, 1, x)))), x)
    with np.errstate(divide="ignore"):
        exponent = np.floor(np.log10(np.where(x == 0, 1, x)))
    leading = (x / (10.0**exponent)).astype(int)
    return np.clip(leading, 1, 9)


def fit_vendor_account_stats(train_df: pd.DataFrame) -> dict:
    genuine = train_df.loc[train_df["is_fraud"] == 0]

    by_vendor_amt = genuine.groupby("vendor_id")["DMBTR"].agg(["mean", "std", "median", "count"])
    by_account_amt = genuine.groupby("HKONT")["DMBTR"].agg(["mean", "std"])

    vendor_freq_rank = genuine.groupby("vendor_id").size().rank(ascending=False, method="min").to_dict()

    benford_by_vendor = {}
    for vendor_id, g in genuine.groupby("vendor_id"):
        if len(g) < MIN_VENDOR_TXNS_FOR_BENFORD:
            continue
        digits = _leading_digit(g["DMBTR"].to_numpy())
        observed = np.array([(digits == d).mean() for d in range(1, 10)])
        benford_by_vendor[vendor_id] = float(np.abs(observed - _BENFORD_EXPECTED).sum())
    global_benford_deviation = float(np.mean(list(benford_by_vendor.values()))) if benford_by_vendor else 0.0

    return {
        "vendor_amt": by_vendor_amt.to_dict("index"),
        "account_amt": by_account_amt.to_dict("index"),
        "vendor_freq_rank": vendor_freq_rank,
        "max_freq_rank": float(max(vendor_freq_rank.values())) if vendor_freq_rank else 1.0,
        "benford_by_vendor": benford_by_vendor,
        "global_benford_deviation": global_benford_deviation,
    }


def _velocity_by_vendor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["vendor_id", "posting_date"]).copy()
    times = df["posting_date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    day_ns = 24 * 3_600_000_000_000
    window_ns = 7 * day_ns

    count_7d = np.empty(len(df))
    days_since_last = np.empty(len(df))
    for _, idx in df.groupby("vendor_id", sort=False).indices.items():
        idx = np.sort(idx)
        t = times[idx]
        cutoffs = t - window_ns
        left = np.searchsorted(t, cutoffs, side="right")
        count_7d[idx] = np.arange(len(t)) - left + 1
        gap = np.empty(len(t))
        gap[0] = 999.0
        gap[1:] = (t[1:] - t[:-1]) / day_ns
        days_since_last[idx] = gap

    df["txn_count_7d_vendor"] = count_7d
    df["days_since_last_vendor_txn"] = days_since_last
    return df


def add_features(df: pd.DataFrame, stats: dict, seed: int) -> pd.DataFrame:
    df = _velocity_by_vendor(df)

    vendor_amt = df["vendor_id"].map(lambda v: stats["vendor_amt"].get(v, {}))
    df["amount_zscore_by_vendor"] = [
        (row["DMBTR"] - va.get("mean", row["DMBTR"])) / (va.get("std") or 1.0)
        for row, va in zip(df.to_dict("records"), vendor_amt)
    ]
    df["amount_to_vendor_median_ratio"] = [
        row["DMBTR"] / va["median"] if va.get("median") else 1.0
        for row, va in zip(df.to_dict("records"), vendor_amt)
    ]

    account_amt = df["HKONT"].map(lambda a: stats["account_amt"].get(a, {}))
    df["amount_zscore_by_account"] = [
        (row["DMBTR"] - aa.get("mean", row["DMBTR"])) / (aa.get("std") or 1.0)
        for row, aa in zip(df.to_dict("records"), account_amt)
    ]

    df["vendor_frequency_rank"] = df["vendor_id"].map(lambda v: stats["vendor_freq_rank"].get(v, stats["max_freq_rank"] + 1))
    df["benford_first_digit_deviation"] = df["vendor_id"].map(
        lambda v: stats["benford_by_vendor"].get(v, stats["global_benford_deviation"])
    )

    df["is_round_amount"] = (df["DMBTR"] % 100 == 0).astype(int)
    df["is_weekend"] = (pd.to_datetime(df["posting_date"]).dt.dayofweek >= 5).astype(int)
    df["posting_lag_days"] = (pd.to_datetime(df["posting_date"]) - pd.to_datetime(df["doc_date"])).dt.days

    rng_hours = np.random.default_rng(seed)
    df["hour_of_day"] = rng_hours.choice(np.arange(24), size=len(df), p=_business_hour_weights())
    return df


def _business_hour_weights() -> np.ndarray:
    weights = np.full(24, 0.5)
    weights[8:18] = 4.0
    return weights / weights.sum()


NUMERIC_FEATURES = [
    "amount_zscore_by_vendor",
    "amount_zscore_by_account",
    "txn_count_7d_vendor",
    "days_since_last_vendor_txn",
    "is_round_amount",
    "is_weekend",
    "hour_of_day",
    "posting_lag_days",
    "benford_first_digit_deviation",
    "vendor_frequency_rank",
    "amount_to_vendor_median_ratio",
    "DMBTR",
]
CATEGORICAL_FEATURES = ["WAERS", "BUKRS", "KTOSL", "PRCTR", "BSCHL", "HKONT"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def fit_category_levels(*dfs: pd.DataFrame) -> dict[str, list]:
    """Canonical category levels per categorical column, fit across every
    split -- a rare BUKRS/PRCTR code that only lands in one CV fold or split
    must not become an xgboost-unrecognized category at scoring time (this
    exact crash is why this function exists: a fold trained without a fixed
    vocabulary hit `XGBoostError: Found a category not in the training set`
    on a code that only appeared in a held-out fold)."""
    combined = pd.concat([df[CATEGORICAL_FEATURES] for df in dfs], ignore_index=True)
    return {col: sorted(combined[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES}


def _as_categorical(series: pd.Series, levels: list | None) -> pd.Series:
    dtype = pd.CategoricalDtype(categories=levels) if levels is not None else "category"
    return series.astype(dtype)


def build_xy(df: pd.DataFrame, category_levels: dict[str, list] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = _as_categorical(X[col], category_levels[col] if category_levels else None)
    y = df["is_fraud"].astype("int8")
    return X, y


# ---------------------------------------------------------------------------
# Model (agents/transaction/model_erp.py, verbatim)
# ---------------------------------------------------------------------------

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "n_estimators": 500,
    "enable_categorical": True,
    "early_stopping_rounds": 30,
}


class TransactionAgent:
    def __init__(self, seed: int = RANDOM_SEED, category_levels: dict[str, list] | None = None):
        self.seed = seed
        self.category_levels = category_levels
        self.iforest = None
        self.xgb_model = None
        self.calibrator = None

    def _iforest_scores(self, X_numeric: pd.DataFrame) -> np.ndarray:
        return -self.iforest.decision_function(X_numeric)

    def fit(self, X_train, y_train, X_val, y_val) -> "TransactionAgent":
        self.iforest = IsolationForest(n_estimators=200, contamination="auto", random_state=self.seed)
        self.iforest.fit(X_train.loc[y_train == 0, NUMERIC_FEATURES])

        X_train = X_train.copy()
        X_val = X_val.copy()
        X_train["iforest_score"] = self._iforest_scores(X_train[NUMERIC_FEATURES])
        X_val["iforest_score"] = self._iforest_scores(X_val[NUMERIC_FEATURES])

        n_pos = int(y_train.sum())
        scale_pos_weight = (len(y_train) - n_pos) / max(n_pos, 1)
        params = dict(XGB_PARAMS, random_state=self.seed, scale_pos_weight=scale_pos_weight)
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_raw = self.xgb_model.predict_proba(X_val)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(val_raw, y_val)
        return self

    def _feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLUMNS].copy()
        for col in CATEGORICAL_FEATURES:
            X[col] = _as_categorical(X[col], self.category_levels[col] if self.category_levels else None)
        X["iforest_score"] = self._iforest_scores(X[NUMERIC_FEATURES])
        return X

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self._feature_frame(df)
        raw = self.xgb_model.predict_proba(X)[:, 1]
        return self.calibrator.predict(raw)


# ---------------------------------------------------------------------------
# Load + engineer (scripts/run_fusion.py::load_engineered_splits, verbatim)
# ---------------------------------------------------------------------------

print("loading...")
train = pd.read_parquet(TRAIN_PARQUET)
val = pd.read_parquet(VAL_PARQUET)
test = pd.read_parquet(TEST_PARQUET)
manifest = pd.read_parquet(MANIFEST_PARQUET).set_index("row_id")
print(f"train={len(train):,} val={len(val):,} test={len(test):,} manifest={len(manifest):,}")

print("engineering features...")
stats = fit_vendor_account_stats(train)
splits = {
    "train": add_features(train, stats, seed=RANDOM_SEED),
    "val": add_features(val, stats, seed=RANDOM_SEED),
    "test": add_features(test, stats, seed=RANDOM_SEED),
}
category_levels = fit_category_levels(splits["train"], splits["val"], splits["test"])

# ---------------------------------------------------------------------------
# OOF for train's imaged rows, full-fit scoring for val/test's imaged rows
# (scripts/run_fusion.py::transaction_oof_scores + main(), verbatim)
# ---------------------------------------------------------------------------


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
        print(f"  fold {fold + 1}/{N_FOLDS} done")
    return oof


X_train, y_train = build_xy(splits["train"], category_levels)
X_val, y_val = build_xy(splits["val"], category_levels)
print("fitting full-train agent (for val/test scoring)...")
full_agent = TransactionAgent(seed=RANDOM_SEED, category_levels=category_levels).fit(X_train, y_train, X_val, y_val)

rows = []
for split_name, df in splits.items():
    split_manifest = manifest.loc[manifest["split"] == split_name]
    sub_ids = split_manifest.index.intersection(df.index)

    if split_name == "train":
        print("computing 5-fold OOF for train's imaged rows...")
        txn_scores = transaction_oof_scores(splits["train"], splits["val"], sub_ids.to_numpy(), category_levels).to_dict()
    else:
        txn_scores = dict(zip(sub_ids, full_agent.score(df.loc[sub_ids])))

    for row_id in sub_ids:
        rows.append({"row_id": int(row_id), "split": split_name, "transaction_score": float(txn_scores[row_id])})
    print(f"{split_name}: {len(sub_ids):,} scores")

out = pd.DataFrame(rows)
out.to_parquet(OUT_DIR / "transaction_scores.parquet", index=False)
(OUT_DIR / "transaction_scores_meta.json").write_text(
    json.dumps({"n_rows": len(out), "n_by_split": out["split"].value_counts().to_dict(), "seed": RANDOM_SEED}, indent=2)
)
print(f"wrote transaction_scores.parquet ({len(out):,} rows) to /kaggle/working")
