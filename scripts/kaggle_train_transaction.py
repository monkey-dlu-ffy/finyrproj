"""Self-contained transaction-agent training script for a Kaggle notebook.

Not imported by anything in this repo -- it is a paste target. Mirrors, line
for line, the logic in data_pipeline/{features,typology,splits}.py and
agents/transaction/{features,model,evaluate}.py so a model trained here is
the same model this repo would produce, just on a machine with real RAM.

How to use
----------
1. On kaggle.com, New Notebook.
2. Add data -> search "Credit Card Transactions Fraud Detection Dataset"
   (kartik2112/fraud-detection) -> Add. This is the exact Sparkov dataset
   already in this repo's data/raw/ (same fraudTrain.csv / fraudTest.csv).
3. Settings -> Accelerator: CPU is fine (xgboost hist method); no GPU needed.
4. Paste this whole file into one cell, Run All.
5. From the notebook's Output/Data pane, download:
     transaction_xgb.json
     transaction_metrics.json
6. Locally: drop transaction_xgb.json into models/, transaction_metrics.json
   into experiments/, then tell Claude -- it will load the model, run SHAP
   (that part is cheap enough for this machine), and verify the metrics.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Reproducibility -- must match config.RANDOM_SEED in the repo
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

VAL_SIZE = 0.20

# ---------------------------------------------------------------------------
# Locate the Kaggle input
# ---------------------------------------------------------------------------
CANDIDATES = list(Path("/kaggle/input").rglob("fraudTrain.csv"))
if not CANDIDATES:
    raise FileNotFoundError(
        "fraudTrain.csv not found under /kaggle/input -- add the "
        "kartik2112/fraud-detection dataset to this notebook first."
    )
TRAIN_CSV = CANDIDATES[0]
TEST_CSV = TRAIN_CSV.parent / "fraudTest.csv"

OUT_DIR = Path("/kaggle/working")

DATE_COLS = ["trans_date_trans_time", "dob"]
CATEGORY_COLS = ["merchant", "category", "gender", "state", "job"]

# ---------------------------------------------------------------------------
# Load  (data_pipeline scripts/build_dataset.py::_load)
# ---------------------------------------------------------------------------


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=DATE_COLS)
    for col in CATEGORY_COLS:
        df[col] = df[col].astype("category")
    df["is_fraud"] = df["is_fraud"].astype("int8")
    return df


print("loading...")
train_df = load(TRAIN_CSV)
test_df = load(TEST_CSV)
print(f"train={len(train_df):,} test={len(test_df):,}")

# ---------------------------------------------------------------------------
# Feature engineering  (data_pipeline/features.py)
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_time_features(df):
    dt = df["trans_date_trans_time"]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_age(df):
    df["age_years"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    return df


def add_geo_distance(df):
    df["geo_distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    return df


def fit_category_amount_stats(df):
    genuine = df.loc[df["is_fraud"] == 0]
    return genuine.groupby("category", observed=True)["amt"].agg(["mean", "std"]).to_dict("index")


def add_amount_zscore(df, stats):
    cat = df["category"].astype(str)
    means = cat.map(lambda c: stats.get(c, {}).get("mean", np.nan))
    stds = cat.map(lambda c: stats.get(c, {}).get("std", np.nan)).replace(0, np.nan)
    df["amt_zscore_category"] = (df["amt"] - means) / stds
    return df


def _rolling_window_stats(times, amts, window_ns):
    cutoffs = times - window_ns
    left = np.searchsorted(times, cutoffs, side="right")
    cumsum = np.concatenate(([0.0], np.cumsum(amts)))
    idx = np.arange(len(times))
    count = (idx - left + 1).astype(float)
    total = cumsum[idx + 1] - cumsum[left]
    return count, total


def add_velocity_features(df):
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    times_ns = df["trans_date_trans_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    amts = df["amt"].to_numpy(dtype="float64")
    count_1h = np.empty(len(df), dtype="float64")
    sum_24h = np.empty(len(df), dtype="float64")
    hour_ns = 3_600_000_000_000
    day_ns = 24 * hour_ns
    starts = df.groupby("cc_num", observed=True, sort=False).indices
    for _, positions in starts.items():
        positions = np.sort(positions)
        g_times = times_ns[positions]
        g_amts = amts[positions]
        c1h, _ = _rolling_window_stats(g_times, g_amts, hour_ns)
        _, s24h = _rolling_window_stats(g_times, g_amts, day_ns)
        count_1h[positions] = c1h
        sum_24h[positions] = s24h
    df["velocity_count_1h"] = count_1h
    df["velocity_sum_24h"] = sum_24h
    return df


def add_all_features(df, category_amount_stats):
    df = add_velocity_features(df)
    df = add_time_features(df)
    df = add_age(df)
    df = add_geo_distance(df)
    df = add_amount_zscore(df, category_amount_stats)
    return df


# ---------------------------------------------------------------------------
# Typology tags  (data_pipeline/typology.py)
# ---------------------------------------------------------------------------

VELOCITY_MIN_COUNT_1H = 3
GEO_ANOMALY_PCTILE = 0.95
HIGH_VALUE_PCTILE = 0.99


def fit_typology_thresholds(df):
    genuine = df.loc[df["is_fraud"] == 0]
    geo_p95 = float(genuine["geo_distance_km"].quantile(GEO_ANOMALY_PCTILE))
    amt_p99_by_category = {
        str(k): float(v)
        for k, v in genuine.groupby("category", observed=True)["amt"].quantile(HIGH_VALUE_PCTILE).to_dict().items()
    }
    return {"geo_distance_km_p95": geo_p95, "amt_p99_by_category": amt_p99_by_category}


def add_typology_columns(df, thresholds):
    cat = df["category"].astype(str)
    df["typ_online"] = cat.str.endswith("_net")
    df["typ_pos"] = cat.str.endswith("_pos")
    df["typ_geo_anomaly"] = df["geo_distance_km"] > thresholds["geo_distance_km_p95"]
    df["typ_velocity"] = df["velocity_count_1h"] >= VELOCITY_MIN_COUNT_1H
    amt_thresh = cat.map(thresholds["amt_p99_by_category"])
    df["typ_high_value"] = df["amt"] > amt_thresh
    df["typ_doc_mismatch"] = False
    return df


TYPOLOGY_COLUMNS = (
    "typ_online",
    "typ_pos",
    "typ_geo_anomaly",
    "typ_velocity",
    "typ_high_value",
    "typ_doc_mismatch",
)

print("engineering features...")
amount_stats = fit_category_amount_stats(train_df)
train_df = add_all_features(train_df, amount_stats)
typ_thresholds = fit_typology_thresholds(train_df)
train_df = add_typology_columns(train_df, typ_thresholds)

test_df = add_all_features(test_df, amount_stats)
test_df = add_typology_columns(test_df, typ_thresholds)

# ---------------------------------------------------------------------------
# Split  (data_pipeline/splits.py)
# ---------------------------------------------------------------------------
train_df, val_df = train_test_split(
    train_df, test_size=VAL_SIZE, stratify=train_df["is_fraud"], random_state=RANDOM_SEED
)
print(f"split train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

# ---------------------------------------------------------------------------
# Model feature contract  (agents/transaction/features.py)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "amt",
    "city_pop",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "age_years",
    "geo_distance_km",
    "amt_zscore_category",
    "velocity_count_1h",
    "velocity_sum_24h",
]
CATEGORICAL_FEATURES = ["merchant", "category", "gender", "state", "job"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_xy(df):
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    y = df["is_fraud"].astype("int8")
    return X, y


X_train, y_train = build_xy(train_df)
X_val, y_val = build_xy(val_df)
X_test, y_test = build_xy(test_df)

# ---------------------------------------------------------------------------
# Train  (agents/transaction/model.py)
# ---------------------------------------------------------------------------
n_pos = int(y_train.sum())
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)

params = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=RANDOM_SEED,
    n_estimators=2000,
    enable_categorical=True,
    early_stopping_rounds=50,
    scale_pos_weight=scale_pos_weight,
)

print("training...")
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
print(f"best_iteration={model.best_iteration}")

# ---------------------------------------------------------------------------
# Evaluate  (agents/transaction/evaluate.py)
# ---------------------------------------------------------------------------
val_score = model.predict_proba(X_val)[:, 1]
test_score = model.predict_proba(X_test)[:, 1]


def rank_metrics(y_true, y_score):
    return {"pr_auc": float(average_precision_score(y_true, y_score)), "roc_auc": float(roc_auc_score(y_true, y_score))}


def threshold_at_precision(y_true, y_score, target_precision):
    precision, _, thresholds = precision_recall_curve(y_true, y_score)
    precision = precision[:-1]
    ok = np.where(precision >= target_precision)[0]
    if len(ok) == 0:
        return float(thresholds.max())
    return float(thresholds[ok[0]])


def recall_at_threshold(y_true, y_score, threshold):
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def recall_by_typology(df, y_score, threshold):
    pred = (y_score >= threshold).astype(int)
    fraud = df.loc[df["is_fraud"] == 1].copy()
    fraud_pred = pred[df["is_fraud"] == 1]
    out = {}
    for col in TYPOLOGY_COLUMNS:
        mask = fraud[col].to_numpy()
        n = int(mask.sum())
        out[col] = {"n": n, "recall": float(fraud_pred[mask].mean()) if n else float("nan")}
    return out


# Operating point chosen on val (never on test), target 50% precision.
target_precision = 0.5
threshold = threshold_at_precision(y_val, val_score, target_precision)

metrics = {
    "seed": RANDOM_SEED,
    "best_iteration": int(model.best_iteration),
    "val": {
        **rank_metrics(y_val, val_score),
        "operating_point": recall_at_threshold(y_val, val_score, threshold),
    },
    "test": {
        **rank_metrics(y_test, test_score),
        "operating_point": recall_at_threshold(y_test, test_score, threshold),
        "recall_by_typology": recall_by_typology(test_df, test_score, threshold),
    },
    "feature_columns": FEATURE_COLUMNS,
}
print(json.dumps(metrics, indent=2))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
model.save_model(str(OUT_DIR / "transaction_xgb.json"))
(OUT_DIR / "transaction_metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT_DIR / "transaction_feature_stats.json").write_text(
    json.dumps(
        {"category_amount_stats": {str(k): v for k, v in amount_stats.items()}, "typology_thresholds": typ_thresholds},
        indent=2,
        default=str,
    )
)
print("wrote transaction_xgb.json, transaction_metrics.json, transaction_feature_stats.json to /kaggle/working")
