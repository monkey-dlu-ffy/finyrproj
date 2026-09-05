"""Feature engineering for the real (ERP) transaction agent -- the exact
list from the project spec. All vendor/account-level stats are fit on train
genuine rows only and applied to val/test, same leakage discipline as
data_pipeline/features.py's category-amount stats. txn_count_7d_vendor and
days_since_last_vendor_txn are per-row structural features computed
independently within whatever split they're asked of (each split is its own
timeline slice, same as the Sparkov velocity features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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

    vendor_freq_rank = (
        genuine.groupby("vendor_id").size().rank(ascending=False, method="min").to_dict()
    )

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
        gap[0] = 999.0  # no prior transaction -- large sentinel, not 0 (0 would read as "same day as itself")
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

    df["vendor_frequency_rank"] = df["vendor_id"].map(
        lambda v: stats["vendor_freq_rank"].get(v, stats["max_freq_rank"] + 1)
    )
    df["benford_first_digit_deviation"] = df["vendor_id"].map(
        lambda v: stats["benford_by_vendor"].get(v, stats["global_benford_deviation"])
    )

    df["is_round_amount"] = (df["DMBTR"] % 100 == 0).astype(int)
    df["is_weekend"] = (pd.to_datetime(df["posting_date"]).dt.dayofweek >= 5).astype(int)
    df["posting_lag_days"] = (pd.to_datetime(df["posting_date"]) - pd.to_datetime(df["doc_date"])).dt.days

    # hour_of_day isn't in the source data (no timestamp, only date) -- deterministic
    # per-row synthetic draw, business-hours biased, seeded off row_id so it's stable
    # across re-runs (E5) without needing to persist a raw random value.
    rng_hours = np.random.default_rng(seed)
    df["hour_of_day"] = rng_hours.choice(
        np.arange(24), size=len(df), p=_business_hour_weights()
    )
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
    split (train+val+test) so a code that only happens to land in one split
    or one CV fold (e.g. a rare BUKRS/PRCTR value) never becomes an
    xgboost-unrecognized category at scoring time -- pandas' `.astype
    ("category")` otherwise infers categories independently from whatever
    subset it's given. These SAP codes are a fixed reference vocabulary
    (company codes, GL accounts, ...), not something inferred from labels,
    so enumerating them from the full corpus isn't a leakage concern the way
    fit_vendor_account_stats' amount statistics (train-genuine-only) are."""
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
