"""Feature engineering for the Sparkov transaction table.

Every stat-fitting function (`fit_category_amount_stats`) must be called on
fraudTrain only; the returned stats are then applied to val/test via the
corresponding `add_*` function so no test-set distribution leaks into a
feature definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["trans_date_trans_time"]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    df["age_years"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    return df


def add_geo_distance(df: pd.DataFrame) -> pd.DataFrame:
    df["geo_distance_km"] = haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    return df


def fit_category_amount_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-category amt mean/std, fit on genuine (is_fraud==0) rows only."""
    genuine = df.loc[df["is_fraud"] == 0]
    return genuine.groupby("category", observed=True)["amt"].agg(["mean", "std"]).to_dict("index")


def add_amount_zscore(df: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    cat = df["category"].astype(str)
    means = cat.map(lambda c: stats.get(c, {}).get("mean", np.nan))
    stds = cat.map(lambda c: stats.get(c, {}).get("std", np.nan)).replace(0, np.nan)
    df["amt_zscore_category"] = (df["amt"] - means) / stds
    return df


def _rolling_window_stats(
    times: np.ndarray, amts: np.ndarray, window_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group trailing window count/sum, inclusive of the current row:
    for row i, includes every j<=i with times[j] > times[i] - window_ns.
    `times` must be sorted ascending within the group. Implemented as a
    two-pointer scan over a searchsorted cutoff rather than pandas'
    groupby().rolling(time_offset), which was verified (see PROGRESS.md
    Phase 1) to silently mix windows across group boundaries when the full
    frame isn't globally time-sorted -- only sorted within each group."""
    cutoffs = times - window_ns
    left = np.searchsorted(times, cutoffs, side="right")
    cumsum = np.concatenate(([0.0], np.cumsum(amts)))
    idx = np.arange(len(times))
    count = (idx - left + 1).astype(float)
    total = cumsum[idx + 1] - cumsum[left]
    return count, total


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing 1h/24h txn count/sum per card, inclusive of the current row.
    Requires (and produces) the frame sorted by (cc_num, time) -- callers must
    not assume row order survives."""
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    times_ns = df["trans_date_trans_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    amts = df["amt"].to_numpy(dtype="float64")

    count_1h = np.empty(len(df), dtype="float64")
    sum_24h = np.empty(len(df), dtype="float64")
    hour_ns = 3_600_000_000_000
    day_ns = 24 * hour_ns

    starts = df.groupby("cc_num", observed=True, sort=False).indices
    for _, positions in starts.items():
        positions = np.sort(positions)  # already ordered, but be explicit
        g_times = times_ns[positions]
        g_amts = amts[positions]
        c1h, _ = _rolling_window_stats(g_times, g_amts, hour_ns)
        _, s24h = _rolling_window_stats(g_times, g_amts, day_ns)
        count_1h[positions] = c1h
        sum_24h[positions] = s24h

    df["velocity_count_1h"] = count_1h
    df["velocity_sum_24h"] = sum_24h
    return df


def add_all_features(df: pd.DataFrame, category_amount_stats: dict) -> pd.DataFrame:
    df = add_velocity_features(df)  # sorts by (cc_num, time); do first
    df = add_time_features(df)
    df = add_age(df)
    df = add_geo_distance(df)
    df = add_amount_zscore(df, category_amount_stats)
    return df
