"""Rule-based fraud typology tagging.

Sparkov ships a single `is_fraud` label with no typology metadata, so each
typology is a post-hoc rule over engineered features. A row can match more
than one typology (E7 recall is computed per bucket independently, not as a
partition). Percentile thresholds are fit on genuine (is_fraud==0) fraudTrain
rows only, so a typology means "anomalous relative to normal behaviour," and
never leaks test-set distribution.

`typ_doc_mismatch` (the 6th typology) is added in Phase 2 once synthetic
documents exist; this module seeds it to False so downstream code can rely on
the column always being present.
"""

from __future__ import annotations

import pandas as pd

VELOCITY_MIN_COUNT_1H = 3
GEO_ANOMALY_PCTILE = 0.95
HIGH_VALUE_PCTILE = 0.99


def fit_typology_thresholds(df: pd.DataFrame) -> dict:
    """Fit on a frame that already has engineered features (geo_distance_km,
    velocity_count_1h) and the raw is_fraud column -- i.e. fraudTrain only."""
    genuine = df.loc[df["is_fraud"] == 0]
    geo_p95 = float(genuine["geo_distance_km"].quantile(GEO_ANOMALY_PCTILE))
    amt_p99_by_category = {
        str(k): float(v)
        for k, v in genuine.groupby("category", observed=True)["amt"]
        .quantile(HIGH_VALUE_PCTILE)
        .to_dict()
        .items()
    }
    return {"geo_distance_km_p95": geo_p95, "amt_p99_by_category": amt_p99_by_category}


def add_typology_columns(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    cat = df["category"].astype(str)
    df["typ_online"] = cat.str.endswith("_net")
    df["typ_pos"] = cat.str.endswith("_pos")
    df["typ_geo_anomaly"] = df["geo_distance_km"] > thresholds["geo_distance_km_p95"]
    df["typ_velocity"] = df["velocity_count_1h"] >= VELOCITY_MIN_COUNT_1H
    amt_thresh = cat.map(thresholds["amt_p99_by_category"])
    df["typ_high_value"] = df["amt"] > amt_thresh
    df["typ_doc_mismatch"] = False  # populated in Phase 2
    return df
