"""Feature contract for the transaction agent's model.

Numeric/engineered features come from data_pipeline.features. Categorical
columns are passed as pandas `category` dtype and handled natively by
xgboost (enable_categorical=True) rather than one-hot encoded, so a new
merchant/job value at inference time degrades gracefully instead of being
silently dropped by a fitted encoder's vocabulary.

Deliberately excluded: cc_num, name/address fields, trans_num, unix_time,
dob, trans_date_trans_time, raw lat/long/merch_lat/merch_long (superseded by
geo_distance_km), zip (not ordinal), and every typ_* column (those are
post-hoc typology tags for evaluation breakdown, not model inputs -- they're
threshold restatements of features already in this list, so including them
would be redundant rather than leaky, but they stay out of X on principle:
the model should not see its own evaluation slicing).
"""

from __future__ import annotations

import pandas as pd

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

LABEL_COLUMN = "is_fraud"

TYPOLOGY_COLUMNS = (
    "typ_online",
    "typ_pos",
    "typ_geo_anomaly",
    "typ_velocity",
    "typ_high_value",
    "typ_doc_mismatch",
)


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    y = df[LABEL_COLUMN].astype("int8")
    return X, y
