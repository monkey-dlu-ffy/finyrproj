"""Gradient-boosted anomaly detector over ledger rows.

xgboost with native categorical support (enable_categorical=True) and
scale_pos_weight for the ~0.6% fraud prevalence -- PR-AUC is the early
stopping metric, not accuracy or plain AUC, because both are misleading
this far into class imbalance (a model that predicts "never fraud" already
scores 99.4% accuracy and a deceptively high ROC-AUC).
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from config import RANDOM_SEED

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": RANDOM_SEED,
    "n_estimators": 2000,
    "enable_categorical": True,
    "early_stopping_rounds": 50,
}


def _scale_pos_weight(y: pd.Series) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return n_neg / max(n_pos, 1)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = RANDOM_SEED,
) -> xgb.XGBClassifier:
    params = dict(XGB_PARAMS)
    params["random_state"] = seed
    params["scale_pos_weight"] = _scale_pos_weight(y_train)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model
