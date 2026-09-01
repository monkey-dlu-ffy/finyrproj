"""Transaction agent (real spec): xgboost + Isolation Forest + isotonic
calibration, wrapped behind AgentEnvelope.

Isolation Forest is trained separately, on genuine-only rows, and its score
is fed to xgboost as an extra feature rather than fused after the fact --
it's the model's only unsupervised signal, useful for typologies (like
threshold_split) whose individual rows aren't extreme on any one supervised
feature but are jointly unusual.

Calibration: IsotonicRegression fit directly on (val raw score, val label)
rather than sklearn's CalibratedClassifierCV(cv='prefit') -- that API has
shifted across recent sklearn versions (prefit is being replaced by
FrozenEstimator) and a hand-rolled fit on an already-trained model's val
scores is the same computation CalibratedClassifierCV would do here anyway.
Uncalibrated xgboost probabilities are not usable as a risk score on a
dashboard -- they're a rank, not a probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression

from agents.envelope import AgentEnvelope
from agents.transaction.features_erp import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from config import RANDOM_SEED

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
    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.iforest: IsolationForest | None = None
        self.xgb_model: xgb.XGBClassifier | None = None
        self.calibrator: IsotonicRegression | None = None

    def _iforest_scores(self, X_numeric: pd.DataFrame) -> np.ndarray:
        # higher = more anomalous (sklearn's decision_function is inverted)
        return -self.iforest.decision_function(X_numeric)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> "TransactionAgent":
        self.iforest = IsolationForest(
            n_estimators=200, contamination="auto", random_state=self.seed
        )
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
            X[col] = X[col].astype("category")
        X["iforest_score"] = self._iforest_scores(X[NUMERIC_FEATURES])
        return X

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self._feature_frame(df)
        raw = self.xgb_model.predict_proba(X)[:, 1]
        return self.calibrator.predict(raw)

    def analyze(self, row: pd.Series, top_k_evidence: list[dict] | None = None) -> AgentEnvelope:
        df = row.to_frame().T
        score = float(self.score(df)[0])
        evidence = top_k_evidence or [
            {"type": "feature", "detail": f, "value": row[f]} for f in NUMERIC_FEATURES[:5]
        ]
        return AgentEnvelope(agent="transaction", row_id=int(row.name), score=score, evidence=evidence)
