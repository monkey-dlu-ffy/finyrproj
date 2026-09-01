"""Per-row evidence for flagged transactions.

The dashboard and xai layer must read evidence, not model internals (see
agents/__init__.py), so this returns a plain list of {feature, value,
shap_value} dicts per row -- ranked by |shap_value| -- rather than a raw
SHAP Explanation object tied to this model version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def make_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def explain_rows(explainer: shap.TreeExplainer, X: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
    """X must use the same column order/dtypes build_xy produced for training."""
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # some SHAP/xgboost combos return [class0, class1]
        shap_values = shap_values[-1]

    out = []
    for i in range(len(X)):
        row = X.iloc[i]
        contribs = shap_values[i]
        order = np.argsort(-np.abs(contribs))[:top_k]
        out.append(
            [
                {
                    "feature": X.columns[j],
                    "value": row.iloc[j].item() if hasattr(row.iloc[j], "item") else row.iloc[j],
                    "shap_value": float(contribs[j]),
                }
                for j in order
            ]
        )
    return out
