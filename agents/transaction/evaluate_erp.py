"""Same ranking/threshold logic as evaluate.py, reused as-is; only the
per-typology breakdown differs because the ERP corpus stores one
`typology` string column (mutually exclusive by construction, see
linked_pipeline/typologies.py) instead of six typ_* boolean columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.transaction.evaluate import rank_metrics, recall_at_threshold, threshold_at_precision  # noqa: F401

TYPOLOGIES = ("duplicate_invoice", "ghost_vendor", "threshold_split", "desc_amount_mismatch", "altered_amount")


def recall_by_typology(df: pd.DataFrame, y_score: np.ndarray, threshold: float) -> dict:
    pred = (y_score >= threshold).astype(int)
    out = {}
    for typ in TYPOLOGIES:
        mask = (df["typology"] == typ).to_numpy()
        n = int(mask.sum())
        out[typ] = {"n": n, "recall": float(pred[mask].mean()) if n else float("nan")}
    return out
