"""Synthesizes the fields the raw Schreyer table doesn't carry: posting date,
invoice/PO numbers, and a line-item description. All downstream (temporal
split, posting_lag_days, txn_count_7d_vendor, duplicate/threshold typologies,
desc_amount_mismatch) depends on these existing first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATE_START = pd.Timestamp("2025-01-01")
DATE_END = pd.Timestamp("2026-06-30")
WEEKEND_RATE = 0.08  # fraction of postings falling on Sat/Sun

# (description, value_tier) -- "low" tier text is what desc_amount_mismatch
# pairs with a high-tier amount.
_DESCRIPTIONS = {
    "low": [
        "Office Supplies", "Miscellaneous Expense", "Postage & Courier",
        "Stationery Order", "Small Tools", "Cleaning Supplies",
    ],
    "medium": [
        "IT Consulting Services", "Software License Renewal", "Travel & Entertainment",
        "Facility Maintenance", "Professional Services Retainer", "Training & Certification",
    ],
    "high": [
        "Raw Materials Purchase", "Equipment Lease", "Capital Equipment Purchase",
        "Construction Contract Milestone", "Bulk Inventory Procurement", "Plant Machinery",
    ],
}
# Fraction of rows whose tier is redrawn uniformly at random instead of
# following their amount band. This is the overlap knob: at 0.0 the tier is a
# deterministic, invertible function of the amount (any "anomaly" a document
# agent reports would just be that inverse, i.e. fitted to the generator); at
# 1.0 it degenerates to the KTOSL-hash version this replaced, where
# descriptions carried no amount information at all. 0.22 leaves genuine
# low-tier-description/high-amount rows in the corpus at a realistic rate, so
# desc_amount_mismatch is detectable but not trivially separable.
TIER_NOISE = 0.22
TIER_ORDER = ["low", "medium", "high"]


def _amount_ranked_tiers(amounts: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Assign each row a description value tier from its amount tertile, then
    redraw TIER_NOISE of them at random.

    Why this replaces the old KTOSL-hash assignment
    -----------------------------------------------
    `_tier_for_ktosl` bucketed each GL txn type into low/medium/high by a hash
    of its code -- stable across runs, but arbitrary with respect to value.
    Measured on the 311K-row train split that left every description sharing
    one amount distribution: log10(DMBTR) class means spanned 5.616 (Small
    Tools) to 5.644 (Travel & Entertainment), a 0.028 spread against a 0.565
    within-class std -- 0.05 sigma of separation, Spearman(tier, DMBTR)
    = -0.003.

    Two things followed. The document agent's one load-bearing signal,
    `amount_atypical_for_description`, was inert by construction (it scored
    ROC-AUC 0.5005 and the fusion meta-classifier gave it a -3.48
    coefficient, so dropping it *improved* the ensemble). And
    desc_amount_mismatch -- nominally the one typology visible from a single
    document -- was not actually an anomaly in the data: the reconciliation
    rule that "caught" it (tier == low and DMBTR > q90) is the injector's
    literal inverse, so its recall of 1.0 measured leakage, not detection.

    Ranking the tier by amount is also just what a real ledger looks like:
    "Office Supplies" for $3M is the anomaly precisely because office
    supplies are normally cheap. DMBTR itself is untouched -- it comes from
    the real Schreyer table and feeds the transaction agent, which never
    reads `description` (see agents/transaction/features_erp.py's
    FEATURE_COLUMNS) -- and no typology injector reads descriptions either,
    so the same rows are selected for every typology as before. Only the
    `description`/`description_tier` columns move.
    """
    # rank-based, not fixed cut points: robust to DMBTR's long right tail, and
    # gives three equally-populated tiers whatever the raw scale.
    pct = amounts.rank(method="first", pct=True).to_numpy()
    idx = np.clip((pct * 3).astype(int), 0, 2)
    redraw = rng.random(len(idx)) < TIER_NOISE
    idx[redraw] = rng.integers(0, 3, size=int(redraw.sum()))
    return np.array(TIER_ORDER, dtype=object)[idx]


def add_dates(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    total_days = (DATE_END - DATE_START).days
    offsets = rng.integers(0, total_days + 1, size=len(df))
    dates = DATE_START + pd.to_timedelta(offsets, unit="D")

    dow = dates.dayofweek.to_numpy()
    is_weekend = dow >= 5
    # nudge most weekend draws onto the nearest weekday, keep WEEKEND_RATE worth as-is
    keep_weekend = rng.random(len(df)) < WEEKEND_RATE
    nudge = is_weekend & ~keep_weekend
    dates = pd.DatetimeIndex(dates)
    if nudge.any():
        shift = np.where(dow[nudge] == 5, 2, 1)  # Sat -> +2 (Mon), Sun -> +1 (Mon)
        dates = dates.to_numpy().copy()
        dates[nudge] = dates[nudge] + shift.astype("timedelta64[D]")
        dates = pd.DatetimeIndex(dates)

    df = df.copy()
    df["posting_date"] = dates
    # document date precedes posting date by a small lag (posting_lag_days)
    lag_days = rng.integers(0, 15, size=len(df))
    df["doc_date"] = df["posting_date"] - pd.to_timedelta(lag_days, unit="D")
    return df


def add_invoice_ids(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    years = df["posting_date"].dt.strftime("%y%m")
    df["invoice_number"] = [f"INV-{y}-{i:06d}" for y, i in zip(years, range(len(df)))]

    rng = np.random.default_rng(seed)
    has_po = rng.random(len(df)) >= 0.15  # 15% of genuine rows have no PO on file
    po_ids = rng.integers(100_000, 999_999, size=len(df))
    df["po_number"] = [f"PO-{p}" if has else None for p, has in zip(po_ids, has_po)]
    return df


def add_descriptions(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Line-item description + its value tier, ranked by the row's amount.

    Draws from one rng in a fixed order (tiers first, then the within-tier
    text choice) so the column is reproducible for a given seed.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    tiers = _amount_ranked_tiers(df["DMBTR"], rng)
    df["description"] = [
        _DESCRIPTIONS[t][rng.integers(len(_DESCRIPTIONS[t]))] for t in tiers
    ]
    df["description_tier"] = tiers
    return df


def add_invoice_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Face amount printed on the invoice image. Equal to the ledger amount
    (DMBTR) unless altered_amount typology injection overwrites it later."""
    df = df.copy()
    df["invoice_amount"] = df["DMBTR"].round(2)
    return df
