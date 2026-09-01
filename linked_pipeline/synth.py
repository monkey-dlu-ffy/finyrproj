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
KTOSL_TIER = {}  # filled lazily by _tier_for_ktosl


def _tier_for_ktosl(ktosl: str) -> str:
    """Deterministic KTOSL -> value tier so a given GL txn type consistently
    reads as low/medium/high, rather than random noise per row."""
    if ktosl not in KTOSL_TIER:
        tiers = ["low", "medium", "high"]
        KTOSL_TIER[ktosl] = tiers[hash(ktosl) % 3]
    return KTOSL_TIER[ktosl]


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
    df = df.copy()
    rng = np.random.default_rng(seed)
    tiers = df["KTOSL"].map(_tier_for_ktosl)
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
