"""Injects the 5 fraud typologies onto an already-synthesized linked frame
(dates, vendor_id, invoice_number, po_number, description, invoice_amount
must already exist). Injection is mutually exclusive per row -- each fraud
row gets exactly one typology -- so E7 per-typology recall isn't muddied by
overlap. Target ~2% combined, roughly balanced across the 5.

Run in this order because threshold_split and duplicate_invoice consume
*sets* of candidate rows (need untouched neighbors), while desc_amount_mismatch
and altered_amount only mutate one row at a time -- doing the multi-row ones
first avoids picking a row two typologies both want.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_FRAUD_RATE = 0.02
GHOST_VENDOR_NAMES = [
    "Quikpay Trading LLC", "Zenith Overseas Supply", "Rapidline Ventures",
    "Omni Trade Partners", "Blackstone Freight Co", "Starline Import Export",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def inject_threshold_split(df: pd.DataFrame, n_instances: int, threshold: float, seed: int) -> pd.DataFrame:
    """Pick a genuine high-value row, replace it and 2-3 neighbors (same
    vendor, same week) with amounts just under `threshold` that sum above it."""
    df = df.copy()
    rng = _rng(seed)
    candidates = df.index[(df["is_fraud"] == 0) & (df["DMBTR"] > threshold)].to_numpy().copy()
    rng.shuffle(candidates)

    used = set()
    picked = 0
    for anchor in candidates:
        if picked >= n_instances:
            break
        vendor = df.at[anchor, "vendor_id"]
        week_start = df.at[anchor, "posting_date"] - pd.Timedelta(days=3)
        week_end = df.at[anchor, "posting_date"] + pd.Timedelta(days=3)
        same_vendor = df.index[
            (df["vendor_id"] == vendor)
            & (df["posting_date"] >= week_start)
            & (df["posting_date"] <= week_end)
            & (df["is_fraud"] == 0)
        ]
        same_vendor = [i for i in same_vendor if i not in used]
        n_split = rng.integers(3, 5)
        if len(same_vendor) < n_split:
            continue

        group = rng.choice(same_vendor, size=n_split, replace=False)
        total = df.at[anchor, "DMBTR"] * rng.uniform(1.1, 1.4)
        splits = np.diff(np.sort(np.concatenate([[0], rng.uniform(0, 1, n_split - 1), [1]]))) * total
        splits = np.minimum(splits, threshold * 0.97)  # each piece stays just under the limit

        common_po = f"PO-{rng.integers(100_000, 999_999)}"
        for idx, amt in zip(group, splits):
            df.at[idx, "DMBTR"] = round(float(amt), 2)
            df.at[idx, "invoice_amount"] = round(float(amt), 2)
            df.at[idx, "po_number"] = common_po
            df.at[idx, "is_fraud"] = 1
            df.at[idx, "typology"] = "threshold_split"
            used.add(idx)
        picked += 1
    return df


def inject_duplicate_invoice(df: pd.DataFrame, n_instances: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    rng = _rng(seed)
    candidates = df.index[df["is_fraud"] == 0].to_numpy().copy()
    rng.shuffle(candidates)
    picked = 0
    i = 0
    while picked < n_instances and i + 1 < len(candidates):
        a, b = candidates[i], candidates[i + 1]
        i += 2
        # The duplicate is the later posting, as in a real double payment --
        # the first posting is the legitimate one. Picking b at random made
        # the fraud copy the later one in only 50.3% of pairs, so no rule
        # could tell the copies apart. BELNR breaks posting-date ties.
        if (df.at[b, "posting_date"], df.at[b, "BELNR"]) < (df.at[a, "posting_date"], df.at[a, "BELNR"]):
            a, b = b, a
        df.at[b, "invoice_number"] = df.at[a, "invoice_number"]
        df.at[b, "vendor_id"] = df.at[a, "vendor_id"]
        df.at[b, "is_fraud"] = 1
        df.at[b, "typology"] = "duplicate_invoice"
        picked += 1
    return df


def inject_ghost_vendor(df: pd.DataFrame, n_instances: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    rng = _rng(seed)
    candidates = df.index[df["is_fraud"] == 0].to_numpy().copy()
    rows = rng.choice(candidates, size=min(n_instances, len(candidates)), replace=False)
    for idx in rows:
        ghost = GHOST_VENDOR_NAMES[rng.integers(len(GHOST_VENDOR_NAMES))]
        df.at[idx, "vendor_id"] = f"GHOST-{ghost}"
        df.at[idx, "is_fraud"] = 1
        df.at[idx, "typology"] = "ghost_vendor"
    return df


def inject_desc_amount_mismatch(df: pd.DataFrame, n_instances: int, seed: int) -> pd.DataFrame:
    from linked_pipeline.synth import _DESCRIPTIONS

    df = df.copy()
    rng = _rng(seed)
    high_amt = df["DMBTR"].quantile(0.9)
    candidates = df.index[(df["is_fraud"] == 0) & (df["DMBTR"] > high_amt)].to_numpy().copy()
    rows = rng.choice(candidates, size=min(n_instances, len(candidates)), replace=False)
    for idx in rows:
        df.at[idx, "description"] = _DESCRIPTIONS["low"][rng.integers(len(_DESCRIPTIONS["low"]))]
        df.at[idx, "description_tier"] = "low"
        df.at[idx, "is_fraud"] = 1
        df.at[idx, "typology"] = "desc_amount_mismatch"
    return df


def inject_altered_amount(df: pd.DataFrame, n_instances: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    rng = _rng(seed)
    candidates = df.index[df["is_fraud"] == 0].to_numpy().copy()
    rows = rng.choice(candidates, size=min(n_instances, len(candidates)), replace=False)
    for idx in rows:
        direction = rng.choice([-1, 1])
        pct = rng.uniform(0.15, 0.6) * direction
        df.at[idx, "invoice_amount"] = round(float(df.at[idx, "DMBTR"] * (1 + pct)), 2)
        df.at[idx, "is_fraud"] = 1
        df.at[idx, "typology"] = "altered_amount"
    return df


def inject_all_typologies(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["is_fraud"] = 0
    df["typology"] = "none"

    n_total = int(round(len(df) * TARGET_FRAUD_RATE))
    n_each = max(n_total // 5, 1)
    threshold = df["DMBTR"].quantile(0.98)  # "sub-limit" approval threshold

    df = inject_threshold_split(df, n_instances=max(n_each // 3, 1), threshold=threshold, seed=seed)
    df = inject_duplicate_invoice(df, n_instances=n_each, seed=seed + 1)
    df = inject_ghost_vendor(df, n_instances=n_each, seed=seed + 2)
    df = inject_desc_amount_mismatch(df, n_instances=n_each, seed=seed + 3)
    df = inject_altered_amount(df, n_instances=n_each, seed=seed + 4)
    return df
