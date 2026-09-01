"""Temporal, vendor-disjoint split. A pure calendar cut would put most
vendors' rows on both sides (they transact throughout), so vendors are
sorted by their median posting_date and assigned to train/val/test as whole
blocks -- train vendors skew earlier, test vendors skew later, and no vendor
crosses the boundary. GHOST-* vendor_ids from inject_ghost_vendor are true
one-offs (one row each) so they don't distort the median-date ordering.
"""

from __future__ import annotations

import pandas as pd


def temporal_vendor_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vendor_median_date = df.groupby("vendor_id")["posting_date"].median().sort_values()
    vendor_row_counts = df["vendor_id"].value_counts()

    ordered_vendors = vendor_median_date.index.to_list()
    cum = 0
    total = len(df)
    train_cut = total * train_frac
    val_cut = total * (train_frac + val_frac)

    train_vendors, val_vendors, test_vendors = [], [], []
    for v in ordered_vendors:
        cum += int(vendor_row_counts.get(v, 0))
        if cum <= train_cut:
            train_vendors.append(v)
        elif cum <= val_cut:
            val_vendors.append(v)
        else:
            test_vendors.append(v)

    train = df[df["vendor_id"].isin(train_vendors)].copy()
    val = df[df["vendor_id"].isin(val_vendors)].copy()
    test = df[df["vendor_id"].isin(test_vendors)].copy()

    assert set(train["vendor_id"]) & set(val["vendor_id"]) == set()
    assert set(train["vendor_id"]) & set(test["vendor_id"]) == set()
    assert set(val["vendor_id"]) & set(test["vendor_id"]) == set()

    return train, val, test
