"""Synthetic vendor master. The raw Schreyer table has no vendor field --
every categorical column (WAERS/BUKRS/KTOSL/PRCTR/BSCHL/HKONT) is an internal
SAP code, not an external counterparty -- so vendor identity is built here,
not extracted.

Each vendor gets a "home" HKONT (GL account) it's correlated with, so
ghost_vendor and threshold_split have real structure to violate instead of
being invisible against fully-random noise.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

N_VENDORS = 300
HOME_ACCOUNT_HIT_RATE = 0.85  # fraction of a vendor's rows that land on its home HKONT

_PREFIXES = [
    "Northgate", "Silverline", "Meridian", "Cobalt", "Harborview", "Bluecrest",
    "Ridgeline", "Summit", "Ironwood", "Fieldstone", "Amberly", "Westbrook",
    "Cascade", "Redwood", "Sterling", "Grayson", "Lakeview", "Foxhollow",
    "Brightwater", "Cedar", "Pinnacle", "Vantage", "Clearwater", "Highland",
    "Anchor", "Beacon", "Crestview", "Driftwood", "Elmhurst", "Falcon",
]
_SUFFIXES = [
    "Supply Co.", "Logistics", "Industrial", "Consulting Group", "Materials",
    "Technologies", "Manufacturing", "Services", "Freight", "Solutions",
    "Contractors", "Trading Co.", "Systems", "Partners", "Equipment Ltd.",
]


def build_vendor_master(hkont_values: pd.Series, seed: int) -> pd.DataFrame:
    """One row per vendor: vendor_id, vendor_name, home_hkont."""
    rng = random.Random(seed)
    unique_accounts = sorted(hkont_values.unique().tolist())

    names = set()
    while len(names) < N_VENDORS:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_SUFFIXES)}"
        names.add(name)
    names = sorted(names)  # deterministic order before shuffling

    rng.shuffle(names)
    home_accounts = [rng.choice(unique_accounts) for _ in range(N_VENDORS)]

    return pd.DataFrame(
        {
            "vendor_id": [f"V{idx:04d}" for idx in range(N_VENDORS)],
            "vendor_name": names,
            "home_hkont": home_accounts,
        }
    )


def assign_vendors(df: pd.DataFrame, vendor_master: pd.DataFrame, seed: int) -> pd.Series:
    """One vendor_id per row of df, biased toward vendors whose home_hkont
    matches the row's HKONT."""
    rng = np.random.default_rng(seed)
    by_account = vendor_master.groupby("home_hkont")["vendor_id"].apply(list).to_dict()
    all_vendor_ids = vendor_master["vendor_id"].to_numpy()

    out = np.empty(len(df), dtype=object)
    accounts = df["HKONT"].to_numpy()
    hits = rng.random(len(df)) < HOME_ACCOUNT_HIT_RATE
    for i, (acct, hit) in enumerate(zip(accounts, hits)):
        pool = by_account.get(acct) if hit else None
        if not pool:
            pool = all_vendor_ids
        out[i] = pool[rng.integers(len(pool))]
    return pd.Series(out, index=df.index, name="vendor_id")
