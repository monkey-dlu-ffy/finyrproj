"""Phase 1 (real spec): build the linked ledger+document corpus from the
Schreyer SAP-ERP dataset (dummy/erp/fraud_dataset_v2.csv).

    python scripts/build_linked_corpus.py

The raw table has no vendor/date/invoice#/PO#/description -- those are
synthesized (linked_pipeline/vendors.py, synth.py) before the 5 typologies
are injected (linked_pipeline/typologies.py) and the frame is split
temporally with vendor disjointness enforced (linked_pipeline/split.py).

N_ROWS=None uses the full raw table (533,009 rows) -- the original 5,000-row
subsample was a time-budget choice for the first build session; scaling up
per an explicit user decision to use the whole ledger (see the 2-day
scale-up plan). Image generation still caps at a fixed count regardless of
N_ROWS (scripts/generate_invoices.py), since disk, not row count, is the
binding constraint there.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LINKED_DIR, RANDOM_SEED, ensure_dirs, seed_everything
from linked_pipeline.split import temporal_vendor_split
from linked_pipeline.synth import add_dates, add_descriptions, add_invoice_amount, add_invoice_ids
from linked_pipeline.typologies import inject_all_typologies
from linked_pipeline.vendors import assign_vendors, build_vendor_master

RAW_ERP_CSV = Path(__file__).resolve().parent.parent / "dummy" / "erp" / "fraud_dataset_v2.csv"
N_ROWS = None  # None = use the full raw table


def main() -> int:
    seed_everything()
    ensure_dirs()
    LINKED_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    raw = pd.read_csv(RAW_ERP_CSV)
    if N_ROWS is not None:
        df = raw.sample(n=N_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df = raw.reset_index(drop=True)
    print(f"loaded {len(df):,}/{len(raw):,} rows ({time.time() - t0:.1f}s)")

    vendor_master = build_vendor_master(df["HKONT"], seed=RANDOM_SEED)
    df["vendor_id"] = assign_vendors(df, vendor_master, seed=RANDOM_SEED)
    df = add_dates(df, seed=RANDOM_SEED)
    df = add_invoice_ids(df, seed=RANDOM_SEED)
    df = add_descriptions(df, seed=RANDOM_SEED)
    df = add_invoice_amount(df)
    print(f"synthesized vendor/date/invoice/description ({time.time() - t0:.1f}s)")

    df = inject_all_typologies(df, seed=RANDOM_SEED)
    print(f"injected typologies: fraud_rate={df['is_fraud'].mean():.4%}")
    print(df.loc[df["is_fraud"] == 1, "typology"].value_counts())

    train, val, test = temporal_vendor_split(df)
    print(f"split train={len(train):,} val={len(val):,} test={len(test):,} (vendor-disjoint, temporal)")

    for name, part in (("train", train), ("val", val), ("test", test)):
        out = LINKED_DIR / f"{name}.parquet"
        part.reset_index(drop=True).to_parquet(out, index=False)
        print(f"wrote {out} ({len(part):,} rows, fraud_rate={part['is_fraud'].mean():.4%})")

    vendor_master.to_json(LINKED_DIR / "vendor_master.json", orient="records", indent=2)

    stats = {
        "n_rows": len(df),
        "fraud_rate": float(df["is_fraud"].mean()),
        "typology_counts": df.loc[df["is_fraud"] == 1, "typology"].value_counts().to_dict(),
        "date_range": [str(df["posting_date"].min()), str(df["posting_date"].max())],
        "n_vendors": int(df["vendor_id"].nunique()),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
    }
    (LINKED_DIR / "corpus_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"wrote {LINKED_DIR / 'corpus_stats.json'}")
    print(f"total ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
