"""Renders + degrades invoice images for a subset of the linked corpus.

    python scripts/generate_invoices.py

Not every one of the 5,000 linked rows gets an image -- at ~0.9s/doc
(reportlab render + pdf2image rasterize + PIL degrade) the full corpus is
~75 minutes, more than this session's time budget allows. Every injected
fraud row gets one (that's what the document/reconciliation agents need to
show a signal on) plus a matching genuine sample, ~16% of each split.

Writes data/linked/images/{split}/{row_id}.jpg and
data/linked/images/manifest.parquet (row_id -> split, image path, template,
is_fraud, typology) so downstream steps don't need to re-derive the subset.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LINKED_DIR, RANDOM_SEED, seed_everything
from linked_pipeline.invoices import render_and_degrade, resolve_vendor_name

IMAGE_DIR = LINKED_DIR / "images"
IMAGE_FRACTION = 0.16


def select_subset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_target = int(np.ceil(len(df) * IMAGE_FRACTION))
    fraud = df[df["is_fraud"] == 1]
    n_genuine = max(n_target - len(fraud), 0)
    genuine_pool = df.index[df["is_fraud"] == 0].to_numpy().copy()
    rng.shuffle(genuine_pool)
    genuine = df.loc[genuine_pool[:n_genuine]]
    return pd.concat([fraud, genuine]).sort_index()


def main() -> int:
    seed_everything()
    vendor_master = pd.read_json(LINKED_DIR / "vendor_master.json")
    vendor_name_by_id = dict(zip(vendor_master["vendor_id"], vendor_master["vendor_name"]))

    manifest_rows = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(LINKED_DIR / f"{split}.parquet")
        subset = select_subset(df, seed=RANDOM_SEED)
        out_dir = IMAGE_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        for row_id, row in subset.iterrows():
            d = row.to_dict()
            d["vendor_name"] = resolve_vendor_name(d["vendor_id"], vendor_name_by_id)
            img = render_and_degrade(d, seed=int(row_id) + hash(split) % 1000)
            path = out_dir / f"{row_id}.jpg"
            img.save(path, format="JPEG", quality=85)
            manifest_rows.append(
                {
                    "row_id": int(row_id),
                    "split": split,
                    "image_path": str(path.relative_to(LINKED_DIR)),
                    "vendor_name": d["vendor_name"],
                    "invoice_number": d["invoice_number"],
                    "invoice_amount": d["invoice_amount"],
                    "ledger_amount": d["DMBTR"],
                    "is_fraud": int(d["is_fraud"]),
                    "typology": d["typology"],
                }
            )
        print(f"{split}: {len(subset)} images written ({time.time() - t0:.1f}s)")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(IMAGE_DIR / "manifest.parquet", index=False)
    print(f"wrote {IMAGE_DIR / 'manifest.parquet'} ({len(manifest)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
