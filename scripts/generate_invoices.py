"""Renders + degrades invoice images for a subset of the linked corpus.

    python scripts/generate_invoices.py

At full ledger scale (533,009 rows) one image per row would need ~110GB
(observed ~213KB/image) -- far beyond this machine's free disk. Instead the
image count is capped at N_IMAGES_TARGET regardless of ledger size, split
proportionally across train/val/test by row count. Every injected fraud row
still gets one (that's what the document/reconciliation agents need to show
a signal on); the rest of the budget is filled with a matched genuine
sample.

Rendering (reportlab render + pdf2image rasterize + PIL degrade) is
~0.9s/doc single-threaded, so it's parallelized across CPU_WORKERS processes
-- CPU-bound, not RAM-heavy, so this is safe on a RAM-constrained machine
unlike model training (see scripts/kaggle_train_transaction_erp.py).

A disk-safety check after the first DISK_SAFETY_CHECK_AFTER images
extrapolates the final total from observed file sizes and aborts before
filling the disk if it would exceed DISK_SAFETY_BUDGET_BYTES. On abort,
already-written image files are left in place (not deleted) but no
manifest is written, so downstream steps won't pick up a partial run.

Writes data/linked/images/{split}/{row_id}.jpg and
data/linked/images/manifest.parquet (row_id -> split, image path, vendor,
invoice fields, is_fraud, typology) so downstream steps don't need to
re-derive the subset.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LINKED_DIR, RANDOM_SEED, seed_everything
from linked_pipeline.invoices import render_and_degrade, resolve_vendor_name

IMAGE_DIR = LINKED_DIR / "images"
N_IMAGES_TARGET = 20_000
CPU_WORKERS = min(8, mp.cpu_count())
DISK_SAFETY_CHECK_AFTER = 500
DISK_SAFETY_BUDGET_BYTES = 7 * 1024**3  # leaves 2GB+ buffer on a 9.2GB-free disk


def select_subset(df: pd.DataFrame, seed: int, n_target: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fraud = df[df["is_fraud"] == 1]
    n_genuine = max(n_target - len(fraud), 0)
    genuine_pool = df.index[df["is_fraud"] == 0].to_numpy().copy()
    rng.shuffle(genuine_pool)
    genuine = df.loc[genuine_pool[:n_genuine]]
    return pd.concat([fraud, genuine]).sort_index()


def _render_job(job: tuple[dict, int, str]) -> int:
    """Runs in a worker process. Returns the written file's byte size so the
    parent can run the disk-safety projection."""
    row_dict, seed, path_str = job
    path = Path(path_str)
    img = render_and_degrade(row_dict, seed=seed)
    img.save(path, format="JPEG", quality=85)
    return path.stat().st_size


def main() -> int:
    seed_everything()
    vendor_master = pd.read_json(LINKED_DIR / "vendor_master.json")
    vendor_name_by_id = dict(zip(vendor_master["vendor_id"], vendor_master["vendor_name"]))

    splits_data = {s: pd.read_parquet(LINKED_DIR / f"{s}.parquet") for s in ("train", "val", "test")}
    total_rows = sum(len(d) for d in splits_data.values())

    manifest_rows = []
    jobs = []
    for split, df in splits_data.items():
        n_target = round(N_IMAGES_TARGET * len(df) / total_rows)
        subset = select_subset(df, seed=RANDOM_SEED, n_target=n_target)
        out_dir = IMAGE_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for row_id, row in subset.iterrows():
            d = row.to_dict()
            d["vendor_name"] = resolve_vendor_name(d["vendor_id"], vendor_name_by_id)
            path = out_dir / f"{row_id}.jpg"
            seed = int(row_id) + hash(split) % 1000
            jobs.append((d, seed, str(path)))
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
        print(f"{split}: {len(subset):,} images queued (target {n_target:,}/{len(df):,} rows)")

    print(f"rendering {len(jobs):,} images across {CPU_WORKERS} processes")
    t0 = time.time()
    sizes = []
    with mp.Pool(processes=CPU_WORKERS) as pool:
        for n_done, size in enumerate(pool.imap_unordered(_render_job, jobs, chunksize=16), start=1):
            sizes.append(size)
            if n_done == DISK_SAFETY_CHECK_AFTER:
                avg = sum(sizes) / len(sizes)
                projected = avg * len(jobs)
                print(f"disk-safety check: avg {avg / 1024:.1f}KB/image, projected total {projected / 1024**3:.2f}GB")
                if projected > DISK_SAFETY_BUDGET_BYTES:
                    pool.terminate()
                    raise SystemExit(
                        f"Aborting: projected {projected / 1024**3:.2f}GB exceeds the "
                        f"{DISK_SAFETY_BUDGET_BYTES / 1024**3:.0f}GB safety budget. "
                        f"{n_done:,}/{len(jobs):,} images already written under {IMAGE_DIR} "
                        "-- lower N_IMAGES_TARGET or the JPEG quality/DPI and re-run."
                    )
            if n_done % 2000 == 0:
                print(f"  {n_done:,}/{len(jobs):,} rendered ({time.time() - t0:.1f}s)")
    print(f"rendered {len(jobs):,} images in {time.time() - t0:.1f}s")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(IMAGE_DIR / "manifest.parquet", index=False)
    total_bytes = sum(sizes)
    print(f"wrote {IMAGE_DIR / 'manifest.parquet'} ({len(manifest):,} rows, {total_bytes / 1024**3:.2f}GB on disk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
