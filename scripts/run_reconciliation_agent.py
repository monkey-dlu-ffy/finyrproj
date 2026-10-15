"""Sanity-checks + runs the reconciliation agent over the document-linked
subset (data/linked/images/manifest.parquet). Uses the manifest's ground-
truth vendor_name/invoice_amount as `extracted_*` fields for now (fast path,
no dependency on OCR); scripts/apply_ocr_extraction.py swaps in the real
noisy OCR-extracted fields once Kaggle returns them, same rule logic either
way.

Per spec: "desc_amount_mismatch and altered_amount cases should light this
agent up. If they don't, your injection script is broken." -- so this prints
per-typology fire rates as the actual check, not just an aggregate score.

    python scripts/run_reconciliation_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.reconciliation.rules import APPROVAL_LIMIT_QUANTILE, ReconciliationAgent
from config import EXPERIMENTS_DIR, LINKED_DIR
from linked_pipeline.invoices import resolve_vendor_name


def load_manifest_with_ledger_fields() -> pd.DataFrame:
    manifest = pd.read_parquet(LINKED_DIR / "images" / "manifest.parquet")
    vendor_master = pd.read_json(LINKED_DIR / "vendor_master.json")
    vendor_name_by_id = dict(zip(vendor_master["vendor_id"], vendor_master["vendor_name"]))

    parts = []
    for split in manifest["split"].unique():
        ledger = pd.read_parquet(LINKED_DIR / f"{split}.parquet")
        sub = manifest.loc[manifest["split"] == split].set_index("row_id")
        joined = ledger.loc[sub.index].copy()
        # ledger-side resolved name, distinct from extracted_vendor_name -- comparing
        # vendor_id ("V0078") to a resolved name would make vendor_fuzzy_mismatch fire
        # on nearly every row regardless of fraud status.
        joined["vendor_name"] = joined["vendor_id"].map(lambda v: resolve_vendor_name(v, vendor_name_by_id))
        joined["extracted_amount"] = sub["invoice_amount"]
        joined["extracted_vendor_name"] = sub["vendor_name"]
        joined["split"] = split
        parts.append(joined)
    return pd.concat(parts)


def main() -> int:
    df = load_manifest_with_ledger_fields()

    train = pd.read_parquet(LINKED_DIR / "train.parquet")
    genuine = train.loc[train["is_fraud"] == 0]
    high_amount_threshold = float(genuine["DMBTR"].quantile(0.9))

    vendor_master = pd.read_json(LINKED_DIR / "vendor_master.json")
    agent = ReconciliationAgent(
        vendor_master_ids=set(vendor_master["vendor_id"]),
        high_amount_threshold=high_amount_threshold,
        approval_limit=float(train["DMBTR"].quantile(APPROVAL_LIMIT_QUANTILE)),
    )
    full_ledger = pd.concat(
        [
            pd.read_parquet(LINKED_DIR / f"{s}.parquet").reset_index(names="row_id").assign(split=s)
            for s in ("train", "val", "test")
        ]
    )
    envelopes = agent.analyze_batch(df, ledger=full_ledger)

    df = df.copy()
    df["recon_score"] = [e.score for e in envelopes]
    df["recon_evidence"] = [e.evidence for e in envelopes]

    print("--- genuine rows: mean score (should be low) ---")
    print(df.loc[df["is_fraud"] == 0, "recon_score"].mean())

    print("--- fire rate by typology (spec's sanity check) ---")
    for typ in ("duplicate_invoice", "ghost_vendor", "threshold_split", "desc_amount_mismatch", "altered_amount"):
        sub = df.loc[df["typology"] == typ]
        if len(sub) == 0:
            print(f"  {typ:<22} n=0 (none in the document-linked subset)")
            continue
        fired_frac = (sub["recon_score"] > 0).mean()
        print(f"  {typ:<22} n={len(sub):<3} fired={fired_frac:.0%} mean_score={sub['recon_score'].mean():.3f}")

    out = df[["is_fraud", "typology", "recon_score", "recon_evidence", "split"]].reset_index()
    out.to_json(EXPERIMENTS_DIR / "reconciliation_results.json", orient="records", indent=2, default_handler=str)
    print(f"wrote {EXPERIMENTS_DIR / 'reconciliation_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
