"""Checks the live-scoring path (agents/pipeline.py) two ways.

    py -3.13 scripts/smoke_live_scoring.py [--parity-rows 500]

1. Parity: test-split postings re-scored one at a time through LivePipeline
   should reproduce their batch scores in experiments/fusion_scores.parquet.
   That is the evidence the live path is the same fitted model, not a
   re-implementation of it. The posting itself is dropped from the history
   it is checked against, and given the hour_of_day the batch run drew for
   it (the corpus has no posting time; see agents/pipeline.py).

2. Out-of-corpus scenarios: hand-built postings that were never in the
   corpus, each asserting the rule it should trip. Scored in sequence with
   each one remembered, as POST /score does, so the duplicate and split
   scenarios see the postings before them. In memory only -- nothing is
   written to the API's submissions table.

Exits non-zero if any scenario fails or any parity row's risk tier differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.pipeline import LivePipeline
from agents.transaction.features_erp import add_features
from config import EXPERIMENTS_DIR, LINKED_DIR, RANDOM_SEED

LEDGER_KEYS = [
    "BELNR", "vendor_id", "DMBTR", "posting_date", "doc_date", "invoice_number", "po_number",
    "description", "WAERS", "BUKRS", "KTOSL", "PRCTR", "BSCHL", "HKONT",
]
TOL = 1e-6


def _ledger(row: pd.Series, **overrides) -> dict:
    out = {k: (None if not isinstance(row[k], str) and pd.isna(row[k]) else row[k]) for k in LEDGER_KEYS}
    out["BELNR"] = int(out["BELNR"])
    out["DMBTR"] = float(out["DMBTR"])
    out.update(overrides)
    return out


def parity(pipe: LivePipeline, n_rows: int) -> int:
    test = pd.read_parquet(LINKED_DIR / "test.parquet")
    batch = pd.read_parquet(EXPERIMENTS_DIR / "fusion_scores.parquet")
    batch = batch.loc[batch["split"] == "test"].set_index("row_id")
    manifest = pd.read_parquet(LINKED_DIR / "images" / "manifest.parquet")
    manifest = manifest.loc[manifest["split"] == "test"].set_index("row_id")
    hours = add_features(test, pipe.vendor_stats, seed=RANDOM_SEED)["hour_of_day"]

    ids = batch.sample(n=min(n_rows, len(batch)), random_state=RANDOM_SEED).index
    diffs = {k: [] for k in ("transaction_score", "document_score", "reconciliation_score", "fused_score")}
    tiers_agree = 0
    for row_id in ids:
        row = test.loc[row_id]
        ledger = _ledger(row, row_id=int(row_id), split="test", hour_of_day=int(hours.loc[row_id]))
        invoice = {
            "vendor_name": manifest.at[row_id, "vendor_name"],
            "amount": float(manifest.at[row_id, "invoice_amount"]),
            "invoice_number": manifest.at[row_id, "invoice_number"],
            "po_number": ledger["po_number"],
        }
        live = pipe.score(ledger, invoice, exclude_belnr=ledger["BELNR"])
        for k in diffs:
            diffs[k].append(abs(live[k] - float(batch.at[row_id, k])))
        tiers_agree += live["risk_tier"] == batch.at[row_id, "risk_tier"]

    print(f"Parity over {len(ids)} test postings re-scored live:")
    for k, d in diffs.items():
        d = np.array(d)
        print(f"  {k:22s} exact (<{TOL:g}): {(d < TOL).mean():6.1%}   max |diff| {d.max():.2e}")
    print(f"  risk tier agrees: {tiers_agree}/{len(ids)}")
    return len(ids) - tiers_agree


def scenarios(pipe: LivePipeline) -> int:
    train = pd.read_parquet(LINKED_DIR / "train.parquet")
    fused = pd.read_parquet(EXPERIMENTS_DIR / "fusion_scores.parquet")
    # Template: a genuine train posting no rule fired on and that the batch
    # run scored low, from the vendor with the most genuine postings.
    clean_ids = fused.loc[(fused["split"] == "train") & (fused["risk_tier"] == "low") & (fused["reconciliation_score"] == 0), "row_id"]
    pool = train.loc[train.index.isin(clean_ids) & (train["is_fraud"] == 0) & train["po_number"].notna()]
    vendor = pool["vendor_id"].value_counts().index[0]
    template = pool.loc[pool["vendor_id"] == vendor].iloc[0]
    limit = pipe.recon_agent.approval_limit
    high_amount = pipe.recon_agent.high_amount_threshold

    n = [0]

    def submit(ledger: dict, invoice: dict | None = None) -> dict:
        n[0] += 1
        ledger = {**ledger, "BELNR": pipe.next_belnr(), "row_id": -n[0], "split": "live"}
        invoice = invoice or {
            "vendor_name": pipe.vendor_name_by_id.get(ledger["vendor_id"], ledger["vendor_id"]),
            "amount": ledger["DMBTR"],
            "invoice_number": ledger["invoice_number"],
            "po_number": ledger["po_number"],
        }
        result = pipe.score(ledger, invoice)
        pipe.remember(ledger)
        result["BELNR"] = ledger["BELNR"]
        return result

    def fired(result: dict) -> set[str]:
        return {e["type"] for e in result["recon_evidence"]}

    base = _ledger(template, invoice_number="INV-LIVE-000001", po_number="PO-LIVE-0001")
    failures = 0

    def check(name: str, result: dict, ok: bool, expect: str) -> None:
        nonlocal failures
        failures += not ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name:34s} fused {result['fused_score']:.3f} ({result['risk_tier']:4s})"
            f"  rules: {sorted(fired(result)) or '-'}   expected: {expect}"
        )

    print(f"Out-of-corpus scenarios (template: vendor {vendor}, ${base['DMBTR']:,.2f}, '{base['description']}'):")

    r1 = submit(base)
    check("1 clean posting", r1, not fired(r1) and r1["risk_tier"] == "low", "no rules, low tier")

    r2 = submit(
        {**base, "invoice_number": "INV-LIVE-000002"},
        {
            "vendor_name": pipe.vendor_name_by_id[vendor],
            "amount": round(base["DMBTR"] * 1.10, 2),
            "invoice_number": "INV-LIVE-000002",
            "po_number": base["po_number"],
        },
    )
    check("2 invoice 10% above ledger", r2, "amount_mismatch" in fired(r2), "amount_mismatch")

    r3 = submit({**base, "vendor_id": "V9999", "invoice_number": "INV-LIVE-000003", "po_number": "PO-LIVE-0003"})
    check("3 vendor not in master", r3, "vendor_absent_from_master" in fired(r3), "vendor_absent_from_master")

    later = (pd.Timestamp(base["posting_date"]) + pd.Timedelta(days=3)).date().isoformat()
    r4 = submit({**base, "posting_date": later})
    dup = next((e for e in r4["recon_evidence"] if e["type"] == "duplicate_invoice_number"), None)
    check(
        "4 same invoice resubmitted",
        r4,
        dup is not None and any(m["BELNR"] == r1["BELNR"] for m in dup["matches"]),
        f"duplicate_invoice_number citing BELNR {r1['BELNR']}",
    )

    piece = round(limit * 0.45, 2)
    pieces = []
    for i in range(3):
        day = (pd.Timestamp(base["posting_date"]) + pd.Timedelta(days=i * 2)).date().isoformat()
        pieces.append(
            submit({**base, "DMBTR": piece, "posting_date": day, "invoice_number": f"INV-LIVE-1000{i}", "po_number": "PO-LIVE-SPLIT"})
        )
    check(
        "5 3 pieces under limit, 1 PO, 4 days",
        pieces[2],
        "split_purchase" in fired(pieces[2]) and "split_purchase" not in fired(pieces[1]),
        f"split_purchase on the 3rd only (each ${piece:,.2f}, limit ${limit:,.2f})",
    )

    low_desc = "Office Supplies"
    r6 = submit(
        {**base, "description": low_desc, "DMBTR": round(high_amount * 1.5, 2), "invoice_number": "INV-LIVE-000006", "po_number": "PO-LIVE-0006"}
    )
    check("6 low-tier text on a large amount", r6, "description_tier_mismatch" in fired(r6), "description_tier_mismatch")

    # Median: the first call pays one-time warm-up.
    latencies = [r["latency_s"] for r in (r1, r2, r3, r4, *pieces, r6)]
    print(f"  live latency: median {np.median(latencies) * 1000:.0f} ms/posting")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-rows", type=int, default=500)
    args = parser.parse_args()

    pipe = LivePipeline.load()
    failures = 0
    if args.parity_rows:
        failures += parity(pipe, args.parity_rows)
        print()
    failures += scenarios(pipe)
    print(f"\n{'all checks passed' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
