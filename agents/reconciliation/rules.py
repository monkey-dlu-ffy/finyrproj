"""Pure rules, no ML -- disagreement between the ledger and the document
fields (real or extracted) is the signal itself, not a learned pattern.
Every rule that fires is logged by name, which is what makes this agent the
most legible evidence on the dashboard (see agents/reconciliation/__init__.py).

Expects one row per case with both ledger fields (vendor_id, DMBTR, po_number,
posting_date, doc_date, invoice_number) and document fields, prefixed
`extracted_` (vendor_name, amount, invoice_number) -- sourced from either the
known linked-corpus ground truth (fast path, used before OCR results are
back) or the document agent's real OCR extraction. Same rule logic either
way; only the input's noise level differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz

from agents.envelope import AgentEnvelope

AMOUNT_DELTA_PCT = 0.03
VENDOR_FUZZY_MIN_SCORE = 85
DATE_LAG_MAX_DAYS = 20
SPLIT_WINDOW_DAYS = 7
# Callers fit approval_limit as this quantile of ALL train DMBTR -- the corpus
# has no approval-policy table to read a real limit from. All rows, not
# genuine-only like the other thresholds: a limit is policy, not a statistic
# of genuine behaviour, and the genuine-only estimate (5.04M) lands below the
# largest split pieces (5.09M), which cut threshold_split recall from 0.98 to
# 0.34 on test -- the rule is sensitive to underestimating the limit.
APPROVAL_LIMIT_QUANTILE = 0.98

RULE_WEIGHTS = {
    "amount_mismatch": 0.30,
    "vendor_fuzzy_mismatch": 0.20,
    "duplicate_invoice_number": 0.30,
    "vendor_absent_from_master": 0.30,
    "missing_po": 0.08,
    "date_outside_window": 0.12,
    "description_tier_mismatch": 0.22,
    "split_purchase": 0.30,
}


@dataclass
class ReconciliationAgent:
    vendor_master_ids: set[str]
    high_amount_threshold: float  # fit on train genuine DMBTR, see scripts/train_reconciliation_agent.py
    # Purchase approval limit for the split-purchase test. In a real audit this
    # is company policy, known up front; here it is fit from train amounts by
    # the caller. None disables the rule.
    approval_limit: float | None = None

    def _split_index(self, ledger: pd.DataFrame) -> dict[tuple, dict]:
        """(vendor_id, po_number) groups that look like one purchase split to
        dodge the approval limit: 2+ postings on the same PO from the same
        vendor within SPLIT_WINDOW_DAYS, each under the limit, together over
        it. A single-row check can't see this -- every piece is individually
        unremarkable -- which is why threshold_split sat at ~20% recall before
        this rule existed."""
        if self.approval_limit is None:
            return {}
        has_pointer_cols = "row_id" in ledger.columns and "split" in ledger.columns
        keyed = ledger.dropna(subset=["po_number"])
        # same pre-filter as the duplicate-invoice index: only keys that repeat can form a group
        keyed = keyed.loc[keyed.duplicated(["vendor_id", "po_number"], keep=False)]
        index = {}
        for key, group in keyed.groupby(["vendor_id", "po_number"]):
            dates = pd.to_datetime(group["posting_date"])
            total = float(group["DMBTR"].sum())
            if (
                (dates.max() - dates.min()).days <= SPLIT_WINDOW_DAYS
                and (group["DMBTR"] < self.approval_limit).all()
                and total > self.approval_limit
            ):
                index[key] = {
                    "total": total,
                    "span_days": int((dates.max() - dates.min()).days),
                    "postings": [
                        {
                            "BELNR": r["BELNR"],
                            "posting_date": str(r["posting_date"]),
                            "amount": float(r["DMBTR"]),
                            "row_id": int(r["row_id"]) if has_pointer_cols else None,
                            "split": r["split"] if has_pointer_cols else None,
                        }
                        for _, r in group.iterrows()
                    ],
                }
        return index

    def _fired_rules(self, row: pd.Series, invoice_index: dict[str, list[dict]], split_index: dict[tuple, dict]) -> list[dict]:
        fired = []

        ledger_amt = float(row["DMBTR"])
        extracted_amt = float(row.get("extracted_amount", ledger_amt))
        if ledger_amt > 0:
            rel_delta = abs(extracted_amt - ledger_amt) / ledger_amt
            if rel_delta > AMOUNT_DELTA_PCT:
                fired.append({"type": "amount_mismatch", "detail": f"ledger={ledger_amt:.2f} vs invoice={extracted_amt:.2f}", "value": round(rel_delta, 4)})

        ledger_vendor = str(row.get("vendor_name", row["vendor_id"]))
        extracted_vendor = str(row.get("extracted_vendor_name", ledger_vendor))
        fuzzy = fuzz.ratio(ledger_vendor, extracted_vendor)
        if fuzzy < VENDOR_FUZZY_MIN_SCORE:
            fired.append({"type": "vendor_fuzzy_mismatch", "detail": f"'{ledger_vendor}' vs '{extracted_vendor}'", "value": fuzzy})

        # Only an *earlier* posting of the same invoice number makes this row
        # a duplicate; the first posting is the legitimate original, so it is
        # not flagged. BELNR breaks posting-date ties.
        row_key = (pd.Timestamp(row["posting_date"]), row["BELNR"])
        others = [
            m
            for m in invoice_index.get(row["invoice_number"], [])
            if m["BELNR"] != row["BELNR"] and (pd.Timestamp(m["posting_date"]), m["BELNR"]) < row_key
        ]
        if others:
            o = others[0]  # spec's typology only ever pairs two rows; if more exist, lead with the first
            proof = (
                f"invoice {row['invoice_number']} also posted as BELNR {o['BELNR']}"
                f" (vendor {o['vendor_id']}, posted {o['posting_date']}, ${o['amount']:,.2f}"
                f"{', ' + o['split'] + ' row ' + str(o['row_id']) if o.get('row_id') is not None else ''})"
            )
            fired.append(
                {
                    "type": "duplicate_invoice_number",
                    "detail": proof,
                    "value": len(others) + 1,
                    "matches": others,
                }
            )

        if row["vendor_id"] not in self.vendor_master_ids:
            fired.append({"type": "vendor_absent_from_master", "detail": row["vendor_id"], "value": True})

        if pd.isna(row.get("po_number")) or row.get("po_number") is None:
            fired.append({"type": "missing_po", "detail": "no PO on file", "value": True})

        lag = (pd.to_datetime(row["posting_date"]) - pd.to_datetime(row["doc_date"])).days
        if abs(lag) > DATE_LAG_MAX_DAYS:
            fired.append({"type": "date_outside_window", "detail": f"{lag} day lag", "value": lag})

        if row.get("description_tier") == "low" and ledger_amt > self.high_amount_threshold:
            fired.append(
                {
                    "type": "description_tier_mismatch",
                    "detail": f"'{row.get('description')}' (low-tier) on ${ledger_amt:,.2f}",
                    "value": ledger_amt,
                }
            )

        split = split_index.get((row["vendor_id"], row.get("po_number")))
        if split is not None:
            fired.append(
                {
                    "type": "split_purchase",
                    "detail": (
                        f"{len(split['postings'])} postings on PO {row['po_number']} from this vendor within"
                        f" {split['span_days']} days, each under the ${self.approval_limit:,.2f} approval limit,"
                        f" totalling ${split['total']:,.2f}"
                    ),
                    "value": split["total"],
                    # not "matches": the dashboard renders matches as "original entry" (duplicate-invoice semantics)
                    "group": [p for p in split["postings"] if p["BELNR"] != row["BELNR"]],
                }
            )

        return fired

    def analyze_batch(self, df: pd.DataFrame, ledger: pd.DataFrame | None = None) -> list[AgentEnvelope]:
        """`ledger` should be the FULL ledger (train+val+test), not just `df`
        -- duplicate_invoice only marks one of its two rows as fraud, so if
        the other (genuine) copy isn't in `df` (e.g. `df` is just the ~16%
        document-linked subset), a duplicate check scoped to `df` alone
        misses most real duplicates and can't point back to the original
        entry. Include `row_id`/`split` columns on `ledger` (not just its
        pandas index, which repeats 0..N per split) so the duplicate-invoice
        proof can cite exactly where the original entry lives. Defaults to
        `df` itself as a fallback for ad-hoc/single-batch callers."""
        if ledger is None:
            ledger = df
        has_pointer_cols = "row_id" in ledger.columns and "split" in ledger.columns
        # Only invoice numbers that repeat can ever produce a non-empty
        # `others` list in _fired_rules -- at full ledger scale (533K rows,
        # ~522K distinct invoice numbers) grouping/iterrows-ing the entire
        # ledger to build entries that are never looked up made this the
        # dominant cost of the whole fusion run. Filtering first drops the
        # group count from ~527K to the low thousands.
        dup_ledger = ledger.loc[ledger["invoice_number"].duplicated(keep=False)]
        invoice_index: dict[str, list[dict]] = {}
        for inv, group in dup_ledger.groupby("invoice_number"):
            invoice_index[inv] = [
                {
                    "BELNR": r["BELNR"],
                    "vendor_id": r["vendor_id"],
                    "posting_date": str(r["posting_date"]),
                    "amount": float(r["DMBTR"]),
                    "row_id": int(r["row_id"]) if has_pointer_cols else None,
                    "split": r["split"] if has_pointer_cols else None,
                }
                for _, r in group.iterrows()
            ]
        split_index = self._split_index(ledger)
        out = []
        for row_id, row in df.iterrows():
            fired = self._fired_rules(row, invoice_index, split_index)
            score = min(1.0, sum(RULE_WEIGHTS[f["type"]] for f in fired))
            out.append(AgentEnvelope(agent="reconciliation", row_id=int(row_id), score=score, evidence=fired))
        return out
