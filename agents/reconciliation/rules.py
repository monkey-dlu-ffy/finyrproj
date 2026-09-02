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

RULE_WEIGHTS = {
    "amount_mismatch": 0.30,
    "vendor_fuzzy_mismatch": 0.20,
    "duplicate_invoice_number": 0.30,
    "vendor_absent_from_master": 0.30,
    "missing_po": 0.08,
    "date_outside_window": 0.12,
    "description_tier_mismatch": 0.22,
}


@dataclass
class ReconciliationAgent:
    vendor_master_ids: set[str]
    high_amount_threshold: float  # fit on train genuine DMBTR, see scripts/train_reconciliation_agent.py

    def _fired_rules(self, row: pd.Series, invoice_number_counts: pd.Series) -> list[dict]:
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

        if invoice_number_counts.get(row["invoice_number"], 1) > 1:
            fired.append({"type": "duplicate_invoice_number", "detail": row["invoice_number"], "value": int(invoice_number_counts[row["invoice_number"]])})

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

        return fired

    def analyze_batch(self, df: pd.DataFrame, invoice_number_counts: pd.Series | None = None) -> list[AgentEnvelope]:
        """invoice_number_counts should normally come from the full ledger,
        not just `df` -- duplicate_invoice only marks one of its two rows as
        fraud, so if the other (genuine) copy isn't in `df` (e.g. `df` is
        just the ~16% document-linked subset), a duplicate check scoped to
        `df` alone misses most real duplicates. Defaults to `df` itself only
        as a fallback for ad-hoc/single-batch callers."""
        if invoice_number_counts is None:
            invoice_number_counts = df["invoice_number"].value_counts()
        out = []
        for row_id, row in df.iterrows():
            fired = self._fired_rules(row, invoice_number_counts)
            score = min(1.0, sum(RULE_WEIGHTS[f["type"]] for f in fired))
            out.append(AgentEnvelope(agent="reconciliation", row_id=int(row_id), score=score, evidence=fired))
        return out
