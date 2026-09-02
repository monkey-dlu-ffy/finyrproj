"""Document agent: emits extracted fields + a text-risk score as an
AgentEnvelope. Field extraction itself (OCR + regex/fuzzy-match) lives in
scripts/kaggle_ocr_extract.py (Kaggle, PaddleOCR is too slow on this
machine -- see README "Document pipeline", ~71s/page); this module is the
agent-side wrapper that scores whatever extraction result it's given,
whether that's the Kaggle OCR output or the linked-corpus ground truth used
as a fast-path stand-in before OCR results are back (same interface either
way -- extracted_fields.parquet and the manifest's ground-truth columns
carry the same field names).

Text-risk is keyword/rule-based (not a fine-tuned sentence-transformer
classifier) -- a deliberate time-budget cut, see the follow-up note in
README/kaggle_ocr_extract.py about the same tradeoff on LayoutLMv3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from agents.envelope import AgentEnvelope

VAGUE_DESCRIPTIONS = {
    "Miscellaneous Expense",
    "Consulting Services",
    "IT Consulting Services",
    "Professional Services Retainer",
}
ROUND_AMOUNT_MODULUS = 1000
LOW_CONFIDENCE_MATCH_SCORE = 80


@dataclass
class DocumentAgent:
    weights: dict = field(
        default_factory=lambda: {"vague_description": 0.4, "round_amount": 0.25, "low_extraction_confidence": 0.35}
    )

    def analyze_row(self, row: pd.Series) -> AgentEnvelope:
        fired = []

        description = row.get("description") or ""
        if description in VAGUE_DESCRIPTIONS:
            fired.append({"type": "vague_description", "detail": description, "value": True})

        amount = row.get("extracted_amount")
        if amount is not None and amount % ROUND_AMOUNT_MODULUS == 0 and amount > 0:
            fired.append({"type": "round_amount", "detail": f"${amount:,.2f}", "value": True})

        confidence = row.get("vendor_match_score", 100)  # ground-truth fast path defaults to full confidence
        if confidence < LOW_CONFIDENCE_MATCH_SCORE:
            fired.append({"type": "low_extraction_confidence", "detail": f"vendor match={confidence}", "value": confidence})

        score = min(1.0, sum(self.weights[f["type"]] for f in fired))
        evidence = fired + [
            {
                "type": "extracted_field",
                "detail": k,
                "value": row.get(k),
            }
            for k in ("extracted_vendor_name", "extracted_invoice_number", "extracted_amount", "extracted_po_number")
            if k in row
        ]
        return AgentEnvelope(agent="document", row_id=int(row.name), score=score, evidence=evidence)

    def analyze_batch(self, df: pd.DataFrame) -> list[AgentEnvelope]:
        return [self.analyze_row(row) for _, row in df.iterrows()]
