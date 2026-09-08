"""Document agent: emits extracted fields + a document-only risk score as an
AgentEnvelope. Field extraction itself (OCR + regex/fuzzy-match) lives in
scripts/kaggle_ocr_extract.py (Kaggle, PaddleOCR is too slow on this
machine -- see README "Document pipeline", ~71s/page); this module is the
agent-side wrapper that scores whatever extraction result it's given,
whether that's the Kaggle OCR output or the linked-corpus ground truth used
as a fast-path stand-in before OCR results are back (same interface either
way -- extracted_fields.parquet and the manifest's ground-truth columns
carry the same field names).

Scope: every signal here is computable from ONE invoice in isolation. Any
check needing the ledger row (amount disagreement, vendor-not-in-master) or
a second document (repeated invoice number) belongs to the reconciliation
agent -- duplicating those would double-count the same evidence into the
fusion meta-classifier.

Why this was rewritten, and why the rewrite alone wasn't enough
--------------------------------------------------------------
The first version scored ROC-AUC 0.504 on the full test split -- chance --
and the fusion meta-classifier gave it a NEGATIVE coefficient, i.e. it was
actively noise. Three causes were measured and addressed here:

  * `vague_description` fired on ~15% of rows at a fraud/genuine lift of
    ~1.04. Its phrase set was hand-listed and split across value tiers
    rather than tracking risk. Kept as displayed evidence at a near-zero
    weight, not as a claim that the line item is suspicious.
  * `round_amount` tested `amount % 1000 == 0`, firing on 0.59% of rows.
    Replaced with a multi-scale digit-preference check, which is the actual
    forensic signal -- fabricated amounts cluster on round figures.
  * `low_extraction_confidence` could never fire: the ground-truth fast path
    pins vendor_match_score to 100. Retained deliberately (it is the agent's
    strongest feature once real OCR lands) but it contributes nothing until
    then.

The score is graded rather than a sum of booleans -- the old version could
only emit 4 distinct values, so it carried almost no rank information for
the meta-classifier to use.

None of that moved the number. The agent stayed at ROC-AUC 0.5005 because
the fault was in the CORPUS, not here: `linked_pipeline/synth.py` used to
assign a line item's value tier from a hash of its GL transaction type, so
descriptions carried no information about amount at all (0.05 sigma between
class means, Spearman -0.003) and `amount_atypical_for_description` had
nothing to detect. See `_amount_ranked_tiers` there, and the README section
"The document agent was blind because of the generator". With the tier
ranked by amount, that feature's lift went 1.17 -> 5.47 and this agent
scores ROC-AUC 0.594 with a +2.96 fusion coefficient.

Ceiling, stated honestly
------------------------
Four of the five injected typologies are defined as ledger/document
disagreements (altered_amount, ghost_vendor) or cross-row patterns
(duplicate_invoice, threshold_split), and are invisible to a single-document
agent by construction -- measured per-typology, this agent sits at ~0.50 on
all three of altered_amount, duplicate_invoice and ghost_vendor, which is
the expected result, not a defect to tune away.

It earns its score on desc_amount_mismatch (0.853) and, without anything
aiming at it, threshold_split (0.618) -- split rows are pushed just under an
approval limit, which lands them at an odd amount for their description.
That second one is the evidence the check generalises rather than inverting
the injector: `amount_atypical_for_description` is a general statistical
check (amount z-score within its description class, fit on train genuine
rows only), so it also fires on genuine outliers at ~5.6%.

Against the full ensemble the agent is largely redundant with the
reconciliation agent, which flags the same rows by rule: dropping it leaves
PR-AUC statistically unchanged (0.792 vs 0.789). "Went from harmful to
independently informative" is the supportable claim; "lifted the ensemble"
is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agents.envelope import AgentEnvelope

# Evidence display only -- measured lift over genuine rows is 1.03.
VAGUE_DESCRIPTIONS = {
    "Miscellaneous Expense",
    "IT Consulting Services",
    "Professional Services Retainer",
}

# Fabricated amounts cluster on round figures; the coarser the multiple, the
# less likely it is to arise naturally from a real posting.
ROUND_AMOUNT_SCALES = ((100, 0.08), (500, 0.14), (1000, 0.20), (5000, 0.25))

# |z| within the description class: below SOFT contributes nothing (ordinary
# spread), at/above HARD contributes the full weight, linear in between.
AMOUNT_Z_SOFT = 2.0
AMOUNT_Z_HARD = 6.0

LOW_CONFIDENCE_MATCH_SCORE = 80

WEIGHTS = {
    "amount_atypical_for_description": 0.55,
    "round_amount": 0.25,
    "missing_po": 0.10,
    "vague_description": 0.05,
    "low_extraction_confidence": 0.35,
}


@dataclass
class DocumentFieldStats:
    """Per-description amount distribution, fit on TRAIN GENUINE rows only.

    Genuine-only keeps an injected outlier from widening the very band it
    should be caught by; train-only keeps test rows out of their own
    baseline.
    """

    log_amount_mean: dict
    log_amount_std: dict
    global_log_amount_mean: float
    global_log_amount_std: float

    def z(self, description, amount) -> float:
        if amount is None or not np.isfinite(amount) or amount <= 0:
            return 0.0
        mean = self.log_amount_mean.get(description, self.global_log_amount_mean)
        std = self.log_amount_std.get(description, self.global_log_amount_std)
        if not std or not np.isfinite(std) or std <= 0:
            std = self.global_log_amount_std
        if not std or not np.isfinite(std) or std <= 0:
            return 0.0
        return float(abs(np.log10(amount) - mean) / std)


def fit_document_stats(train_df: pd.DataFrame) -> DocumentFieldStats:
    """Fit the per-description amount baseline. Call once on the train split
    and pass the result to DocumentAgent -- see scripts/run_fusion.py."""
    genuine = train_df.loc[train_df["is_fraud"] == 0] if "is_fraud" in train_df else train_df
    if not len(genuine):
        genuine = train_df
    amounts = pd.to_numeric(genuine["invoice_amount"], errors="coerce")
    log_amt = np.log10(amounts.clip(lower=1.0))
    grouped = log_amt.groupby(genuine["description"])
    global_std = float(log_amt.std())
    return DocumentFieldStats(
        log_amount_mean=grouped.mean().to_dict(),
        log_amount_std=grouped.std().to_dict(),
        global_log_amount_mean=float(log_amt.mean()),
        global_log_amount_std=global_std if global_std and np.isfinite(global_std) else 1.0,
    )


def _graded(z: float) -> float:
    """|z| -> 0..1, flat below SOFT, saturating at HARD."""
    if z <= AMOUNT_Z_SOFT:
        return 0.0
    return min(1.0, (z - AMOUNT_Z_SOFT) / (AMOUNT_Z_HARD - AMOUNT_Z_SOFT))


@dataclass
class DocumentAgent:
    """`stats` is optional so ad-hoc/single-row callers still work, but the
    amount-vs-description signal -- the only one carrying real weight on this
    corpus -- is inert without it. Fit it with fit_document_stats(train)."""

    stats: DocumentFieldStats | None = None
    weights: dict = field(default_factory=lambda: dict(WEIGHTS))

    def analyze_row(self, row: pd.Series) -> AgentEnvelope:
        fired = []

        raw_amount = row.get("extracted_amount")
        amount = float(raw_amount) if raw_amount is not None and pd.notna(raw_amount) else None
        description = row.get("description") or ""

        if self.stats is not None and amount is not None:
            z = self.stats.z(description, amount)
            graded = _graded(z)
            if graded > 0:
                fired.append(
                    {
                        "type": "amount_atypical_for_description",
                        "detail": f"${amount:,.2f} on '{description}' -- {z:.1f} sigma from that line item's usual amount",
                        "value": round(z, 3),
                        "weight": round(self.weights["amount_atypical_for_description"] * graded, 4),
                    }
                )

        if amount is not None and amount > 0:
            for modulus, weight in reversed(ROUND_AMOUNT_SCALES):
                if amount % modulus == 0:
                    fired.append(
                        {
                            "type": "round_amount",
                            "detail": f"${amount:,.2f} is an exact multiple of {modulus:,}",
                            "value": modulus,
                            "weight": weight,
                        }
                    )
                    break

        po = row.get("po_number")
        if po is None or pd.isna(po):
            fired.append({"type": "missing_po", "detail": "no PO printed on the invoice", "value": True, "weight": self.weights["missing_po"]})

        if description in VAGUE_DESCRIPTIONS:
            fired.append({"type": "vague_description", "detail": description, "value": True, "weight": self.weights["vague_description"]})

        # Inert on the ground-truth fast path (score pinned to 100); the
        # agent's strongest feature once real OCR supplies a match score.
        confidence = row.get("vendor_match_score", 100)
        if confidence < LOW_CONFIDENCE_MATCH_SCORE:
            fired.append({"type": "low_extraction_confidence", "detail": f"vendor match={confidence}", "value": confidence, "weight": self.weights["low_extraction_confidence"]})

        score = min(1.0, sum(f["weight"] for f in fired))
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
