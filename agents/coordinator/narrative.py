"""Coordinator: the only component that talks to the local LLM. Narrates
evidence the specialist agents already computed -- never decides fraud
itself (see agents/coordinator/__init__.py); the fused score from
scripts/run_fusion.py already made that call.

Fed only: agent scores, top SHAP features (if given), fired reconciliation
rules, extracted invoice fields. Never the label: redact_case() drops the
label columns, and masks ghost-vendor IDs, which spell the label out.

The prompt separates "the entry under review" from the other entries a rule
cites, because it does not otherwise: on a duplicate_invoice case the model
attached the duplicated entry's BELNR and posting date to this entry's amount,
and grounding_check passed it -- that check verifies every number traces back
to the evidence, not that each number is attached to the right row. num_predict is capped --
at ~3 tok/s (README, Phase 0 measurement) an uncapped generation is minutes
long; capping to ~120 tokens keeps a batch of narratives inside this
session's time budget.

Grounding check: every number in the narrative must appear (allowing
$/,/rounding noise) in the evidence it was given, or the narrative is
rejected. This is the anti-hallucination mechanism, not a formatting nicety.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from config import LLM_NUM_CTX, LLM_SEED, LLM_TEMPERATURE, OLLAMA_HOST, OLLAMA_MODEL
from linked_pipeline.typologies import mask_ghost_vendor_ids

NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
MAX_TOKENS = 130

# Ground truth the case dict carries (it is a fusion_scores row) that must
# never reach the model. typology was in the prompt itself until this was added.
LABEL_KEYS = ("typology", "is_fraud")

PROMPT_TEMPLATE = """You are a fraud audit coordinator. Summarize the evidence below into a \
finding of at most {max_words} words for a human reviewer. State only what the \
evidence supports; do not speculate, invent numbers, or guess at causes not listed here.

Case: row {row_id}

The entry under review. Every field here describes THIS entry:
{doc_fields}

Agent scores (0=low risk, 1=high risk):
  transaction_agent:    {transaction_score:.2f}
  document_agent:       {document_score:.2f}
  reconciliation_agent: {reconciliation_score:.2f}
  fused risk score:     {fused_score:.2f} (tier: {risk_tier})

Reconciliation rules fired. These compare the entry above against the ledger and against other entries. Where a rule names a second posting, that posting's document number, date and amount belong to the OTHER entry -- never restate them as if they were the entry under review:
{recon_evidence}

Write the finding, then a one-line recommended action."""


def redact_case(case: dict) -> dict:
    """The case as the model may see it: no label columns, no vendor ID that
    names the label -- ghost-vendor IDs reach the prompt through
    vendor_absent_from_master's detail, a duplicate's cited vendor, or the
    vendor_name fallback. narrate() builds the prompt AND runs the grounding
    check on this, so a pseudonym the model repeats still counts as grounded."""
    mask = mask_ghost_vendor_ids
    redacted = {k: v for k, v in case.items() if k not in LABEL_KEYS}
    redacted["recon_evidence"] = [
        {**e, "detail": mask(e.get("detail")), "value": mask(e.get("value"))}
        for e in case.get("recon_evidence", [])
    ]
    redacted["doc_fields"] = {k: mask(v) for k, v in case.get("doc_fields", {}).items()}
    return redacted


def _format_evidence(recon_evidence: list[dict]) -> str:
    if not recon_evidence:
        return "  (none fired)"
    return "\n".join(f"  - {e['type']}: {e['detail']}" for e in recon_evidence)


def _format_fields(fields: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in fields.items() if v is not None)


def build_prompt(case: dict, max_words: int = 120) -> str:
    return PROMPT_TEMPLATE.format(
        max_words=max_words,
        row_id=case["row_id"],
        transaction_score=case["transaction_score"],
        document_score=case["document_score"],
        reconciliation_score=case["reconciliation_score"],
        fused_score=case["fused_score"],
        risk_tier=case.get("risk_tier", "n/a"),
        recon_evidence=_format_evidence(case.get("recon_evidence", [])),
        doc_fields=_format_fields(case.get("doc_fields", {})),
    )


def _evidence_numbers(case: dict) -> set[str]:
    numbers = set()
    for key in ("transaction_score", "document_score", "reconciliation_score", "fused_score"):
        numbers.add(f"{case[key]:.2f}")
        numbers.add(f"{case[key]:.1f}")
    for e in case.get("recon_evidence", []):
        numbers.update(NUMBER_RE.findall(str(e.get("value", ""))))
        numbers.update(NUMBER_RE.findall(str(e.get("detail", ""))))
    for v in case.get("doc_fields", {}).values():
        numbers.update(NUMBER_RE.findall(str(v)))
    numbers.add(str(case["row_id"]))
    return {n.replace(",", "").rstrip(".") for n in numbers if n}


def grounding_check(narrative: str, case: dict) -> tuple[bool, list[str]]:
    """Every number in the narrative must trace back to the evidence given."""
    allowed = _evidence_numbers(case)
    found = {n.replace(",", "").rstrip(".") for n in NUMBER_RE.findall(narrative)}
    ungrounded = sorted(n for n in found if n and n not in allowed and float(n or 0) not in (0, 1))
    return len(ungrounded) == 0, ungrounded


@dataclass
class CoordinatorAgent:
    host: str = OLLAMA_HOST
    model: str = OLLAMA_MODEL

    def narrate(self, case: dict, max_retries: int = 1) -> dict:
        case = redact_case(case)
        prompt = build_prompt(case)
        if "GHOST" in prompt.upper():
            raise RuntimeError(f"row {case['row_id']}: ghost-vendor label leaked into the prompt after masking")
        for attempt in range(max_retries + 1):
            resp = httpx.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": LLM_TEMPERATURE,
                        "seed": LLM_SEED,
                        "num_ctx": LLM_NUM_CTX,
                        "num_predict": MAX_TOKENS,
                    },
                },
                timeout=600.0,
            )
            resp.raise_for_status()
            narrative = resp.json().get("response", "").strip()
            grounded, ungrounded = grounding_check(narrative, case)
            if grounded:
                return {"narrative": narrative, "grounded": True, "ungrounded_numbers": [], "attempts": attempt + 1}
        return {"narrative": narrative, "grounded": False, "ungrounded_numbers": ungrounded, "attempts": max_retries + 1}
