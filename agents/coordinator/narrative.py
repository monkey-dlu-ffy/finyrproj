"""Coordinator: the only component that talks to the local LLM. Narrates
evidence the specialist agents already computed -- never decides fraud
itself (see agents/coordinator/__init__.py); the fused score from
scripts/run_fusion.py already made that call.

Fed only: agent scores, top SHAP features (if given), fired reconciliation
rules, extracted invoice fields. Never the label. num_predict is capped --
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

NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
MAX_TOKENS = 130

PROMPT_TEMPLATE = """You are a fraud audit coordinator. Summarize the evidence below into a \
finding of at most {max_words} words for a human reviewer. State only what the \
evidence supports; do not speculate, invent numbers, or guess at causes not listed here.

Case: row {row_id}, typology under review: {typology}

Agent scores (0=low risk, 1=high risk):
  transaction_agent:    {transaction_score:.2f}
  document_agent:       {document_score:.2f}
  reconciliation_agent: {reconciliation_score:.2f}
  fused risk score:     {fused_score:.2f} (tier: {risk_tier})

Reconciliation rules fired:
{recon_evidence}

Extracted document fields:
{doc_fields}

Write the finding, then a one-line recommended action."""


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
        typology=case.get("typology", "unknown"),
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
        prompt = build_prompt(case)
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
