"""Shared output contract. Every agent (transaction/document/reconciliation/
coordinator) returns evidence, not model internals -- the dashboard and the
LLM narrative both read this, never a raw model object (see agents/__init__.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentEnvelope:
    agent: str
    row_id: int
    score: float | None  # calibrated risk in [0,1]; None if the agent only flags (reconciliation)
    evidence: list[dict] = field(default_factory=list)  # [{"type": ..., "detail": ..., "value": ...}]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"agent": self.agent, "row_id": self.row_id, "score": self.score, "evidence": self.evidence, "meta": self.meta}
