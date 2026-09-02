"""FastAPI scoring/review service. Reads precomputed agent + fusion scores
(experiments/fusion_scores.parquet, reconciliation_results.json) rather than
re-running the agents per request -- CPU inference here is cheap, but the
coordinator's LLM call is not (~3 tok/s, README Phase 0), so /case/{id}
generates its narrative lazily and caches it, and /analyze only re-scores
the numeric agents, never blocking on the LLM.

SQLite audit_log is append-only: every /analyze call and every auditor
decision is a new row, never an update, so the log stays a true record of
what was shown and decided, not just the latest state.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.coordinator.narrative import CoordinatorAgent
from config import EXPERIMENTS_DIR, LINKED_DIR

DB_PATH = EXPERIMENTS_DIR / "audit_log.sqlite3"
app = FastAPI(title="fraud-audit dashboard API")

_state: dict = {}


def _load_state() -> dict:
    if _state:
        return _state
    meta_df = pd.read_parquet(EXPERIMENTS_DIR / "fusion_scores.parquet")
    # row_id is only unique WITHIN a split (see scripts/run_fusion.py) -- the
    # dashboard reviews the held-out test split only, both to sidestep that
    # collision and because reviewing training rows isn't a real workflow.
    meta_df = meta_df.loc[meta_df["split"] == "test"]
    recon_path = EXPERIMENTS_DIR / "reconciliation_results.json"
    recon = pd.read_json(recon_path) if recon_path.exists() else pd.DataFrame()
    if len(recon):
        recon = recon.loc[recon["split"] == "test"]

    fusion_results = json.loads((EXPERIMENTS_DIR / "fusion_results.json").read_text())
    _state["meta_df"] = meta_df.set_index("row_id")
    _state["recon_df"] = recon.set_index("row_id") if len(recon) else recon
    _state["threshold"] = fusion_results["f1_threshold"]
    _state["coordinator"] = CoordinatorAgent()
    _state["narrative_cache"] = {}
    return _state


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            score REAL,
            evidence_json TEXT,
            timestamp REAL NOT NULL,
            auditor_decision TEXT
        )"""
    )
    return conn


class Decision(BaseModel):
    decision: str  # accept | reject | escalate
    reviewer: str | None = None


@app.post("/analyze")
def analyze(row_id: int):
    state = _load_state()
    if row_id not in state["meta_df"].index:
        raise HTTPException(404, f"row_id {row_id} not in the document-linked corpus")
    row = state["meta_df"].loc[row_id]
    t0 = time.time()

    conn = _db()
    conn.execute(
        "INSERT INTO audit_log (case_id, agent_id, score, evidence_json, timestamp, auditor_decision) VALUES (?,?,?,?,?,NULL)",
        (row_id, "fusion", float(row["fused_score"]), json.dumps(row.to_dict(), default=str), time.time()),
    )
    conn.commit()
    conn.close()

    return {
        "row_id": row_id,
        "fused_score": float(row["fused_score"]),
        "risk_tier": row["risk_tier"],
        "transaction_score": float(row["transaction_score"]),
        "document_score": float(row["document_score"]),
        "reconciliation_score": float(row["reconciliation_score"]),
        "typology": row["typology"],
        "latency_s": time.time() - t0,  # E6
    }


@app.get("/cases")
def list_cases(tier: str | None = None, limit: int = 100):
    state = _load_state()
    df = state["meta_df"]
    if tier:
        df = df.loc[df["risk_tier"] == tier]
    df = df.sort_values("fused_score", ascending=False).head(limit)
    return df.reset_index().to_dict(orient="records")


@app.get("/case/{row_id}")
def get_case(row_id: int):
    state = _load_state()
    if row_id not in state["meta_df"].index:
        raise HTTPException(404, "not found")
    row = state["meta_df"].loc[row_id].to_dict()
    row["row_id"] = row_id

    recon_evidence = []
    if len(state["recon_df"]) and row_id in state["recon_df"].index:
        recon_evidence = state["recon_df"].loc[row_id, "recon_evidence"]

    manifest = pd.read_parquet(LINKED_DIR / "images" / "manifest.parquet")
    manifest = manifest.loc[manifest["split"] == "test"].set_index("row_id")
    doc_fields = {}
    image_path = None
    if row_id in manifest.index:
        m = manifest.loc[row_id]
        doc_fields = {
            "vendor_name": m["vendor_name"],
            "invoice_number": m["invoice_number"],
            "invoice_amount": m["invoice_amount"],
            "ledger_amount": m["ledger_amount"],
        }
        image_path = m["image_path"]

    if row_id not in state["narrative_cache"]:
        case = {**row, "recon_evidence": recon_evidence, "doc_fields": doc_fields}
        try:
            state["narrative_cache"][row_id] = state["coordinator"].narrate(case)
        except Exception as exc:  # Ollama unavailable -- degrade, don't 500 the whole case view
            state["narrative_cache"][row_id] = {"narrative": f"(narrative unavailable: {exc})", "grounded": False}

    return {
        **row,
        "recon_evidence": recon_evidence,
        "doc_fields": doc_fields,
        "image_path": image_path,
        "narrative": state["narrative_cache"][row_id],
    }


@app.post("/case/{row_id}/decision")
def record_decision(row_id: int, decision: Decision):
    if decision.decision not in ("accept", "reject", "escalate"):
        raise HTTPException(400, "decision must be accept|reject|escalate")
    conn = _db()
    conn.execute(
        "INSERT INTO audit_log (case_id, agent_id, score, evidence_json, timestamp, auditor_decision) VALUES (?,?,?,?,?,?)",
        (row_id, "auditor", None, json.dumps({"reviewer": decision.reviewer}), time.time(), decision.decision),
    )
    conn.commit()
    conn.close()
    return {"row_id": row_id, "recorded": decision.decision}
