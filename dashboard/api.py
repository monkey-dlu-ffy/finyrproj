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

# The ledger columns a reviewer needs to judge a case, in display order. The
# dashboard shows the actual dataset row -- BELNR (the SAP document number)
# included -- rather than the manifest's summary, because a duplicate case is
# only reviewable against the *entry* it duplicates: same invoice_number and
# vendor, different BELNR/posting_date. The manifest carries neither BELNR nor
# the posting dates, so it can't answer that on its own.
LEDGER_DISPLAY_COLS = [
    "BELNR", "vendor_id", "invoice_number", "po_number", "description",
    "description_tier", "posting_date", "doc_date", "DMBTR", "invoice_amount",
    "is_fraud", "typology",
]
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

    # Loaded once, not per request: /case/{id} previously re-read the whole
    # image manifest from parquet on every call. Only LEDGER_DISPLAY_COLS are
    # read -- the full 533K-row ledger at 20 columns is a needless resident
    # cost on this machine, and nothing here needs the ERP feature columns.
    ledger = {}
    for split in ("train", "val", "test"):
        part = pd.read_parquet(LINKED_DIR / f"{split}.parquet", columns=LEDGER_DISPLAY_COLS)
        # row_id is the positional index each split.parquet was written with
        # (reset_index(drop=True)), which is what recon evidence cites.
        ledger[split] = part
    manifest = pd.read_parquet(LINKED_DIR / "images" / "manifest.parquet")

    fusion_results = json.loads((EXPERIMENTS_DIR / "fusion_results.json").read_text())
    _state["ledger"] = ledger
    # keyed across ALL splits: a duplicate's original is normally in the same
    # split (pairs share a vendor, and splitting is vendor-disjoint) but not
    # always -- 2 of 560 test duplicates cite a row in another split.
    _state["image_path_by_key"] = dict(
        zip(zip(manifest["split"], manifest["row_id"]), manifest["image_path"])
    )
    # The ledger stores vendor_id; the human-readable name is resolved once at
    # render time and only recorded in the manifest, so carry it across for
    # rows that have one.
    _state["image_vendor_name"] = dict(
        zip(zip(manifest["split"], manifest["row_id"]), manifest["vendor_name"])
    )
    _state["meta_df"] = meta_df.set_index("row_id")
    _state["recon_df"] = recon.set_index("row_id") if len(recon) else recon
    _state["threshold"] = fusion_results["f1_threshold"]
    _state["coordinator"] = CoordinatorAgent()
    _state["narrative_cache"] = {}
    return _state


def _ledger_entry(split: str, row_id: int) -> dict | None:
    """The dataset row itself, as shown on the dashboard. `image_path` is None
    when the row wasn't part of the imaged subset -- the fields still resolve,
    so a case is never left with nothing to show."""
    state = _load_state()
    part = state["ledger"].get(split)
    if part is None or row_id not in part.index:
        return None
    row = part.loc[row_id]
    entry = {"row_id": int(row_id), "split": split}
    for col in LEDGER_DISPLAY_COLS:
        value = row[col]
        if col in ("posting_date", "doc_date"):
            entry[col] = str(pd.Timestamp(value).date())
        elif col in ("DMBTR", "invoice_amount"):
            entry[col] = float(value)
        elif col in ("is_fraud",):
            entry[col] = int(value)
        elif value is None or (not isinstance(value, str) and pd.isna(value)):
            entry[col] = None
        else:
            entry[col] = str(value)
    entry["image_path"] = state["image_path_by_key"].get((split, int(row_id)))
    return entry


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

    # The case's own dataset row, BELNR included. Sourced from the ledger
    # rather than the image manifest so a case still renders its identifying
    # fields when it has no image, and so the document number is available at
    # all -- the manifest never carried BELNR.
    case_entry = _ledger_entry("test", row_id)
    image_path = case_entry["image_path"] if case_entry else None
    # document_number/posting_date are here so the coordinator can tell THIS
    # entry from the one it duplicates. Without them the only BELNR in the
    # prompt was the other row's (it appears in the duplicate rule's detail
    # string), and the narrative attributed that document number and this
    # case's amount to the same entry. The grounding check did not catch it:
    # it verifies every number traces back to the evidence, not that each is
    # attached to the right row.
    doc_fields = {
        "document_number": case_entry["BELNR"] if case_entry else None,
        "vendor_name": case_entry["vendor_id"] if case_entry else None,
        "invoice_number": case_entry["invoice_number"] if case_entry else None,
        "invoice_amount": case_entry["invoice_amount"] if case_entry else None,
        "ledger_amount": case_entry["DMBTR"] if case_entry else None,
        "posting_date": case_entry["posting_date"] if case_entry else None,
    }
    if case_entry:
        vendor_name = state["image_vendor_name"].get(("test", row_id))
        if vendor_name:
            doc_fields["vendor_name"] = vendor_name
            case_entry["vendor_name"] = vendor_name

    # duplicate_invoice_number's evidence cites the entry this case duplicates
    # by (split, row_id) -- resolve it to that full dataset row so the two can
    # be read side by side. Pairs share an invoice_number and vendor and differ
    # in BELNR/posting_date, which is the whole of the finding, so citing the
    # BELNR alone (as this did) left the reviewer unable to see it.
    duplicate_original = None
    for e in recon_evidence:
        if e.get("type") == "duplicate_invoice_number" and e.get("matches"):
            match = e["matches"][0]
            orig_row_id, orig_split = match.get("row_id"), match.get("split")
            if orig_row_id is not None and orig_split is not None:
                duplicate_original = _ledger_entry(orig_split, int(orig_row_id))
                if duplicate_original is not None:
                    name = state["image_vendor_name"].get((orig_split, int(orig_row_id)))
                    if name:
                        duplicate_original["vendor_name"] = name
            if duplicate_original is None:
                # Pre-pointer evidence, or a row outside the loaded splits:
                # fall back to what the rule itself recorded.
                duplicate_original = {**match, "image_path": None}
            break

    narrative = state["narrative_cache"].get(row_id)
    if narrative is None:
        case = {**row, "recon_evidence": recon_evidence, "doc_fields": doc_fields}
        try:
            narrative = state["coordinator"].narrate(case)
            state["narrative_cache"][row_id] = narrative
        except Exception as exc:  # Ollama unavailable -- degrade, don't 500 the whole case view
            # Deliberately NOT cached. The cache exists because generation is
            # slow (~30-60s), not because failures are worth keeping: caching
            # one meant every case opened while the daemon was down stayed
            # broken after it came back, until the service was restarted.
            narrative = {"narrative": f"(narrative unavailable: {exc})", "grounded": False}

    return {
        **row,
        "recon_evidence": recon_evidence,
        "doc_fields": doc_fields,
        "image_path": image_path,
        "case_entry": case_entry,
        "duplicate_original": duplicate_original,
        "narrative": narrative,
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
