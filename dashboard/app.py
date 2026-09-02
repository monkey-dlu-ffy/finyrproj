"""Streamlit review UI. Talks to dashboard/api.py over HTTP (run both:
`uvicorn dashboard.api:app --port 8811` then `streamlit run dashboard/app.py`)
rather than importing it directly, so the FastAPI service stays the single
source of truth for scoring/audit-log writes -- the spec's own separation
(FastAPI backend + Streamlit frontend), not a shortcut merge of the two.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import streamlit as st

API = "http://127.0.0.1:8811"
LINKED_DIR = Path(__file__).resolve().parent.parent / "data" / "linked"

st.set_page_config(page_title="fraud-audit review queue", layout="wide")
st.title("Fraud Audit — Case Review Queue")

tier = st.sidebar.selectbox("Risk tier", ["all", "high", "medium", "low"])
params = {} if tier == "all" else {"tier": tier}

try:
    cases = httpx.get(f"{API}/cases", params=params, timeout=10).json()
except httpx.ConnectError:
    st.error(f"Can't reach the API at {API}. Run: uvicorn dashboard.api:app --port 8811")
    st.stop()

if not cases:
    st.info("No cases for this filter.")
    st.stop()

left, right = st.columns([1, 1.4])

with left:
    st.subheader(f"{len(cases)} cases")
    for c in cases:
        label = f"#{c['row_id']} · {c['typology']} · {c['risk_tier']} ({c['fused_score']:.2f})"
        if st.button(label, key=f"case_{c['row_id']}", use_container_width=True):
            st.session_state["selected"] = c["row_id"]

with right:
    row_id = st.session_state.get("selected", cases[0]["row_id"])
    st.subheader(f"Case #{row_id}")

    with st.spinner("Loading case (first view generates an LLM narrative, ~1min on this machine)..."):
        detail = httpx.get(f"{API}/case/{row_id}", timeout=180).json()

    img_left, fields_right = st.columns(2)
    with img_left:
        img_path = detail.get("image_path")
        if img_path:
            full_path = LINKED_DIR / img_path
            if full_path.exists():
                st.image(str(full_path), caption="Invoice image", use_container_width=True)
    with fields_right:
        st.metric("Fused risk score", f"{detail['fused_score']:.2f}", detail["risk_tier"])
        st.write("**Agent scores**")
        st.write(
            {
                "transaction": round(detail["transaction_score"], 3),
                "document": round(detail["document_score"], 3),
                "reconciliation": round(detail["reconciliation_score"], 3),
            }
        )
        st.write("**Document fields**")
        st.json(detail.get("doc_fields", {}))

    st.write("**Reconciliation rules fired**")
    evidence = detail.get("recon_evidence") or []
    if evidence:
        for e in evidence:
            st.warning(f"**{e['type']}** — {e['detail']}")
    else:
        st.success("No rules fired.")

    st.write("**Coordinator narrative**")
    narrative = detail.get("narrative", {})
    st.write(narrative.get("narrative", "(unavailable)"))
    if narrative.get("grounded") is False:
        st.error(f"Ungrounded numbers detected: {narrative.get('ungrounded_numbers')}")

    st.write("**Decision**")
    d1, d2, d3 = st.columns(3)
    for col, decision in zip((d1, d2, d3), ("accept", "reject", "escalate")):
        if col.button(decision.capitalize(), key=f"decide_{decision}_{row_id}"):
            httpx.post(f"{API}/case/{row_id}/decision", json={"decision": decision}, timeout=10)
            st.success(f"Recorded: {decision}")
