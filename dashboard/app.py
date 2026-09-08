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
import pandas as pd
import streamlit as st

API = "http://127.0.0.1:8811"
LINKED_DIR = Path(__file__).resolve().parent.parent / "data" / "linked"

# Ledger fields shown for a case, in reading order: identity first, then the
# two amounts (they disagree under altered_amount), then the classification.
DISPLAY_FIELDS = [
    ("BELNR", "Document no (BELNR)"),
    ("row_id", "Dataset row"),
    ("split", "Split"),
    ("vendor_name", "Vendor"),
    ("vendor_id", "Vendor ID"),
    ("invoice_number", "Invoice no"),
    ("po_number", "PO no"),
    ("description", "Description"),
    ("description_tier", "Description tier"),
    ("doc_date", "Invoice date"),
    ("posting_date", "Posting date"),
    ("DMBTR", "Ledger amount"),
    ("invoice_amount", "Invoice amount"),
    ("typology", "Typology"),
]


def _fmt(key: str, value) -> str:
    if value is None:
        return "—"
    if key in ("DMBTR", "invoice_amount"):
        return f"${float(value):,.2f}"
    return str(value)


def _entry_fields(entry: dict) -> list[tuple[str, str]]:
    return [(label, _fmt(key, entry.get(key))) for key, label in DISPLAY_FIELDS if key in entry]


def ent_label(entry: dict) -> str:
    row_id = entry.get("row_id")
    return f"#{row_id}" if row_id is not None else f"BELNR {entry.get('BELNR', '?')}"


def _entry_diff(this_entry: dict, other: dict) -> pd.DataFrame:
    """Both dataset rows in one table, flagging the fields that differ -- a
    duplicate is defined by what matches (invoice no, vendor) against what
    doesn't (BELNR, posting date), so both halves are the evidence."""
    rows = []
    for key, label in DISPLAY_FIELDS:
        if key not in this_entry and key not in other:
            continue
        a, b = this_entry.get(key), other.get(key)
        rows.append(
            {
                "Field": label,
                "This case": _fmt(key, a),
                "Duplicated entry": _fmt(key, b),
                "": "" if a == b else "differs",
            }
        )
    return pd.DataFrame(rows).set_index("Field")


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
        st.write("**Ledger entry (dataset row)**")
        entry = detail.get("case_entry")
        if entry:
            st.table(pd.DataFrame(_entry_fields(entry), columns=["field", "value"]).set_index("field"))
        else:
            st.json(detail.get("doc_fields", {}))

    st.write("**Reconciliation rules fired**")
    evidence = detail.get("recon_evidence") or []
    if evidence:
        for e in evidence:
            st.warning(f"**{e['type']}** — {e['detail']}")
            for m in e.get("matches") or []:
                where = f" ({m['split']} row {m['row_id']})" if m.get("row_id") is not None else ""
                st.caption(
                    f"↳ original entry: BELNR {m['BELNR']}{where} · vendor {m['vendor_id']}"
                    f" · posted {m['posting_date']} · ${m['amount']:,.2f}"
                )
    else:
        st.success("No rules fired.")

    original = detail.get("duplicate_original")
    if original:
        st.write("**Duplicate invoice — this entry vs. the entry it duplicates**")
        # The two rows share invoice_number and vendor and differ in
        # BELNR/posting_date; differing fields are marked so the reviewer sees
        # which is which without diffing two JSON blobs by eye.
        this_entry = detail.get("case_entry") or {}
        st.dataframe(_entry_diff(this_entry, original), use_container_width=True)

        dup_col, orig_col = st.columns(2)
        for col, label, ent in (
            (dup_col, f"This case · #{row_id}", this_entry),
            (orig_col, f"Duplicated entry · {ent_label(original)}", original),
        ):
            with col:
                st.caption(label)
                path = ent.get("image_path")
                if path and (LINKED_DIR / path).exists():
                    st.image(str(LINKED_DIR / path), use_container_width=True)
                else:
                    st.info("No invoice image was rendered for this row (it is outside the imaged subset).")

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
