"""Score-a-new-entry tab of the review UI: a posting typed in here was never in
the corpus, and the API scores it at request time -- all three agents and the
fusion meta-classifier (POST /score, agents/pipeline.py). Nothing on this
page is read from the batch score files.

The invoice fields are typed in too; there is no OCR. An uploaded invoice
image is shown next to the result for the reviewer and never read.
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pandas as pd
import streamlit as st

from linked_pipeline.typologies import mask_ghost_vendor_ids

API = "http://127.0.0.1:8811"
OTHER_VENDOR = "Other (type a vendor ID)"
SAP_CODES = ("WAERS", "BUKRS", "KTOSL", "PRCTR", "BSCHL", "HKONT")


def _masked(value):
    # Same masking the review queue applies (dashboard/app.py::_masked).
    if isinstance(value, dict):
        return {k: _masked(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_masked(v) for v in value]
    return mask_ghost_vendor_ids(value)


@st.cache_data(ttl=600)
def _options() -> dict:
    return httpx.get(f"{API}/live/options", timeout=300).json()


def _entry_form(options: dict) -> dict | None:
    """The posting and its invoice. Returns the /score payload on submit."""
    vendors = {f"{v['vendor_id']} · {v['vendor_name']}": v for v in options["vendors"]}
    levels = options["category_levels"]

    st.subheader("Ledger entry")
    c1, c2 = st.columns(2)
    vendor_label = c1.selectbox("Vendor", list(vendors) + [OTHER_VENDOR])
    if vendor_label == OTHER_VENDOR:
        vendor_id = c1.text_input("Vendor ID", value="V9999")
        vendor_name = c1.text_input("Vendor name", value="Unlisted Supplier Co.")
    else:
        vendor_id = vendors[vendor_label]["vendor_id"]
        vendor_name = vendors[vendor_label]["vendor_name"]
    amount = c2.number_input("Amount (DMBTR)", min_value=0.01, value=12_500.00, step=100.0, format="%.2f")
    description = c1.selectbox("Line-item description", options["descriptions"])
    invoice_number = c2.text_input("Invoice number", value="INV-LIVE-000001")
    po_number = c1.text_input("PO number (blank = none)", value="PO-900001")
    posting_date = c2.date_input("Posting date", value=date(2026, 3, 16))
    doc_date = c1.date_input("Invoice date", value=date(2026, 3, 16) - timedelta(days=4))
    hour = c2.number_input("Posting hour", min_value=0, max_value=23, value=10)

    with st.expander("SAP codes"):
        cols = st.columns(3)
        codes = {code: cols[i % 3].selectbox(code, levels[code], key=f"sap_{code}") for i, code in enumerate(SAP_CODES)}

    st.subheader("Invoice")
    matches = st.checkbox("The invoice shows the same vendor, amount, invoice number and PO as the ledger", value=True)
    if matches:
        inv = {"vendor_name": vendor_name, "amount": amount, "invoice_number": invoice_number, "po_number": po_number}
    else:
        d1, d2 = st.columns(2)
        inv = {
            "vendor_name": d1.text_input("Vendor name on invoice", value=vendor_name),
            "amount": d2.number_input("Amount on invoice", min_value=0.01, value=float(amount), step=100.0, format="%.2f"),
            "invoice_number": d1.text_input("Invoice number on invoice", value=invoice_number),
            "po_number": d2.text_input("PO on invoice (blank = none)", value=po_number),
        }
    image = st.file_uploader("Invoice image (optional, shown only -- not read)", type=["png", "jpg", "jpeg"])
    if image is not None:
        st.session_state["live_image"] = image.getvalue()

    if not st.button("Score this entry", type="primary"):
        return None
    return {
        "ledger": {
            "vendor_id": vendor_id,
            "DMBTR": amount,
            "posting_date": posting_date.isoformat(),
            "doc_date": doc_date.isoformat(),
            "invoice_number": invoice_number,
            "po_number": po_number or None,
            "description": description,
            "hour_of_day": int(hour),
            **codes,
        },
        "invoice": {**inv, "po_number": inv["po_number"] or None},
    }


def _show_result(result: dict) -> None:
    result = _masked(result)
    sub_id = result["submission_id"]
    st.subheader(f"Submission #{sub_id}")
    st.caption(f"Scored live in {result['latency_s'] * 1000:.0f} ms · document number {result['ledger']['BELNR']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fused risk score", f"{result['fused_score']:.2f}", result["risk_tier"])
    m2.metric("Transaction agent", f"{result['transaction_score']:.2f}")
    m3.metric("Document agent", f"{result['document_score']:.2f}")
    m4.metric("Reconciliation agent", f"{result['reconciliation_score']:.2f}")
    st.caption(f"High tier at fused score ≥ {result['f1_threshold']:.2f} (the F1-optimal threshold on validation).")

    if st.session_state.get("live_image"):
        st.image(st.session_state["live_image"], caption="Uploaded invoice (not read by the system)", width=360)

    st.write("**Reconciliation rules fired**")
    if result["recon_evidence"]:
        for e in result["recon_evidence"]:
            st.markdown(f"- `{e['type']}` — {e['detail']}")
    else:
        st.write("None.")

    st.write("**Document signals**")
    doc_signals = [e for e in result["document_evidence"] if e["type"] != "extracted_field"]
    if doc_signals:
        for e in doc_signals:
            st.markdown(f"- `{e['type']}` — {e['detail']}")
    else:
        st.write("None.")

    with st.expander("Transaction features"):
        st.table(pd.DataFrame(result["transaction_evidence"])[["detail", "value"]].set_index("detail"))

    if st.button("Generate narrative (local LLM, ~1 min)", key=f"narr_{sub_id}"):
        with st.spinner("Generating narrative..."):
            narrative = httpx.post(f"{API}/submission/{sub_id}/narrative", timeout=600).json()
        st.session_state[f"narrative_{sub_id}"] = _masked(narrative)
    narrative = st.session_state.get(f"narrative_{sub_id}")
    if narrative:
        st.info(narrative["narrative"])
        if not narrative.get("grounded"):
            st.caption("Not grounded: the narrative cites numbers absent from the evidence.")

    cols = st.columns(3)
    for col, decision in zip(cols, ("accept", "reject", "escalate")):
        if col.button(decision.capitalize(), key=f"live_decide_{decision}_{sub_id}"):
            httpx.post(f"{API}/submission/{sub_id}/decision", json={"decision": decision}, timeout=10)
            st.success(f"Recorded: {decision}")


def score_page() -> None:
    st.title("Score a new entry")
    st.caption(
        "Enter a posting and its invoice. All three agents and the fusion model score it now, "
        "against the corpus ledger plus every earlier submission. Invoice fields are entered, not OCR'd."
    )
    try:
        options = _options()
    except httpx.ConnectError:
        st.error(f"Can't reach the API at {API}. Run: uvicorn dashboard.api:app --port 8811")
        st.stop()

    form_col, result_col = st.columns([1, 1.2])
    with form_col:
        payload = _entry_form(options)
    if payload is not None:
        with st.spinner("Scoring..."):
            resp = httpx.post(f"{API}/score", json=payload, timeout=120)
        if resp.status_code != 200:
            result_col.error(f"Scoring failed ({resp.status_code}): {resp.text}")
        else:
            st.session_state["live_result"] = resp.json()
    with result_col:
        if "live_result" in st.session_state:
            _show_result(st.session_state["live_result"])

    with st.expander("Earlier submissions"):
        subs = httpx.get(f"{API}/submissions", timeout=10).json()["submissions"]
        if subs:
            st.dataframe(pd.DataFrame(_masked(subs)).drop(columns=["timestamp"]), hide_index=True)
        else:
            st.write("None yet.")
