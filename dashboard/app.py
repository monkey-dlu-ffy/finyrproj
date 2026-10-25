"""Streamlit review UI. Talks to dashboard/api.py over HTTP (run both:
`uvicorn dashboard.api:app --port 8811` then `streamlit run dashboard/app.py`)
rather than importing it directly, so the FastAPI service stays the single
source of truth for scoring/audit-log writes -- the spec's own separation
(FastAPI backend + Streamlit frontend), not a shortcut merge of the two.

Two pages: the review queue, and the Metrics page (dashboard/metrics_page.py),
which reads the score files directly -- it reports on the whole test split,
not on any one case, so it has nothing to take from the API.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import duckdb
import httpx
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.metrics_page import metrics_page
from linked_pipeline.typologies import mask_ghost_vendor_ids

API = "http://127.0.0.1:8811"
PROJECT_DIR = Path(__file__).resolve().parent.parent
LINKED_DIR = PROJECT_DIR / "data" / "linked"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
VENDOR_MASTER = LINKED_DIR / "vendor_master.json"
# Readable over both themes: a translucent amber, not a fixed light fill.
HIGHLIGHT = "background-color: rgba(255, 191, 0, 0.35)"
PARQUET_PAGE_SIZE = 100
AMOUNT_COLS = ("DMBTR", "WRBTR", "invoice_amount", "ledger_amount")
# The injected ground truth. The reviewer is the one deciding whether a case
# is fraud, so no view on this screen shows them, even though the API's
# responses and the source files carry them.
LABEL_COLS = ("typology", "is_fraud")


def _masked(value):
    """`value` with every GHOST-<name> vendor ID swapped for its VND-<5 digits>
    pseudonym, at any depth -- the ID itself spells out the ghost_vendor label
    (see LABEL_COLS). Applied to everything the API returns, so every view
    built from it inherits the mask."""
    if isinstance(value, dict):
        return {k: _masked(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_masked(v) for v in value]
    return mask_ghost_vendor_ids(value)

# Ledger fields shown for a case, in reading order: identity first, then the
# two amounts (they disagree under altered_amount).
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


def _open_file(path: Path, reveal: bool = False) -> None:
    """Open a source file in its default app or, with `reveal`, show it
    selected in the file manager. This runs where the Streamlit server runs
    -- the reviewer's own machine in this setup -- and only for files inside
    the project."""
    path = path.resolve()
    if not path.is_relative_to(PROJECT_DIR) or not path.exists():
        st.error(f"File not found: {path}")
        return
    if sys.platform == "win32":
        if reveal:
            # explorer expects /select,"<path>" verbatim, so pass one
            # pre-built command line rather than an argv list.
            subprocess.Popen(f'explorer /select,"{path}"')
        else:
            os.startfile(path)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(path.parent if reveal else path)])


def _doc_row(title: str, path: Path, note: str, key: str, actions: list[tuple[str, Callable[[], None]]]) -> None:
    """One source document: what it is, where it lives, and its buttons. Show
    in folder is always last, since every document is a file on disk."""
    actions = actions + [("Show in folder", lambda: _open_file(path, reveal=True))]
    text, *buttons = st.columns([3] + [1.2] * len(actions))
    text.markdown(f"**{title}** · {note}  \n`{path.relative_to(PROJECT_DIR)}`")
    for col, (label, act) in zip(buttons, actions):
        if col.button(label, key=f"{key}_{label}", use_container_width=True):
            act()


@st.dialog("Parquet file", width="large")
def _parquet_dialog(path: Path, split: str, row_id: int) -> None:
    """The whole file, a page at a time, opening on the page that holds this
    case's row. Windows has no default app for .parquet, so this is the
    one-click way to read one; DuckDB reads only the page shown, in place."""
    columns = pq.read_schema(path).names
    n_rows = pq.ParquetFile(path).metadata.num_rows
    src = f"read_parquet('{path.as_posix()}', file_row_number = true)"
    con = duckdb.connect()
    if {"split", "row_id"} <= set(columns):
        # manifest / experiment outputs: rows are keyed by (split, row_id) columns
        hit = con.execute(
            f"SELECT file_row_number FROM {src} WHERE split = ? AND row_id = ?", [split, row_id]
        ).fetchone()
        target = int(hit[0]) if hit else None
    else:
        # a ledger split file: row_id IS the row's position in the file
        target = row_id

    n_pages = max(1, -(-n_rows // PARQUET_PAGE_SIZE))
    page = st.number_input(
        f"Page (of {n_pages:,})", min_value=1, max_value=n_pages, step=1,
        value=1 if target is None else target // PARQUET_PAGE_SIZE + 1,
        key=f"parquet_page_{path}_{split}_{row_id}",
    )
    first = (page - 1) * PARQUET_PAGE_SIZE
    df = (
        con.execute(
            f"SELECT * FROM {src} WHERE file_row_number >= ? AND file_row_number < ? ORDER BY file_row_number",
            [first, first + PARQUET_PAGE_SIZE],
        )
        .df()
        .set_index("file_row_number")
        .rename_axis("file row")
        .drop(columns=list(LABEL_COLS), errors="ignore")
    )
    # Read straight from disk, not through the API, so it needs its own mask.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(mask_ghost_vendor_ids)
    if target is None:
        where = f"no row for {split} row {row_id} in this file"
    elif target in df.index:
        where = f"this case is file row **{target:,}**, highlighted"
    else:
        where = f"this case is file row **{target:,}**, on page {target // PARQUET_PAGE_SIZE + 1:,}"
    st.markdown(
        f"`{path.relative_to(PROJECT_DIR)}` · {n_rows:,} rows × {len(df.columns)} columns shown · "
        f"showing file rows {first:,}–{first + len(df) - 1:,} · {where}"
    )
    styled = df.style.apply(lambda r: [HIGHLIGHT if r.name == target else ""] * len(r), axis=1).format(
        {c: "${:,.2f}" for c in AMOUNT_COLS if c in df.columns}, na_rep="—"
    )
    st.dataframe(styled, use_container_width=True, height=560)
    if path.parent == LINKED_DIR:
        st.caption("The same row from a terminal, in the project folder:")
        st.code(f'py -3.13 ledger_sql.py "select * from {split} where row_id = {row_id}"', language="powershell")


@st.dialog("Vendor master", width="large")
def _vendor_dialog(vendor_id: str | None) -> None:
    vendors = pd.read_json(VENDOR_MASTER)
    st.markdown(
        f"`{VENDOR_MASTER.relative_to(PROJECT_DIR)}` · {len(vendors)} vendors, the list the"
        " `vendor_absent_from_master` rule checks against."
    )
    if vendor_id in set(vendors["vendor_id"]):
        st.success(f"{vendor_id} is on the vendor master (highlighted).")
    else:
        st.error(f"{vendor_id} is not on the vendor master, so `vendor_absent_from_master` fires for it.")
    query = st.text_input("Filter", placeholder="vendor ID or name")
    if query:
        hit = vendors["vendor_id"].str.contains(query, case=False, regex=False) | vendors[
            "vendor_name"
        ].str.contains(query, case=False, regex=False)
        vendors = vendors.loc[hit]
    styled = vendors.style.apply(
        lambda r: [HIGHLIGHT if r["vendor_id"] == vendor_id else ""] * len(r), axis=1
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


@st.dialog("Source record", width="large")
def _record_dialog(title: str, path: Path, record: dict) -> None:
    st.markdown(f"**{title}** · `{path.relative_to(PROJECT_DIR)}`")
    st.json(record)


st.set_page_config(page_title="fraud-audit review", layout="wide")


def review_queue() -> None:
    st.title("Fraud Audit — Case Review Queue")

    tier = st.sidebar.selectbox("Risk tier", ["all", "high", "low"])
    page_size = st.sidebar.selectbox("Cases per page", [25, 50, 100, 200], index=2)

    # Page resets to 1 whenever the filter or page size changes, so a page number
    # that was valid for "all" can't point past the end of a smaller tier.
    if st.session_state.get("page_key") != (tier, page_size):
        st.session_state["page_key"] = (tier, page_size)
        st.session_state["page"] = 1


    def _fetch_page(page: int) -> dict:
        params = {"limit": page_size, "offset": (page - 1) * page_size}
        if tier != "all":
            params["tier"] = tier
        return httpx.get(f"{API}/cases", params=params, timeout=10).json()


    try:
        result = _fetch_page(st.session_state["page"])
    except httpx.ConnectError:
        st.error(f"Can't reach the API at {API}. Run: uvicorn dashboard.api:app --port 8811")
        st.stop()

    total = result["total"]
    if not total:
        st.info("No cases for this filter.")
        st.stop()

    n_pages = (total + page_size - 1) // page_size
    page = st.sidebar.number_input("Page", min_value=1, max_value=n_pages, step=1, key="page")
    cases = _masked(result["cases"])


    def _step_page(delta: int) -> None:
        # A callback, not an inline assignment: Streamlit forbids writing a
        # widget's session_state key after the widget has rendered in this run.
        st.session_state["page"] += delta

    left, right = st.columns([1, 1.4])

    with left:
        first = result["offset"] + 1
        st.subheader(f"{total} cases")
        st.caption(f"Showing {first}–{first + len(cases) - 1} · page {page} of {n_pages}")
        prev_col, next_col = st.columns(2)
        prev_col.button(
            "← Previous", disabled=page <= 1, on_click=_step_page, args=(-1,), use_container_width=True
        )
        next_col.button(
            "Next →", disabled=page >= n_pages, on_click=_step_page, args=(1,), use_container_width=True
        )
        for c in cases:
            label = f"#{c['row_id']} · {c['risk_tier']} ({c['fused_score']:.2f})"
            if st.button(label, key=f"case_{c['row_id']}", use_container_width=True):
                st.session_state["selected"] = c["row_id"]

    with right:
        row_id = st.session_state.get("selected", cases[0]["row_id"])
        st.subheader(f"Case #{row_id}")

        with st.spinner("Loading case (first view generates an LLM narrative, ~1min on this machine)..."):
            detail = _masked(httpx.get(f"{API}/case/{row_id}", timeout=180).json())

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

        st.write("**Source documents**")
        st.caption("Every file this case was scored from. Open and Show in folder act on this machine.")
        _doc_row(
            "Ledger entry", LINKED_DIR / "test.parquet", f"row {row_id}", f"doc_{row_id}_ledger",
            [("View file", lambda: _parquet_dialog(LINKED_DIR / "test.parquet", "test", row_id))],
        )
        if img_path:
            _doc_row(
                "Invoice image", LINKED_DIR / img_path, "rendered invoice", f"doc_{row_id}_image",
                [("Open", lambda: _open_file(LINKED_DIR / img_path))],
            )
        else:
            st.caption("Invoice image: none rendered for this row (it is outside the imaged subset).")
        manifest_record = {
            k: v for k, v in (detail.get("manifest_record") or {}).items() if k not in LABEL_COLS
        }
        if manifest_record:
            _doc_row(
                "Invoice fields", LINKED_DIR / "images" / "manifest.parquet",
                "what the agents read from the invoice", f"doc_{row_id}_manifest",
                [
                    ("View record", lambda: _record_dialog("Invoice fields", LINKED_DIR / "images" / "manifest.parquet", manifest_record)),
                    ("View file", lambda: _parquet_dialog(LINKED_DIR / "images" / "manifest.parquet", "test", row_id)),
                ],
            )
        case_vendor = (detail.get("case_entry") or {}).get("vendor_id")
        _doc_row(
            "Vendor master", VENDOR_MASTER, f"vendor {case_vendor}", f"doc_{row_id}_vendors",
            [("View vendors", lambda: _vendor_dialog(case_vendor))],
        )
        for i, rel in enumerate(detail.get("related_entries") or []):
            rel_actions = [
                ("View file", lambda s=rel["split"], r=rel["row_id"]: _parquet_dialog(LINKED_DIR / f"{s}.parquet", s, r))
            ]
            if rel.get("image_path"):
                rel_actions.append(("Open invoice", lambda p=rel["image_path"]: _open_file(LINKED_DIR / p)))
            _doc_row(
                rel["role"], LINKED_DIR / f"{rel['split']}.parquet",
                f"{rel['split']} row {rel['row_id']} · BELNR {rel['BELNR']}"
                + ("" if rel.get("image_path") else " · no invoice image"),
                f"doc_{row_id}_related_{i}", rel_actions,
            )
        recon_file = EXPERIMENTS_DIR / "reconciliation_results.json"
        recon_record = {
            "row_id": row_id, "split": "test",
            "recon_score": detail["reconciliation_score"], "recon_evidence": detail.get("recon_evidence") or [],
        }
        _doc_row(
            "Reconciliation results", recon_file, "rules fired for this row", f"doc_{row_id}_recon",
            [("View record", lambda: _record_dialog("Reconciliation results", recon_file, recon_record))],
        )
        fusion_file = EXPERIMENTS_DIR / "fusion_scores.parquet"
        fusion_record = {
            k: detail.get(k)
            for k in ("row_id", "split", "transaction_score", "document_score",
                      "reconciliation_score", "fused_score", "risk_tier")
        }
        _doc_row(
            "Fusion scores", fusion_file, "agent scores and risk tier", f"doc_{row_id}_fusion",
            [
                ("View record", lambda: _record_dialog("Fusion scores", fusion_file, fusion_record)),
                ("View file", lambda: _parquet_dialog(fusion_file, "test", row_id)),
            ],
        )

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
            # st.table, not st.dataframe: dataframe scrolls inside a fixed height,
            # which hid the amount rows -- the two amounts are the point of the
            # comparison, so every row has to be on screen at once.
            st.table(_entry_diff(this_entry, original))

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


pages = st.navigation(
    [
        st.Page(review_queue, title="Review queue", default=True),
        st.Page(metrics_page, title="Metrics", url_path="metrics"),
    ],
    position="top",
)
pages.run()
