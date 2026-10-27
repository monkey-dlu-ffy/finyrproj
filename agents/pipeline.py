"""Live scoring: one new ledger posting plus its invoice fields, through the
transaction, document and reconciliation agents and the fusion
meta-classifier, computed at request time.

Everything scored here was fit once by scripts/run_fusion.py and saved to
BUNDLE_PATH -- the same fitted objects that produced
experiments/fusion_scores.parquet, not a re-implementation. The batch scores
remain what the paper's metrics are computed from; this module is what lets
a posting that was never in the corpus be scored at all
(scripts/smoke_live_scoring.py checks both: hand-built entries, and parity
against the batch scores).

Invoice fields arrive as structured input (vendor name, amount, invoice
number, PO), as an e-invoicing feed would supply them. There is no OCR in
this path -- an uploaded invoice image is shown to the reviewer, never read.

The two cross-posting rules (duplicate invoice, split purchase) and the
vendor velocity features need the postings already on file. That history is
the full corpus ledger plus every earlier live submission, so submitting the
same invoice twice flags the second.
"""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from agents.document.extract import DocumentAgent
from agents.reconciliation.rules import ReconciliationAgent
from agents.transaction.features_erp import CATEGORICAL_FEATURES, add_features
from config import LINKED_DIR, MODELS_DIR, RANDOM_SEED
from linked_pipeline.invoices import resolve_vendor_name
from linked_pipeline.synth import _DESCRIPTIONS

BUNDLE_PATH = MODELS_DIR / "live_pipeline.joblib"

# Every column either cross-posting rule or the velocity features read.
HISTORY_COLS = ["BELNR", "vendor_id", "invoice_number", "po_number", "posting_date", "doc_date", "DMBTR", "HKONT"]

# Description text -> value tier is fixed by the generator (each text is
# drawn from exactly one tier), so a new posting's tier is a lookup.
TIER_BY_DESCRIPTION = {d: tier for tier, texts in _DESCRIPTIONS.items() for d in texts}
DESCRIPTIONS = sorted(TIER_BY_DESCRIPTION)

# Shown as the transaction agent's evidence: the engineered features a
# reviewer can read without the model.
TRANSACTION_EVIDENCE_FEATURES = [
    "amount_zscore_by_vendor",
    "amount_to_vendor_median_ratio",
    "amount_zscore_by_account",
    "txn_count_7d_vendor",
    "days_since_last_vendor_txn",
    "posting_lag_days",
    "is_weekend",
    "is_round_amount",
]

# The corpus has no posting time; add_features draws hour_of_day at random.
# A live posting carries its own, defaulting to mid-morning.
DEFAULT_HOUR = 10


def risk_tier(score: float, f1_threshold: float) -> str:
    """Two tiers: high = flagged at the val-selected F1 threshold, low = not.

    There is no medium because the scores have no middle. They fall into three
    groups by which rules fired -- none (val fraud rate 1.2%), only missing_po
    (0%), any other rule (99.1%) -- and nothing scores in [0.14, 0.8) on val or
    test. Every three-tier cut tried put one of those groups in medium under a
    label it didn't earn: a 0.3/0.7 band left it empty; "flagged at only the
    looser threshold" filled it with 82 frauds that are 100% fraud; adding
    "a rule fired" mixed those with 240 missing_po-only rows that are 0%."""
    return "high" if score >= f1_threshold else "low"


def load_history() -> pd.DataFrame:
    """The full ledger (train+val+test) with the row_id/split pointers the
    reconciliation evidence cites."""
    parts = [
        pd.read_parquet(LINKED_DIR / f"{split}.parquet", columns=HISTORY_COLS).reset_index(names="row_id").assign(split=split)
        for split in ("train", "val", "test")
    ]
    return pd.concat(parts, ignore_index=True)


class LivePipeline:
    def __init__(self, bundle: dict, history: pd.DataFrame):
        self.meta_clf = bundle["meta_clf"]
        self.feature_cols = bundle["feature_cols"]
        self.f1_threshold = bundle["f1_threshold"]
        self.txn_agent = bundle["txn_agent"]
        self.vendor_stats = bundle["vendor_stats"]
        self.doc_agent = DocumentAgent(stats=bundle["doc_stats"])
        self.recon_agent: ReconciliationAgent = bundle["recon_agent"]
        self.vendor_name_by_id = bundle["vendor_name_by_id"]
        self.category_levels = self.txn_agent.category_levels

        self.history = history
        self._by_vendor = {v: g for v, g in history.groupby("vendor_id")}
        # Sorted invoice numbers + positions: a binary search per lookup,
        # instead of ~520K per-invoice groups held in memory.
        order = np.argsort(history["invoice_number"].to_numpy(), kind="stable")
        self._inv_sorted = history["invoice_number"].to_numpy()[order]
        self._inv_order = order
        self.live = pd.DataFrame(columns=HISTORY_COLS + ["row_id", "split"])
        self._max_belnr = int(history["BELNR"].max())

    @classmethod
    def load(cls) -> "LivePipeline":
        if not BUNDLE_PATH.exists():
            raise FileNotFoundError(f"{BUNDLE_PATH} not found -- run scripts/run_fusion.py first")
        return cls(joblib.load(BUNDLE_PATH), load_history())

    def next_belnr(self) -> int:
        live_max = int(self.live["BELNR"].max()) if len(self.live) else 0
        return max(self._max_belnr, live_max) + 1

    def remember(self, ledger: dict) -> None:
        """Add a scored live posting to the history later postings are checked against."""
        rec = self._ledger_record(ledger)
        row = pd.DataFrame([{c: rec[c] for c in HISTORY_COLS + ["row_id", "split"]}])
        self.live = row if self.live.empty else pd.concat([self.live, row], ignore_index=True)

    def _ledger_record(self, ledger: dict) -> dict:
        rec = dict(ledger)
        rec["posting_date"] = pd.Timestamp(rec["posting_date"])
        rec["doc_date"] = pd.Timestamp(rec["doc_date"])
        rec["DMBTR"] = float(rec["DMBTR"])
        rec["po_number"] = rec.get("po_number") or None
        rec.setdefault("split", "live")
        return rec

    def _vendor_history(self, vendor_id: str, exclude_belnr: int | None) -> pd.DataFrame:
        hist = self._by_vendor.get(vendor_id, self.history.iloc[0:0])
        if len(self.live):
            hist = pd.concat([hist, self.live.loc[self.live["vendor_id"] == vendor_id]], ignore_index=True)
        if exclude_belnr is not None:
            hist = hist.loc[hist["BELNR"] != exclude_belnr]
        return hist

    def _invoice_history(self, invoice_number: str, exclude_belnr: int | None) -> pd.DataFrame:
        lo = np.searchsorted(self._inv_sorted, invoice_number, side="left")
        hi = np.searchsorted(self._inv_sorted, invoice_number, side="right")
        hist = self.history.iloc[self._inv_order[lo:hi]]
        if len(self.live):
            hist = pd.concat([hist, self.live.loc[self.live["invoice_number"] == invoice_number]], ignore_index=True)
        if exclude_belnr is not None:
            hist = hist.loc[hist["BELNR"] != exclude_belnr]
        return hist

    def score(self, ledger: dict, invoice: dict, exclude_belnr: int | None = None) -> dict:
        """`ledger`: vendor_id, DMBTR, posting_date, doc_date, invoice_number,
        po_number, description, the six SAP codes (WAERS, BUKRS, KTOSL, PRCTR,
        BSCHL, HKONT), BELNR, row_id, and optionally hour_of_day.
        `invoice`: vendor_name, amount, invoice_number, po_number.
        `exclude_belnr` drops that posting from the history -- only for
        re-scoring a posting that is already in it (the parity check)."""
        t0 = time.time()
        rec = self._ledger_record(ledger)
        rec["description_tier"] = TIER_BY_DESCRIPTION.get(rec.get("description"))
        rec["vendor_name"] = resolve_vendor_name(rec["vendor_id"], self.vendor_name_by_id)
        row_id = int(rec["row_id"])

        vendor_hist = self._vendor_history(rec["vendor_id"], exclude_belnr)

        # --- transaction agent: the entry's features against its vendor's history
        # History rows only feed the velocity window; only the entry is scored,
        # so they need no SAP codes beyond HKONT.
        # Only the 7-day window before the entry, plus the last posting before
        # that window (days_since_last_vendor_txn), can change its features;
        # engineering the vendor's whole history made a call take seconds.
        day = rec["posting_date"]
        prior = vendor_hist.loc[vendor_hist["posting_date"] <= day]
        in_window = prior["posting_date"] >= day - pd.Timedelta(days=7)
        before = prior.loc[~in_window]
        before = before.loc[before["posting_date"] == before["posting_date"].max()].tail(1)
        velocity_hist = pd.concat([before, prior.loc[in_window]])
        entry_frame = pd.DataFrame([{c: rec.get(c) for c in HISTORY_COLS + CATEGORICAL_FEATURES + ["row_id", "split"]}])
        frame = pd.concat([velocity_hist[HISTORY_COLS + ["row_id", "split"]], entry_frame], ignore_index=True)
        # add_features' sort is stable, so postings on the same day keep this
        # order, and the 7-day count includes only those before the entry. A
        # live posting goes after everything already on file; a corpus posting
        # being re-scored goes back to its place in its split, as in the batch
        # run (vendors never span splits, so row_id orders its history).
        is_live = (frame["split"] == "live").to_numpy()
        frame["_order"] = np.where(is_live, len(self.history) + frame.index.to_numpy(), frame["row_id"].astype(float))
        frame = frame.sort_values("_order", kind="stable").drop(columns="_order")
        engineered = add_features(frame, self.vendor_stats, seed=RANDOM_SEED)
        feat_row = engineered.loc[engineered["BELNR"] == rec["BELNR"]].copy()
        feat_row["hour_of_day"] = int(rec.get("hour_of_day") or DEFAULT_HOUR)
        txn_score = float(self.txn_agent.score(feat_row)[0])
        txn_evidence = [
            {"type": "feature", "detail": f, "value": round(float(feat_row[f].iloc[0]), 4)} for f in TRANSACTION_EVIDENCE_FEATURES
        ]

        # --- document agent: the invoice in isolation
        doc_row = pd.Series(
            {
                "description": rec.get("description"),
                "extracted_amount": float(invoice["amount"]),
                "po_number": invoice.get("po_number") or None,
                "extracted_vendor_name": invoice.get("vendor_name"),
                "extracted_invoice_number": invoice.get("invoice_number"),
                "extracted_po_number": invoice.get("po_number") or None,
            },
            name=row_id,
        )
        doc_env = self.doc_agent.analyze_row(doc_row)

        # --- reconciliation agent: ledger vs invoice, and vs other postings
        recon_row = pd.Series(
            {
                **{k: rec.get(k) for k in HISTORY_COLS + ["row_id", "split", "description", "description_tier", "vendor_name"]},
                "extracted_amount": float(invoice["amount"]),
                "extracted_vendor_name": invoice.get("vendor_name") or rec["vendor_name"],
            }
        )
        po = rec.get("po_number")
        same_po = vendor_hist.loc[vendor_hist["po_number"] == po] if po else vendor_hist.iloc[0:0]
        cross_hist = pd.concat([self._invoice_history(rec["invoice_number"], exclude_belnr), same_po], ignore_index=True)
        cross_hist = cross_hist.drop_duplicates(subset=["BELNR"])
        recon_env = self.recon_agent.analyze_one(recon_row, cross_hist)

        # --- fusion
        scores = {
            "transaction_score": txn_score,
            "document_score": float(doc_env.score),
            "reconciliation_score": float(recon_env.score),
        }
        fused = float(self.meta_clf.predict_proba(pd.DataFrame([scores])[self.feature_cols])[:, 1][0])

        return {
            **scores,
            "fused_score": fused,
            "risk_tier": risk_tier(fused, self.f1_threshold),
            "f1_threshold": self.f1_threshold,
            "transaction_evidence": txn_evidence,
            "document_evidence": doc_env.evidence,
            "recon_evidence": recon_env.evidence,
            "latency_s": time.time() - t0,
        }
