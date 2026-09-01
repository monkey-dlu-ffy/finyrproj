"""Specialist audit agents.

Each subpackage owns one evidence modality and exposes a uniform surface so the
coordinator can fan out without knowing internals:

    transaction/     numeric/behavioural signals over ledger rows
    document/        OCR + layout evidence from scanned invoices & receipts
    reconciliation/  cross-source agreement between transactions and documents
    coordinator/     fuses specialist findings into a single ranked case file

Every agent is expected to return evidence, not just a score -- the XAI layer
and the dashboard both read the evidence, not the model internals.
"""
