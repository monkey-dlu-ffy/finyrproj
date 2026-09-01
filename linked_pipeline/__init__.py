"""Builds the linked ledger+document corpus from the Schreyer SAP-ERP dataset.

Separate from data_pipeline/ (which builds the Sparkov credit-card reference
track, kept as-is) -- this package targets the real Phase 1 spec: vendor
master + injected typologies + temporal, vendor-disjoint split, feeding
scripts/build_linked_corpus.py.
"""
