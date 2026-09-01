"""Reconciliation agent: cross-source agreement checks.

Matches extracted document fields against ledger transactions and reports
disagreements (amount drift, missing counterparty, duplicate invoice, date
inconsistency). Disagreement is the signal -- this agent produces findings,
not probabilities.
"""
