"""Document agent: evidence extracted from scanned invoices and receipts.

Owns the OCR stack (paddleocr + pdf2image) and downstream field extraction.
Consumes data/raw PDFs and images, writes normalised field records to
data/linked for the reconciliation agent to match against ledger rows.
"""
