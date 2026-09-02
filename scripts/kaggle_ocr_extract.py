"""Self-contained OCR + field-extraction script for a Kaggle notebook.

Not imported by anything in this repo -- paste target, same pattern as
scripts/kaggle_train_transaction.py. Phase 3 fine-tuning LayoutLMv3 on
DocILE was cut for this session's time budget (a real fine-tune + upload of
DocILE's ~1.6GB is its own multi-hour task); this substitutes a regex/
fuzzy-match extractor over PaddleOCR text, which is enough to feed the
document agent and reconciliation agent honestly -- it's just not a learned
extractor. Revisit LayoutLMv3 fine-tuning as follow-up work.

How to use
----------
1. Zip data/linked/images/ (train/, val/, test/ subfolders + manifest.parquet)
   and data/linked/vendor_master.json locally, upload both as a Kaggle
   dataset (or two).
2. New Kaggle notebook, add that dataset, CPU is fine.
3. Paste this whole file into one cell, Run All.
4. Download extracted_fields.parquet from Output.
5. Drop it into data/linked/images/ locally, tell Claude -- reconciliation
   and document-agent scoring switch from the ground-truth fast path to real
   OCR-extracted fields.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
from paddleocr import PaddleOCR
from rapidfuzz import fuzz, process

IMAGES_ROOT = next(Path("/kaggle/input").rglob("manifest.parquet")).parent
VENDOR_MASTER_PATH = next(Path("/kaggle/input").rglob("vendor_master.json"))
OUT_DIR = Path("/kaggle/working")

INVOICE_RE = re.compile(r"INV-\d{4,6}-\d{4,6}")
PO_RE = re.compile(r"PO-\d{5,7}")
AMOUNT_RE = re.compile(r"\$?([\d,]+\.\d{2})")

manifest = pd.read_parquet(IMAGES_ROOT / "manifest.parquet")
vendor_master = pd.read_json(VENDOR_MASTER_PATH)
vendor_names = vendor_master["vendor_name"].tolist()

ocr = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False, show_log=False)


def extract_fields(text_lines: list[str]) -> dict:
    full_text = "\n".join(text_lines)

    inv_match = INVOICE_RE.search(full_text)
    po_match = PO_RE.search(full_text)

    amounts = [float(m.replace(",", "")) for m in AMOUNT_RE.findall(full_text)]
    total_amount = max(amounts) if amounts else None  # TOTAL line is usually the largest figure on the page

    best_vendor, vendor_score = None, 0
    for line in text_lines:
        match = process.extractOne(line, vendor_names, scorer=fuzz.partial_ratio)
        if match and match[1] > vendor_score:
            best_vendor, vendor_score = match[0], match[1]

    return {
        "extracted_invoice_number": inv_match.group(0) if inv_match else None,
        "extracted_po_number": po_match.group(0) if po_match else None,
        "extracted_amount": total_amount,
        "extracted_vendor_name": best_vendor,
        "vendor_match_score": vendor_score,
        "ocr_line_count": len(text_lines),
    }


records = []
t0 = time.time()
for i, row in manifest.iterrows():
    img_path = IMAGES_ROOT / row["split"] / f"{row['row_id']}.jpg"

    result = ocr.ocr(str(img_path), cls=True)
    lines = [line[1][0] for line in (result[0] or [])] if result and result[0] else []

    fields = extract_fields(lines)
    fields["row_id"] = int(row["row_id"])
    fields["split"] = row["split"]
    records.append(fields)

    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(manifest)} ({time.time() - t0:.1f}s)")

out = pd.DataFrame(records)
out.to_parquet(OUT_DIR / "extracted_fields.parquet", index=False)
print(f"wrote {OUT_DIR / 'extracted_fields.parquet'} ({len(out)} rows, {time.time() - t0:.1f}s total)")

match_rate = out["extracted_invoice_number"].notna().mean()
print(f"invoice# regex match rate: {match_rate:.1%}")
print(f"vendor fuzzy-match mean score: {out['vendor_match_score'].mean():.1f}")
