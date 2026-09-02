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
Via the Kaggle CLI (what was actually used -- avoids the web UI entirely,
including its file-download step, which crashed on this machine):
    kaggle datasets create -p <folder containing images/> -r zip
    kaggle kernels push -p <folder containing this script + kernel-metadata.json>
    kaggle kernels status <user>/<kernel-slug>          # poll until COMPLETE
    kaggle kernels output <user>/<kernel-slug> -p <local dir>

Or by hand: new Kaggle notebook, add the images dataset, paste this file
into a script cell, Run All, download extracted_fields.parquet from Output.

Either way: drop extracted_fields.parquet into data/linked/images/ locally,
tell Claude -- reconciliation and document-agent scoring switch from the
ground-truth fast path to real OCR-extracted fields.

Status: attempted, not completed. Two real runs, two different failures:
  - CPU (all 792 images): worked correctly, but at ~44s/image -- close to
    this project's local RAM-thrashing rate even on Kaggle's real hardware,
    apparently just an inherent cost of unbatched full-page PP-OCRv6
    detection on CPU. 100/792 images took 4364s; the full run would have
    been ~8.5 hours.
  - GPU (150-image subset, enable_gpu=true): pip-installing paddlepaddle-gpu
    downgrades several nvidia-cu12 packages (cublas/cudnn/nccl/etc.) to
    versions incompatible with the torch+cu128 build Kaggle's image ships
    preinstalled. paddleocr's own import chain pulls in `modelscope` ->
    `torch`, which then fails: "undefined symbol: ncclCommShrink". Not a
    quick fix -- would need pinning the exact nvidia-cu12 package set torch
    expects before/instead of letting paddlepaddle-gpu's pip resolution
    choose, or a from-scratch conda/docker env, neither of which fit this
    session's time budget.
Decision: abandoned for this session in favor of the ground-truth fast path
already powering the delivered fusion/dashboard results (see
agents/document/extract.py, agents/reconciliation/rules.py). Revisit either
by fixing the CUDA pin above, or by accepting the ~2hr CPU runtime on a
smaller image subset.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Kaggle's default image doesn't ship paddleocr/rapidfuzz, and this runs as a
# script kernel (kernel_type="script"), not a notebook -- `!pip install`
# Jupyter magic only works in .ipynb cells, so install via subprocess instead.
#
# CPU-only PP-OCRv6 detection over a full 200-DPI page measured at ~44s/image
# on Kaggle's CPU (this project's first attempt: 100 images = 4364s) -- close
# to unusable for hundreds of images. Try paddlepaddle-gpu first (several
# CUDA tags, since Kaggle's exact CUDA version isn't known ahead of time);
# fall back to CPU only if every GPU wheel fails to install.
def _sh(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args]).returncode


_has_gpu = False
try:
    _smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    if _smi.returncode == 0:
        for _tag in ("cu126", "cu129", "cu118", "cu120"):
            if _sh("paddlepaddle-gpu==3.3.1", "-i", f"https://www.paddlepaddle.org.cn/packages/stable/{_tag}/") == 0:
                _has_gpu = True
                print(f"installed paddlepaddle-gpu via {_tag}")
                break
except Exception:
    pass
if not _has_gpu:
    _sh("paddlepaddle")
    print("no GPU wheel installed -- falling back to CPU paddlepaddle")
_sh("paddleocr", "rapidfuzz")

os.environ["FLAGS_use_mkldnn"] = "0"  # see README "paddlepaddle oneDNN crash" -- CPU-path safeguard, harmless on GPU

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

def ocr_lines(ocr: PaddleOCR, image_path) -> list[str]:
    """PaddleOCR 3.x's predict() returns dict-like results with rec_texts;
    fall back to the pre-3.x .ocr() tuple format if an older paddleocr ends
    up installed instead -- same dual-path proven in
    experiments/smoke_document.py."""
    texts: list[str] = []
    try:
        results = ocr.predict(str(image_path))
        for res in results:
            texts.extend(res.get("rec_texts", []))
    except AttributeError:
        results = ocr.ocr(str(image_path))
        for page in results:
            for line in page or []:
                texts.append(line[1][0])
    return texts


# Matches the exact config verified working on this project's CPU
# (experiments/smoke_document.py) -- paddleocr 3.x's extra doc-orientation/
# unwarping pipeline stages are what actually trigger the oneDNN crash
# documented in README, not just the core detector; disable them too, not
# just enable_mkldnn.
import paddle  # noqa: E402 -- must come after the pip install block above

print(f"paddle compiled with CUDA: {paddle.is_compiled_with_cuda()}")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)


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

    lines = ocr_lines(ocr, img_path)

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
