"""End-to-end smoke test for the document pipeline.

reportlab -> PDF -> pdf2image (poppler) -> paddleocr -> text

Proves the whole chain executes on this machine and that known strings survive
the round trip. Run it before designing the document agent around any of these
libraries.

    python experiments/smoke_document.py

Writes nothing to data/ -- it uses a temp dir, so it stays reproducible and
leaves no untracked artifacts.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# paddlepaddle 3.3.1 crashes in its oneDNN backend on this CPU:
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<pir::DoubleAttribute>]  (onednn_instruction.cc:118)
# Falling back to the plain CPU kernels avoids it. Must be set before paddle
# is imported anywhere.
os.environ.setdefault("FLAGS_use_mkldnn", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import seed_everything

# Strings we will look for after the round trip. Chosen to exercise the field
# types the document agent actually cares about.
INVOICE_NO = "INV-4471"
VENDOR = "Northwind Supplies Ltd"
AMOUNT = "48200.00"
DATE = "2026-03-14"


def make_invoice(path: Path) -> None:
    """Synthesise a one-page invoice PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 60, "INVOICE")

    c.setFont("Helvetica", 12)
    rows = [
        f"Invoice No: {INVOICE_NO}",
        f"Vendor: {VENDOR}",
        f"Date: {DATE}",
        "",
        "Description                 Qty        Amount",
        "Industrial fittings         120      48200.00",
        "",
        f"TOTAL USD {AMOUNT}",
    ]
    y = height - 110
    for row in rows:
        c.drawString(50, y, row)
        y -= 22

    c.save()


def rasterise(pdf_path: Path, out_dir: Path):
    """PDF -> PIL images via poppler. Raises if poppler is missing."""
    from pdf2image import convert_from_path

    return convert_from_path(str(pdf_path), dpi=200, output_folder=str(out_dir))


def ocr_image(image_path: Path) -> list[str]:
    """Run PaddleOCR and return recognised text lines.

    PaddleOCR's API moved between 2.x and 3.x; try the current entry point
    first and fall back so this does not silently break on a version bump.
    """
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False,
                    enable_mkldnn=False)

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


def main() -> int:
    seed_everything()
    print("document pipeline smoke test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        pdf_path = tmpdir / "invoice.pdf"
        t0 = time.perf_counter()
        make_invoice(pdf_path)
        print(f"[  ok  ] reportlab    wrote {pdf_path.stat().st_size} B "
              f"in {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        try:
            images = rasterise(pdf_path, tmpdir)
        except Exception as exc:
            print(f"[ FAIL ] pdf2image    {type(exc).__name__}: {exc}")
            print("         is poppler on PATH? see README -> System dependencies")
            return 1
        img_path = tmpdir / "page1.png"
        images[0].save(img_path)
        print(f"[  ok  ] pdf2image    {len(images)} page(s), {images[0].size} px "
              f"in {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        try:
            texts = ocr_image(img_path)
        except Exception as exc:
            print(f"[ FAIL ] paddleocr    {type(exc).__name__}: {exc}")
            return 1
        ocr_s = time.perf_counter() - t0
        print(f"[  ok  ] paddleocr    {len(texts)} text lines in {ocr_s:.2f}s")

    blob = " ".join(texts)
    print("-" * 60)
    print("recognised:", " | ".join(texts[:12]))
    print("-" * 60)

    # Round-trip check on the fields the document agent will extract.
    checks = {
        "invoice no": INVOICE_NO,
        "vendor": VENDOR.split()[0],   # OCR often splits multi-word vendors
        "amount": AMOUNT,
        "date": DATE,
    }
    failed = [name for name, needle in checks.items() if needle not in blob]

    for name, needle in checks.items():
        mark = "  ok  " if needle in blob else " FAIL "
        print(f"[{mark}] {name:<12} expected {needle!r}")

    print("=" * 60)
    if failed:
        print(f"round trip INCOMPLETE -- missed: {', '.join(failed)}")
        print(f"OCR wall time {ocr_s:.1f}s per page on this CPU")
        return 1

    print(f"document pipeline OK -- {ocr_s:.1f}s per page on this CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
