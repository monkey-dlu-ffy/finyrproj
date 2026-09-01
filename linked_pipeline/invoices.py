"""Renders each linked-corpus row as an invoice PDF, rasterizes it, and
degrades the image. 4 visually distinct layouts (not 1) so the document
agent's extractor has to generalize across templates instead of memorizing
one fixed coordinate grid.

reportlab, not the spec's Jinja2+WeasyPrint -- WeasyPrint needs a GTK3
runtime that's routinely broken on Windows, and reportlab+pdf2image is the
exact stack already verified end-to-end on this machine
(experiments/smoke_document.py, Phase 0).
"""

from __future__ import annotations

import io
import random

import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

TEMPLATES = [
    {"name": "classic", "accent": colors.HexColor("#1f3a5f"), "font": "Helvetica", "header_align": "left"},
    {"name": "modern", "accent": colors.HexColor("#0b6e4f"), "font": "Helvetica-Bold", "header_align": "center"},
    {"name": "minimal", "accent": colors.HexColor("#333333"), "font": "Times-Roman", "header_align": "left"},
    {"name": "bold", "accent": colors.HexColor("#8a1c1c"), "font": "Courier", "header_align": "right"},
]


def _draw_invoice(c: canvas.Canvas, row: dict, template: dict) -> None:
    width, height = letter
    accent = template["accent"]
    font = template["font"]

    c.setFillColor(accent)
    c.rect(0, height - 1.1 * inch, width, 1.1 * inch, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont(font, 20)
    title_x = {"left": 0.6 * inch, "center": width / 2 - 1.2 * inch, "right": width - 3.2 * inch}[template["header_align"]]
    c.drawString(title_x, height - 0.7 * inch, "INVOICE")

    c.setFillColor(colors.black)
    c.setFont(font, 10)
    y = height - 1.5 * inch
    lines = [
        f"Vendor: {row['vendor_name']}",
        f"Invoice #: {row['invoice_number']}",
        f"PO #: {row['po_number'] or 'N/A'}",
        f"Invoice Date: {row['doc_date']}",
        f"Posting Date: {row['posting_date']}",
    ]
    for line in lines:
        c.drawString(0.6 * inch, y, line)
        y -= 0.28 * inch

    y -= 0.3 * inch
    c.setStrokeColor(accent)
    c.line(0.6 * inch, y, width - 0.6 * inch, y)
    y -= 0.35 * inch
    c.setFont(font, 11)
    c.drawString(0.6 * inch, y, "Description")
    c.drawString(width - 2.2 * inch, y, "Amount")
    y -= 0.25 * inch
    c.setFont(font, 10)
    c.drawString(0.6 * inch, y, row["description"])
    c.drawString(width - 2.2 * inch, y, f"${row['invoice_amount']:,.2f}")

    y -= 0.5 * inch
    c.setFont(font, 12)
    c.drawString(width - 3 * inch, y, f"TOTAL: ${row['invoice_amount']:,.2f}")


def render_pdf_bytes(row: dict, template_idx: int) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _draw_invoice(c, row, TEMPLATES[template_idx % len(TEMPLATES)])
    c.save()
    return buf.getvalue()


def _add_noise(img: Image.Image, rng: np.random.Generator, sigma: float = 6.0) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    noise = rng.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _add_scan_lines(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    for _ in range(rng.integers(2, 6)):
        y = int(rng.integers(0, img.height))
        draw.line([(0, y), (img.width, y)], fill=(120, 120, 120, 40), width=1)
    return img


def degrade(img: Image.Image, seed: int) -> Image.Image:
    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)

    angle = py_rng.uniform(-2, 2)
    img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    img = _add_noise(img, rng)
    if py_rng.random() < 0.3:
        img = _add_scan_lines(img, rng)

    buf = io.BytesIO()
    quality = py_rng.randint(60, 75)
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resolve_vendor_name(vendor_id: str, vendor_name_by_id: dict[str, str]) -> str:
    """GHOST-* ids (see linked_pipeline/typologies.py::inject_ghost_vendor)
    carry their display name inline since they're deliberately absent from
    vendor_master."""
    if vendor_id.startswith("GHOST-"):
        return vendor_id.split("GHOST-", 1)[1]
    return vendor_name_by_id.get(vendor_id, vendor_id)


def render_and_degrade(row: dict, seed: int) -> Image.Image:
    template_idx = seed % len(TEMPLATES)
    pdf_bytes = render_pdf_bytes(row, template_idx)
    pages = convert_from_bytes(pdf_bytes, dpi=200)
    return degrade(pages[0], seed)
