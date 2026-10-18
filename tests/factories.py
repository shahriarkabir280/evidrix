"""Builders for test documents, so tests don't depend on binary fixture files."""

import io

from fpdf import FPDF
from pypdf import PdfReader, PdfWriter


def make_pdf(pages: list[str], *, title: str | None = None) -> bytes:
    """A text PDF with one page per entry; an empty string gives a blank page."""
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    if title:
        pdf.set_title(title)
    for text in pages:
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


PAPER_BODY = "We train the generator and the discriminator with an equilibrium term."


def make_paper_pdf(
    title_lines: list[str], *, metadata_title: str | None = None, stamp: str = ""
) -> bytes:
    """A first page like a preprint's: a title in large type above body text."""
    pdf = FPDF()
    if metadata_title:
        pdf.set_title(metadata_title)
    pdf.add_page()
    pdf.set_font("Helvetica", size=17)
    for i, line in enumerate(title_lines):
        pdf.text(30, 30 + 8 * i, line)
    pdf.set_font("Helvetica", size=10)
    pdf.set_xy(30, 60)
    pdf.multi_cell(150, 5, f"{PAPER_BODY} " * 3)
    if stamp:
        pdf.set_font("Helvetica", size=20)
        with pdf.rotation(90, x=10, y=200):
            pdf.text(10, 200, stamp)
    return bytes(pdf.output())


def make_image_only_pdf() -> bytes:
    """Stands in for a scanned PDF: the page has drawings but no text layer."""
    pdf = FPDF()
    pdf.add_page()
    pdf.rect(20, 20, 100, 60, style="F")
    return bytes(pdf.output())


def encrypt_pdf(data: bytes, *, user_password: str, owner_password: str = "owner") -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(data)))
    writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm="AES-256")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
