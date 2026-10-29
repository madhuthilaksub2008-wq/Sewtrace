"""The printable ArUco marker (SPEC 9.1): one A4 page, the black square exactly 100 mm."""

from __future__ import annotations

from io import BytesIO

import cv2
from reportlab.lib.colors import black
from reportlab.pdfgen.canvas import Canvas

from sewtrace.config import MARKER_DICT, MARKER_ID, MARKER_SIZE_MM
from sewtrace.output.layout import A4_MM
from sewtrace.output.pdf import PAGE_PT, mm

TOP_MM = 60.0  # distance from the page top to the marker; the quiet zone around it stays white


def marker_square_pt() -> tuple[float, float, float, float]:
    """The black square: x, y (bottom left), w, h in pt. Centred on the page width."""
    x_mm = (A4_MM[0] - MARKER_SIZE_MM) / 2
    return mm(x_mm), mm(A4_MM[1] - TOP_MM - MARKER_SIZE_MM), mm(MARKER_SIZE_MM), mm(MARKER_SIZE_MM)


def marker_cells() -> list[list[bool]]:
    """Cell grid of the marker, True = black, including the 1 cell border."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT))
    n = dictionary.markerSize + 2
    image = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, n)  # 1 px per cell
    return [[bool(v < 128) for v in row] for row in image]


def marker_pdf_bytes() -> bytes:
    """Vector PDF. Each run of black cells in a row is one rectangle, so no hairline gaps show."""
    buf = BytesIO()
    c = Canvas(buf, pagesize=PAGE_PT, pageCompression=0, invariant=1)
    c.setTitle("Sewtrace marker")
    x0, y0, side, _ = marker_square_pt()
    cells = marker_cells()
    cell = side / len(cells)
    c.setFillColor(black)
    for r, row in enumerate(cells):
        col = 0
        while col < len(row):
            if not row[col]:
                col += 1
                continue
            end = col
            while end < len(row) and row[end]:
                end += 1
            c.rect(x0 + col * cell, y0 + side - (r + 1) * cell, (end - col) * cell, cell, stroke=0, fill=1)
            col = end
    c.setFont("Helvetica", 11)
    cx = PAGE_PT[0] / 2
    c.drawCentredString(cx, mm(A4_MM[1] - 30), "Sewtrace marker")
    c.drawCentredString(cx, mm(A4_MM[1] - 38), "Print at 100% (Actual size). Do not use 'Fit to page'.")
    c.drawCentredString(cx, y0 - mm(10), f"The black square must be {MARKER_SIZE_MM:g} mm wide. Check it with a ruler.")
    c.showPage()
    c.save()
    return buf.getvalue()
