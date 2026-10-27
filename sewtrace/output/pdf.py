"""T21: tiled A4 PDF at 1:1. Owner: Bala."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from reportlab.lib.colors import Color, black
from reportlab.pdfgen.canvas import Canvas

from sewtrace.output.layout import (
    A4_MM, BRAND, MARGIN_MM, OVERLAP_MM, PanelArt, Sheet, Text, cut_label, layout, tile_label,
)
from sewtrace.types import PanelSpec

PT_PER_MM = 72.0 / 25.4
PAGE_PT = (A4_MM[0] * PT_PER_MM, A4_MM[1] * PT_PER_MM)
FONT_PER_MM = 1.35  # Text.size (mm, cap-ish) -> font size (mm)
TEST_SQUARE_MM = 100.0
INCH_MM = 25.4
FOLD_BLUE = Color(0.1, 0.35, 0.7)
GREY = Color(0.55, 0.55, 0.55)
LIGHT = Color(0.85, 0.85, 0.85)


def mm(v: float) -> float:
    return v * PT_PER_MM


def cover_test_square_pt() -> tuple[float, float, float, float]:
    """The 100 mm test square on the cover page: x, y (bottom left), w, h in pt."""
    x, top = 15.0, 32.0
    return mm(x), mm(A4_MM[1] - top - TEST_SQUARE_MM), mm(TEST_SQUARE_MM), mm(TEST_SQUARE_MM)


def cover_inch_square_pt() -> tuple[float, float, float, float]:
    x, top = 15.0 + TEST_SQUARE_MM + 15.0, 32.0
    return mm(x), mm(A4_MM[1] - top - INCH_MM), mm(INCH_MM), mm(INCH_MM)


class _Page:
    """Maps sheet mm (y down) onto the current page in pt (y up)."""

    def __init__(self, c: Canvas, x0: float, y0: float, ox: float, oy: float, scale: float = 1.0):
        self.c, self.x0, self.y0, self.ox, self.oy, self.s = c, x0, y0, ox, oy, scale

    def P(self, x: float, y: float) -> tuple[float, float]:
        return mm(self.ox + (x - self.x0) * self.s), mm(A4_MM[1] - self.oy - (y - self.y0) * self.s)

    def poly(self, pts: np.ndarray, closed: bool = False, fill: bool = False) -> None:
        p = self.c.beginPath()
        p.moveTo(*self.P(*pts[0]))
        for q in pts[1:]:
            p.lineTo(*self.P(*q))
        if closed:
            p.close()
        self.c.drawPath(p, stroke=1, fill=1 if fill else 0)

    def text(self, t: Text, font: str = "Helvetica") -> None:
        c = self.c
        c.saveState()
        c.translate(*self.P(t.x, t.y))
        c.rotate(t.angle)
        c.setFont(font, mm(t.size * FONT_PER_MM * self.s))
        (c.drawCentredString if t.anchor == "middle" else c.drawString)(0, 0, t.text)
        c.restoreState()


def _draw_panel(pg: _Page, art: PanelArt, detail: bool = True) -> None:
    c = pg.c
    c.setStrokeColor(black)
    c.setFillColor(black)
    c.setLineWidth(0.8 if detail else 0.4)
    c.setDash()
    pg.poly(art.cut, closed=True)
    if not detail:
        return
    c.setLineWidth(0.5)
    c.setDash([mm(4), mm(2.5)])
    for s in art.stitch:
        pg.poly(s, closed=art.stitch_closed)
    c.setDash()
    if art.fold is not None:
        c.setStrokeColor(FOLD_BLUE)
        c.setLineWidth(1.2)
        c.setDash([mm(8), mm(2), mm(1.5), mm(2)])
        pg.poly(art.fold)
        c.setDash()
        c.setStrokeColor(black)
    c.setLineWidth(0.6)
    for line in art.lines():
        pg.poly(line)
    for t in art.texts:
        c.setFillColor(FOLD_BLUE if t.text == "PLACE ON FOLD" else black)
        pg.text(t, "Helvetica-Bold" if t.text == art.name.upper() else "Helvetica")
    c.setFillColor(black)


def _tile_map(c: Canvas, sheet: Sheet, x: float, top: float, w: float, h: float,
              current: tuple[int, int] | None, panels: bool, labels: bool) -> None:
    """Grid of tiles (sheet scaled into the box), inked tiles white, skipped tiles grey."""
    s = min(w / sheet.width, h / sheet.height)
    pg = _Page(c, 0.0, 0.0, x, top, s)
    inked = set(sheet.tiles())
    c.setLineWidth(0.4)
    c.setDash()
    for r in range(sheet.rows):
        for col in range(sheet.cols):
            x0, y0, x1, y1 = sheet.tile_rect(r, col)
            px, py = pg.P(x0, y1)
            qx, qy = pg.P(x1, y0)
            fill = black if (r, col) == current else (Color(1, 1, 1) if (r, col) in inked else LIGHT)
            c.setFillColor(fill)
            c.setStrokeColor(GREY)
            c.rect(px, py, qx - px, qy - py, stroke=1, fill=1)
    if panels:
        for art in sheet.panels:
            _draw_panel(pg, art, detail=False)
    if labels:
        for r in range(sheet.rows):
            for col in range(sheet.cols):
                x0, y0, x1, y1 = sheet.tile_rect(r, col)
                c.setFillColor(black if (r, col) in inked else GREY)
                c.setFont("Helvetica-Bold", 9)
                cx, cy = pg.P((x0 + x1) / 2, (y0 + y1) / 2)
                c.drawCentredString(cx, cy - 3, tile_label(r, col))
    c.setFillColor(black)
    c.setStrokeColor(black)


def _cover(c: Canvas, spec: PanelSpec, sheet: Sheet, size: str | None) -> None:
    W, H = PAGE_PT
    c.setFont("Helvetica-Bold", 16)
    title = f"{BRAND} pattern: {spec.garment.replace('_', ' ')}" + (f", size {size}" if size else "")
    c.drawString(mm(15), H - mm(20), title)
    c.setFont("Helvetica", 9)
    c.drawString(mm(15), H - mm(26), f"{len(sheet.tiles())} pattern pages + this page. Scale 1:1, units mm.")

    c.setLineWidth(0.5)
    c.setDash()
    c.rect(*cover_test_square_pt(), stroke=1, fill=0)
    x, y, w, h = cover_test_square_pt()
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(x + w / 2, y + h / 2 + 8, "100 mm test square")
    c.setFont("Helvetica", 8)
    c.drawCentredString(x + w / 2, y + h / 2 - 6, "must measure 100 mm;")
    c.drawCentredString(x + w / 2, y + h / 2 - 16, "print at 100% / actual size")
    c.rect(*cover_inch_square_pt(), stroke=1, fill=0)
    ix, iy, iw, ih = cover_inch_square_pt()
    c.drawString(ix, iy - 10, "1 inch")

    c.setFont("Helvetica", 8)
    notes = [
        "1. Print every page at 100% / actual size",
        "   (turn off 'fit to page'). Check the squares.",
        "2. Lay pages out as in the map below.",
        "3. Trim the top and left margins of each page",
        "   to the dashed overlap line, match the",
        "   circled marks and tape.",
        "4. Cut on the solid line. The dashed line is",
        "   the stitch line; seam allowance is included.",
        "5. Notches: snip into the allowance only.",
    ]
    for i, line in enumerate(notes):
        c.drawString(ix, iy - mm(14) - i * 11, line)

    map_top = 32.0 + TEST_SQUARE_MM + 10.0
    c.setFont("Helvetica-Bold", 11)
    c.drawString(mm(15), H - mm(map_top), "Tile map")
    _tile_map(c, sheet, 15.0, map_top + 4.0, 180.0, 85.0, None, panels=True, labels=True)

    top = map_top + 4.0 + 85.0 + 12.0
    c.setFont("Helvetica-Bold", 11)
    c.drawString(mm(15), H - mm(top), "Pieces")
    c.drawString(mm(110), H - mm(top), "Sewing order")
    c.setFont("Helvetica", 9)
    for i, p in enumerate(spec.panels):
        extra = f", {p.length_rule}" if p.length_rule else ""
        c.drawString(mm(15), H - mm(top + 7 + 5 * i),
                     f"{p.name.capitalize()}: {cut_label(p)}, {p.seam_allowance_mm:g} mm allowance{extra}")
    for i, step in enumerate(spec.sewing_order):
        c.drawString(mm(110), H - mm(top + 7 + 5 * i), f"{i + 1}. {step}")


def _alignment_mark(pg: _Page, x: float, y: float) -> None:
    c = pg.c
    cx, cy = pg.P(x, y)
    c.circle(cx, cy, mm(3), stroke=1, fill=0)
    c.line(cx - mm(4), cy, cx + mm(4), cy)
    c.line(cx, cy - mm(4), cx, cy + mm(4))


def _tile(c: Canvas, sheet: Sheet, r: int, col: int) -> None:
    x0, y0, x1, y1 = sheet.tile_rect(r, col)
    pg = _Page(c, x0, y0, MARGIN_MM, MARGIN_MM)
    W, H = PAGE_PT
    c.saveState()
    clip = c.beginPath()
    clip.rect(mm(MARGIN_MM), mm(MARGIN_MM), W - 2 * mm(MARGIN_MM), H - 2 * mm(MARGIN_MM))
    c.clipPath(clip, stroke=0, fill=0)
    for art in sheet.panels:
        _draw_panel(pg, art)
    # Overlap lines and alignment marks (same sheet points on neighbouring tiles).
    c.setStrokeColor(GREY)
    c.setLineWidth(0.4)
    c.setDash([mm(2), mm(2)])
    if col > 0:
        pg.poly(np.array([[x0 + OVERLAP_MM, y0], [x0 + OVERLAP_MM, y1]]))
    if r > 0:
        pg.poly(np.array([[x0, y0 + OVERLAP_MM], [x1, y0 + OVERLAP_MM]]))
    c.setDash()
    c.setStrokeColor(black)
    c.setLineWidth(0.5)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    half = OVERLAP_MM / 2
    if col > 0:
        _alignment_mark(pg, x0 + half, my)
    if col < sheet.cols - 1:
        _alignment_mark(pg, x1 - half, my)
    if r > 0:
        _alignment_mark(pg, mx, y0 + half)
    if r < sheet.rows - 1:
        _alignment_mark(pg, mx, y1 - half)
    c.restoreState()

    # Crop marks at the printable-area corners, in the margin.
    c.setLineWidth(0.4)
    m, L = mm(MARGIN_MM), mm(6)
    for X, Y, sx, sy in ((m, m, -1, -1), (W - m, m, 1, -1), (m, H - m, -1, 1), (W - m, H - m, 1, 1)):
        c.line(X, Y, X + sx * L, Y)
        c.line(X, Y, X, Y + sy * L)
    label = tile_label(r, col)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(mm(MARGIN_MM + 2), mm(3), label)
    c.drawString(mm(MARGIN_MM + 2), H - mm(MARGIN_MM - 3), label)
    c.setFont("Helvetica", 7)
    c.drawString(mm(MARGIN_MM + 14), mm(4), f"{BRAND}  |  page {label}  |  print at 100%, 1:1")
    cell = min(2.0, 5.5 / sheet.rows, 60.0 / sheet.cols)
    _tile_map(c, sheet, A4_MM[0] - MARGIN_MM - cell * sheet.cols * 190 / 180 - 2, A4_MM[1] - MARGIN_MM + 2.0,
              cell * sheet.cols * 190 / 180, cell * sheet.rows * 277 / 267, (r, col), panels=False, labels=False)


def render_pdf(spec: PanelSpec, path: Path, size: str | None = None) -> Path:
    """Tiled A4 pattern with test square, labels, grain, notches, sewing order page."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet = layout(spec, size)
    c = Canvas(str(path), pagesize=PAGE_PT, pageCompression=0)
    c.setTitle(f"{BRAND} pattern" + (f" {size}" if size else ""))
    c.setAuthor(BRAND)
    _cover(c, spec, sheet, size)
    c.showPage()
    for r, col in sheet.tiles():
        _tile(c, sheet, r, col)
        c.showPage()
    c.save()
    return path
