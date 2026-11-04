"""T22: SVG and DXF cut files at 1:1. Owner: Bala."""

from __future__ import annotations

from pathlib import Path

import ezdxf
import svgwrite
from ezdxf.enums import TextEntityAlignment

from sewtrace.output.layout import BRAND, Sheet, layout
from sewtrace.types import PanelSpec

FONT_PER_MM = 1.35  # Text.size -> font size, same as the PDF
DXF_LAYERS = {"CUT": 7, "STITCH": 8, "NOTCH": 1, "TEXT": 5, "MARK": 3}


def _pts(arr) -> list[tuple[float, float]]:
    return [(round(float(x), 3), round(float(y), 3)) for x, y in arr]


def render_svg(spec: PanelSpec, path: Path, size: str | None = None, sheet: Sheet | None = None) -> Path:
    """One sheet at 1:1: width/height in mm, viewBox in mm."""
    path = Path(path)
    sheet = sheet or layout(spec, size)
    w, h = round(sheet.width, 3), round(sheet.height, 3)
    dwg = svgwrite.Drawing(str(path), size=(f"{w}mm", f"{h}mm"), viewBox=f"0 0 {w} {h}", profile="full")
    dwg.set_desc(title=f"{BRAND} pattern", desc="1 user unit = 1 mm")
    for art in sheet.panels:
        g = dwg.g(id=f"panel-{art.name}", fill="none", stroke="black")
        g.add(dwg.polygon(_pts(art.cut), id=f"{art.name}-cut", stroke_width=0.5))
        for s in art.stitch:
            el = dwg.polygon if art.stitch_closed else dwg.polyline
            g.add(el(_pts(s), stroke_width=0.3, stroke_dasharray="4,2.5", stroke="#555"))
        if art.fold is not None:
            g.add(dwg.polyline(_pts(art.fold), stroke="#1a59b3", stroke_width=0.7,
                               stroke_dasharray="8,2,1.5,2"))
        for line in art.lines():
            g.add(dwg.polyline(_pts(line), stroke_width=0.35))
        for t in art.texts:
            attrs = dict(insert=(round(t.x, 3), round(t.y, 3)), font_size=round(t.size * FONT_PER_MM, 3),
                         font_family="Helvetica, Arial, sans-serif", text_anchor=t.anchor,
                         fill="#1a59b3" if t.text == "PLACE ON FOLD" else "black", stroke="none")
            if t.angle:
                attrs["transform"] = f"rotate({-t.angle:g} {t.x:.3f} {t.y:.3f})"
            g.add(dwg.text(t.text, **attrs))
        dwg.add(g)
    dwg.save()
    return path


def render_dxf(spec: PanelSpec, path: Path, size: str | None = None, sheet: Sheet | None = None) -> Path:
    """DXF in mm ($INSUNITS = 4), y up, closed LWPOLYLINE cut lines on layer CUT."""
    path = Path(path)
    sheet = sheet or layout(spec, size)
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    for name, color in DXF_LAYERS.items():
        doc.layers.add(name, color=color, linetype="DASHED" if name == "STITCH" else "CONTINUOUS")
    msp = doc.modelspace()
    H = sheet.height

    def flip(arr):
        return [(float(x), float(H - y)) for x, y in arr]

    for art in sheet.panels:
        msp.add_lwpolyline(flip(art.cut), close=True, dxfattribs={"layer": "CUT"})
        for s in art.stitch:
            msp.add_lwpolyline(flip(s), close=art.stitch_closed, dxfattribs={"layer": "STITCH"})
        for n in art.notches:
            msp.add_lwpolyline(flip(n), dxfattribs={"layer": "NOTCH"})
        if art.fold is not None:
            msp.add_lwpolyline(flip(art.fold), dxfattribs={"layer": "MARK", "linetype": "DASHDOT"})
        for line in art.fold_bracket + [art.grain] + [a for a in _heads(art)]:
            msp.add_lwpolyline(flip(line), dxfattribs={"layer": "MARK"})
        for t in art.texts:
            align = TextEntityAlignment.BOTTOM_CENTER if t.anchor == "middle" else TextEntityAlignment.BOTTOM_LEFT
            msp.add_text(t.text, height=t.size, rotation=t.angle,
                         dxfattribs={"layer": "TEXT"}).set_placement((t.x, H - t.y), align=align)
    doc.saveas(str(path))
    return path


def _heads(art):
    from sewtrace.output.layout import arrow_heads
    return arrow_heads(art.grain)
