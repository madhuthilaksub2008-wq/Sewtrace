"""Shared render geometry for the PDF, SVG, DXF and preview outputs.

Everything here is in sheet mm, x right, y DOWN. Renderers that need y up
(PDF, DXF) flip with `sheet.height - y`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point as SPoint, Polygon, box
from shapely.ops import polylabel, unary_union

from sewtrace.types import Panel, PanelSpec

A4_MM = (210.0, 297.0)
MARGIN_MM = 10.0
OVERLAP_MM = 10.0
PRINT_W = A4_MM[0] - 2 * MARGIN_MM  # 190
PRINT_H = A4_MM[1] - 2 * MARGIN_MM  # 277
STEP_W = PRINT_W - OVERLAP_MM  # 180
STEP_H = PRINT_H - OVERLAP_MM  # 267
GAP_MM = 15.0
INSET_MM = 5.0
NOTCH_IN_MM = 6.0  # into the seam allowance
NOTCH_OUT_MM = 3.0  # past the cut line
FOLD_EPS = 1e-6
BRAND = "Sewtrace"


@dataclass
class Text:
    x: float
    y: float
    text: str
    size: float  # cap height-ish, mm
    angle: float = 0.0  # degrees, counter-clockwise as seen on the page
    anchor: str = "middle"  # "start" | "middle"

    def bounds(self) -> Polygon:
        w = 0.6 * self.size * len(self.text)
        x0 = self.x - (w / 2 if self.anchor == "middle" else 0)
        b = box(x0, self.y - self.size, x0 + w, self.y + 0.3 * self.size)
        if self.angle:
            from shapely import affinity
            b = affinity.rotate(b, -self.angle, origin=(self.x, self.y))
        return b


@dataclass
class PanelArt:
    """One panel's render geometry, placed on the sheet."""

    name: str
    cut: np.ndarray  # closed polygon, last point not repeated
    stitch: list[np.ndarray]  # polylines (open where the fold edge was dropped)
    stitch_closed: bool
    fold: np.ndarray | None  # 2 points on the cut line, or None
    fold_bracket: list[np.ndarray]  # polylines of the "place on fold" bracket
    grain: np.ndarray  # 2 points, arrow heads on both ends
    notches: list[np.ndarray]  # 2-point ticks across the cut line
    texts: list[Text]
    seam_allowance: float

    def lines(self) -> list[np.ndarray]:
        """Everything except the cut line and stitch line, as polylines (incl. arrow heads)."""
        out = list(self.fold_bracket) + [self.grain] + arrow_heads(self.grain) + list(self.notches)
        return out

    def ink(self):
        """Area that counts as pattern ink, for tile skipping."""
        return unary_union([Polygon(self.cut).buffer(NOTCH_OUT_MM + 0.5)] + [t.bounds() for t in self.texts])


@dataclass
class Sheet:
    width: float
    height: float
    cols: int
    rows: int
    panels: list[PanelArt] = field(default_factory=list)

    def tile_rect(self, row: int, col: int) -> tuple[float, float, float, float]:
        """Printable area of tile (row, col) in sheet mm: x0, y0, x1, y1."""
        x0, y0 = col * STEP_W, row * STEP_H
        return x0, y0, x0 + PRINT_W, y0 + PRINT_H

    def tile_has_ink(self, row: int, col: int) -> bool:
        b = box(*self.tile_rect(row, col))
        return any(p.ink().intersects(b) for p in self.panels)

    def tiles(self) -> list[tuple[int, int]]:
        """Non-empty tiles, row-major."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if self.tile_has_ink(r, c)]


def tile_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + row)}{col + 1}"


def cut_label(panel: Panel) -> str:
    return f"Cut {panel.cut}"


# ---- per-panel geometry (panel coordinates) --------------------------------

def cut_polygon(panel: Panel) -> Polygon:
    """Stitch line offset outward by the seam allowance; no allowance along the fold."""
    poly = Polygon(panel.outline_mm)
    if not poly.is_valid:
        poly = poly.buffer(0)
    cut = poly.buffer(panel.seam_allowance_mm, join_style="mitre", mitre_limit=5.0)
    if panel.on_fold:
        minx, miny, maxx, maxy = cut.bounds
        cut = cut.intersection(box(0.0, miny - 1, maxx + 1, maxy + 1))
        if cut.geom_type != "Polygon":
            cut = max(cut.geoms, key=lambda g: g.area)
    return cut


def _ring(poly: Polygon) -> np.ndarray:
    return np.asarray(poly.exterior.coords)[:-1]


def _stitch_lines(panel: Panel) -> tuple[list[np.ndarray], bool]:
    pts = np.asarray(panel.outline_mm, float)
    if not panel.on_fold:
        return [pts], True
    n = len(pts)
    on_fold = [abs(pts[i, 0]) < FOLD_EPS and abs(pts[(i + 1) % n, 0]) < FOLD_EPS for i in range(n)]
    if not any(on_fold):
        return [pts], True
    # Start right after a fold segment so the kept runs are contiguous.
    start = (next(i for i in range(n) if on_fold[i]) + 1) % n
    while on_fold[start % n]:
        start += 1
    lines, cur = [], [pts[start % n]]
    for k in range(n):
        i = (start + k) % n
        if on_fold[i]:
            if len(cur) > 1:
                lines.append(np.asarray(cur))
            cur = [pts[(i + 1) % n]]
        else:
            cur.append(pts[(i + 1) % n])
    if len(cur) > 1:
        lines.append(np.asarray(cur))
    return lines, False


def _fold_segment(cut: np.ndarray) -> np.ndarray | None:
    on = cut[np.abs(cut[:, 0]) < 1e-3]
    if len(on) < 2:
        return None
    return np.array([[0.0, on[:, 1].min()], [0.0, on[:, 1].max()]])


def _default_notches(panel: Panel) -> list[tuple[float, float]]:
    """Arc-length midpoint of the armhole / sleeve cap, when the spec gives none."""
    out = []
    for e in panel.edges:
        if e.name in ("armhole", "cap"):
            line = LineString(panel.edge_points(e.name))
            p = line.interpolate(0.5, normalized=True)
            out.append((p.x, p.y))
    return out


def _notch_ticks(panel: Panel, cut: Polygon) -> list[np.ndarray]:
    sa = panel.seam_allowance_mm
    ring = cut.exterior
    stitch = Polygon(panel.outline_mm)
    ticks = []
    for pt in panel.notches or _default_notches(panel):
        p = np.asarray(pt, float)
        q = ring.interpolate(ring.project(SPoint(p)))
        d = np.array([q.x, q.y]) - p
        if np.linalg.norm(d) < 1e-6:
            continue
        d /= np.linalg.norm(d)
        # Point the tick outward even if the stitch point is slightly off the line.
        if stitch.contains(SPoint(*(p + d))):
            d = -d
        ticks.append(np.array([p + d * (sa - NOTCH_IN_MM), p + d * (sa + NOTCH_OUT_MM)]))
    return ticks


def _vertical_span(poly: Polygon, x: float) -> tuple[float, float]:
    minx, miny, maxx, maxy = poly.bounds
    seg = poly.intersection(LineString([(x, miny - 1), (x, maxy + 1)]))
    if seg.is_empty:
        return miny, maxy
    if seg.geom_type != "LineString":
        seg = max(seg.geoms, key=lambda g: g.length)
    ys = [c[1] for c in seg.coords]
    return min(ys), max(ys)


def panel_art(panel: Panel, size: str | None = None) -> PanelArt:
    """Render geometry for one panel, in panel coordinates (not yet placed)."""
    cut = cut_polygon(panel)
    cut_pts = _ring(cut)
    stitch_lines, closed = _stitch_lines(panel)
    stitch_poly = Polygon(panel.outline_mm)
    minx, miny, maxx, maxy = stitch_poly.bounds
    w, h = maxx - minx, maxy - miny

    fold = _fold_segment(cut_pts) if panel.on_fold else None
    bracket: list[np.ndarray] = []
    texts: list[Text] = []
    if fold is not None:
        y0, y1 = fold[:, 1]
        inset = min(40.0, 0.2 * (y1 - y0))
        a, b = y0 + inset, y1 - inset
        bx = 12.0
        bracket = [np.array([[0.0, a], [bx, a], [bx, b], [0.0, b]])]
        bracket += [np.array([[2.5, a - 1.5], [0.0, a], [2.5, a + 1.5]]),
                    np.array([[2.5, b - 1.5], [0.0, b], [2.5, b + 1.5]])]
        texts.append(Text(bx + 6.0, (a + b) / 2, "PLACE ON FOLD", 4.5, angle=90.0))

    if panel.grain == "vertical":
        gx = minx + (0.3 if panel.on_fold else 0.25) * w
        lo, hi = _vertical_span(stitch_poly, gx)
        span = hi - lo
        grain = np.array([[gx, lo + 0.2 * span], [gx, hi - 0.2 * span]])
        room = stitch_poly.intersection(box(gx + 8, miny, maxx, maxy))
        big = max(5.0, min(10.0, w / 12))
    else:
        gy = miny + 0.22 * h
        grain = np.array([[minx + 0.1 * w, gy], [maxx - 0.1 * w, gy]])
        room = stitch_poly.intersection(box(minx, gy + 3, maxx, maxy))
        big = max(4.0, min(8.0, h / 6))
    small = max(3.0, 0.5 * big)
    if room.geom_type != "Polygon":
        room = max(getattr(room, "geoms", [stitch_poly]), key=lambda g: g.area)
    centre = polylabel(room, tolerance=1.0)
    # Prefer the centroid unless it sits much closer to an edge than the label pole.
    c = room.centroid
    if room.contains(c) and room.exterior.distance(c) >= 0.7 * room.exterior.distance(centre):
        centre = c
    tx, ty = centre.x, centre.y

    info = [cut_label(panel)] + ([f"Size {size}"] if size else []) + (
        [panel.length_rule] if panel.length_rule else [])
    foot = [f"{panel.seam_allowance_mm:g} mm seam allowance incl.", BRAND]
    lines = [(panel.name.upper(), big)] + [(t, small) for t in info] + [(t, 0.8 * small) for t in foot]
    if sum(1.5 * s for _, s in lines) > 0.8 * (room.bounds[3] - room.bounds[1]):
        lines = [(panel.name.upper(), big), ("  ·  ".join(info), small), ("  ·  ".join(foot), 0.8 * small)]
    y = ty - sum(1.5 * s for _, s in lines) / 2
    for text, s in lines:
        y += s
        texts.append(Text(tx, y, text, s))
        y += 0.5 * s
    texts.append(Text(grain[:, 0].mean() + (3.0 if panel.grain == "vertical" else 0.0),
                      grain[:, 1].mean() - (0 if panel.grain == "vertical" else 2.0),
                      "GRAIN", 3.0, angle=90.0 if panel.grain == "vertical" else 0.0,
                      anchor="middle"))

    return PanelArt(
        name=panel.name, cut=cut_pts, stitch=stitch_lines, stitch_closed=closed,
        fold=fold, fold_bracket=bracket, grain=grain, notches=_notch_ticks(panel, cut),
        texts=texts, seam_allowance=panel.seam_allowance_mm,
    )


def arrow_heads(seg: np.ndarray, length: float = 5.0, half: float = 2.0) -> list[np.ndarray]:
    a, b = np.asarray(seg[0], float), np.asarray(seg[1], float)
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    n = np.array([-d[1], d[0]])
    return [np.array([tip - d * length * s + n * half, tip, tip - d * length * s - n * half])
            for tip, s in ((b, 1), (a, -1))]


def translate(art: PanelArt, dx: float, dy: float) -> PanelArt:
    off = np.array([dx, dy])
    return PanelArt(
        name=art.name, cut=art.cut + off, stitch=[s + off for s in art.stitch],
        stitch_closed=art.stitch_closed, fold=None if art.fold is None else art.fold + off,
        fold_bracket=[b + off for b in art.fold_bracket], grain=art.grain + off,
        notches=[n + off for n in art.notches],
        texts=[Text(t.x + dx, t.y + dy, t.text, t.size, t.angle, t.anchor) for t in art.texts],
        seam_allowance=art.seam_allowance,
    )


# ---- sheet layout ----------------------------------------------------------

def _bounds(art: PanelArt) -> tuple[float, float, float, float]:
    return art.ink().bounds


def _shelf_pack(arts: list[PanelArt], width: float) -> tuple[list[PanelArt], float] | None:
    order = sorted(range(len(arts)), key=lambda i: -(_bounds(arts[i])[3] - _bounds(arts[i])[1]))
    placed: list[PanelArt | None] = [None] * len(arts)
    x, y, shelf_h = INSET_MM, INSET_MM, 0.0
    for i in order:
        bx0, by0, bx1, by1 = _bounds(arts[i])
        w, h = bx1 - bx0, by1 - by0
        if INSET_MM + w > width - INSET_MM:
            return None
        if x + w > width - INSET_MM:
            x, y, shelf_h = INSET_MM, y + shelf_h + GAP_MM, 0.0
        placed[i] = translate(arts[i], x - bx0, y - by0)
        x += w + GAP_MM
        shelf_h = max(shelf_h, h)
    return [p for p in placed if p is not None], y + shelf_h + INSET_MM


def _rows_for(height: float) -> int:
    return max(1, math.ceil(max(height - PRINT_H, 0.0) / STEP_H) + 1)


def layout(spec: PanelSpec, size: str | None = None, max_cols: int = 8) -> Sheet:
    """Place all panels on a sheet of whole A4 tiles, picking the column count
    that needs the fewest printed pages."""
    arts = [panel_art(p, size) for p in spec.panels]
    best: Sheet | None = None
    best_key = None
    for cols in range(1, max_cols + 1):
        width = PRINT_W + (cols - 1) * STEP_W
        packed = _shelf_pack(arts, width)
        if packed is None:
            continue
        placed, used_h = packed
        rows = _rows_for(used_h)
        sheet = Sheet(width=width, height=PRINT_H + (rows - 1) * STEP_H, cols=cols, rows=rows, panels=placed)
        key = (len(sheet.tiles()), cols * rows)
        if best_key is None or key < best_key:
            best, best_key = sheet, key
    if best is None:
        raise ValueError("a panel is wider than the widest sheet")
    return best


# ---- PNG preview -----------------------------------------------------------

def _cv_dashed(img, pts, color, thick, dash, gap, closed):
    import cv2
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    on, carry = True, dash
    for a, b in zip(pts[:-1], pts[1:]):
        seg = np.linalg.norm(b - a)
        t = 0.0
        while t < seg:
            step = min(carry, seg - t)
            if on:
                p0, p1 = a + (b - a) * t / seg, a + (b - a) * (t + step) / seg
                cv2.line(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), color, thick, cv2.LINE_AA)
            t += step
            carry -= step
            if carry <= 1e-9:
                on = not on
                carry = dash if on else gap


def _cv_text(img, t: Text, k: float, color=(30, 30, 30)):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = t.size * k / 22.0
    thick = max(1, int(round(scale * 1.5)))
    (tw, th), base = cv2.getTextSize(t.text, font, scale, thick)
    if not t.angle:
        x = t.x * k - (tw / 2 if t.anchor == "middle" else 0)
        cv2.putText(img, t.text, (int(x), int(t.y * k)), font, scale, color, thick, cv2.LINE_AA)
        return
    # Rotated text: draw on a patch, rotate 90, blend.
    patch = np.full((th + base + 4, tw + 4, 3), 255, np.uint8)
    cv2.putText(patch, t.text, (2, th + 2), font, scale, color, thick, cv2.LINE_AA)
    patch = np.rot90(patch, k=int(round(t.angle / 90)) % 4)
    ph, pw = patch.shape[:2]
    x0, y0 = int(t.x * k - pw / 2), int(t.y * k - ph / 2)
    H, W = img.shape[:2]
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + pw, W), min(y0 + ph, H)
    if xe > xs and ye > ys:
        sub = patch[ys - y0 : ye - y0, xs - x0 : xe - x0]
        img[ys:ye, xs:xe] = np.minimum(img[ys:ye, xs:xe], sub)


def render_preview(spec: PanelSpec, path: Path, px_per_mm: float = 2.0, size: str | None = None,
                   show_tiles: bool = True, sheet: Sheet | None = None) -> Path:
    """Rasterise the whole sheet to a PNG for the UI."""
    import cv2
    sheet = sheet or layout(spec, size)
    k = px_per_mm
    W, H = int(math.ceil(sheet.width * k)) + 1, int(math.ceil(sheet.height * k)) + 1
    img = np.full((H, W, 3), 255, np.uint8)

    def P(a):
        return np.round(np.asarray(a) * k).astype(np.int32)

    if show_tiles:
        inked = set(sheet.tiles())
        for r in range(sheet.rows):
            for c in range(sheet.cols):
                x0, y0, x1, y1 = sheet.tile_rect(r, c)
                col = (225, 200, 170) if (r, c) in inked else (235, 235, 235)
                cv2.rectangle(img, tuple(P((x0, y0))), tuple(P((x1, y1))), col, 1)
                cv2.putText(img, tile_label(r, c), tuple(P((x0 + 3, y0 + 9))), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35 * k, col, 1, cv2.LINE_AA)
    for art in sheet.panels:
        cv2.fillPoly(img, [P(art.cut)], (246, 250, 252))
        cv2.polylines(img, [P(art.cut)], True, (20, 20, 20), max(1, int(round(0.6 * k))), cv2.LINE_AA)
        for s in art.stitch:
            _cv_dashed(img, s * k, (90, 90, 90), 1, 4 * k, 2.5 * k, art.stitch_closed)
        if art.fold is not None:
            _cv_dashed(img, art.fold * k, (180, 90, 20), max(1, int(round(0.8 * k))), 8 * k, 3 * k, False)
        for line in art.lines():
            cv2.polylines(img, [P(line)], False, (40, 40, 40), 1, cv2.LINE_AA)
        for t in art.texts:
            _cv_text(img, t, k, (180, 90, 20) if t.text == "PLACE ON FOLD" else (30, 30, 30))
    cv2.imwrite(str(path), img)
    return Path(path)
