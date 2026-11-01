"""Manufacturing outputs from the pattern: cut plan, cut path, bill of materials, assembly
instructions and one zip of the run. Deterministic geometry and text only. No model call.

The cutter profile (speeds, pierce time) is ILLUSTRATIVE: a sample profile in config.toml
[manufacture], not a real machine. Fabric is cut flat from a roll, so every on-fold half panel is
unfolded first (``unfold_spec``) and the cut polygon includes the seam allowance.

Roll frame, mm: x across the roll (0 to roll_width_mm), y along the roll from its start, y down.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import svgwrite
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from sewtrace.config import AppConfig, ManufactureConfig, get_config
from sewtrace.output.layout import cut_polygon, unfold_spec
from sewtrace.reason.seams import edge_length
from sewtrace.types import Panel, PanelSpec, Seam

ILLUSTRATIVE = "Illustrative cutter profile. Sample profile, not a real machine."
SIDES = 2  # every seam in spec.seams is one side of the garment; the other side is its mirror
TOL_MM = 1e-6  # float tolerance for the overlap and gap checks
POINT_DECIMALS = 2

FILE_NAMES = {
    "cut_path": "cut-path.csv", "toolpath": "toolpath-illustrative.txt", "cutplan": "cutplan.svg",
    "cutplan_json": "cutplan.json", "bom": "bom.csv", "instructions": "instructions.md",
}
BUNDLE_NAME = "bundle.zip"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # the earliest date a zip can hold


class PieceTooWide(ValueError):
    """A piece is wider than the usable roll width. Names the piece and says what to change."""


class NestingError(RuntimeError):
    """The packer produced overlapping pieces or pieces outside the roll. A bug, never a user error."""


# ---- cut plan (nesting) -------------------------------------------------------

@dataclass
class Piece:
    id: str  # "front", "sleeve_1"
    panel: str
    polygon: Polygon  # cut line incl. seam allowance, panel frame, whole piece
    rotation_deg: int  # turn that puts the grain line along the roll length


@dataclass
class Placement:
    piece: Piece
    polygon: Polygon  # roll frame
    x_mm: float  # lower-left corner of the bounding box
    y_mm: float


@dataclass
class CutPlan:
    roll_width_mm: float
    margin_mm: float
    gap_mm: float
    placements: list[Placement]
    used_length_mm: float
    overlap_area_mm2: float
    min_gap_mm: float

    @property
    def fabric_area_mm2(self) -> float:
        return self.roll_width_mm * self.used_length_mm

    @property
    def piece_area_mm2(self) -> float:
        return sum(p.polygon.area for p in self.placements)

    @property
    def utilisation(self) -> float:
        return self.piece_area_mm2 / self.fabric_area_mm2

    @property
    def waste_mm2(self) -> float:
        return self.fabric_area_mm2 - self.piece_area_mm2


def cut_quantity(panel: Panel) -> int:
    """N of "Cut N" (panel.cut is "1 on fold", "2", ...). One when the text has no number."""
    found = re.match(r"\s*(\d+)", panel.cut)
    return max(1, int(found.group(1))) if found else 1


def pieces_to_cut(spec: PanelSpec) -> list[Piece]:
    """Every piece to cut, quantities expanded, as whole pieces (cut flat)."""
    pieces: list[Piece] = []
    for panel in unfold_spec(spec).panels:
        qty = cut_quantity(panel)
        # The grain line must lie along the roll length. A panel drawn with a horizontal grain line is turned once.
        rotation = 0 if panel.grain == "vertical" else 90
        poly = cut_polygon(panel)
        if rotation:
            poly = affinity.rotate(poly, rotation, origin=poly.centroid)
        for i in range(qty):
            pieces.append(Piece(f"{panel.name}_{i + 1}" if qty > 1 else panel.name, panel.name, poly, rotation))
    return pieces


def _check_plan(placements: list[Placement], roll_width_mm: float, margin_mm: float, gap_mm: float) -> tuple[float, float]:
    """(total overlap area, smallest gap between two pieces). Raises NestingError on an overlap, a gap below
    ``gap_mm`` or a piece outside the margins."""
    for p in placements:
        x0, _, x1, _ = p.polygon.bounds
        if x0 < margin_mm - TOL_MM or x1 > roll_width_mm - margin_mm + TOL_MM:
            raise NestingError(f"{p.piece.id} lies outside the usable roll width")
    overlap, min_gap = 0.0, math.inf
    for a, b in combinations(placements, 2):
        overlap += a.polygon.intersection(b.polygon).area
        min_gap = min(min_gap, a.polygon.distance(b.polygon))
    if overlap > TOL_MM:
        raise NestingError(f"nested pieces overlap by {overlap:.3f} mm2")
    if min_gap < gap_mm - TOL_MM:
        raise NestingError(f"two nested pieces are {min_gap:.3f} mm apart, less than the {gap_mm:g} mm gap")
    return overlap, min_gap


def nest(pieces: list[Piece], cfg: ManufactureConfig | None = None) -> CutPlan:
    """Shelf packer on bounding boxes: tallest first, left to right across the roll, a new shelf when a piece
    does not fit. Deterministic. The polygons are then checked with shapely (no overlap, gap kept)."""
    cfg = cfg or get_config().manufacture
    usable = cfg.roll_width_mm - 2 * cfg.margin_mm
    sized = []
    for piece in pieces:
        x0, y0, x1, y1 = piece.polygon.bounds
        if x1 - x0 > usable + TOL_MM:
            raise PieceTooWide(
                f"{piece.id} is {x1 - x0:.0f} mm wide, but the usable roll width is {usable:.0f} mm "
                f"({cfg.roll_width_mm:g} mm roll, {cfg.margin_mm:g} mm margin on each side). "
                "Set a wider roll_width_mm in config.toml [manufacture], or use a smaller size.")
        sized.append((piece, x1 - x0, y1 - y0))
    sized.sort(key=lambda t: (-t[2], t[0].id))

    shelves: list[dict] = []  # each: y, height, next_x
    placements: list[Placement] = []
    for piece, w, h in sized:
        shelf = next((s for s in shelves if s["next_x"] + w <= cfg.roll_width_mm - cfg.margin_mm + TOL_MM), None)
        if shelf is None:
            y = cfg.margin_mm if not shelves else shelves[-1]["y"] + shelves[-1]["height"] + cfg.gap_mm
            shelf = {"y": y, "height": 0.0, "next_x": cfg.margin_mm}
            shelves.append(shelf)
        x0, y0, _, _ = piece.polygon.bounds
        placed = affinity.translate(piece.polygon, shelf["next_x"] - x0, shelf["y"] - y0)
        placements.append(Placement(piece, placed, shelf["next_x"], shelf["y"]))
        shelf["next_x"] += w + cfg.gap_mm
        shelf["height"] = max(shelf["height"], h)
    used = max(s["y"] + s["height"] for s in shelves) + cfg.margin_mm if shelves else 2 * cfg.margin_mm
    overlap, min_gap = _check_plan(placements, cfg.roll_width_mm, cfg.margin_mm, cfg.gap_mm)
    return CutPlan(cfg.roll_width_mm, cfg.margin_mm, cfg.gap_mm, placements, used, overlap,
                   0.0 if min_gap == math.inf else min_gap)


# ---- cut path -----------------------------------------------------------------

@dataclass
class PiecePath:
    piece: str
    panel: str
    points: list[tuple[float, float]]  # closed: the last point is the first


@dataclass
class CutPath:
    paths: list[PiecePath]
    home: tuple[float, float]
    cut_length_mm: float
    travel_length_mm: float
    duration_s: float


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cut_path(plan: CutPlan, cfg: ManufactureConfig | None = None) -> CutPath:
    """One closed path per placed piece. The next piece is the one with a vertex nearest the head; the path
    starts at that vertex. The head starts at, and returns to, the roll origin. Time is ILLUSTRATIVE."""
    cfg = cfg or get_config().manufacture
    home = (0.0, 0.0)
    todo = {p.piece.id: (p, [(float(x), float(y)) for x, y in orient(p.polygon, 1.0).exterior.coords][:-1])
            for p in plan.placements}
    head, travel, cut = home, 0.0, 0.0
    paths: list[PiecePath] = []
    while todo:
        # Ties break on the piece id, so the order never depends on dict or set order.
        key = min(todo, key=lambda k: (min(_dist(head, v) for v in todo[k][1]), k))
        placement, ring = todo.pop(key)
        k0 = min(range(len(ring)), key=lambda i: (_dist(head, ring[i]), i))
        ring = ring[k0:] + ring[:k0]
        closed = ring + [ring[0]]
        travel += _dist(head, ring[0])
        cut += sum(_dist(a, b) for a, b in zip(closed[:-1], closed[1:]))
        paths.append(PiecePath(placement.piece.id, placement.piece.panel, closed))
        head = ring[0]
    travel += _dist(head, home)
    duration = cut / cfg.cut_speed_mm_s + travel / cfg.travel_speed_mm_s + len(paths) * cfg.pierce_s
    return CutPath(paths, home, cut, travel, duration)


# ---- the whole set ------------------------------------------------------------

@dataclass
class Manufacture:
    spec: PanelSpec
    cfg: ManufactureConfig
    plan: CutPlan
    path: CutPath
    files: dict[str, Path] = field(default_factory=dict)

    def summary(self) -> dict:
        p, c = self.plan, self.path
        return {
            "label": ILLUSTRATIVE, "roll_width_mm": p.roll_width_mm, "used_length_mm": p.used_length_mm,
            "fabric_area_mm2": p.fabric_area_mm2, "piece_area_mm2": p.piece_area_mm2,
            "utilisation": p.utilisation, "waste_mm2": p.waste_mm2, "waste_frac": 1 - p.utilisation,
            "piece_count": len(p.placements), "cut_length_mm": c.cut_length_mm,
            "travel_length_mm": c.travel_length_mm, "est_duration_s": c.duration_s,
            "overlap_area_mm2": p.overlap_area_mm2, "min_gap_mm": p.min_gap_mm,
        }


def plan_manufacture(spec: PanelSpec, cfg: AppConfig | None = None) -> Manufacture:
    """Cut plan and cut path for a spec. Raises PieceTooWide before anything is written."""
    mcfg = (cfg or get_config()).manufacture
    plan = nest(pieces_to_cut(spec), mcfg)
    return Manufacture(spec, mcfg, plan, cut_path(plan, mcfg))


# ---- writers ------------------------------------------------------------------

def _num(v: float) -> str:
    return f"{v:.{POINT_DECIMALS}f}"


def write_cut_path_csv(m: Manufacture, path: Path) -> Path:
    """piece,seq,move,x_mm,y_mm. seq counts every row of the file; the move is travel, pierce or cut."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["piece", "seq", "move", "x_mm", "y_mm"])
    seq = 0
    for pp in m.path.paths:
        rows = [("travel", pp.points[0]), ("pierce", pp.points[0])] + [("cut", pt) for pt in pp.points[1:]]
        for move, (x, y) in rows:
            seq += 1
            w.writerow([pp.piece, seq, move, _num(x), _num(y)])
    w.writerow(["home", seq + 1, "travel", _num(m.path.home[0]), _num(m.path.home[1])])
    path.write_text(buf.getvalue())
    return path


def write_toolpath(m: Manufacture, path: Path) -> Path:
    """Generic G-code. ILLUSTRATIVE: not for any machine."""
    cfg = m.cfg
    lines = [
        "; ILLUSTRATIVE ONLY. Generic G-code from a sample cutter profile.",
        "; NOT FOR ANY MACHINE. Not a real machine, not tested on one. Do not run it.",
        f"; Profile: cut {cfg.cut_speed_mm_s:g} mm/s, travel {cfg.travel_speed_mm_s:g} mm/s, pierce {cfg.pierce_s:g} s.",
        f"; Roll {cfg.roll_width_mm:g} mm wide, used length {m.plan.used_length_mm:.1f} mm. X across the roll, Y along it.",
        "G21 ; units: mm", "G90 ; absolute positions",
    ]
    for pp in m.path.paths:
        x0, y0 = pp.points[0]
        lines += [f"; piece {pp.piece}",
                  f"G0 X{_num(x0)} Y{_num(y0)} F{cfg.travel_speed_mm_s * 60:.0f} ; travel",
                  f"M3 ; tool down (pierce {cfg.pierce_s:g} s)"]
        lines += [f"G1 X{_num(x)} Y{_num(y)} F{cfg.cut_speed_mm_s * 60:.0f}" for x, y in pp.points[1:]]
        lines.append("M5 ; tool up")
    lines += [f"G0 X{_num(m.path.home[0])} Y{_num(m.path.home[1])} F{cfg.travel_speed_mm_s * 60:.0f} ; home", "M30 ; end"]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_cutplan_svg(m: Manufacture, path: Path) -> Path:
    """The roll with the nested pieces, 1:1: width and height in mm, one user unit is 1 mm."""
    p = m.plan
    w, h = round(p.roll_width_mm, 3), round(p.used_length_mm, 3)
    dwg = svgwrite.Drawing(str(path), size=(f"{w}mm", f"{h}mm"), viewBox=f"0 0 {w} {h}", profile="full")
    dwg.set_desc(title="Sewtrace cut plan", desc="1 user unit = 1 mm. Grain runs along the roll length (y). " + ILLUSTRATIVE)
    dwg.add(dwg.rect((0, 0), (w, h), id="roll", fill="#f3efe6", stroke="#555", stroke_width=1))
    dwg.add(dwg.rect((p.margin_mm, p.margin_mm), (w - 2 * p.margin_mm, h - 2 * p.margin_mm), fill="none",
                     stroke="#999", stroke_width=0.6, stroke_dasharray="6,4"))
    for pl in p.placements:
        g = dwg.g(id=f"piece-{pl.piece.id}")
        ring = [(round(x, 3), round(y, 3)) for x, y in pl.polygon.exterior.coords][:-1]
        g.add(dwg.polygon(ring, fill="#ffffff", stroke="#111", stroke_width=0.8))
        c = pl.polygon.representative_point()
        g.add(dwg.text(pl.piece.id, insert=(round(c.x, 2), round(c.y, 2)), font_size=18, text_anchor="middle",
                       font_family="Helvetica, Arial, sans-serif", fill="#111"))
        g.add(dwg.polyline([(round(c.x, 2), round(c.y + 14, 2)), (round(c.x, 2), round(c.y + 54, 2))],
                           stroke="#1a59b3", stroke_width=0.8, fill="none"))
        g.add(dwg.text("grain", insert=(round(c.x + 4, 2), round(c.y + 40, 2)), font_size=9, fill="#1a59b3",
                       font_family="Helvetica, Arial, sans-serif"))
        dwg.add(g)
    dwg.add(dwg.text(f"Roll {p.roll_width_mm:g} mm wide, used {p.used_length_mm:.0f} mm. Utilisation {100 * p.utilisation:.1f} %. "
                     "Selvedge on the left and right.", insert=(p.margin_mm, round(h - 4, 2)), font_size=8,
                     font_family="Helvetica, Arial, sans-serif", fill="#555"))
    dwg.save()
    return path


def write_cutplan_json(m: Manufacture, path: Path) -> Path:
    """Small file for the page: the roll, the summary and the cut paths in cutting order."""
    cfg = m.cfg
    data = {
        "label": ILLUSTRATIVE, "roll_width_mm": m.plan.roll_width_mm, "used_length_mm": round(m.plan.used_length_mm, 2),
        "home": list(m.path.home), "cut_speed_mm_s": cfg.cut_speed_mm_s, "travel_speed_mm_s": cfg.travel_speed_mm_s,
        "pierce_s": cfg.pierce_s, "summary": m.summary(),
        "paths": [{"piece": pp.piece, "panel": pp.panel, "points": [[round(x, 1), round(y, 1)] for x, y in pp.points]}
                  for pp in m.path.paths],
    }
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path


def seam_mm(seam: Seam) -> float:
    """The longer of the two sides of a seam, for the thread estimate."""
    return max(seam.len_a_mm, seam.len_b_mm)


def thread_mm(spec: PanelSpec, factor: float) -> float:
    return factor * SIDES * sum(seam_mm(s) for s in spec.seams)


def neckband_size_mm(spec: PanelSpec) -> tuple[float, float] | None:
    """(length, width) of the cut neckband strip, or None when the spec has none."""
    if not any(p.name == "neckband" for p in spec.panels):
        return None
    x0, y0, x1, y1 = cut_polygon(spec.panel("neckband")).bounds
    return x1 - x0, y1 - y0


def write_bom(m: Manufacture, path: Path) -> Path:
    spec, cfg = m.spec, m.cfg
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["item", "quantity", "unit", "note"])
    w.writerow(["fabric", f"{m.plan.used_length_mm / 1000:.3f}", "m",
                f"roll {cfg.roll_width_mm:g} mm wide; used length of the cut plan, margins included"])
    seam_total = SIDES * sum(seam_mm(s) for s in spec.seams)
    w.writerow(["thread", f"{thread_mm(spec, cfg.thread_factor) / 1000:.1f}", "m",
                f"rule of thumb: seam length x {cfg.thread_factor:g} for lockstitch; {len(spec.seams)} seams x {SIDES} sides "
                f"= {seam_total:.0f} mm; hems and the neckband seam are not counted"])
    band = neckband_size_mm(spec)
    if band is not None:
        rule = spec.panel("neckband").length_rule
        w.writerow(["neckband rib", f"{band[0] / 1000:.3f}", "m",
                    f"one strip {band[0]:.0f} x {band[1]:.0f} mm incl. seam allowance" + (f"; {rule}" if rule else "")])
    w.writerow(["notions", "0", "pcs", "none"])
    path.write_text(buf.getvalue())
    return path


# ---- instructions -------------------------------------------------------------

def seam_line(seam: Seam) -> str:
    """"front.shoulder~back.shoulder 106.6 mm"; both lengths when the two sides differ (sleeve ease)."""
    text = f"{seam.a}~{seam.b} {seam.len_a_mm:.1f} mm"
    if abs(seam.len_b_mm - seam.len_a_mm) > 0.05:
        text += f" to {seam.len_b_mm:.1f} mm"
    return text


def _kind(seam: Seam) -> str:
    ref = seam.a + " " + seam.b
    return next((k for k in ("shoulder", "armhole", "cap", "side", "underarm") if k in ref), "other")


def _hem_lines(spec: PanelSpec) -> list[str]:
    """Real hem lengths: the half-panel hem edges, doubled to the whole garment."""
    def whole(panel: str) -> float | None:
        if not any(p.name == panel for p in spec.panels) or not any(e.name == "hem" for e in spec.panel(panel).edges):
            return None
        return 2 * edge_length(spec.panel(panel), "hem")

    front, back, sleeve = whole("front"), whole("back"), whole("sleeve")
    out = []
    if front is not None and back is not None:
        out.append(f"body hem {front + back:.1f} mm (front {front:.1f} mm, back {back:.1f} mm)")
    if sleeve is not None:
        out.append(f"sleeve hem {sleeve:.1f} mm each")
    return out


def instructions_text(m: Manufacture) -> str:
    spec, plan = m.spec, m.plan
    sa = spec.panels[0].seam_allowance_mm if spec.panels else 0.0
    out = ["# Assembly instructions", "",
           "Made from the pattern of this run. The seam lengths are the real lengths in spec.json.", "",
           "## Pieces to cut", "",
           f"The sizes include {sa:g} mm seam allowance. Cut each piece flat, with the grain line parallel to the selvedge.",
           "", "| Piece | Qty | Cut size (mm) | Note |", "| --- | --- | --- | --- |"]
    for panel in unfold_spec(spec).panels:
        x0, y0, x1, y1 = cut_polygon(panel).bounds
        note = "whole piece (the pattern is a half on the fold)" if spec.panel(panel.name).on_fold else "strip"
        out.append(f"| {panel.name} | {cut_quantity(panel)} | {x1 - x0:.0f} x {y1 - y0:.0f} | {note} |")
    out += ["", f"Fabric: {plan.used_length_mm:.0f} mm of a {plan.roll_width_mm:g} mm wide roll. See cutplan.svg.", "",
            "## Sewing order", "", "Sew with right sides together. One step is one seam group.", ""]
    steps: list[tuple[str, list[str]]] = []
    seams = {k: [seam_line(s) for s in spec.seams if _kind(s) == k] for k in ("shoulder", "armhole", "cap", "side", "underarm", "other")}
    if seams["shoulder"]:
        steps.append(("Sew the shoulder seams, one on each side.", seams["shoulder"]))
    band = neckband_size_mm(spec)
    if band is not None:
        rule = spec.panel("neckband").length_rule
        steps.append(("Sew the short ends of the neckband together to make a ring. Sew the ring to the neckline. "
                      "Stretch the neckband to fit the neckline.",
                      [f"neckband strip {band[0]:.1f} mm long" + (f", {rule}" if rule else "")]))
    sleeve = seams["armhole"] + seams["cap"]
    if sleeve:
        steps.append(("Set each sleeve into its armhole. Match the notches of the sleeve cap to the notches of the "
                      "armhole. Spread any extra sleeve cap length evenly between the notches.", sleeve))
    if seams["side"] or seams["underarm"]:
        steps.append(("Sew each side seam and underarm seam in one pass, from the sleeve hem to the body hem.",
                      seams["side"] + seams["underarm"]))
    if seams["other"]:
        steps.append(("Sew the remaining seams.", seams["other"]))
    steps.append(("Turn up the hems and sew them.", _hem_lines(spec)))
    for i, (text, lines) in enumerate(steps, 1):
        out.append(f"{i}. {text}")
        out += [f"   - {line}" for line in lines]
    return "\n".join(out) + "\n"


def write_instructions(m: Manufacture, path: Path) -> Path:
    path.write_text(instructions_text(m))
    return path


# ---- files and bundle ---------------------------------------------------------

def write_files(m: Manufacture, out_dir: Path) -> dict[str, Path]:
    """All manufacturing files into ``out_dir``. Returns key -> path."""
    out_dir = Path(out_dir)
    paths = {k: out_dir / name for k, name in FILE_NAMES.items()}
    write_cut_path_csv(m, paths["cut_path"])
    write_toolpath(m, paths["toolpath"])
    write_cutplan_svg(m, paths["cutplan"])
    write_cutplan_json(m, paths["cutplan_json"])
    write_bom(m, paths["bom"])
    write_instructions(m, paths["instructions"])
    m.files = paths
    return paths


def write_bundle(out_dir: Path, debug_dir: Path | None = None) -> Path:
    """bundle.zip: every file in ``out_dir`` (not the zip itself) and the debug images, sorted, so the same
    run gives the same entry order."""
    out_dir = Path(out_dir)
    target = out_dir / BUNDLE_NAME
    entries = [(p.name, p) for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != BUNDLE_NAME]
    if debug_dir is not None and Path(debug_dir).is_dir():
        entries += [(f"debug/{p.name}", p) for p in sorted(Path(debug_dir).glob("debug_*.png"))]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in entries:
            # A fixed date, so the same run gives the same bytes (a second recording rewrites no file).
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
    return target
