"""T16: panel inference. Measured half outlines + photos -> PanelSpec.

Owner: Madhu. In mock mode the "model" answer is the deterministic draft
built from the measured outlines, which also serves as the fallback.

The model sees the draft as named edge polylines (downsampled to ~5 mm on
curves, 2 points on straight runs) and answers in the same shape, so code can
rebuild the outline and edge index ranges. Cut, fold, notches, neckband and
seams are always derived by code from the returned geometry.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from shapely.geometry import LineString, Polygon

from sewtrace.api.client import MetaClient
from sewtrace.config import get_config
from sewtrace.mock.fixtures import NECKBAND_RATIO, NECKBAND_WIDTH_MM, default_tee_seams
from sewtrace.reason.seams import check_seams, edge_length
from sewtrace.types import TEE_EDGES, TEE_PANELS, Edge, Outline, Panel, PanelSpec, PhotoSet

log = logging.getLogger(__name__)

PROMPT_STEP_MM = 5.0
SIMPLIFY_MM = 0.3
EDGE_GAP_MM = 2.0
NOTCH_FRAC = 0.4  # armhole notch, fraction of armhole length up from the underarm
DOUBLE_NOTCH_GAP_MM = 6.0
CUTS = {"front": "1 on fold", "back": "1 on fold", "sleeve": "2 on fold"}

_POLYLINE = {"type": "array", "minItems": 2,
             "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}

ANSWER_SCHEMA = {
    "type": "object",
    "required": ["panels"],
    "properties": {
        "panels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "grain", "edges"],
                "properties": {
                    "name": {"enum": ["front", "back", "sleeve"]},
                    "grain": {"enum": ["vertical", "horizontal"]},
                    "edges": {
                        "type": "object",
                        "description": "edge name -> polyline [[x,y],...] in mm, in outline order; "
                                       "each edge starts at the previous edge's last point",
                        "additionalProperties": _POLYLINE,
                    },
                },
            },
        },
        "front_neck_depth_mm": {"type": "number"},
        "back_neck_depth_mm": {"type": "number"},
        "notes": {"type": "string"},
    },
}


def draft_spec(
    front: Outline, front_edges: list[Edge],
    back: Outline, back_edges: list[Edge],
    sleeve: Outline, sleeve_edges: list[Edge],
) -> PanelSpec:
    """Deterministic spec from measured geometry: front, back, sleeve, neckband, seams."""
    panels = [
        _body_panel(name, outline, edges)
        for name, outline, edges in (("front", front, front_edges), ("back", back, back_edges),
                                     ("sleeve", sleeve, sleeve_edges))
    ]
    return _assemble(panels)


def build_prompt(spec: PanelSpec, measurements: dict[str, float]) -> str:
    draft = {"panels": _geometry(spec)["panels"]}
    meas = {k: round(v, 1) for k, v in measurements.items()}
    seam = get_config().seam
    return "\n".join([
        "You are an expert pattern maker. The three photos show one kids' round-neck t-shirt laid",
        "flat: front view, back view, and one sleeve folded flat. A segmentation model measured the",
        "draft below from the photos (accurate to about 2 mm).",
        "Coordinates: mm, x right, y DOWN. front, back and sleeve are HALF panels cut on the fold:",
        "the fold is the line x = 0 and the panel lies at x >= 0. Outlines are stitch lines (no seam",
        "allowance). Body edges run neck, shoulder, armhole, side, hem, fold; sleeve edges run cap,",
        "underarm, hem, fold. Each edge starts at the previous edge's last point; fold ends at neck",
        "(or cap) start.",
        "Infer what the flat photos cannot show directly and correct the draft only where needed:",
        "1. Neck depth: the front neck is usually deeper than the back (kids: back 15-25 mm). The",
        "   back neck is hidden in the front photo; use the back photo. Edit each neck curve.",
        "2. Sleeve cap: the folded sleeve hides the cap shape. Keep a smooth convex cap from (0,0)",
        f"   whose whole length (2 x the half cap) is {seam.cap_ease_min_mm:g} to {seam.cap_ease_max_mm:g} mm",
        "   longer than the front armhole plus the back armhole (ease).",
        "3. Grain: knit tees normally have vertical grain (parallel to the fold) on all three.",
        "Rules: keep every point within 10 mm of the draft unless a photo clearly disagrees; keep",
        "edge names, order and the fold on x = 0; shoulder and side lengths must match front to back;",
        "omit the neckband (code sizes it at 85% of the neckline).",
        f"Measurements (mm): {json.dumps(meas)}",
        f"Draft: {json.dumps(draft, separators=(',', ':'))}",
        "Return ONLY JSON (no prose, no code fence) matching this schema:",
        json.dumps(ANSWER_SCHEMA, separators=(",", ":")),
    ])


def infer_panels(
    client: MetaClient, photos: PhotoSet, draft: PanelSpec, measurements: dict[str, float],
    *, history: list[dict] | None = None,
) -> PanelSpec:
    """Ask the model to refine the draft; validate; fall back to the draft.

    Pass `history` (a list) to receive one record per attempt.
    """
    history = history if history is not None else []
    images = [photos.front, photos.back, photos.sleeve]
    prompt = build_prompt(draft, measurements)
    for attempt in (1, 2):
        calls = client.calls
        answer = client.reason(prompt, images, mock=lambda: _mock_answer(draft))
        spec, problems = parse_answer(answer, draft)
        if spec is not None:
            problems += validate_spec(spec, draft)
        record = {"attempt": attempt, "new_call": client.calls > calls, "problems": problems,
                  "notes": answer.get("notes", "") if isinstance(answer, dict) else ""}
        history.append(record)
        if not problems:
            record["result"] = "accepted"
            log.info("infer_panels: model answer accepted on attempt %d", attempt)
            return spec
        record["result"] = "retry" if attempt == 1 else "fallback"
        log.warning("infer_panels attempt %d invalid: %s", attempt, problems)
        prompt = build_prompt(draft, measurements) + (
            "\nYour previous answer was rejected for: " + "; ".join(problems[:10])
            + ". Fix these and return the full JSON again."
        )
    log.warning("infer_panels: falling back to the measured draft")
    return draft


def validate_spec(spec: PanelSpec, draft: PanelSpec, max_drift_mm: float = 15.0) -> list[str]:
    """Problems found (empty list = valid)."""
    problems = []
    names = [p.name for p in spec.panels]
    problems += [f"missing panel {n}" for n in TEE_PANELS if n not in names]
    for panel in spec.panels:
        pts = np.asarray(panel.outline_mm, dtype=float).reshape(-1, 2)
        if len(pts) < 3:
            problems.append(f"{panel.name}: outline has {len(pts)} points")
            continue
        poly = Polygon(pts)
        if not poly.is_valid:
            problems.append(f"{panel.name}: outline self-intersects")
        if poly.area <= 0:
            problems.append(f"{panel.name}: outline has no area")
        if panel.on_fold and pts[:, 0].min() < -1.0:
            problems.append(f"{panel.name}: on-fold panel crosses x = 0")
        problems += _edge_problems(panel, len(pts))
        ref = next((p for p in draft.panels if p.name == panel.name), None)
        if ref is not None and panel.name in TEE_EDGES and len(ref.outline_mm) >= 3:
            drift = LineString([*pts, pts[0]]).hausdorff_distance(
                LineString([*ref.outline_mm, ref.outline_mm[0]]))
            if drift > max_drift_mm:
                problems.append(f"{panel.name}: outline drifts {drift:.1f} mm from draft")
    return problems


def parse_answer(answer: dict, draft: PanelSpec) -> tuple[PanelSpec | None, list[str]]:
    """Model JSON -> PanelSpec (None plus problems if it cannot be built)."""
    if not isinstance(answer, dict) or not isinstance(answer.get("panels"), list):
        return None, ["answer has no panels list"]
    given = {p.get("name"): p for p in answer["panels"] if isinstance(p, dict)}
    panels, problems = [], []
    for name, edge_names in TEE_EDGES.items():
        raw = given.get(name)
        if raw is None:
            problems.append(f"missing panel {name}")
            continue
        try:
            segs = [np.asarray(raw["edges"][e], dtype=float).reshape(-1, 2) for e in edge_names]
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"{name}: bad or missing edge ({exc})")
            continue
        outline, edges, gaps = _join(list(zip(edge_names, segs)))
        if gaps:
            problems += [f"{name}: {g}" for g in gaps]
            continue
        grain = raw.get("grain", "vertical")
        panel = _body_panel(name, Outline.from_array(outline), edges)
        panels.append(panel.model_copy(update={"grain": grain if grain in ("vertical", "horizontal")
                                                else "vertical"}))
    if problems:
        return None, problems
    return _assemble(panels), []


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for JSON-heavy English)."""
    return len(text) // 4


# ---- helpers -----------------------------------------------------------------

def _body_panel(name: str, outline: Outline, edges: list[Edge]) -> Panel:
    return Panel(name=name, cut=CUTS[name], on_fold=True, outline_mm=list(outline.points),
                 edges=list(edges), seam_allowance_mm=get_config().pattern.seam_allowance_mm, grain="vertical")


def _assemble(panels: list[Panel]) -> PanelSpec:
    """Add notches, the neckband and filled-in seams to front/back/sleeve."""
    by = {p.name: p for p in panels}
    front_arm = edge_length(by["front"], "armhole")
    notch_at = NOTCH_FRAC * front_arm
    half = DOUBLE_NOTCH_GAP_MM / 2
    arm_f = by["front"].edge_points("armhole")[::-1]  # from underarm up
    arm_b = by["back"].edge_points("armhole")[::-1]
    cap = by["sleeve"].edge_points("cap")[::-1]  # from underarm to cap top
    notches = {
        "front": [_point_at(arm_f, notch_at)],
        "back": [_point_at(arm_b, notch_at - half), _point_at(arm_b, notch_at + half)],
        # Cap top = shoulder point; one matching notch (mirrors onto the back half when cut).
        "sleeve": [tuple(map(float, cap[-1])), _point_at(cap, notch_at)],
    }
    panels = [p.model_copy(update={"notches": notches[p.name]}) for p in panels]
    neck = 2 * (edge_length(by["front"], "neck") + edge_length(by["back"], "neck"))
    panels.append(_neckband(neck))
    spec = PanelSpec(panels=panels, seams=default_tee_seams())
    return spec.model_copy(update={"seams": check_seams(spec)})


def _neckband(neckline_mm: float) -> Panel:
    """Same rule as mock.fixtures.truth_spec: rectangle, 85% of the full neckline."""
    length = NECKBAND_RATIO * neckline_mm
    return Panel(
        name="neckband", cut="1", seam_allowance_mm=get_config().pattern.neckband_allowance_mm, grain="horizontal",
        outline_mm=[(0, 0), (length, 0), (length, NECKBAND_WIDTH_MM), (0, NECKBAND_WIDTH_MM)],
        notches=[(length / 2, 0.0)],
        length_rule=f"{int(NECKBAND_RATIO * 100)}% of neckline",
    )


def _point_at(pts: np.ndarray, dist: float) -> tuple[float, float]:
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return (float(np.interp(dist, cum, pts[:, 0])), float(np.interp(dist, cum, pts[:, 1])))


def _join(segs: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray, list[Edge], list[str]]:
    """Edge polylines (sharing end points) -> outline, edge index ranges, gap problems."""
    pts, edges, gaps = [], [], []
    for i, (name, seg) in enumerate(segs):
        if len(seg) < 2:
            gaps.append(f"edge {name} has fewer than 2 points")
            continue
        nxt = segs[(i + 1) % len(segs)][1]
        if len(nxt) and np.linalg.norm(seg[-1] - nxt[0]) > EDGE_GAP_MM:
            gaps.append(f"edge {name} does not meet the next edge")
        start = len(pts)
        pts.extend(seg[:-1])
        edges.append(Edge(name=name, start=start, end=len(pts)))
    if edges:
        edges[-1] = Edge(name=edges[-1].name, start=edges[-1].start, end=0)
    return np.asarray(pts), edges, gaps


def _simplify(seg: np.ndarray, step: float = PROMPT_STEP_MM) -> list[list[float]]:
    """Resample at ~step mm, then drop points within SIMPLIFY_MM of a straight run."""
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
    n = max(1, int(np.ceil(cum[-1] / step)))
    s = np.linspace(0, cum[-1], n + 1)
    res = np.c_[np.interp(s, cum, seg[:, 0]), np.interp(s, cum, seg[:, 1])]
    line = LineString(res).simplify(SIMPLIFY_MM, preserve_topology=False)
    return [[round(x, 1), round(y, 1)] for x, y in line.coords]


def _geometry(spec: PanelSpec) -> dict:
    """Spec as the answer schema: downsampled named edge polylines per panel."""
    return {"panels": [
        {"name": p.name, "grain": p.grain,
         "edges": {e: _simplify(p.edge_points(e)) for e in TEE_EDGES[p.name]}}
        for p in spec.panels if p.name in TEE_EDGES
    ]}


def _mock_answer(draft: PanelSpec) -> dict:
    """Deterministic stand-in: the draft geometry echoed back, with neck depths read off it."""
    answer = _geometry(draft)
    for name in ("front", "back"):
        neck = draft.panel(name).edge_points("neck")
        answer[f"{name}_neck_depth_mm"] = round(float(neck[:, 1].max() - neck[:, 1].min()), 1)
    answer["notes"] = "mock: draft accepted unchanged"
    return answer


def _edge_problems(panel: Panel, n: int) -> list[str]:
    problems = []
    names = [e.name for e in panel.edges]
    problems += [f"{panel.name}: missing edge {e}" for e in TEE_EDGES.get(panel.name, ()) if e not in names]
    for e in panel.edges:
        if not (0 <= e.start < n and 0 <= e.end < n):
            problems.append(f"{panel.name}.{e.name}: index out of range")
        elif edge_length(panel, e.name) <= 0:
            problems.append(f"{panel.name}.{e.name}: non-positive length")
    if panel.name == "neckband":
        pts = np.asarray(panel.outline_mm, dtype=float)
        if np.ptp(pts[:, 0]) <= 0 or np.ptp(pts[:, 1]) <= 0:
            problems.append("neckband: non-positive length or width")
    return problems
