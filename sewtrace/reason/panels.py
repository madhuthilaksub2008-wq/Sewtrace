"""T16: panel inference. Measured half outlines + photos -> PanelSpec.

Owner: Madhu. The model infers only what the photos cannot show (REQ-RSN-04):
front neck depth, back neck depth, sleeve cap shape and grain line. It answers
with the `PanelInference` model below, sent as structured output (REQ-RSN-02).
Code applies those few numbers to the deterministic draft (`reshape`) and checks
that no measured edge moved more than the config tolerance (REQ-RSN-05). A bad
answer gets one retry, then the fallback is used (REQ-RSN-03): the measured draft with the
front neck depth set to back depth plus a config offset. The front outline of a flat-lay
photo follows the back neckline, so the front neck depth of the draft is not a measurement.

Neck width is measured, so it is never in the answer. The armhole is hidden under
the sleeve in a flat-lay photo, but SPEC does not list it as model output. It stays
the code default (config [contour.armhole]); only the cap shape can adapt to it.
In mock mode the "model" answer is read off the fallback spec. For a synthetic photo (its
folder holds a truth.json with "synthetic": true) the front neck depth comes from that file,
so the mock behaves like a good model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import shapely
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from shapely.geometry import LineString, Polygon

from sewtrace.api.client import (
    ApiError, ApiRequestRejected, BudgetExceeded, CacheMiss, MetaClient, PolicyViolation,
)
from sewtrace.config import get_config
from sewtrace.mock.fixtures import NECKBAND_RATIO, NECKBAND_WIDTH_MM, default_tee_seams
from sewtrace.reason.reshape import (
    SleeveCap, cap_width_mm, fit_cap, neck_depth_mm, with_neck_depth, with_sleeve_cap,
)
from sewtrace.reason.seams import check_seams, edge_length
from sewtrace.types import TEE_EDGES, TEE_PANELS, Edge, Outline, Panel, PanelSpec, PhotoSet
from sewtrace.vision.measure import measure_tee

log = logging.getLogger(__name__)

NOTCH_FRAC = 0.4  # armhole notch, fraction of armhole length up from the underarm
DOUBLE_NOTCH_GAP_MM = 6.0
CUTS = {"front": "1 on fold", "back": "1 on fold", "sleeve": "2 on fold"}
# Edges that must come back exactly as measured. The body fold is compared without its neck end,
# which moves with the neck depth (the fold meets the neck curve at the inferred depth).
# Measurements whose value comes from a fit to the neck curve (vision/measure.py _neck_point).
NECK_CURVE_MEASUREMENTS = ("neck_width", "body_length")
MEASURED_EDGES = {
    "front": ("shoulder", "armhole", "side", "hem"),
    "back": ("shoulder", "armhole", "side", "hem"),
    "sleeve": ("underarm", "hem", "fold"),
}


class PanelInference(BaseModel):
    """The whole model answer (REQ-RSN-02, REQ-RSN-04). No other field is allowed."""

    model_config = ConfigDict(extra="forbid")

    front_neck_depth_mm: float = Field(
        gt=0, description="Front neck depth in mm: from the neck point down to the neck curve at the fold.")
    back_neck_depth_mm: float = Field(
        gt=0, description="Back neck depth in mm: from the neck point down to the neck curve at the fold.")
    sleeve_cap: SleeveCap = Field(description="Shape of the half sleeve cap.")
    grain_line: Literal["vertical", "horizontal"] = Field(
        description="Grain direction for front, back and sleeve. Vertical means parallel to the fold.")


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
    """Photos plus a few measured numbers. The only model output is PanelInference."""
    cfg = get_config()
    seam, inf = cfg.seam, cfg.inference
    sleeve = spec.panel("sleeve")
    cap = fit_cap(sleeve.edge_points("cap"))
    meas = {k: round(v, 1) for k, v in measurements.items()}
    neck_half = float(spec.panel("front").edge_points("neck")[-1, 0])
    arm = edge_length(spec.panel("front"), "armhole") + edge_length(spec.panel("back"), "armhole")
    return "\n".join([
        "You are an expert pattern maker. The three photos show one kids' round-neck t-shirt laid",
        "flat: front view, back view, and one sleeve folded flat. A segmentation model measured the",
        "garment from the photos (accurate to about 2 mm). Every measured edge is fixed: you cannot",
        "change it, and you do not return it.",
        "Infer ONLY these four fields. They are the whole answer:",
        f"1. front_neck_depth_mm: front neck depth, {inf.front_neck_depth_min_mm:g} to "
        f"{inf.front_neck_depth_max_mm:g} mm, and at least {inf.front_minus_back_min_mm:g} mm deeper than "
        "back_neck_depth_mm. The front outline shows the back neckline through the neck opening, so "
        "no draft value is given. Read the front neck edge in the front photo.",
        f"2. back_neck_depth_mm: back neck depth, {inf.back_neck_depth_min_mm:g} to "
        f"{inf.back_neck_depth_max_mm:g} mm. Kids' back necks are shallow (about 15 to 25 mm). "
        f"Use the back photo. Draft: "
        f"{neck_depth_mm(spec.panel('back')):.1f}.",
        "3. sleeve_cap: the folded sleeve hides the cap shape. Give two handle lengths in mm of the",
        "   half cap curve (a cubic Bezier from the cap top to the underarm corner, horizontal",
        "   tangents at both ends): crown_run_mm and underarm_run_mm, each 0 to the cap width. Longer",
        f"   handles make a longer, fuller cap. The cap width is {cap_width_mm(sleeve):.1f} mm. The whole",
        f"   cap (2 x the half cap) must be {seam.cap_ease_min_mm:g} to {seam.cap_ease_max_mm:g} mm longer than",
        f"   the front armhole plus the back armhole ({arm:.1f} mm). Draft handles: "
        f"{cap.crown_run_mm:.1f}, {cap.underarm_run_mm:.1f}.",
        "4. grain_line: vertical or horizontal for front, back and sleeve. Knit tees normally have",
        "   vertical grain (parallel to the fold).",
        f"Half neck width (measured, fixed): {neck_half:.1f} mm.",
        f"Measurements (mm): {json.dumps(meas)}",
        "Return ONLY JSON with exactly these four fields, no prose and no other field.",
    ])


def infer_panels(
    client: MetaClient, photos: PhotoSet, draft: PanelSpec, measurements: dict[str, float],
    *, history: list[dict] | None = None,
) -> PanelSpec:
    """Ask the model for the four inferred fields; apply and check them; fall back to the draft.

    Pass `history` (a list) to receive one record per attempt.
    """
    history = history if history is not None else []
    images = [photos.front, photos.back, photos.sleeve]
    fallback = fallback_spec(draft)
    mock_basis = _mock_basis(fallback, photos.front)
    base_prompt = build_prompt(draft, measurements)
    prompt = base_prompt
    for attempt in (1, 2):
        calls = client.calls
        try:
            answer = client.reason(prompt, images, mock=lambda: _mock_answer(mock_basis),
                                   schema=PanelInference.model_json_schema())
        except (BudgetExceeded, CacheMiss, PolicyViolation, ApiRequestRejected):
            raise  # stop the run: more calls are not allowed, not possible, or the request itself is wrong
        except ApiError as exc:  # truncated at max_tokens, not JSON, timeout, HTTP 429/5xx after retries (REQ-RSN-06)
            log.warning("infer_panels attempt %d: the model call failed: %s", attempt, exc)
            answer, spec, problems = None, None, [f"the model call failed: {exc}"]
        else:
            spec, problems = apply_answer(answer, draft)
        record = {"attempt": attempt, "new_call": client.calls > calls, "problems": problems,
                  "answer": answer}
        history.append(record)
        if spec is not None and not problems:
            record["result"] = "accepted"
            log.info("infer_panels: model answer accepted on attempt %d", attempt)
            return spec
        record["result"] = "retry" if attempt == 1 else "fallback"
        log.warning("infer_panels attempt %d rejected: %s", attempt, problems)
        prompt = base_prompt + (
            "\nYour previous answer was rejected for: " + "; ".join(problems[:10])
            + ". Fix these and return the four fields again."
        )
    log.warning("infer_panels: falling back to the draft with the default front neck depth")
    return fallback


def default_front_depth_mm(back_depth_mm: float) -> float:
    """Front neck depth when the model gives none: back depth plus a config offset, kept in range."""
    cfg = get_config().inference
    depth = back_depth_mm + cfg.fallback_front_minus_back_mm
    return min(max(depth, cfg.front_neck_depth_min_mm), cfg.front_neck_depth_max_mm)


def fallback_spec(draft: PanelSpec) -> PanelSpec:
    """The draft with the front neck depth replaced by the default (REQ-RSN-03).

    Only the front neck is edited: it is the one value the draft gets wrong, because the front
    outline follows the back neckline. Every measured edge is untouched."""
    front = with_neck_depth(draft.panel("front"), default_front_depth_mm(neck_depth_mm(draft.panel("back"))))
    return _assemble([front, draft.panel("back"), draft.panel("sleeve")])


def apply_answer(answer: dict, draft: PanelSpec) -> tuple[PanelSpec | None, list[str]]:
    """Parse (pydantic), apply to the draft and validate. Returns (spec or None, problems)."""
    try:
        inference = PanelInference.model_validate(answer)
    except ValidationError as exc:
        return None, [f"answer rejected by the schema: {_errors(exc)}"]
    problems = inference_problems(inference, draft)
    if problems:
        return None, problems
    spec = apply_inference(draft, inference)
    return spec, validate_spec(spec, draft)


def inference_problems(inference: PanelInference, draft: PanelSpec) -> list[str]:
    """Values outside the ranges from config or outside the sleeve geometry."""
    cfg = get_config().inference
    problems = []
    gap = inference.front_neck_depth_mm - inference.back_neck_depth_mm
    if gap < cfg.front_minus_back_min_mm:
        problems.append(
            f"front_neck_depth_mm {inference.front_neck_depth_mm:g} must be at least "
            f"{cfg.front_minus_back_min_mm:g} mm deeper than back_neck_depth_mm {inference.back_neck_depth_mm:g}")
    for name, depth in (("front", inference.front_neck_depth_mm), ("back", inference.back_neck_depth_mm)):
        lo, hi = getattr(cfg, f"{name}_neck_depth_min_mm"), getattr(cfg, f"{name}_neck_depth_max_mm")
        if not lo <= depth <= hi:
            problems.append(f"{name}_neck_depth_mm {depth:g} is outside {lo:g} to {hi:g}")
    width = cap_width_mm(draft.panel("sleeve"))
    for field in ("crown_run_mm", "underarm_run_mm"):
        value = getattr(inference.sleeve_cap, field)
        if value > width:
            problems.append(f"sleeve_cap.{field} {value:g} is more than the cap width {width:.1f}")
    return problems


def apply_inference(draft: PanelSpec, inference: PanelInference) -> PanelSpec:
    """Draft with the four inferred quantities applied. Nothing else is edited."""
    by = {p.name: p for p in draft.panels}
    panels = [
        with_neck_depth(by["front"], inference.front_neck_depth_mm),
        with_neck_depth(by["back"], inference.back_neck_depth_mm),
        with_sleeve_cap(by["sleeve"], inference.sleeve_cap),
    ]
    return _assemble([p.model_copy(update={"grain": inference.grain_line}) for p in panels])


def refresh_derived(spec: PanelSpec) -> PanelSpec:
    """Recompute notches, neckband and seams after the outlines changed (after the seam loop)."""
    return _assemble([p for p in spec.panels if p.name in TEE_EDGES])


def validate_spec(spec: PanelSpec, draft: PanelSpec) -> list[str]:
    """Problems found (empty list = valid): structure, then measured values (REQ-RSN-05)."""
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
    if not problems:
        problems += measured_changes(draft, spec)
    return problems


def measured_changes(draft: PanelSpec, spec: PanelSpec) -> list[str]:
    """Measured values that differ from the draft by more than the config tolerance (REQ-RSN-05).

    Compares the visible edges, the neck width, the cap end points, the fold, and the
    measurements. The neck depth and the cap shape between its end points may change.
    The two measurements that read the neck curve (neck width, body length) are left out of
    the last step: their fit follows the neck curve shape, which a new neck depth changes by
    1 mm or more on a real outline. The neck point itself is compared above, in 2D, within
    the same tolerance.
    """
    tol = get_config().inference.measured_tol_mm
    problems = []
    for name, edges in MEASURED_EDGES.items():
        before, after = draft.panel(name), spec.panel(name)
        for edge in edges:
            moved = _moved_mm(before.edge_points(edge), after.edge_points(edge))
            if moved > tol:
                problems.append(f"{name}.{edge} moved {moved:.1f} mm from the measured edge (limit {tol:g})")
    for name in ("front", "back"):
        before, after = draft.panel(name), spec.panel(name)
        # Only the part of the fold below both neck starts is measured.
        cut = max(before.edge_points("neck")[0, 1], after.edge_points("neck")[0, 1])
        moved = _moved_mm(_fold_below(before.edge_points("fold"), cut), _fold_below(after.edge_points("fold"), cut))
        n0, n1 = before.edge_points("neck"), after.edge_points("neck")
        if len(n0) != len(n1):
            problems.append(f"{name}.neck changed its point count")
            continue
        # Same x everywhere and the same neck point: the neck width stays as measured.
        moved = max(moved, float(np.abs(n0[:, 0] - n1[:, 0]).max()), float(np.linalg.norm(n0[-1] - n1[-1])))
        if moved > tol:
            problems.append(f"{name} fold or neck width moved {moved:.1f} mm from measured (limit {tol:g})")
    c0, c1 = draft.panel("sleeve").edge_points("cap"), spec.panel("sleeve").edge_points("cap")
    ends = max(float(np.linalg.norm(c0[0] - c1[0])), float(np.linalg.norm(c0[-1] - c1[-1])))
    if ends > tol:
        problems.append(f"sleeve.cap end points moved {ends:.1f} mm from measured (limit {tol:g})")
    if problems:
        return problems
    before, after = _measure(draft), _measure(spec)
    return [f"measurement {k} changed by {after[k] - before[k]:+.1f} mm (limit {tol:g})"
            for k in before if k not in NECK_CURVE_MEASUREMENTS and abs(after[k] - before[k]) > tol]


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


def _moved_mm(before: np.ndarray, after: np.ndarray) -> float:
    """Largest distance between two polylines, in mm, and between their end points.

    Densified, so point counts may differ; the end points catch an edge that slid along itself."""
    shape = float(shapely.hausdorff_distance(LineString(before), LineString(after), densify=0.05))
    ends = max(float(np.linalg.norm(before[0] - after[0])), float(np.linalg.norm(before[-1] - after[-1])))
    return max(shape, ends)


def _fold_below(fold: np.ndarray, cut_y: float) -> np.ndarray:
    """Fold polyline from the hem up to y = cut_y (y grows downward), cut by interpolation."""
    kept = fold[fold[:, 1] > cut_y]
    x_cut = float(np.interp(cut_y, fold[::-1, 1], fold[::-1, 0]))
    return np.vstack([kept, [x_cut, cut_y]])


def _measure(spec: PanelSpec) -> dict[str, float]:
    front, back, sleeve = spec.panel("front"), spec.panel("back"), spec.panel("sleeve")
    return measure_tee(front.outline(), front.edges, back.outline(), back.edges, sleeve.outline(), sleeve.edges)


def _errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in e['loc']) or 'answer'}: {e['msg']}" for e in exc.errors()[:10])


def _mock_basis(fallback: PanelSpec, front_photo: str) -> PanelSpec:
    """Spec the mock reads its answer from: the fallback, with the true front neck depth when the
    photo is synthetic. A real photo never gets a truth value (there is none)."""
    depth = _synthetic_front_depth(Path(front_photo))
    if depth is None:
        return fallback
    return _assemble([with_neck_depth(fallback.panel("front"), depth), fallback.panel("back"),
                      fallback.panel("sleeve")])


def _synthetic_front_depth(front_photo: Path) -> float | None:
    """front_neck_depth from truth.json next to a synthetic photo, else None."""
    try:
        truth = json.loads((front_photo.parent / "truth.json").read_text())
        if truth.get("synthetic") is not True:
            return None
        return float(truth["params"]["front_neck_depth"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _mock_answer(draft: PanelSpec) -> dict:
    """Deterministic stand-in built through the same pydantic model: values read off the given spec."""
    return PanelInference(
        front_neck_depth_mm=round(neck_depth_mm(draft.panel("front")), 1),
        back_neck_depth_mm=round(neck_depth_mm(draft.panel("back")), 1),
        sleeve_cap=fit_cap(draft.panel("sleeve").edge_points("cap")),
        grain_line="vertical",
    ).model_dump()
