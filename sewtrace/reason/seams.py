"""T20: sewability check and adjust loop.

Owner: Madhu. Joined seams must match within the seam's own range from config
(SPEC 7.6.1), e.g. the whole sleeve cap against front plus back armhole (ease).

Measured geometry is never edited here (REQ-RSN-05). The only quantity that can
change to fix a seam is the sleeve cap shape, one of the inferred quantities of
REQ-RSN-04: it is the one edge that is hidden in the photos. The model proposes the
two cap handle lengths (`SeamAdjustment`); the code fix scales the handles to hit the
middle of the allowed ease range. The cap end points stay fixed. A shoulder, side or
underarm pair that fails comes from two measured edges, so no step edits it: the loop
stops and `require_seams_ok` raises with the pair named. Neck depth and grain do not
touch any seam, so they are not adjustable here.
"""

from __future__ import annotations

import json
import logging
from functools import partial

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from sewtrace.api.client import ApiError, BudgetExceeded, CacheMiss, MetaClient, PolicyViolation
from sewtrace.config import get_config
from sewtrace.reason.reshape import SleeveCap, cap_width_mm, fit_cap, scale_cap_to_length, with_sleeve_cap
from sewtrace.types import Panel, PanelSpec, Seam

log = logging.getLogger(__name__)

CAP_REF = "sleeve.cap"  # the only seam edge the loop may reshape
MOCK_FIX_FRACTION = 0.5  # the mock model fixes half the error, on one call


class SeamAdjustment(BaseModel):
    """The whole seam-loop model answer: a new sleeve cap shape and nothing else."""

    model_config = ConfigDict(extra="forbid")

    sleeve_cap: SleeveCap


ADJUST_SCHEMA = SeamAdjustment.model_json_schema()


class SeamCheckFailed(Exception):
    """Raised when a seam pair is still out of tolerance; nothing may be exported (REQ-SEW-04)."""

    def __init__(self, failed: list[Seam], seam_log: list[dict] | None = None) -> None:
        self.failed = failed
        self.seam_log = seam_log or []
        self.measurements: dict[str, float] | None = None  # set by pipeline.run for eval
        rows = "; ".join(
            f"{seam_name(s)}: {s.len_a_mm:.1f} mm vs {s.len_b_mm:.1f} mm, difference {s.diff_mm:+.1f} mm "
            f"(allowed {s.min_diff_mm:+.1f} to {s.max_diff_mm:+.1f} mm)"
            for s in failed)
        super().__init__(
            f"Seam check failed for {len(failed)} pair(s), so no pattern was written. {rows}. "
            "Retake the photos or fix the measured edges, then run again.")


def require_seams_ok(spec: PanelSpec, seam_log: list[dict] | None = None) -> None:
    """Raise SeamCheckFailed unless every seam pair of the current outlines is in tolerance."""
    failed = [s for s in check_seams(spec) if not s.ok]
    if failed:
        raise SeamCheckFailed(failed, seam_log)


def edge_length(panel: Panel, edge: str) -> float:
    pts = panel.edge_points(edge)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def check_seams(spec: PanelSpec) -> list[Seam]:
    """Seams with len_a_mm / len_b_mm filled in from the current outlines."""
    return [
        s.model_copy(update={"len_a_mm": _ref_length(spec, s.a),
                             "len_b_mm": s.b_count * _ref_length(spec, s.b)})
        for s in spec.seams
    ]


def adjust_loop(client: MetaClient, spec: PanelSpec, max_calls: int | None = None) -> tuple[PanelSpec, list[dict]]:
    """Model fixes a failing cap seam; deterministic fix after max_calls. Returns spec + log.

    Pairs made only of measured edges cannot be fixed: the loop stops and the caller's
    `require_seams_ok` fails loudly (REQ-SEW-04).
    """
    max_calls = client.config.seam.max_adjust_calls if max_calls is None else max_calls
    history: list[dict] = []
    calls, source, applied = 0, "initial", []
    while True:
        seams = check_seams(spec)
        spec = spec.model_copy(update={"seams": seams})
        failing = [s for s in seams if not s.ok]
        history.append({
            "iteration": len(history),
            "source": source,
            "applied": applied,
            "diffs": {seam_name(s): round(s.diff_mm, 2) for s in seams},
            "failing": [seam_name(s) for s in failing],
            # Same edge on both sides: the diff is 0 by construction and proves nothing.
            "equal_by_construction": [seam_name(s) for s in seams if s.a == s.b.removesuffix(":mirror")],
        })
        if not failing or source == "fallback":
            break
        fixable = [s for s in failing if s.b == CAP_REF]
        if not fixable:
            log.warning("adjust_loop: %s fail between measured edges; nothing may change them",
                        [seam_name(s) for s in failing])
            break
        if calls < max_calls:
            prompt = build_adjust_prompt(spec, fixable)
            calls += 1  # a failed call still counts toward the limit of REQ-SEW-02
            try:
                answer = client.reason(prompt, [], mock=partial(_mock_adjust, spec, fixable), schema=ADJUST_SCHEMA)
            except (BudgetExceeded, CacheMiss, PolicyViolation):
                raise
            except ApiError as exc:  # no usable answer: same as an answer that is skipped
                log.warning("adjust_loop: model call %d failed, answer skipped: %s", calls, exc)
                applied = []
            else:
                spec, applied = apply_adjustment(spec, answer)
            source = "model"
        else:
            spec, applied = fallback_fix(spec)
            source = "fallback"
    log.info("adjust_loop: %d model calls, final source %s, failing %s",
             calls, source, history[-1]["failing"])
    return spec, history


def seam_name(seam: Seam) -> str:
    return f"{seam.a}~{seam.b}"


def build_adjust_prompt(spec: PanelSpec, failing: list[Seam]) -> str:
    """Short text-only prompt: fixed prefix first (cache discount), then the failing cap seam."""
    cfg = get_config().seam
    sleeve = spec.panel("sleeve")
    cap = fit_cap(sleeve.edge_points("cap"))
    rows = [
        f"- {s.a} ({s.len_a_mm:.1f} mm) sews to {s.b}"
        f"{f' x {s.b_count}' if s.b_count > 1 else ''} ({s.len_b_mm:.1f} mm): "
        f"b - a = {s.diff_mm:+.1f} mm, allowed {s.min_diff_mm:+.1f} to {s.max_diff_mm:+.1f}"
        for s in failing
    ]
    return "\n".join([
        "You are a pattern maker checking a kids' t-shirt pattern (mm, half panels on fold).",
        "The sleeve cap does not sew to the armholes within tolerance. You may change only the",
        "sleeve cap shape: two handle lengths of the half cap curve (a cubic Bezier from the cap top",
        "to the underarm corner, horizontal tangents at both ends). Longer handles give a longer cap.",
        "Everything else is measured and fixed. Handles run from 0 to the cap width.",
        "The whole sleeve cap (2 x the half cap) must be "
        f"{cfg.cap_ease_min_mm:g} to {cfg.cap_ease_max_mm:g} mm longer than front armhole plus back armhole (ease).",
        "Return ONLY JSON with exactly this field and no other:",
        json.dumps(ADJUST_SCHEMA, separators=(",", ":")),
        "Failing seam:",
        *rows,
        f"Cap width: {cap_width_mm(sleeve):.1f} mm. Current half cap length: {edge_length(sleeve, 'cap'):.1f} mm.",
        f"Current handles: crown_run_mm {cap.crown_run_mm:.1f}, underarm_run_mm {cap.underarm_run_mm:.1f}.",
    ])


def apply_adjustment(spec: PanelSpec, answer: dict) -> tuple[PanelSpec, list[dict]]:
    """Apply the model's cap shape; an answer that fails the schema or the cap width is skipped."""
    try:
        cap = SeamAdjustment.model_validate(answer).sleeve_cap
    except ValidationError as exc:
        log.warning("skipping adjustment that fails the schema: %s", exc.error_count())
        return spec, []
    sleeve = spec.panel("sleeve")
    if max(cap.crown_run_mm, cap.underarm_run_mm) > cap_width_mm(sleeve):
        log.warning("skipping adjustment with handles longer than the cap width: %r", cap)
        return spec, []
    return _set_cap(spec, cap)


def fallback_fix(spec: PanelSpec) -> tuple[PanelSpec, list[dict]]:
    """Code fix: scale the cap handles so the whole cap sits in the middle of its ease range."""
    cap_seam = next((s for s in check_seams(spec) if s.b == CAP_REF and not s.ok), None)
    if cap_seam is None:
        return spec, []
    pts = spec.panel("sleeve").edge_points("cap")
    return _set_cap(spec, scale_cap_to_length(pts, fit_cap(pts), _target_half_cap(cap_seam)))


def _set_cap(spec: PanelSpec, cap: SleeveCap) -> tuple[PanelSpec, list[dict]]:
    before = edge_length(spec.panel("sleeve"), "cap")
    sleeve = with_sleeve_cap(spec.panel("sleeve"), cap)
    applied = [{"sleeve_cap": cap.model_dump(), "from_mm": round(before, 2),
                "to_mm": round(edge_length(sleeve, "cap"), 2)}]
    return _replace_panel(spec, sleeve), applied


def _ref_length(spec: PanelSpec, ref: str) -> float:
    """Length of a ref; "a.x+b.y" sums edges, a ":mirror" suffix measures the mirrored half."""
    total = 0.0
    for term in ref.split("+"):
        term, _, mode = term.partition(":")
        name, edge = term.split(".")
        panel = spec.panel(name)
        if mode == "mirror":
            panel = panel.model_copy(update={"outline_mm": [(-x, y) for x, y in panel.outline_mm]})
        total += edge_length(panel, edge)
    return total


def _target_half_cap(seam: Seam) -> float:
    """Ideal half-cap length: the middle of the allowed range of the whole cap."""
    return (seam.len_a_mm + (seam.min_diff_mm + seam.max_diff_mm) / 2) / seam.b_count


def _mock_adjust(spec: PanelSpec, failing: list[Seam]) -> dict:
    """Deliberately imperfect stand-in: moves the cap only part way to the target."""
    seam = failing[0]
    pts = spec.panel("sleeve").edge_points("cap")
    current = seam.len_b_mm / seam.b_count
    target = current + MOCK_FIX_FRACTION * (_target_half_cap(seam) - current)
    return SeamAdjustment(sleeve_cap=scale_cap_to_length(pts, fit_cap(pts), target)).model_dump()


def _replace_panel(spec: PanelSpec, panel: Panel) -> PanelSpec:
    panels = [panel if p.name == panel.name else p for p in spec.panels]
    return spec.model_copy(update={"panels": panels})
