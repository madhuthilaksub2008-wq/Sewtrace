"""T20: sewability check and adjust loop.

Owner: Madhu. Joined seams must match within the seam's own range from config
(SPEC 7.6.1), e.g. the whole sleeve cap against front plus back armhole (ease).

The model only chooses which edge to change and its target length; the
reshaping itself is deterministic (`set_edge_length`): the edge's end vertex
slides along the edge tangent and the neighbouring edge follows with a linear
falloff, so straight edges stay straight and the far corners stay put.
"""

from __future__ import annotations

import json
import logging
from functools import partial

import numpy as np

from sewtrace.api.client import MetaClient
from sewtrace.config import get_config
from sewtrace.types import Panel, PanelSpec, Seam

log = logging.getLogger(__name__)

# Edges whose end vertex may move: shoulder point, underarm, hem corner, cap underarm.
ADJUSTABLE_EDGES = {"shoulder", "armhole", "side", "cap"}
MOCK_FIX_FRACTION = 0.5  # the mock model fixes half the error, on one seam per call

ADJUST_SCHEMA = {
    "type": "object",
    "required": ["adjustments"],
    "properties": {
        "adjustments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["panel", "edge", "target_length_mm"],
                "properties": {
                    "panel": {"enum": ["front", "back", "sleeve"]},
                    "edge": {"enum": sorted(ADJUSTABLE_EDGES)},
                    "target_length_mm": {"type": "number"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}


class SeamCheckFailed(Exception):
    """Raised when a seam pair is still out of tolerance; nothing may be exported (REQ-SEW-04)."""

    def __init__(self, failed: list[Seam], seam_log: list[dict] | None = None) -> None:
        self.failed = failed
        self.seam_log = seam_log or []
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
    """Model fixes failing seams; deterministic fallback after max_calls. Returns spec + log."""
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
        })
        if not failing or source == "fallback":
            break
        if calls < max_calls:
            prompt = build_adjust_prompt(spec, failing)
            answer = client.reason(prompt, [], mock=partial(_mock_adjust, spec, failing))
            calls += 1
            spec, applied = apply_adjustments(spec, answer)
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
    """Short text-only prompt: failing seams with lengths and allowed ranges."""
    rows = [
        f"- {s.a} ({s.len_a_mm:.1f} mm) sews to {s.b}"
        f"{f' x {s.b_count}' if s.b_count > 1 else ''} ({s.len_b_mm:.1f} mm): "
        f"b - a = {s.diff_mm:+.1f} mm, allowed {s.min_diff_mm:+.1f} to {s.max_diff_mm:+.1f}"
        for s in failing
    ]
    cfg = get_config().seam
    current = {f"{p.name}.{e.name}": round(edge_length(p, e.name), 1)
               for p in spec.panels for e in p.edges if e.name in ADJUSTABLE_EDGES}
    return "\n".join([
        "You are a pattern maker checking a kids' t-shirt pattern (mm, half panels on fold).",
        "These seams do not sew together within tolerance:",
        *rows,
        f"Current adjustable edge lengths: {json.dumps(current)}",
        "Choose which edge(s) to lengthen or shorten. Prefer changing the back panel or the",
        "sleeve cap; keep the front as measured. The whole sleeve cap (2 x the half cap) must be",
        f"{cfg.cap_ease_min_mm:g} to {cfg.cap_ease_max_mm:g} mm longer than front armhole plus back armhole (ease).",
        "Edge lengths and target_length_mm are for the half panel. Code reshapes the edge to your target.",
        "Return ONLY JSON matching this schema:",
        json.dumps(ADJUST_SCHEMA, separators=(",", ":")),
    ])


def apply_adjustments(spec: PanelSpec, answer: dict) -> tuple[PanelSpec, list[dict]]:
    """Apply the model's target lengths; unknown or wild requests are skipped."""
    applied = []
    for adj in answer.get("adjustments", []) if isinstance(answer, dict) else []:
        try:
            name, edge, target = adj["panel"], adj["edge"], float(adj["target_length_mm"])
            panel = spec.panel(name)
            current = edge_length(panel, edge)
        except (KeyError, TypeError, ValueError, StopIteration):
            log.warning("skipping malformed adjustment %r", adj)
            continue
        if edge not in ADJUSTABLE_EDGES or not 0.5 * current <= target <= 1.5 * current:
            log.warning("skipping adjustment out of bounds %r", adj)
            continue
        spec = _replace_panel(spec, set_edge_length(panel, edge, target))
        applied.append({"panel": name, "edge": edge, "from_mm": round(current, 2),
                        "to_mm": round(edge_length(spec.panel(name), edge), 2)})
    return spec, applied


def fallback_fix(spec: PanelSpec, passes: int = 3) -> tuple[PanelSpec, list[dict]]:
    """Code fix: set each failing seam's b edge to the middle of its allowed range."""
    applied = []
    for _ in range(passes):
        seams = check_seams(spec)
        refs = list(dict.fromkeys(s.b for s in seams if not s.ok and _adjustable(s.b)))
        if not refs:
            break
        for ref in refs:
            target = _targets(check_seams(spec))[ref]
            name, edge = ref.split(".")
            current = edge_length(spec.panel(name), edge)
            spec = _replace_panel(spec, set_edge_length(spec.panel(name), edge, target))
            applied.append({"panel": name, "edge": edge, "from_mm": round(current, 2),
                            "to_mm": round(edge_length(spec.panel(name), edge), 2)})
    return spec, applied


def set_edge_length(panel: Panel, edge: str, target: float, iters: int = 50) -> Panel:
    """Slide the edge's end vertex along its tangent until the edge is `target` mm long."""
    pts = panel.edge_points(edge)
    tangent = pts[-1] - pts[-2]
    tangent /= np.linalg.norm(tangent)
    span = 0.5 * edge_length(panel, edge)
    lo, hi = -span, span

    def length_at(t: float) -> float:
        return edge_length(move_edge_end(panel, edge, t * tangent), edge)

    if length_at(lo) >= target:
        hi = lo
    elif length_at(hi) <= target:
        lo = hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if length_at(mid) < target else (lo, mid)
    return move_edge_end(panel, edge, (lo + hi) / 2 * tangent)


def move_edge_end(panel: Panel, edge: str, delta: np.ndarray) -> Panel:
    """Displace the edge's end vertex by delta; this edge and the next follow linearly."""
    pts = np.asarray(panel.outline_mm, dtype=float)
    e = next(x for x in panel.edges if x.name == edge)
    nxt = next(x for x in panel.edges if x.start == e.end and x.name != edge)
    new = pts.copy()
    for ed, rising in ((e, True), (nxt, False)):
        idx = _indices(ed.start, ed.end, len(pts))
        seg = pts[idx]
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
        frac = cum / cum[-1] if cum[-1] > 0 else np.linspace(0, 1, len(seg))
        w = frac if rising else 1 - frac
        new[idx] = seg + w[:, None] * delta
    return panel.model_copy(update={"outline_mm": [(float(x), float(y)) for x, y in new]})


def _indices(start: int, end: int, n: int) -> list[int]:
    return list(range(start, end + 1)) if start <= end else list(range(start, n)) + list(range(end + 1))


def _adjustable(ref: str) -> bool:
    return ":" not in ref and ref.split(".")[1] in ADJUSTABLE_EDGES


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


def _excess(seam: Seam) -> float:
    return max(seam.min_diff_mm - seam.diff_mm, seam.diff_mm - seam.max_diff_mm, 0.0)


def _targets(seams: list[Seam]) -> dict[str, float]:
    """Ideal half-panel length per b edge: middle of the intersection of its seams' allowed ranges."""
    ranges: dict[str, tuple[float, float]] = {}
    for s in seams:
        lo, hi = (s.len_a_mm + s.min_diff_mm) / s.b_count, (s.len_a_mm + s.max_diff_mm) / s.b_count
        if s.b in ranges:
            lo, hi = max(lo, ranges[s.b][0]), min(hi, ranges[s.b][1])
        ranges[s.b] = (lo, hi)
    return {ref: (lo + hi) / 2 for ref, (lo, hi) in ranges.items()}


def _mock_adjust(spec: PanelSpec, failing: list[Seam]) -> dict:
    """Deliberately imperfect stand-in: worst seam only, half the error."""
    fixable = [s for s in failing if _adjustable(s.b)]
    if not fixable:
        return {"adjustments": [], "notes": "mock: no adjustable edge"}
    worst = max(fixable, key=_excess)
    target = _targets(check_seams(spec))[worst.b]
    current = worst.len_b_mm / worst.b_count
    name, edge = worst.b.split(".")
    return {
        "adjustments": [{"panel": name, "edge": edge,
                         "target_length_mm": round(current + MOCK_FIX_FRACTION * (target - current), 2)}],
        "notes": f"mock: moved {worst.b} part way toward {target:.1f} mm",
    }


def _replace_panel(spec: PanelSpec, panel: Panel) -> PanelSpec:
    panels = [panel if p.name == panel.name else p for p in spec.panels]
    return spec.model_copy(update={"panels": panels})
