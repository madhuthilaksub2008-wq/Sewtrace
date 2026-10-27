"""T16: panel inference. Measured half outlines + photos -> PanelSpec.

Owner: Madhu. In mock mode the "model" answer is the deterministic draft
built from the measured outlines, which also serves as the fallback.
"""

from __future__ import annotations

from sewtrace.api.client import MetaClient
from sewtrace.types import Edge, Outline, PanelSpec, PhotoSet


def draft_spec(
    front: Outline, front_edges: list[Edge],
    back: Outline, back_edges: list[Edge],
    sleeve: Outline, sleeve_edges: list[Edge],
) -> PanelSpec:
    """Deterministic spec from measured geometry: front, back, sleeve, neckband, seams."""
    raise NotImplementedError


def build_prompt(spec: PanelSpec, measurements: dict[str, float]) -> str:
    raise NotImplementedError


def infer_panels(
    client: MetaClient, photos: PhotoSet, draft: PanelSpec, measurements: dict[str, float]
) -> PanelSpec:
    """Ask the model to refine the draft; validate; fall back to the draft."""
    raise NotImplementedError


def validate_spec(spec: PanelSpec, draft: PanelSpec, max_drift_mm: float = 15.0) -> list[str]:
    """Problems found (empty list = valid)."""
    raise NotImplementedError
