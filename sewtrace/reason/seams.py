"""T20: sewability check and adjust loop.

Owner: Madhu. Joined seams must match within SEAM_TOLERANCE_MM (or the seam's
own range, e.g. sleeve cap ease).
"""

from __future__ import annotations

from sewtrace.api.client import MetaClient
from sewtrace.types import Panel, PanelSpec, Seam


def edge_length(panel: Panel, edge: str) -> float:
    raise NotImplementedError


def check_seams(spec: PanelSpec) -> list[Seam]:
    """Seams with len_a_mm / len_b_mm filled in from the current outlines."""
    raise NotImplementedError


def adjust_loop(client: MetaClient, spec: PanelSpec, max_calls: int = 3) -> tuple[PanelSpec, list[dict]]:
    """Model fixes failing seams; deterministic fallback after max_calls. Returns spec + log."""
    raise NotImplementedError
