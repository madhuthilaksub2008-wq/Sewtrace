"""T21: tiled A4 PDF at 1:1. Owner: Bala."""

from __future__ import annotations

from pathlib import Path

from sewtrace.types import PanelSpec


def render_pdf(spec: PanelSpec, path: Path) -> Path:
    """Tiled A4 pattern with test square, labels, grain, notches, sewing order page."""
    raise NotImplementedError
