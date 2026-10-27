"""T22: SVG and DXF cut files at 1:1. Owner: Bala."""

from __future__ import annotations

from pathlib import Path

from sewtrace.types import PanelSpec


def render_svg(spec: PanelSpec, path: Path) -> Path:
    raise NotImplementedError


def render_dxf(spec: PanelSpec, path: Path) -> Path:
    raise NotImplementedError
