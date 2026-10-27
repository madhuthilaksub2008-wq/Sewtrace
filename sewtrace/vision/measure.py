"""T15.4: the six proof measurements from labelled half outlines."""

from __future__ import annotations

from sewtrace.types import Edge, Outline


def measure_tee(
    front: Outline, front_edges: list[Edge], sleeve: Outline, sleeve_edges: list[Edge]
) -> dict[str, float]:
    """Keys = sewtrace.types.MEASUREMENT_KEYS, values in mm (definitions there)."""
    raise NotImplementedError
