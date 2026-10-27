"""T15: SAM mask to a clean half outline in mm.

Owner: Madhu. See sewtrace/types.py for the coordinate convention.
"""

from __future__ import annotations

import numpy as np

from sewtrace.types import Edge, Outline
from sewtrace.vision.marker import Homography


def mask_to_outline(mask: np.ndarray, homography: Homography) -> Outline:
    """Largest external contour of a boolean mask, in the marker mm frame."""
    raise NotImplementedError


def smooth_outline(outline: Outline, spacing_mm: float = 2.0) -> Outline:
    """Resample at even spacing and smooth, keeping sharp corners."""
    raise NotImplementedError


def body_half(outline: Outline) -> Outline:
    """Full body outline -> half panel: find the vertical symmetry axis, mirror
    the two sides into one averaged half, fold on x = 0, neck point at y = 0."""
    raise NotImplementedError


def sleeve_half(outline: Outline) -> Outline:
    """Folded sleeve outline (fold on the left, roughly vertical) -> half panel
    with the fold on x = 0 and the cap top at y = 0."""
    raise NotImplementedError


def label_body_edges(half: Outline) -> list[Edge]:
    """Edges in TEE_EDGES["front"] order: neck, shoulder, armhole, side, hem, fold."""
    raise NotImplementedError


def label_sleeve_edges(half: Outline) -> list[Edge]:
    """Edges in TEE_EDGES["sleeve"] order: cap, underarm, hem, fold."""
    raise NotImplementedError
