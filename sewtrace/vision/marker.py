"""T14: ArUco marker detection and homography, photo pixels to millimetres.

Owner: Bala. The mm frame has its origin at the marker's top-left corner,
x along the marker's top edge, y down along its left edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sewtrace.config import MARKER_ID, MARKER_SIZE_MM


@dataclass
class Homography:
    H: np.ndarray  # 3x3, photo px -> mm

    def px_to_mm(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) pixel points to (N, 2) mm points."""
        return _apply(self.H, pts)

    def mm_to_px(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) mm points to (N, 2) pixel points."""
        return _apply(np.linalg.inv(self.H), pts)


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    h = np.c_[pts, np.ones(len(pts))] @ H.T
    return h[:, :2] / h[:, 2:3]


class MarkerNotFound(RuntimeError):
    pass


def detect_marker(image: np.ndarray, marker_id: int = MARKER_ID) -> np.ndarray:
    """(4, 2) float corners in px, order TL, TR, BR, BL. Raises MarkerNotFound."""
    raise NotImplementedError


def homography_from_marker(corners_px: np.ndarray, marker_size_mm: float = MARKER_SIZE_MM) -> Homography:
    raise NotImplementedError


def photo_homography(image: np.ndarray) -> Homography:
    """detect_marker + homography_from_marker."""
    return homography_from_marker(detect_marker(image))
