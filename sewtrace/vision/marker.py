"""T14: ArUco marker detection and homography, photo pixels to millimetres.

Owner: Bala. The mm frame has its origin at the marker's top-left corner,
x along the marker's top edge, y down along its left edge.

The garment lies several marker widths away, so corner error is amplified by
extrapolation (0.01 px at the corners is already ~0.3 mm at the hem). ArUco
gives rough corners; each side is then re-fitted as a line, and finally the
homography is re-solved from every black/white cell edge of the marker. Edge
points are located by the area under the normalised profile, which, unlike a
threshold crossing, does not wobble with the edge's sub-pixel phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from sewtrace.config import MARKER_DICT, MARKER_ID, MARKER_SIZE_MM


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
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT))
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    if ids is None or marker_id not in ids.ravel():
        raise MarkerNotFound(f"ArUco {MARKER_DICT} id {marker_id} not found")
    quad = corners[list(ids.ravel()).index(marker_id)].reshape(4, 2).astype(float)
    pad = 0.3 * np.linalg.norm(quad[2] - quad[0])
    x0, y0 = np.maximum(np.floor(quad.min(axis=0) - pad), 0).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0) + pad).astype(int)
    crop, origin = gray[y0:y1, x0:x1], np.array([x0, y0], float)
    quad -= origin
    for refine in (refine_corners, refine_grid):
        try:
            better = refine(crop, quad)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.abs(better - quad).max() < 3.0:
            quad = better
    return quad + origin


def refine_corners(gray: np.ndarray, quad: np.ndarray, samples: int = 200) -> np.ndarray:
    """Re-fit each side of the square as a line through sub-pixel edge points."""
    img = gray.astype(np.float32)
    lines = [_fit_side(img, quad, quad[i], quad[(i + 1) % 4], samples) for i in range(4)]
    out = np.empty((4, 2))
    for i in range(4):  # corner i joins side i-1 and side i
        out[i] = _intersect(lines[i - 1], lines[i])
    if np.abs(out - quad).max() > 3.0:
        raise ValueError("refinement moved a corner too far")
    return out


def _fit_side(img: np.ndarray, quad: np.ndarray, a: np.ndarray, b: np.ndarray, samples: int) -> np.ndarray:
    """Line (n, c) with n.x = c, through the dark-to-light crossings along side a-b."""
    side = b - a
    length = np.linalg.norm(side)
    u = side / length
    n = np.array([u[1], -u[0]])
    if np.dot(quad.mean(axis=0) - a, n) > 0:
        n = -n  # n points out of the marker, towards the white card
    cell = length / 6.0  # 4x4 bits plus a one-cell black border
    reach = 0.45 * cell
    t = np.linspace(0.15, 0.85, samples) * length
    s = np.linspace(-reach, reach, int(4 * reach) + 1)
    pts = a + t[:, None, None] * u + s[None, :, None] * n  # (samples, len(s), 2)
    prof = cv2.remap(img, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    k = max(2, len(s) // 4)
    dark = np.median(prof[:, :k], axis=1)
    light = np.median(prof[:, -k:], axis=1)
    found = []
    for j in range(samples):
        if light[j] - dark[j] < 40:
            continue
        mid = (dark[j] + light[j]) / 2
        p = prof[j]
        idx = np.nonzero((p[:-1] < mid) & (p[1:] >= mid))[0]
        if len(idx) != 1:
            continue
        i = idx[0]
        off = s[i] + (mid - p[i]) / (p[i + 1] - p[i]) * (s[i + 1] - s[i])
        found.append(a + t[j] * u + off * n)
    if len(found) < samples // 3:
        raise ValueError("too few edge points")
    line = _fit_line(np.asarray(found))
    # Second pass: area under the normalised profile, centred on the first line.
    # Unlike a threshold crossing it does not wobble with the sub-pixel phase.
    normal, c = line[:2], line[2]
    a = a - (normal @ a - c) * normal
    n = normal * np.sign(normal @ n)
    u = np.array([-n[1], n[0]]) * np.sign(np.array([-n[1], n[0]]) @ u)
    w, ds = 5.0, 0.125
    s = np.arange(-w, w + ds / 2, ds)
    pts = a + t[:, None, None] * u + s[None, :, None] * n
    prof = cv2.remap(img, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    lo, hi = float(np.median(dark)), float(np.median(light))
    frac = np.clip((hi - prof) / (hi - lo), -0.5, 1.5)
    off = -w + (frac[:, :-1] + frac[:, 1:]).sum(axis=1) * ds / 2
    good = np.abs(off) < w - 2.0
    return _fit_line(a + t[good, None] * u + off[good, None] * n)


def _fit_line(pts: np.ndarray) -> np.ndarray:
    """Total least squares line (n, c), n.x = c, with one round of outlier removal."""
    for _ in range(2):
        centre = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centre)
        normal = vt[1]
        resid = (pts - centre) @ normal
        pts = pts[np.abs(resid) < max(3 * resid.std(), 0.05)]
    return np.r_[normal, normal @ pts.mean(axis=0)]


def refine_grid(
    gray: np.ndarray, corners: np.ndarray, marker_id: int = MARKER_ID, marker_size_mm: float = MARKER_SIZE_MM
) -> np.ndarray:
    """Re-solve the corners from every black/white cell edge of the marker.

    Each edge point is located across its edge, then the homography is fitted
    so the points land on their grid lines (x = j * cell or y = i * cell).
    """
    img = gray.astype(np.float32)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT))
    white = np.ones((8, 8), bool)
    white[1:7, 1:7] = cv2.aruco.generateImageMarker(dictionary, marker_id, 6) > 127
    cell = marker_size_mm / 6
    to_px = np.linalg.inv(homography_from_marker(corners, marker_size_mm).H)
    px_per_mm = np.linalg.norm(np.diff(corners, axis=0), axis=1).mean() / marker_size_mm
    t = np.linspace(0.2, 0.8, max(8, int(0.6 * cell * px_per_mm)))
    pts, axis, value = [], [], []
    for k in range(7):
        for m in range(6):
            for ax, first, second in ((0, white[m + 1, k], white[m + 1, k + 1]), (1, white[k, m + 1], white[k + 1, m + 1])):
                if first == second:
                    continue
                along = (m + t) * cell
                fixed = np.full_like(along, k * cell)
                mm = np.c_[fixed, along] if ax == 0 else np.c_[along, fixed]
                d = np.zeros(2)
                d[ax] = 1.0 if second else -1.0  # towards the white cell
                found = _edge_points(img, _apply(to_px, mm), _apply(to_px, mm + 0.1 * d))
                pts.append(found)
                axis.append(np.full(len(found), ax))
                value.append(np.full(len(found), k * cell))
    pts, axis, value = np.vstack(pts), np.concatenate(axis), np.concatenate(value)
    ok = np.isfinite(pts).all(axis=1)
    if ok.sum() < 100:
        raise ValueError("too few grid edge points")
    H = _fit_gridlines(pts[ok], axis[ok], value[ok])
    s = marker_size_mm
    return _apply(np.linalg.inv(H), np.float64([[0, 0], [s, 0], [s, s], [0, s]]))


def _edge_points(img: np.ndarray, p: np.ndarray, toward_light: np.ndarray, w: float = 3.0) -> np.ndarray:
    """Sub-pixel edge position across a dark-to-light edge, by profile area."""
    n = toward_light - p
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ds = 0.125
    s = np.arange(-w - 4, w + 4 + ds / 2, ds)
    q = p[:, None, :] + s[None, :, None] * n[:, None, :]
    prof = cv2.remap(img, q[..., 0].astype(np.float32), q[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    lo = np.median(prof[:, s < -w - 1])
    hi = np.median(prof[:, s > w + 1])
    if hi - lo < 30:
        return np.full_like(p, np.nan)
    inner = np.abs(s) <= w + ds / 2
    frac = np.clip((hi - prof[:, inner]) / (hi - lo), -0.5, 1.5)
    off = -w + (frac[:, :-1] + frac[:, 1:]).sum(axis=1) * ds / 2
    out = p + off[:, None] * n
    out[np.abs(off) > w - 1] = np.nan
    return out


def _fit_gridlines(pts: np.ndarray, axis: np.ndarray, value: np.ndarray) -> np.ndarray:
    """px -> mm homography with point i on the line mm[axis[i]] = value[i] (normalised DLT)."""
    centre = pts.mean(axis=0)
    scale = np.sqrt(2) / np.linalg.norm(pts - centre, axis=1).mean()
    T = np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]])
    p = np.c_[(pts - centre) * scale, np.ones(len(pts))]
    keep = np.ones(len(p), bool)
    weight = np.ones(len(p))
    for _ in range(3):
        A = np.zeros((len(p), 9))
        for ax in (0, 1):
            sel = axis == ax
            A[sel, 3 * ax : 3 * ax + 3] = p[sel]
        A[:, 6:9] = -value[:, None] * p
        A *= weight[:, None]
        _, _, vt = np.linalg.svd(A[keep])
        h = vt[-1].reshape(3, 3)
        denom = p @ h[2]
        mm = (p @ h[:2].T) / denom[:, None]
        resid = mm[np.arange(len(p)), axis] - value
        weight = 1.0 / np.abs(denom)
        keep = np.abs(resid) < max(4 * resid[keep].std(), 0.02)
    H = h @ T
    return H / H[2, 2]


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    return np.linalg.solve(np.array([l1[:2], l2[:2]]), np.array([l1[2], l2[2]]))


def homography_from_marker(corners_px: np.ndarray, marker_size_mm: float = MARKER_SIZE_MM) -> Homography:
    """Photo px -> mm, marker TL at (0, 0), x along the top edge, y down the left edge."""
    s = marker_size_mm
    dst = np.float64([[0, 0], [s, 0], [s, s], [0, s]])
    H, _ = cv2.findHomography(np.asarray(corners_px, np.float64).reshape(4, 2), dst, 0)
    return Homography(H=H / H[2, 2])


def photo_homography(image: np.ndarray) -> Homography:
    """detect_marker + homography_from_marker."""
    return homography_from_marker(detect_marker(image))
