"""T15: SAM mask to a clean half outline in mm.

Owner: Madhu. See sewtrace/types.py for the coordinate convention.

Pipeline per photo:
    outline = smooth_outline(mask_to_outline(mask, H))
    half = body_half(remove_sleeves(outline))   # or sleeve_half(outline)
    edges = label_body_edges(half)              # or label_sleeve_edges(half)

SAM masks the whole t-shirt, sleeves included (REQ-SEG-02), so the front and back
outlines pass through remove_sleeves() first.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from sewtrace.config import get_config
from sewtrace.types import TEE_EDGES, Edge, Outline
from sewtrace.vision.marker import Homography
from sewtrace.vision.measure import _corner, _fit_line

UPSAMPLE = 2           # mask upsampling before tracing, for a sub-pixel edge
CORNER_DEG = 30.0      # turning angle that counts as a corner
CORNER_WINDOW_MM = 4.0
SMOOTH_SIGMA_MM = 1.5
FOLD_TOL_MM = 1.0
JOINT_SPACING_MM = 1.0   # outline spacing for the sleeve joint search
JOINT_TRIM_MM = 4.0      # ends of the sleeve-top run ignored by the line fits (corner rounding)
JOINT_MARGIN_MM = 4.0    # points this close to the split are ignored by both line fits
JOINT_MIN_RUN_MM = 15.0  # shortest straight run on each side of the shoulder point
JOINT_REACH_MM = 20.0    # tangent fit length at the underarm point


# ---- polyline helpers -------------------------------------------------------

def _closed(pts: np.ndarray) -> np.ndarray:
    return np.vstack([pts, pts[:1]])


def _cumlen(pts: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]


def _resample_open(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Even resampling of an open polyline; both end points kept."""
    s = _cumlen(pts)
    if s[-1] <= 0:
        return pts[:1].copy()
    n = max(int(np.ceil(s[-1] / spacing)), 1)
    t = np.linspace(0, s[-1], n + 1)
    return np.c_[np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])]


def _resample_closed(pts: np.ndarray, spacing: float) -> np.ndarray:
    return _resample_open(_closed(pts), spacing)[:-1]


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _centroid(pts: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6 * a)


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _turning(pts: np.ndarray, k: int) -> np.ndarray:
    """Absolute turning angle (deg) at each point of a closed polyline, chord window k."""
    v1 = pts - np.roll(pts, k, axis=0)
    v2 = np.roll(pts, -k, axis=0) - pts
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(axis=1)
    return np.degrees(np.abs(np.arctan2(cross, dot)))


def _peaks(score: np.ndarray, threshold: float, radius: int) -> list[int]:
    """Indices of local maxima above threshold, strongest first, circular suppression."""
    score = score.copy()
    n, out = len(score), []
    while True:
        i = int(np.argmax(score))
        if score[i] < threshold:
            return out
        out.append(i)
        score[[(i + d) % n for d in range(-radius, radius + 1)]] = -np.inf


def _spacing(pts: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(np.diff(_closed(pts), axis=0), axis=1)))


def _orient_like_synth(pts: np.ndarray) -> np.ndarray:
    """Synth outlines run neck -> shoulder -> ... -> fold: positive shoelace area in y-down."""
    return pts if _signed_area(pts) > 0 else pts[::-1]


# ---- public API -------------------------------------------------------------

def mask_to_outline(mask: np.ndarray, homography: Homography) -> Outline:
    """Largest external contour of a boolean mask, in the marker mm frame."""
    mask = np.asarray(mask).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    pad = 4
    x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
    crop = mask[y0 : ys.max() + pad + 1, x0 : xs.max() + pad + 1] * 255
    big = cv2.resize(crop, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_LINEAR)
    contours, _ = cv2.findContours((big > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    px = (c + 0.5) / UPSAMPLE - 0.5 + [x0, y0]
    mm = homography.px_to_mm(px)
    return Outline.from_array(_orient_like_synth(mm))


def smooth_outline(outline: Outline, spacing_mm: float = 2.0) -> Outline:
    """Resample at even spacing and smooth, keeping sharp corners.

    The output starts at the input's first point, which is always kept as a vertex.
    """
    pts = outline.array()
    step = min(0.5, spacing_mm / 2)
    origin = pts[0]
    dense = _resample_closed(pts, step)
    k = max(int(round(CORNER_WINDOW_MM / step)), 1)
    turn = gaussian_filter1d(_turning(dense, k), k / 2, mode="wrap")
    corners = set(_peaks(turn, CORNER_DEG, 2 * k))
    corners.add(int(np.argmin(np.linalg.norm(dense - origin, axis=1))))
    breaks = sorted(corners)
    first = int(np.argmin(np.linalg.norm(dense[breaks] - origin, axis=1)))
    breaks = breaks[first:] + breaks[:first]
    sigma = SMOOTH_SIGMA_MM / step
    n, out = len(dense), []
    for a, b in zip(breaks, breaks[1:] + breaks[:1]):
        idx = np.arange(a, b + (n if b <= a else 0) + 1) % n
        seg = _smooth_pinned(dense[idx], sigma)
        out.append(_resample_open(seg, spacing_mm)[:-1])
    return Outline.from_array(np.vstack(out))


def _smooth_pinned(seg: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of an open run with both end points fixed (odd reflection)."""
    if len(seg) < 5:
        return seg
    m = min(len(seg) - 1, int(3 * sigma) + 1)
    head = 2 * seg[0] - seg[m:0:-1]
    tail = 2 * seg[-1] - seg[-2 : -m - 2 : -1]
    ext = np.vstack([head, seg, tail])
    sm = gaussian_filter1d(ext, sigma, axis=0, mode="nearest")[m : m + len(seg)]
    sm[0], sm[-1] = seg[0], seg[-1]
    return sm


def _signed_turning(pts: np.ndarray, k: int) -> np.ndarray:
    """Turning angle (deg) at each point of a closed polyline, positive where the
    outline is convex when its shoelace area is positive."""
    v1 = pts - np.roll(pts, k, axis=0)
    v2 = np.roll(pts, -k, axis=0) - pts
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.degrees(np.arctan2(cross, (v1 * v2).sum(axis=1)))


@dataclass(frozen=True)
class _Joints:
    """Sleeve joints of a whole-garment outline, found by walking it once.

    ``pts`` is the outline resampled at JOINT_SPACING_MM and rolled so that it starts
    at the first underarm point, then runs along the hem side to the second one
    (index ``other``), then over the sleeves and the neck back to the start.
    """

    pts: np.ndarray
    other: int
    underarm_a: np.ndarray    # at index ``other``
    underarm_b: np.ndarray    # at index 0
    shoulder_a: np.ndarray    # between the first two sleeve-top corners and the neck
    shoulder_b: np.ndarray
    index_a: int              # outline samples nearest the shoulder points
    index_b: int
    shoulder_dir_a: np.ndarray  # unit vectors along the shoulder seam, towards the neck
    shoulder_dir_b: np.ndarray
    side_dir_a: np.ndarray      # unit vectors along the side seam, towards the hem
    side_dir_b: np.ndarray


def _line_ssr(pts: np.ndarray) -> float:
    c, d = _fit_line(pts)
    r = (pts - c) - ((pts - c) @ d)[:, None] * d
    return float((r**2).sum())


def _breakpoint(run: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shoulder point on the run from the sleeve's outer hem corner to the neck corner.

    The run is two straight lines (sleeve top, shoulder seam) that meet at a kink of
    only 10 to 25 degrees, too soft for a corner detector. Find the split that makes
    both lines fit best, then intersect the two fitted lines. Returns the point and
    the unit direction of the shoulder seam (towards the neck).
    """
    s = _cumlen(run)
    lo, hi = JOINT_TRIM_MM + JOINT_MIN_RUN_MM, s[-1] - JOINT_TRIM_MM - JOINT_MIN_RUN_MM
    best: tuple[float, int] | None = None
    for i in np.nonzero((s >= lo) & (s <= hi))[0]:
        left = run[(s >= JOINT_TRIM_MM) & (s <= s[i] - JOINT_MARGIN_MM)]
        right = run[(s >= s[i] + JOINT_MARGIN_MM) & (s <= s[-1] - JOINT_TRIM_MM)]
        cost = _line_ssr(left) + _line_ssr(right)
        if best is None or cost < best[0]:
            best = (cost, int(i))
    if best is None:
        raise ValueError("sleeve top and shoulder seam are too short to find the shoulder point")
    split = best[1]
    c1, d1 = _fit_line(run[(s >= JOINT_TRIM_MM) & (s <= s[split] - JOINT_MARGIN_MM)])
    c2, d2 = _fit_line(run[(s >= s[split] + JOINT_MARGIN_MM) & (s <= s[-1] - JOINT_TRIM_MM)])
    point = run[split]
    A = np.c_[d1, -d2]
    if abs(np.linalg.det(A)) > 1e-3:
        t = np.linalg.solve(A, c2 - c1)
        crossing = c1 + t[0] * d1
        if np.linalg.norm(crossing - point) <= 2 * JOINT_MARGIN_MM:  # else the lines are near parallel
            point = crossing
    if d2 @ (run[-1] - run[0]) < 0:
        d2 = -d2
    return point, d2


def _ring_run(pts: np.ndarray, first: int, last: int) -> np.ndarray:
    """Samples first..last inclusive of a closed polyline, wrapping."""
    n = len(pts)
    return pts[np.arange(first, last + 1 + (n if last < first else 0)) % n]


def _find_joints(outline: Outline) -> _Joints:
    """Shoulder and underarm points of a whole t-shirt outline (REQ-GEO-01).

    Underarm points are the two sharpest concave corners. They split the outline into
    the hem arc (two body hem corners) and the top arc (neck corners and the sleeve
    hem corners: six in all). The shoulder point of each sleeve lies on the run between
    its outer hem corner and the neck corner.
    """
    pts = _resample_closed(_orient_like_synth(outline.array()), JOINT_SPACING_MM)
    n = len(pts)
    k = max(int(round(CORNER_WINDOW_MM / JOINT_SPACING_MM)), 1)
    turn = gaussian_filter1d(_signed_turning(pts, k), k / 2, mode="wrap")
    convex = sorted(_peaks(turn, CORNER_DEG, 2 * k))
    concave = sorted(_peaks(-turn, CORNER_DEG, 2 * k)[:2])
    if len(concave) < 2:
        raise ValueError("found no underarm corners: is the whole t-shirt, with both sleeves, in the mask?")
    u1, u2 = concave
    between = [c for c in convex if u1 < c < u2]
    outside = [c for c in convex if not u1 < c < u2]
    start, other_abs = (u1, u2) if len(between) < len(outside) else (u2, u1)
    pts = np.roll(pts, -start, axis=0)
    other = (other_abs - start) % n
    convex = sorted((c - start) % n for c in convex)
    hem = [c for c in convex if 0 < c < other]
    top = [c for c in convex if c > other]
    if len(top) != 6:
        raise ValueError(
            f"found {len(top)} convex corners over the sleeves and neck, expected 6 "
            "(2 neck, 4 sleeve hem): is the whole t-shirt, with both sleeves, in the mask?"
        )
    c0, c1, c2, c3, c4, c5 = top
    reach = int(JOINT_REACH_MM / JOINT_SPACING_MM)
    h = hem[0] if hem else other // 2

    def underarm(into_lo: int, into_hi: int, out_lo: int, out_hi: int) -> np.ndarray:
        return _corner(_ring_run(pts, into_lo, into_hi), _ring_run(pts, out_lo, out_hi), JOINT_REACH_MM, JOINT_REACH_MM)

    prev = hem[-1] if hem else 0
    under_a = underarm(other - min(reach, other - prev - 2), other, other, other + min(reach, c0 - other - 2))
    under_b = underarm(n - min(reach, n - c5 - 2), n, 0, min(reach, h - 2))

    shoulder_a, dir_a = _breakpoint(pts[c1 : c2 + 1])
    shoulder_b, dir_b = _breakpoint(pts[c3 : c4 + 1][::-1])  # walked from the sleeve to the neck
    ia = c1 + int(np.argmin(np.linalg.norm(pts[c1 : c2 + 1] - shoulder_a, axis=1)))
    ib = c3 + int(np.argmin(np.linalg.norm(pts[c3 : c4 + 1] - shoulder_b, axis=1)))
    side_a = pts[max(other - reach, 0)] - under_a
    side_b = pts[min(reach, other)] - under_b
    return _Joints(pts, other, under_a, under_b, shoulder_a, shoulder_b, ia, ib,
                   dir_a, dir_b, side_a / np.linalg.norm(side_a), side_b / np.linalg.norm(side_b))


def find_sleeve_joints(outline: Outline) -> tuple[np.ndarray, np.ndarray]:
    """(shoulder points, underarm points) of a whole t-shirt outline: two (2, 2) arrays, mm.

    Row 0 is the sleeve met first on the outline walk (a), row 1 the other (b).
    """
    j = _find_joints(outline)
    return np.vstack([j.shoulder_a, j.shoulder_b]), np.vstack([j.underarm_a, j.underarm_b])


def _perp_towards(direction: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to ``direction`` that points to the side of ``toward``."""
    perp = np.array([-direction[1], direction[0]])
    perp /= np.linalg.norm(perp)
    return perp if perp @ toward > 0 else -perp


def default_armhole(
    shoulder: np.ndarray,
    underarm: np.ndarray,
    shoulder_dir: np.ndarray,
    side_dir: np.ndarray,
    centre: np.ndarray,
) -> np.ndarray:
    """Armhole from the shoulder point to the underarm point, about 1 mm apart.

    The flat-lay photo shows only the two end points. The default is a cubic Bezier
    that leaves the shoulder seam at a right angle and meets the side seam at a right
    angle, both bending towards the body centre (handle lengths in config).
    """
    cfg = get_config().contour.armhole
    chord = float(np.linalg.norm(underarm - shoulder))
    ctrl1 = shoulder + cfg.shoulder_handle * chord * _perp_towards(shoulder_dir, centre - shoulder)
    ctrl2 = underarm + cfg.underarm_handle * chord * _perp_towards(side_dir, centre - underarm)
    t = np.linspace(0, 1, max(int(np.ceil(chord)), 8))[:, None]
    return ((1 - t) ** 3 * shoulder + 3 * (1 - t) ** 2 * t * ctrl1
            + 3 * (1 - t) * t**2 * ctrl2 + t**3 * underarm)


def cut_sleeves(outline: Outline) -> tuple[Outline, np.ndarray]:
    """Whole t-shirt outline -> body outline: cut each sleeve off at the shoulder point
    and the underarm point and close the gap with the default armhole (REQ-GEO-01).

    Only the two joints are visible in a flat-lay photo, so the armhole shape here is
    a default, not a measurement. The model infers the real shape (REQ-RSN-04).
    Also returns a boolean array, True for each point of the body outline that lies on
    a default armhole, so callers can leave those points out of an edge comparison.
    """
    j = _find_joints(outline)
    pts, other = j.pts, j.other
    centre = _centroid(pts)
    arm_a = default_armhole(j.shoulder_a, j.underarm_a, j.shoulder_dir_a, j.side_dir_a, centre)
    arm_b = default_armhole(j.shoulder_b, j.underarm_b, j.shoulder_dir_b, j.side_dir_b, centre)
    parts = [
        (j.underarm_b[None], False),
        (pts[1:other], False),
        (j.underarm_a[None], True),
        (arm_a[::-1][1:-1], True),
        (j.shoulder_a[None], True),
        (pts[j.index_a + 1 : j.index_b], False),
        (j.shoulder_b[None], True),
        (arm_b[1:-1], True),
    ]
    body = np.vstack([p for p, _ in parts])
    invented = np.concatenate([np.full(len(p), flag) for p, flag in parts])
    return Outline.from_array(body), invented


def remove_sleeves(outline: Outline) -> Outline:
    """cut_sleeves() without the armhole flags."""
    return cut_sleeves(outline)[0]


def body_half(outline: Outline) -> Outline:
    """Full body outline -> half panel: find the vertical symmetry axis, mirror
    the two sides into one averaged half, fold on x = 0, neck point at y = 0."""
    pts = _resample_closed(outline.array(), 1.0)
    c = _centroid(pts)
    centred = pts - c
    tree = cKDTree(centred)

    def cost(v: np.ndarray) -> float:
        theta, dx = v
        q = centred @ _rot(-theta).T
        mirrored = np.c_[2 * dx - q[:, 0], q[:, 1]] @ _rot(theta).T
        d, _ = tree.query(mirrored)
        return float(np.mean(np.minimum(d, 20.0)))

    grid = np.radians(np.arange(-30, 30.5, 1.0))
    t0 = grid[int(np.argmin([cost(np.array([t, 0.0])) for t in grid]))]
    res = minimize(cost, np.array([t0, 0.0]), method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-6, "initial_simplex": [[t0, 0], [t0 + 0.01, 0], [t0, 2.0]]})
    theta, dx = res.x
    q = centred @ _rot(-theta).T - [dx, 0]

    top, bottom = _axis_crossings(q)
    if (q[:, 1].max() - bottom) > (top - q[:, 1].min()):  # neck recess is at the bottom: flip
        q = -q
    q = _orient_like_synth(q)
    right, left = _split_sides(q)
    avg = _average_sides(right, left * [-1, 1])
    return _close_half(avg, spacing=2.0)


def _axis_crossings(q: np.ndarray) -> tuple[float, float]:
    """y of the top and bottom places where a closed outline crosses x = 0."""
    x, y = q[:, 0], q[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    hit = np.nonzero((x <= 0) != (xn <= 0))[0]
    ys = y[hit] + (yn[hit] - y[hit]) * (0 - x[hit]) / (xn[hit] - x[hit])
    return float(ys.min()), float(ys.max())


def _split_sides(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right (x > 0) and left runs, each ordered top to bottom, with axis end points."""
    n = len(q)
    x = q[:, 0]
    hit = np.nonzero((x <= 0) != (np.roll(x, -1) <= 0))[0]
    cross = []
    for i in hit:
        a, b = q[i], q[(i + 1) % n]
        t = a[0] / (a[0] - b[0])
        cross.append((i, a + t * (b - a)))
    cross.sort(key=lambda c: c[1][1])
    (i_top, p_top), (i_bot, p_bot) = cross[0], cross[-1]

    def run(i: int, j: int) -> np.ndarray:  # points strictly after i up to j, cyclic
        idx = np.arange(i + 1, j + 1 + (n if j < i else 0)) % n
        return q[idx]

    a = np.vstack([p_top, run(i_top, i_bot), p_bot])
    b = np.vstack([p_bot, run(i_bot, i_top), p_top])[::-1]
    if np.median(a[:, 0]) < np.median(b[:, 0]):
        a, b = b, a
    return a, b


def _average_sides(right: np.ndarray, left_mirrored: np.ndarray) -> np.ndarray:
    """Midpoint of each right point and its nearest point on the mirrored left run."""
    dense = _resample_open(left_mirrored, 0.25)
    _, j = cKDTree(dense).query(right)
    avg = (right + dense[j]) / 2
    avg[0, 0] = avg[-1, 0] = 0.0
    return avg


def _close_half(chain: np.ndarray, spacing: float) -> Outline:
    """Chain from the top fold point round to the bottom fold point -> closed half outline,
    smoothed, fold on x = 0, the topmost point (neck point / cap top) on y = 0."""
    chain = chain.copy()
    chain[0, 0] = chain[-1, 0] = 0.0
    fold = _resample_open(np.vstack([chain[-1], chain[0]]), spacing)[1:-1]
    half = np.vstack([chain, fold])
    half = smooth_outline(Outline.from_array(half), spacing).array()
    half[:, 0] = np.maximum(half[:, 0], 0)
    half[:, 1] -= half[:, 1].min()
    return Outline.from_array(half)


def sleeve_half(outline: Outline) -> Outline:
    """Folded sleeve outline (fold on the left, roughly vertical) -> half panel
    with the fold on x = 0 and the cap top at y = 0."""
    pts = _resample_closed(outline.array(), 1.0)
    c = pts.mean(axis=0)
    centred = pts - c

    def support(theta: float) -> tuple[int, np.ndarray]:
        q = centred @ _rot(-theta).T
        near = q[:, 0] < q[:, 0].min() + FOLD_TOL_MM
        return int(near.sum()), near

    grid = np.radians(np.arange(-30, 30.05, 0.25))
    theta = grid[int(np.argmax([support(t)[0] for t in grid]))]
    _, near = support(theta)
    # Refine: total least squares line through the fold points.
    fp = pts[near]
    _, _, vt = np.linalg.svd(fp - fp.mean(axis=0))
    d = vt[0] if vt[0][1] > 0 else -vt[0]
    theta = np.arctan2(-d[0], d[1])  # angle of the fold from +y
    q = (pts - fp.mean(axis=0)) @ _rot(-theta).T
    q = _orient_like_synth(q)

    on_fold = np.abs(q[:, 0]) < FOLD_TOL_MM
    chain = _longest_run(q, ~on_fold)
    if chain[0, 1] > chain[-1, 1]:
        chain = chain[::-1]
    return _close_half(chain, spacing=2.0)


def _longest_run(q: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Longest cyclic run of points where keep is True."""
    n = len(q)
    if keep.all():
        return q
    start = int(np.argmin(keep))  # a False index
    order = (np.arange(n) + start) % n
    best, cur = (0, 0), None
    for pos, i in enumerate(order):
        if keep[i]:
            cur = pos if cur is None else cur
            if pos - cur + 1 > best[1] - best[0]:
                best = (cur, pos + 1)
        else:
            cur = None
    return q[order[best[0] : best[1]]]


def _edges(names: tuple[str, ...], corners: list[int]) -> list[Edge]:
    bounds = list(corners) + [0]
    return [Edge(name=nm, start=bounds[i], end=bounds[i + 1]) for i, nm in enumerate(names)]


def label_body_edges(half: Outline) -> list[Edge]:
    """Edges in TEE_EDGES["front"] order: neck, shoulder, armhole, side, hem, fold."""
    pts = half.array()
    n = len(pts)
    k = max(int(round(CORNER_WINDOW_MM / _spacing(pts))), 1)
    turn = _turning(pts, k)
    top = int(np.argmin(pts[:, 1]))  # neck point is the highest point; snap to the sharpest corner near it
    near = np.arange(max(top - k - 1, 1), top + k + 2)
    neck_pt = int(near[np.argmax(turn[near])])
    hem_fold = _bottom_fold_corner(pts)
    inner = np.full(n, -np.inf)
    inner[neck_pt + 2 * k : hem_fold - 2 * k] = turn[neck_pt + 2 * k : hem_fold - 2 * k]
    shoulder, underarm, hem_side = sorted(_peaks(inner, 0.0, 2 * k)[:3])
    return _edges(TEE_EDGES["front"], [0, neck_pt, shoulder, underarm, hem_side, hem_fold])


def label_sleeve_edges(half: Outline) -> list[Edge]:
    """Edges in TEE_EDGES["sleeve"] order: cap, underarm, hem, fold."""
    pts = half.array()
    n = len(pts)
    k = max(int(round(CORNER_WINDOW_MM / _spacing(pts))), 1)
    turn = _turning(pts, k)
    hem_fold = _bottom_fold_corner(pts)
    inner = np.full(n, -np.inf)
    inner[2 * k : hem_fold - 2 * k] = turn[2 * k : hem_fold - 2 * k]
    cap_end, hem_side = sorted(_peaks(inner, 0.0, 2 * k)[:2])
    return _edges(TEE_EDGES["sleeve"], [0, cap_end, hem_side, hem_fold])


def _bottom_fold_corner(pts: np.ndarray) -> int:
    """Index of the fold's bottom end (x = 0, largest y)."""
    on_fold = pts[:, 0] < 0.5
    idx = np.nonzero(on_fold)[0]
    return int(idx[np.argmax(pts[idx, 1])])
