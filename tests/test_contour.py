from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial import cKDTree

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings
from sewtrace.mock import synth
from sewtrace.types import TEE_EDGES, Outline, Panel
from sewtrace.vision import contour
from sewtrace.vision.marker import Homography

VIEWS = ("front", "back", "sleeve")
TEE_NAMES = [p.name for p in synth.TEES]


# test_measure.py imports these fixtures, which makes pytest register them once per
# module; _SESSION keeps the expensive work to one run per session.
_SESSION: dict = {}


@pytest.fixture(scope="session")
def tees(tmp_path_factory) -> dict[str, tuple[Path, dict]]:
    if "tees" not in _SESSION:
        out = tmp_path_factory.mktemp("raw")
        _SESSION["tees"] = {
            p.name: (out / p.name, synth.make_tee(p, out / p.name, seed=i)) for i, p in enumerate(synth.TEES)
        }
    return _SESSION["tees"]


@pytest.fixture(scope="session")
def session_client(tmp_path_factory) -> MetaClient:
    if "client" not in _SESSION:
        cache = tmp_path_factory.mktemp("cache")
        _SESSION["client"] = MetaClient(
            Settings(api_key="", mock=True, cache_dir=cache)
        )
    return _SESSION["client"]


def truth_homography(truth: dict, view: str) -> Homography:
    return Homography(np.linalg.inv(np.array(truth["photos"][view]["world_mm_to_photo_px"])))


def extract(mask: np.ndarray, H: Homography, view: str):
    outline = contour.smooth_outline(contour.mask_to_outline(mask, H))
    if view == "sleeve":
        half = contour.sleeve_half(outline)
        return half, contour.label_sleeve_edges(half)
    half = contour.body_half(outline)
    return half, contour.label_body_edges(half)


@pytest.fixture(scope="session")
def halves(tees, session_client) -> dict[tuple[str, str], tuple]:
    if "halves" not in _SESSION:
        out = {}
        for name, (d, truth) in tees.items():
            for view in VIEWS:
                mask = session_client.segment(d / f"{view}.jpg", "sleeve" if view == "sleeve" else "body")
                out[name, view] = extract(mask, truth_homography(truth, view), view)
        _SESSION["halves"] = out
    return _SESSION["halves"]


def length(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def outline_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-point distance between two closed outlines."""
    da = contour._resample_closed(a, 0.25)
    db = contour._resample_closed(b, 0.25)
    return max(cKDTree(db).query(a)[0].mean(), cKDTree(da).query(b)[0].mean())


def truth_edge_lengths(truth: dict, view: str) -> dict[str, float]:
    pts = np.array(truth["photos"][view]["half_outline_mm"])
    panel = Panel(name=view, cut="", outline_mm=[tuple(p) for p in pts], edges=[
        {"name": k, "start": a, "end": b} for k, (a, b) in truth["photos"][view]["edges"].items()])
    return {e.name: length(panel.edge_points(e.name)) for e in panel.edges}


def edge_lengths(half: Outline, edges) -> dict[str, float]:
    panel = Panel(name="x", cut="", outline_mm=half.points, edges=edges)
    return {e.name: length(panel.edge_points(e.name)) for e in edges}


# ---- unit tests ---------------------------------------------------------------

def test_mask_to_outline_square():
    mask = np.zeros((200, 300), bool)
    mask[50:150, 100:250] = True  # 150 x 100 px, edges at pixel boundaries
    H = Homography(np.diag([0.5, 0.5, 1.0]))
    pts = contour.mask_to_outline(mask, H).array()
    assert pts[:, 0].min() == pytest.approx(49.75, abs=0.3)
    assert pts[:, 0].max() - pts[:, 0].min() == pytest.approx(75, abs=0.5)
    assert pts[:, 1].max() - pts[:, 1].min() == pytest.approx(50, abs=0.5)


def test_smooth_outline_keeps_corners_and_spacing():
    rng = np.random.default_rng(0)
    square = np.array([[0, 0], [100, 0], [100, 60], [0, 60]], float)
    dense = contour._resample_closed(square, 0.3)
    dense += rng.normal(0, 0.15, dense.shape)
    out = contour.smooth_outline(Outline.from_array(dense), 2.0).array()
    steps = np.linalg.norm(np.diff(np.vstack([out, out[:1]]), axis=0), axis=1)
    assert np.all(steps < 2.2) and np.all(steps > 1.0)
    for corner in square:
        assert np.linalg.norm(out - corner, axis=1).min() < 0.75
    assert np.linalg.norm(out[0] - dense[0]) < 0.5


# ---- synthetic tees -------------------------------------------------------------

@pytest.mark.parametrize("name", TEE_NAMES)
@pytest.mark.parametrize("view", VIEWS)
def test_half_outline_matches_truth(halves, tees, name, view):
    half, _ = halves[name, view]
    truth = np.array(tees[name][1]["photos"][view]["half_outline_mm"])
    pts = half.array()
    assert outline_distance(pts, truth) < 1.5
    assert pts[:, 0].min() >= 0
    assert np.abs(pts[0] - truth[0]).max() < 1.5  # starts at neck/fold or cap top


@pytest.mark.parametrize("name", TEE_NAMES)
@pytest.mark.parametrize("view", VIEWS)
def test_edge_labels_match_truth(halves, tees, name, view):
    half, edges = halves[name, view]
    panel = "sleeve" if view == "sleeve" else "front"
    assert [e.name for e in edges] == list(TEE_EDGES[panel])
    for a, b in zip(edges, edges[1:]):
        assert a.end == b.start
    assert edges[-1].end == edges[0].start == 0
    got, want = edge_lengths(half, edges), truth_edge_lengths(tees[name][1], view)
    for k in want:
        assert got[k] == pytest.approx(want[k], abs=3.0), k


def rotate_photo(mask: np.ndarray, H: Homography, deg: float, frame_deg: float):
    """Rotate the mask image by deg and the mm frame by frame_deg, keeping H consistent."""
    h, w = mask.shape
    A = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0), [0, 0, 1]])
    rotated = cv2.warpAffine(mask.astype(np.uint8), A[:2], (w, h), flags=cv2.INTER_NEAREST) > 0
    t = np.radians(frame_deg)
    R = np.array([[np.cos(t), -np.sin(t), 40], [np.sin(t), np.cos(t), -25], [0, 0, 1]])
    return rotated, Homography(R @ H.H @ np.linalg.inv(A))


@pytest.mark.parametrize("view", VIEWS)
def test_rotated_photo(tees, session_client, view):
    d, truth = tees["tee3"]
    mask = session_client.segment(d / f"{view}.jpg", "sleeve" if view == "sleeve" else "body")
    for deg, frame_deg in ((7.0, -3.0), (-5.0, 4.0)):
        rmask, H = rotate_photo(mask, truth_homography(truth, view), deg, frame_deg)
        half, edges = extract(rmask, H, view)
        ref = np.array(truth["photos"][view]["half_outline_mm"])
        assert outline_distance(half.array(), ref) < 1.5
        got, want = edge_lengths(half, edges), truth_edge_lengths(truth, view)
        for k in want:
            assert got[k] == pytest.approx(want[k], abs=3.0), k


def test_body_half_upside_down(tees, session_client):
    d, truth = tees["tee1"]
    mask = session_client.segment(d / "front.jpg", "body")
    rmask, H = rotate_photo(mask, truth_homography(truth, "front"), 180.0, 0.0)
    half, _ = extract(rmask, H, "front")
    ref = np.array(truth["photos"]["front"]["half_outline_mm"])
    assert outline_distance(half.array(), ref) < 1.5

