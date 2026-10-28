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
    half = contour.body_half(contour.remove_sleeves(outline))
    return half, contour.label_body_edges(half)


@pytest.fixture(scope="session")
def halves(tees, session_client) -> dict[tuple[str, str], tuple]:
    if "halves" not in _SESSION:
        out = {}
        for name, (d, truth) in tees.items():
            for view in VIEWS:
                mask = session_client.segment(d / f"{view}.jpg", "sleeve" if view == "sleeve" else "t-shirt")
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


def visible_half(truth: dict, view: str) -> np.ndarray:
    """Half outline the mask shows. On the front that has the BACK neckline: the back panel is
    seen through the neck opening (synth.silhouette_half), so the front neck is not visible."""
    photo = truth["photos"][view]
    return np.array(photo.get("visible_half_outline_mm", photo["half_outline_mm"]))


def truth_edge_lengths(truth: dict, view: str) -> dict[str, float]:
    pts = visible_half(truth, view)
    panel = Panel(name=view, cut="", outline_mm=[tuple(p) for p in pts], edges=[
        {"name": k, "start": a, "end": b} for k, (a, b) in truth["photos"][view]["edges"].items()])
    return {e.name: length(panel.edge_points(e.name)) for e in panel.edges}


def edge_lengths(half: Outline, edges) -> dict[str, float]:
    panel = Panel(name="x", cut="", outline_mm=half.points, edges=edges)
    return {e.name: length(panel.edge_points(e.name)) for e in edges}


# ---- unit tests ---------------------------------------------------------------

def test_mask_to_outline_square():
    """REQ-GEO-01: mask to outline square."""
    mask = np.zeros((200, 300), bool)
    mask[50:150, 100:250] = True  # 150 x 100 px, edges at pixel boundaries
    H = Homography(np.diag([0.5, 0.5, 1.0]))
    pts = contour.mask_to_outline(mask, H).array()
    assert pts[:, 0].min() == pytest.approx(49.75, abs=0.3)
    assert pts[:, 0].max() - pts[:, 0].min() == pytest.approx(75, abs=0.5)
    assert pts[:, 1].max() - pts[:, 1].min() == pytest.approx(50, abs=0.5)


def test_smooth_outline_keeps_corners_and_spacing():
    """REQ-GEO-01: smooth outline keeps corners and spacing."""
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
    """REQ-GEO-01: half outline matches truth."""
    half, _ = halves[name, view]
    truth = visible_half(tees[name][1], view)
    pts = half.array()
    assert outline_distance(pts, truth) < 1.5
    assert pts[:, 0].min() >= 0
    assert np.abs(pts[0] - truth[0]).max() < 1.5  # starts at neck/fold or cap top


@pytest.mark.parametrize("name", TEE_NAMES)
@pytest.mark.parametrize("view", VIEWS)
def test_edge_labels_match_truth(halves, tees, name, view):
    """REQ-GEO-03: edge labels match truth."""
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
    """REQ-GEO-01: rotated photo."""
    d, truth = tees["tee3"]
    mask = session_client.segment(d / f"{view}.jpg", "sleeve" if view == "sleeve" else "t-shirt")
    for deg, frame_deg in ((7.0, -3.0), (-5.0, 4.0)):
        rmask, H = rotate_photo(mask, truth_homography(truth, view), deg, frame_deg)
        half, edges = extract(rmask, H, view)
        ref = visible_half(truth, view)
        assert outline_distance(half.array(), ref) < 1.5
        got, want = edge_lengths(half, edges), truth_edge_lengths(truth, view)
        for k in want:
            assert got[k] == pytest.approx(want[k], abs=3.0), k


def test_body_half_upside_down(tees, session_client):
    """REQ-GEO-02: body half upside down."""
    d, truth = tees["tee1"]
    mask = session_client.segment(d / "front.jpg", "t-shirt")
    rmask, H = rotate_photo(mask, truth_homography(truth, "front"), 180.0, 0.0)
    half, _ = extract(rmask, H, "front")
    ref = visible_half(truth, "front")
    assert outline_distance(half.array(), ref) < 1.5



# ---- whole-garment masks: sleeve removal (REQ-SEG-02, REQ-GEO-01, SPEC 10.1) -----------

def whole_outline(tees, client, name: str, view: str = "front") -> Outline:
    d, truth = tees[name]
    mask = client.segment(d / f"{view}.jpg", "t-shirt")
    return contour.smooth_outline(contour.mask_to_outline(mask, truth_homography(truth, view)))


@pytest.mark.parametrize("name", TEE_NAMES)
@pytest.mark.parametrize("view", ("front", "back"))
def test_req_geo_01_shoulder_and_underarm_points_match_truth(tees, session_client, name, view):
    """REQ-GEO-01: both sleeve joints are found on the whole-garment outline: underarm points
    within 1 mm (measured 0.5 mm worst), shoulder points within 2 mm (1.4 mm worst: the
    shoulder seam and sleeve top meet at a kink of only about 15 degrees)."""
    _, truth = tees[name]
    p, org = truth["params"], np.array(truth["photos"][view]["origin_mm"])
    shoulder, underarm = contour.find_sleeve_joints(whole_outline(tees, session_client, name, view))
    want_s = np.array([[org[0] + sx * p["shoulder_half"], org[1] + p["shoulder_drop"]] for sx in (1, -1)])
    want_u = np.array([[org[0] + sx * p["chest_half"], org[1] + p["shoulder_drop"] + p["armhole_depth"]] for sx in (1, -1)])
    for got, want, tol in ((shoulder, want_s, 2.0), (underarm, want_u, 1.0)):
        dist = np.linalg.norm(got[:, None] - want[None], axis=2)
        assert dist.min(axis=1).max() < tol and dist.min(axis=0).max() < tol  # each joint matched once


@pytest.mark.parametrize("name", TEE_NAMES)
def test_req_geo_01_body_half_from_whole_garment_vs_truth(tees, session_client, name):
    """REQ-GEO-01: whole-garment mask -> body half outline vs truth. Measured over the 5 tees:
    outside the armhole mean 0.12 to 0.23 mm, max 0.72 mm; the default armhole (its shape is
    not visible in the photo) mean 0.67 to 1.04 mm, max 1.79 mm, length error up to 0.83 mm;
    whole outline mean 0.38 mm worst. The synth armhole is a drafting-style curve, so these
    armhole numbers flatter the default; a real armhole will differ (fix item 4)."""
    half, edges = extract_mask_free(tees, session_client, name)
    want = visible_half(tees[name][1], "front")
    a, b = tees[name][1]["photos"]["front"]["edges"]["armhole"]
    got = half.array()
    got_tree = cKDTree(contour._resample_closed(got, 0.25))
    d_rest = got_tree.query(np.vstack([want[:a], want[b + 1 :]]))[0]
    d_arm = got_tree.query(contour._resample_open(want[a : b + 1], 0.5))[0]
    assert d_rest.mean() < 0.5 and d_rest.max() < 1.5
    assert d_arm.mean() < 1.5 and d_arm.max() < 2.5
    assert outline_distance(got, want) < 1.0  # including the armhole
    arm_got = edge_lengths(half, edges)["armhole"]
    assert arm_got == pytest.approx(truth_edge_lengths(tees[name][1], "front")["armhole"], abs=1.5)


def extract_mask_free(tees, client, name):
    return extract(client.segment(tees[name][0] / "front.jpg", "t-shirt"), truth_homography(tees[name][1], "front"), "front")


def test_req_geo_01_reviewer_repro_whole_garment_sidecar(tees):
    """The review repro: the raw whole-garment sidecar mask (sleeves included) used to give
    chest +67.9, body length -33.1, shoulder +313.0 and armhole depth +37.5 mm."""
    from sewtrace.vision.measure import measure_tee

    d, truth = tees["tee1"]
    raw = cv2.imread(str(d / "front.masks" / "t-shirt.png"), cv2.IMREAD_GRAYSCALE) > 127
    front = extract(raw, truth_homography(truth, "front"), "front")
    back_raw = cv2.imread(str(d / "back.masks" / "t-shirt.png"), cv2.IMREAD_GRAYSCALE) > 127
    back = extract(back_raw, truth_homography(truth, "back"), "back")
    sleeve_mask = cv2.imread(str(d / "sleeve.masks" / "sleeve.png"), cv2.IMREAD_GRAYSCALE) > 127
    sleeve = extract(sleeve_mask, truth_homography(truth, "sleeve"), "sleeve")
    got = measure_tee(*front, *back, *sleeve)
    for k in ("chest_width", "body_length", "shoulder_width", "armhole_depth"):
        assert got[k] == pytest.approx(truth["measurements"][k], abs=2.0), k


def test_req_geo_01_body_only_mask_is_an_error(tees):
    """A mask without sleeves has no joints: fail loudly instead of cutting the body."""
    d, truth = tees["tee1"]
    body = cv2.imread(str(d / "front.masks" / "body.png"), cv2.IMREAD_GRAYSCALE) > 127
    outline = contour.smooth_outline(contour.mask_to_outline(body, truth_homography(truth, "front")))
    with pytest.raises(ValueError, match="sleeves"):
        contour.remove_sleeves(outline)


def test_req_geo_01_default_armhole_geometry():
    """The default armhole starts and ends on the joints, bows towards the body centre and
    leaves the shoulder seam and meets the side seam at right angles."""
    shoulder, underarm = np.array([140.0, 12.0]), np.array([160.0, 152.0])
    arm = contour.default_armhole(shoulder, underarm, np.array([-1.0, -0.15]) / 1.011,
                                  np.array([0.02, 1.0]), np.array([0.0, 200.0]))
    assert np.allclose(arm[0], shoulder) and np.allclose(arm[-1], underarm)
    assert arm[:, 0].min() < shoulder[0]  # bows towards the centre (x = 0)
    start, end = arm[1] - arm[0], arm[-1] - arm[-2]
    assert abs(start @ np.array([-1.0, -0.15]) / np.linalg.norm(start) / 1.011) < 0.05
    assert abs(end @ np.array([0.02, 1.0]) / np.linalg.norm(end)) < 0.05
