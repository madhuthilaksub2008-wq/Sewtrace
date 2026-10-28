import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sewtrace.config import get_config
from sewtrace.mock import synth
from sewtrace.vision import marker
from sewtrace.vision.marker import (
    Homography,
    MarkerNotFound,
    MarkerRejected,
    _apply,
    detect_marker,
    homography_from_marker,
    photo_homography,
)

VIEWS = ("front", "back", "sleeve")


@pytest.fixture(scope="session")
def tees(tmp_path_factory) -> list[Path]:
    out = tmp_path_factory.mktemp("marker_tees")
    dirs = []
    for i, p in enumerate(synth.TEES):
        synth.make_tee(p, out / p.name, seed=i)
        dirs.append(out / p.name)
    return dirs


def _square_scene(rng: np.random.Generator, size: int = 900) -> tuple[np.ndarray, np.ndarray]:
    """Marker (300 px) on white, warped by a random camera. Returns photo, true corners px."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    world = np.full((size, size), 240, np.uint8)
    world[300:600, 300:600] = cv2.aruco.generateImageMarker(d, 0, 300)
    cam = synth._camera_homography(size, size, rng)
    photo = cv2.warpPerspective(world, cam, (size, size), borderMode=cv2.BORDER_REPLICATE)
    photo = cv2.GaussianBlur(photo, (0, 0), 0.8)
    # Pixel centres sit on integers, so the black square spans 299.5 .. 599.5.
    corners = np.array([[299.5, 299.5], [599.5, 299.5], [599.5, 599.5], [299.5, 599.5]])
    return photo, _apply(cam, corners)


def test_known_square_corners():
    """REQ-SCL-01: known square corners."""
    rng = np.random.default_rng(3)
    for _ in range(3):
        photo, truth = _square_scene(rng)
        corners = detect_marker(photo)
        assert np.abs(corners - truth).max() < 0.03


def test_homography_maps_corners_to_marker_square():
    """REQ-SCL-02: homography maps corners to marker square."""
    corners = np.array([[110.0, 205.0], [402.0, 198.0], [410.0, 497.0], [104.0, 503.0]])
    hom = homography_from_marker(corners, 100.0)
    expect = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    assert np.allclose(hom.px_to_mm(corners), expect, atol=1e-9)
    pts = np.array([[0.0, 0.0], [-350.0, 820.0], [1200.0, -40.0]])
    assert np.allclose(hom.px_to_mm(hom.mm_to_px(pts)), pts, atol=1e-8)


def test_corner_order_follows_marker_not_image():
    """REQ-SCL-02: corner order follows marker not image."""
    photo, truth = _square_scene(np.random.default_rng(5))
    rotated = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE)
    h = photo.shape[0]
    # (x, y) -> (h - 1 - y, x) under a clockwise quarter turn.
    expect = np.c_[h - 1 - truth[:, 1], truth[:, 0]]
    assert np.abs(detect_marker(rotated) - expect).max() < 0.03


def test_marker_not_found():
    """REQ-CAP-02: marker not found."""
    with pytest.raises(MarkerNotFound):
        detect_marker(np.full((400, 400, 3), 200, np.uint8))


def _garment_error(tee_dir: Path, view: str) -> np.ndarray:
    """Per-point mm error of photo_homography on the garment outline of one photo."""
    truth = json.loads((tee_dir / "truth.json").read_text())["photos"][view]
    H_world = np.array(truth["world_mm_to_photo_px"])
    # The synth paints the marker from pixel index `off`, so its drawn edge is half
    # a world pixel before marker_corners_mm (pixel centres sit on integers).
    marker_tl = np.array(truth["marker_corners_mm"][0]) - 0.5 / synth.PX_PER_MM
    half = np.array(truth["half_outline_mm"])
    outline = synth.mirror_full(half) if view != "sleeve" else half
    world = outline + truth["origin_mm"]
    hom = photo_homography(cv2.imread(str(tee_dir / f"{view}.jpg")))
    assert isinstance(hom, Homography)
    measured = hom.px_to_mm(_apply(H_world, world))
    return np.linalg.norm(measured - (world - marker_tl), axis=1)


@pytest.fixture(scope="session")
def garment_errors(tees) -> dict[tuple[int, str], np.ndarray]:
    return {(i, v): _garment_error(d, v) for i, d in enumerate(tees) for v in VIEWS}


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("tee", range(len(synth.TEES)))
def test_garment_points_within_tolerance(garment_errors, tee, view):
    """REQ-SCL-02: garment points within tolerance."""
    err = garment_errors[tee, view]
    print(f"{synth.TEES[tee].name} {view}: max {err.max():.3f} mm, mean {err.mean():.3f} mm")
    assert err.max() < 0.8


def test_garment_error_overall(garment_errors):
    """REQ-SCL-02: garment error overall."""
    errs = np.concatenate(list(garment_errors.values()))
    print(f"all photos: max {errs.max():.3f} mm, mean {errs.mean():.3f} mm, p95 {np.percentile(errs, 95):.3f} mm")
    assert errs.mean() < 0.2
    assert np.percentile(errs, 95) < 0.5


# ---- REQ-SCL-02, REQ-SCL-03, REQ-CAP-02 ------------------------------------------------

@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("tee", range(len(synth.TEES)))
def test_req_scl_02_measured_marker_side_is_100_mm(tees, tee, view):
    """REQ-SCL-02: on tilted synthetic photos, the true marker corners (from the synth
    camera, not from the detector) are 100 mm +/- 1.0 mm apart after the homography."""
    d = tees[tee]
    truth = json.loads((d / "truth.json").read_text())["photos"][view]
    # The drawn marker edge is half a world pixel before marker_corners_mm (see _garment_error).
    corners_world = np.array(truth["marker_corners_mm"]) - 0.5 / synth.PX_PER_MM
    hom = photo_homography(cv2.imread(str(d / f"{view}.jpg")))
    mm = hom.px_to_mm(_apply(np.array(truth["world_mm_to_photo_px"]), corners_world))
    sides = np.linalg.norm(mm - np.roll(mm, -1, axis=0), axis=1)
    assert np.abs(sides - 100.0).max() <= 1.0
    assert hom.marker_error_mm <= get_config().marker.tolerance_mm


def _bowed_marker_photo(amp_px: float) -> np.ndarray:
    """Marker on a card that bows by amp_px in the middle (not flat), then a tilted camera."""
    size = 900
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    world = np.full((size, size), 240, np.uint8)
    world[300:600, 300:600] = cv2.aruco.generateImageMarker(d, 0, 300)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    bow = amp_px * np.sin(np.pi * np.clip((xx - 300) / 300, 0, 1)) * np.sin(np.pi * np.clip((yy - 300) / 300, 0, 1))
    world = cv2.remap(world, xx, yy + bow, cv2.INTER_LINEAR, borderValue=240)
    cam = synth._camera_homography(size, size, np.random.default_rng(3))
    photo = cv2.warpPerspective(world, cam, (size, size), borderMode=cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(photo, (0, 0), 0.8)


def test_req_scl_03_flat_marker_passes_and_reports_error():
    """REQ-SCL-03: a flat tilted marker is accepted; its error is far below the tolerance."""
    hom = photo_homography(_bowed_marker_photo(0.0), "front photo f.jpg")
    assert 0.0 <= hom.marker_error_mm < 0.1 * get_config().marker.tolerance_mm


def test_req_scl_03_distorted_marker_is_rejected_with_retake_advice():
    """REQ-SCL-03: a marker that is not flat (3 mm bow) is rejected; the message names the
    photo and tells the user to take it again from directly above."""
    with pytest.raises(MarkerRejected, match=r"front photo f\.jpg.*directly above"):
        photo_homography(_bowed_marker_photo(9.0), "front photo f.jpg")


def test_req_scl_03_error_over_tolerance_is_rejected(monkeypatch):
    """REQ-SCL-03: the limit is marker.tolerance_mm; the message gives the error and the limit."""
    photo, _ = _square_scene(np.random.default_rng(3))
    monkeypatch.setattr(marker, "marker_error_mm", lambda *a, **k: 1.4)
    with pytest.raises(MarkerRejected, match=r"s\.jpg: marker error 1\.40 mm is more than 1 mm.*directly above"):
        photo_homography(photo, "s.jpg")
    monkeypatch.setattr(marker, "marker_error_mm", lambda *a, **k: 0.99)
    assert photo_homography(photo, "s.jpg").marker_error_mm == 0.99


def test_req_cap_02_missing_or_partial_marker_names_the_photo(tees):
    """REQ-CAP-02: no marker, or only part of it, is rejected with the photo's name."""
    with pytest.raises(MarkerNotFound, match=r"back photo b\.jpg"):
        photo_homography(np.full((400, 400, 3), 200, np.uint8), "back photo b.jpg")
    d = tees[0]
    image = cv2.imread(str(d / "front.jpg"))
    truth = json.loads((d / "truth.json").read_text())["photos"]["front"]
    px = _apply(np.array(truth["world_mm_to_photo_px"]), np.array(truth["marker_corners_mm"]))
    x0, y0 = px.min(axis=0).astype(int)
    x1, y1 = px.max(axis=0).astype(int)
    image[y0 : (y0 + y1) // 2, x0:x1] = 245  # cover the top half of the marker
    with pytest.raises(MarkerNotFound, match=r"front photo f\.jpg"):
        photo_homography(image, "front photo f.jpg")


def _fake_grid(monkeypatch, found: float):
    """Grid points whose residuals are 0.00 to 1.00 mm, for a photo where px = mm."""
    pts = np.c_[np.linspace(0, 1, 101), np.zeros(101)]
    monkeypatch.setattr(marker, "_grid_edge_points",
                        lambda *a, **k: (pts, np.zeros(101, int), np.zeros(101), found))
    return np.float64([[0, 0], [100, 0], [100, 100], [0, 100]])


def _marker_cfg(**update):
    cfg = get_config()
    return cfg.model_copy(update={"marker": cfg.marker.model_copy(update=update)})


def test_req_api_05_marker_percentile_comes_from_config(monkeypatch):
    """REQ-API-05, REQ-SCL-03: the percentile of the marker error is marker.error_percentile."""
    corners = _fake_grid(monkeypatch, found=1.0)
    gray = np.zeros((10, 10), np.uint8)
    assert marker.marker_error_mm(gray, corners) == pytest.approx(0.95, abs=1e-6)
    monkeypatch.setattr(marker, "get_config", lambda: _marker_cfg(error_percentile=50))
    assert marker.marker_error_mm(gray, corners) == pytest.approx(0.5, abs=1e-6)


def test_req_api_05_min_grid_found_comes_from_config(monkeypatch):
    """REQ-API-05, REQ-SCL-03: the share of marker edges that must be found is marker.min_grid_found."""
    corners = _fake_grid(monkeypatch, found=0.85)
    gray = np.zeros((10, 10), np.uint8)
    with pytest.raises(ValueError, match="85%"):
        marker.marker_error_mm(gray, corners)
    monkeypatch.setattr(marker, "get_config", lambda: _marker_cfg(min_grid_found=0.8))
    marker.marker_error_mm(gray, corners)
