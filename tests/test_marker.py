import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sewtrace.mock import synth
from sewtrace.vision.marker import (
    Homography,
    MarkerNotFound,
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
    rng = np.random.default_rng(3)
    for _ in range(3):
        photo, truth = _square_scene(rng)
        corners = detect_marker(photo)
        assert np.abs(corners - truth).max() < 0.03


def test_homography_maps_corners_to_marker_square():
    corners = np.array([[110.0, 205.0], [402.0, 198.0], [410.0, 497.0], [104.0, 503.0]])
    hom = homography_from_marker(corners, 100.0)
    expect = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    assert np.allclose(hom.px_to_mm(corners), expect, atol=1e-9)
    pts = np.array([[0.0, 0.0], [-350.0, 820.0], [1200.0, -40.0]])
    assert np.allclose(hom.px_to_mm(hom.mm_to_px(pts)), pts, atol=1e-8)


def test_corner_order_follows_marker_not_image():
    photo, truth = _square_scene(np.random.default_rng(5))
    rotated = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE)
    h = photo.shape[0]
    # (x, y) -> (h - 1 - y, x) under a clockwise quarter turn.
    expect = np.c_[h - 1 - truth[:, 1], truth[:, 0]]
    assert np.abs(detect_marker(rotated) - expect).max() < 0.03


def test_marker_not_found():
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
    err = garment_errors[tee, view]
    print(f"{synth.TEES[tee].name} {view}: max {err.max():.3f} mm, mean {err.mean():.3f} mm")
    assert err.max() < 0.8


def test_garment_error_overall(garment_errors):
    errs = np.concatenate(list(garment_errors.values()))
    print(f"all photos: max {errs.max():.3f} mm, mean {errs.mean():.3f} mm, p95 {np.percentile(errs, 95):.3f} mm")
    assert errs.mean() < 0.2
    assert np.percentile(errs, 95) < 0.5
