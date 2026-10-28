from pathlib import Path

import cv2
import numpy as np
import pytest

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings
from sewtrace.mock import synth
from sewtrace.mock.proof import make_proof
from sewtrace.vision.marker import photo_homography
from sewtrace.vision.verify import ProofResult, edge_gaps, gap_result, overlay_image, verify_proof

CASES = {
    "perfect": {},
    "scaled": {"scale": 1.02},
    "shifted": {"offset_mm": (4.0, 0.0)},
}


@pytest.fixture(scope="session")
def proofs(tmp_path_factory) -> tuple[Path, dict[str, dict]]:
    tee_dir = tmp_path_factory.mktemp("proof") / "tee1"
    synth.make_tee(synth.TEES[0], tee_dir, seed=0)
    return tee_dir, {name: make_proof(tee_dir, seed=7, name=name, **kw) for name, kw in CASES.items()}


@pytest.fixture
def client(tmp_path) -> MetaClient:
    return MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path))


def _verify(proofs, client, name: str) -> tuple[ProofResult, dict]:
    tee_dir, truth = proofs
    photo = tee_dir / f"{name}.jpg"
    hom = photo_homography(cv2.imread(str(photo)))
    return verify_proof(photo, client, hom), truth[name]


def test_proof_writes_photo_and_masks(proofs):
    tee_dir, _ = proofs
    for mask in ("paper", "body"):
        m = cv2.imread(str(tee_dir / "perfect.masks" / f"{mask}.png"), cv2.IMREAD_GRAYSCALE)
        assert m is not None and (m > 127).sum() > 10_000
    assert (tee_dir / "perfect.jpg").exists() and (tee_dir / "perfect.json").exists()


def test_edge_gaps_square():
    inner = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    outer = inner * 1.04 - 2.0
    pts, d = edge_gaps(outer, inner, step_mm=2.0)
    corner = np.isclose(np.abs(pts - 50), 52).all(axis=1)  # outer corners sit 2*sqrt(2) out
    assert corner.sum() == 4
    assert np.allclose(d[~corner], 2.0) and np.allclose(d[corner], 2 * np.sqrt(2))
    r = gap_result(pts, d, tol_mm=3.0)
    assert r.frac_within_tol == 1.0 and r.max_mm == pytest.approx(2 * np.sqrt(2))


def test_perfect_paper_has_small_gap(proofs, client):
    r, truth = _verify(proofs, client, "perfect")
    print(f"perfect: mean {r.mean_mm:.2f} max {r.max_mm:.2f} p95 {r.p95_mm:.2f} mm")
    assert truth["expected_gap_mean_mm"] < 1e-6
    assert r.mean_mm < 1.0
    assert r.frac_within_tol > 0.98
    assert len(r.points_mm) == len(r.distances_mm) > 500


@pytest.mark.parametrize("name", ["scaled", "shifted"])
def test_gap_matches_known_paper_error(proofs, client, name):
    r, truth = _verify(proofs, client, name)
    expected = truth["expected_gap_mean_mm"]
    print(f"{name}: measured mean {r.mean_mm:.2f} mm, expected {expected:.2f} mm")
    assert expected > 2.0
    assert r.mean_mm == pytest.approx(expected, abs=0.5)
    assert r.max_mm == pytest.approx(truth["expected_gap_max_mm"], abs=1.0)
    assert r.frac_within_tol < 0.9


def test_overlay_marks_gaps_red_and_green(proofs, client, tmp_path):
    tee_dir, _ = proofs
    r, _ = _verify(proofs, client, "scaled")
    hom = photo_homography(cv2.imread(str(tee_dir / "scaled.jpg")))
    out = overlay_image(tee_dir / "scaled.jpg", r, hom, tmp_path / "overlay.png")
    img = cv2.imread(str(out))
    b, g, rr = (img[..., i].astype(int) for i in range(3))
    assert ((rr > 200) & (g < 60) & (b < 60)).sum() > 100  # red: gap over tolerance
    assert ((g > 180) & (rr < 80) & (b < 80)).sum() > 100  # green: within tolerance
