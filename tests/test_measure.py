import numpy as np
import pytest

from sewtrace.mock import synth
from sewtrace.types import MEASUREMENT_KEYS, Edge, Outline
from sewtrace.vision.measure import measure_tee
from test_contour import (  # noqa: F401  (shared session fixtures)
    TEE_NAMES,
    extract,
    halves,
    rotate_photo,
    session_client,
    tees,
    truth_homography,
)

TOL_MM = 2.0


def _edges(ranges: dict) -> list[Edge]:
    return [Edge(name=k, start=a, end=b) for k, (a, b) in ranges.items()]


@pytest.mark.parametrize("name", TEE_NAMES)
def test_measurements_match_truth(halves, tees, name):
    got = measure_tee(*halves[name, "front"], *halves[name, "sleeve"])
    want = tees[name][1]["measurements"]
    assert set(got) == set(MEASUREMENT_KEYS)
    for k in MEASUREMENT_KEYS:
        assert got[k] == pytest.approx(want[k], abs=TOL_MM), k


def test_measurements_on_rotated_photos(tees, session_client):
    d, truth = tees["tee4"]
    parts = []
    for view, deg, frame_deg in (("front", 6.0, -2.0), ("sleeve", -4.0, 3.0)):
        mask = session_client.segment(d / f"{view}.jpg", "sleeve" if view == "sleeve" else "body")
        rmask, H = rotate_photo(mask, truth_homography(truth, view), deg, frame_deg)
        parts.extend(extract(rmask, H, view))
    got = measure_tee(*parts)
    for k in MEASUREMENT_KEYS:
        assert got[k] == pytest.approx(truth["measurements"][k], abs=TOL_MM), k


def test_measurements_on_exact_truth_outlines():
    """Exact synth geometry with its own edge labels gives the parameters back."""
    p = synth.TEES[2]
    body, be = synth.body_half(p)
    sleeve, se = synth.sleeve_half(p)
    got = measure_tee(Outline.from_array(body), _edges(be), Outline.from_array(sleeve), _edges(se))
    for k, v in synth.measurements(p).items():
        assert got[k] == pytest.approx(v, abs=0.5), k
    assert np.isfinite(list(got.values())).all()
