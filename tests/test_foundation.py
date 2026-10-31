import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings
from sewtrace.mock import synth
from sewtrace.mock.fixtures import truth_spec


@pytest.fixture(scope="session")
def tee_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("raw")
    synth.make_tee(synth.TEES[0], out / "tee1", seed=0)
    return out / "tee1"


@pytest.fixture
def client(tmp_path) -> MetaClient:
    return MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path))


def test_synth_writes_photos_and_masks(tee_dir):
    """C-08: synth writes photos and masks."""
    for view in ("front", "back", "sleeve"):
        assert (tee_dir / f"{view}.jpg").exists()
        assert (tee_dir / f"{view}.masks" / "t-shirt.png").exists()
    truth = json.loads((tee_dir / "truth.json").read_text())
    # SPEC 10.2: chest is 25 mm below the underarm point; the side seam flares 4 mm over 228 mm.
    assert truth["measurements"]["chest_width"] == pytest.approx(2 * (160 + 4 * 25 / 228), abs=0.01)


def test_sleeve_cap_matches_armhole_plus_ease(tee_dir):
    """REQ-SEW-01: sleeve cap matches armhole plus ease."""
    spec = truth_spec(tee_dir)
    cap = spec.panel("sleeve").edge_points("cap")
    arm = spec.panel("front").edge_points("armhole")
    diff = synth._length(cap) - synth._length(arm)
    assert diff == pytest.approx(synth.TEES[0].cap_ease, abs=0.5)


def test_edges_cover_outline_in_order(tee_dir):
    """REQ-GEO-03: edges cover outline in order."""
    front = truth_spec(tee_dir).panel("front")
    assert [e.name for e in front.edges] == ["neck", "shoulder", "armhole", "side", "hem", "fold"]
    for a, b in zip(front.edges, front.edges[1:]):
        assert a.end == b.start
    assert front.edges[-1].end == front.edges[0].start == 0


def test_mock_segment_is_cached(client, tee_dir):
    """REQ-API-02: mock segment is cached."""
    m1 = client.segment(tee_dir / "front.jpg", "t-shirt")
    m2 = client.segment(tee_dir / "front.jpg", "t-shirt")
    assert client.calls == 1
    assert np.array_equal(m1, m2) and m1.sum() > 0


def test_reason_mock_and_cache(client):
    """REQ-API-02: reason mock and cache."""
    answer = client.reason("hello", mock=lambda: {"ok": 1})
    assert answer == {"ok": 1}
    assert client.reason("hello", mock=lambda: {"ok": 2}) == {"ok": 1}
    assert client.calls == 1


def test_req_seg_02_mock_serves_whole_garment_for_tshirt(client, tee_dir):
    """REQ-SEG-02: the mock answers "t-shirt" with the whole garment, sleeves included,
    and "sleeve" with the sleeve photo mask. It never serves the sleeveless body."""
    side = tee_dir / "front.masks"

    def sidecar(name: str) -> np.ndarray:
        return cv2.imread(str(side / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127

    got = client.segment(tee_dir / "front.jpg", "t-shirt")
    whole, body = sidecar("t-shirt"), sidecar("body")
    assert (got & whole).sum() / whole.sum() > 0.98 and (got & ~whole).sum() / whole.sum() < 0.02
    assert whole.sum() > 1.2 * body.sum()  # sleeves are in
    assert (whole & ~body).sum() > 0.2 * whole.sum()
    assert (got & ~body).sum() > 0.2 * got.sum()  # not the body mask
    assert client.segment(tee_dir / "sleeve.jpg", "sleeve").sum() > 0


def test_req_cap_01_sleeve_photo_shows_whole_tshirt_with_exact_sleeve(client, tee_dir):
    """REQ-CAP-01, REQ-CAP-03, REQ-SEG-02: the sleeve photo shows the whole t-shirt, as a real
    capture does. The "sleeve" mask is the folded half sleeve alone, with the known outline;
    the "t-shirt" mask is the whole garment and the sleeve touches the body at the cap."""
    side = tee_dir / "sleeve.masks"
    truth = json.loads((tee_dir / "truth.json").read_text())["photos"]["sleeve"]

    def sidecar(name: str) -> np.ndarray:
        return cv2.imread(str(side / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127

    sleeve, whole, body = sidecar("sleeve"), sidecar("t-shirt"), sidecar("body")
    assert whole.sum() > 5 * sleeve.sum()
    assert (whole & sleeve).sum() == sleeve.sum()  # the sleeve is part of the garment
    assert (sleeve & body).sum() < 0.01 * sleeve.sum()  # no overlap but edge pixels of the warp
    assert cv2.dilate(sleeve.astype(np.uint8), np.ones((5, 5))).astype(bool)[body].sum() > 100  # seam
    n, _ = cv2.connectedComponents(whole.astype(np.uint8))
    assert n == 2  # one garment, attached
    # The sleeve mask area is the half sleeve polygon area, in mm2, within 1 %.
    h = np.asarray(truth["world_mm_to_photo_px"])
    outline = np.asarray(truth["half_outline_mm"]) + truth["origin_mm"]
    px = cv2.perspectiveTransform(outline[None].astype(np.float64), h)[0]
    assert sleeve.sum() == pytest.approx(cv2.contourArea(px.astype(np.float32)), rel=0.01)
    # The mock SAM serves the sleeve alone for "sleeve" and the whole garment for "t-shirt".
    got = client.segment(tee_dir / "sleeve.jpg", "sleeve")
    assert (got & sleeve).sum() / sleeve.sum() > 0.98 and got.sum() < 1.1 * sleeve.sum()
    assert client.segment(tee_dir / "sleeve.jpg", "t-shirt").sum() > 5 * sleeve.sum()


def test_req_seg_02_prompts_come_from_config():
    from sewtrace.config import get_config

    prompts = get_config().sam.prompts
    assert (prompts.garment, prompts.sleeve, prompts.paper) == ("t-shirt", "sleeve", "paper")
    assert not hasattr(prompts, "mock_body")
