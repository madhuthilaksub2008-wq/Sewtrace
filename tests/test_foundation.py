import json
from pathlib import Path

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
    for view in ("front", "back", "sleeve"):
        assert (tee_dir / f"{view}.jpg").exists()
        assert (tee_dir / f"{view}.masks" / "garment.png").exists()
    truth = json.loads((tee_dir / "truth.json").read_text())
    assert truth["measurements"]["chest_width"] == 320


def test_sleeve_cap_matches_armhole_plus_ease(tee_dir):
    spec = truth_spec(tee_dir)
    cap = spec.panel("sleeve").edge_points("cap")
    arm = spec.panel("front").edge_points("armhole")
    diff = synth._length(cap) - synth._length(arm)
    assert diff == pytest.approx(synth.TEES[0].cap_ease, abs=0.5)


def test_edges_cover_outline_in_order(tee_dir):
    front = truth_spec(tee_dir).panel("front")
    assert [e.name for e in front.edges] == ["neck", "shoulder", "armhole", "side", "hem", "fold"]
    for a, b in zip(front.edges, front.edges[1:]):
        assert a.end == b.start
    assert front.edges[-1].end == front.edges[0].start == 0


def test_mock_segment_is_cached(client, tee_dir):
    m1 = client.segment(tee_dir / "front.jpg", "body")
    m2 = client.segment(tee_dir / "front.jpg", "body")
    assert client.calls == 1
    assert np.array_equal(m1, m2) and m1.sum() > 0


def test_reason_mock_and_cache(client):
    answer = client.reason("hello", mock=lambda: {"ok": 1})
    assert answer == {"ok": 1}
    assert client.reason("hello", mock=lambda: {"ok": 2}) == {"ok": 1}
    assert client.calls == 1
