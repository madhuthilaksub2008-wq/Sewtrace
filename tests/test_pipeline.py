import json

import pytest

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings
from sewtrace.mock import synth
from sewtrace.pipeline import run
from sewtrace.types import MEASUREMENT_KEYS


def test_photos_to_pattern_end_to_end(tmp_path):
    tee = tmp_path / "tee3"
    synth.make_tee(synth.TEES[2], tee, seed=2)
    client = MetaClient(Settings(api_key="", api_base_url="", mock=True, reasoning_model="spark", cache_dir=tmp_path / "cache"))

    result = run(tee, tmp_path / "out", client)

    truth = json.loads((tee / "truth.json").read_text())["measurements"]
    for k in MEASUREMENT_KEYS:
        assert result.measurements[k] == pytest.approx(truth[k], abs=2.0), k
    assert all(s.ok for s in result.spec.seams)
    for path in result.files.values():
        assert path.exists() and path.stat().st_size > 0

    again = run(tee, tmp_path / "out2", client)
    assert again.model_calls == 0  # every call served from cache
