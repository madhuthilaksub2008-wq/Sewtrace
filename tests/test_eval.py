"""Ground truth for eval: synthetic truth.json, real tee data/ground_truth/<tee>.json."""

import json

import pytest

from sewtrace.eval import GroundTruth, GroundTruthMissing, evaluate, ground_truth
from sewtrace.mock import synth
from sewtrace.types import MEASUREMENT_KEYS

SIX = dict(chest_width=320.0, body_length=380.0, shoulder_width=280.0,
           sleeve_length=110.0, neck_width=124.0, armhole_depth=140.0)


def _tape(tmp_path, tee: str, data: dict) -> None:
    folder = tmp_path / "ground_truth"
    folder.mkdir(exist_ok=True)
    (folder / f"{tee}.json").write_text(json.dumps(data))


def test_req_geo_04_synthetic_tee_uses_truth_json(tmp_path):
    """REQ-GEO-04 (SPEC 10.1): a tee folder with truth.json needs no tape file."""
    tee = tmp_path / "synth" / "tee1"
    truth = synth.make_tee(synth.TEES[0], tee, seed=0)["measurements"]
    assert ground_truth(tee, tmp_path / "ground_truth") == truth


def test_req_geo_04_real_tee_reads_tape_file_and_means_readings(tmp_path):
    """REQ-GEO-04 (SPEC 10.3): a real tee takes the mean of its readings, in mm."""
    tee = tmp_path / "raw" / "tee1"
    tee.mkdir(parents=True)
    _tape(tmp_path, "tee1", {**SIX, "chest_width": [320.0, 321.0], "neck_width": [124.0, 125.0, 126.5]})
    got = ground_truth(tee, tmp_path / "ground_truth")
    assert got["chest_width"] == pytest.approx(320.5)
    assert got["neck_width"] == pytest.approx(125.1666, abs=1e-3)
    assert got["body_length"] == 380.0 and set(got) == set(MEASUREMENT_KEYS)


def test_req_geo_04_real_tee_without_tape_file_fails_clearly(tmp_path):
    """REQ-GEO-04 (SPEC 10.1): no fallback to synthetic parameters; the error says what to do."""
    tee = tmp_path / "raw" / "tee1"
    tee.mkdir(parents=True)
    with pytest.raises(GroundTruthMissing, match=r"tee1 has no truth\.json and no .*ground_truth.tee1\.json.*SPEC 10\.3"):
        ground_truth(tee, tmp_path / "ground_truth")


def test_req_geo_04_readings_that_disagree_need_a_third():
    """REQ-GEO-04 (SPEC 10.3 step 5): two readings more than 2 mm apart are rejected."""
    with pytest.raises(ValueError, match="Measure a third time"):
        GroundTruth.model_validate({**SIX, "chest_width": [320.0, 323.0]})
    assert GroundTruth.model_validate({**SIX, "chest_width": [320.0, 323.0, 321.0]}).means_mm()["chest_width"] \
        == pytest.approx(321.3333, abs=1e-3)


def test_req_geo_04_tape_file_must_have_all_six_values():
    """REQ-GEO-04: a missing, extra or non-positive value is an error."""
    short = {k: v for k, v in SIX.items() if k != "neck_width"}
    for bad in (short, {**SIX, "extra": 1.0}, {**SIX, "body_length": 0.0}):
        with pytest.raises(ValueError):
            GroundTruth.model_validate(bad)


def test_req_geo_04_eval_stops_before_any_run_when_truth_is_missing(tmp_path, monkeypatch):
    """REQ-GEO-04: eval on a real tee folder with no tape file fails before the pipeline runs."""
    import sewtrace.eval as ev

    raw = tmp_path / "raw"
    synth.make_tee(synth.TEES[0], raw / "tee1", seed=0)
    (raw / "tee1" / "truth.json").unlink()  # now it looks like a real tee
    monkeypatch.setattr(ev, "run", lambda *a, **k: pytest.fail("pipeline ran"))
    with pytest.raises(GroundTruthMissing, match="tee1"):
        evaluate(raw, tmp_path / "out")
