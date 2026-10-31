import json
from pathlib import Path

import pytest
from perturb import set_edge_length

from sewtrace import cli, pipeline
from sewtrace.api.client import MetaClient
from sewtrace.config import Settings
from sewtrace.mock import synth
from sewtrace.pipeline import run
from sewtrace.reason import seams
from sewtrace.reason.seams import SeamCheckFailed
from sewtrace.types import MEASUREMENT_KEYS


def test_photos_to_pattern_end_to_end(tmp_path):
    """REQ-CAP-01: photos to pattern end to end."""
    tee = tmp_path / "tee3"
    synth.make_tee(synth.TEES[2], tee, seed=2)
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache"))

    result = run(tee, tmp_path / "out", client)

    truth = json.loads((tee / "truth.json").read_text())["measurements"]
    for k in MEASUREMENT_KEYS:
        assert result.measurements[k] == pytest.approx(truth[k], abs=2.0), k
    assert all(s.ok for s in result.spec.seams)
    for path in result.files.values():
        assert path.exists() and path.stat().st_size > 0

    again = run(tee, tmp_path / "out2", client)
    assert again.model_calls == 0  # every call served from cache


def test_req_rsn_05_correct_front_depth_is_accepted_first_try_in_the_pipeline(tmp_path, caplog):
    """REQ-RSN-04, REQ-RSN-05, SPEC 10.1: the front outline follows the back neckline, yet the exported
    front neck depth is the inferred one. Circular on the front depth: the mock reads the synthetic
    truth.json, so the value is copied. It proves the deep neck passes the check first try."""
    from sewtrace.reason.reshape import neck_depth_mm
    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache"))
    with caplog.at_level("WARNING", logger="sewtrace.reason.panels"):
        result = run(tee, tmp_path / "out", client)
    # The deeper front neck must pass the measured-value check on a real (noisy) outline, first try.
    assert not [r for r in caplog.records if "infer_panels" in r.getMessage()], caplog.text
    params = json.loads((tee / "truth.json").read_text())["params"]
    assert neck_depth_mm(result.spec.panel("front")) == pytest.approx(params["front_neck_depth"], abs=0.1)
    assert neck_depth_mm(result.spec.panel("back")) == pytest.approx(params["back_neck_depth"], abs=3.0)


def _unfixable(monkeypatch):
    """The back shoulder is 6 mm too long. Two measured edges disagree, so nothing may fix it."""
    real = pipeline.adjust_loop

    def broken(client, spec):
        back = spec.panel("back")
        longer = set_edge_length(back, "shoulder", seams.edge_length(back, "shoulder") + 6)
        spec = spec.model_copy(update={"panels": [longer if p.name == "back" else p for p in spec.panels]})
        return real(client, spec)

    monkeypatch.setattr(pipeline, "adjust_loop", broken)


def test_req_sew_04_failed_seams_stop_before_any_file(tmp_path, monkeypatch):
    """REQ-SEW-04: run() raises, names each failed pair, renders nothing."""
    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache"))
    _unfixable(monkeypatch)
    for name in ("render_pdf", "render_svg", "render_dxf"):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: pytest.fail("render called"))

    out = tmp_path / "out"
    with pytest.raises(SeamCheckFailed, match=r"front\.shoulder~back\.shoulder.*\+") as err:
        run(tee, out, client)
    assert "front.shoulder~back.shoulder" in [seams.seam_name(s) for s in err.value.failed]
    assert not out.exists() or not any(out.iterdir())


def test_req_sew_03_small_side_mismatch_is_fixed_and_exported(tmp_path, monkeypatch):
    """REQ-SEW-03, D-02: back side 2 mm long. The code fix balances it, files are written,
    and the six reported measurements are the measured ones (taken before the fix)."""
    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache"))
    real = pipeline.adjust_loop
    seen = {}

    def skewed(client, spec):
        back = spec.panel("back")
        longer = set_edge_length(back, "side", seams.edge_length(back, "side") + 2.0)
        spec = spec.model_copy(update={"panels": [longer if p.name == "back" else p for p in spec.panels]})
        seen["before"] = spec
        return real(client, spec)

    monkeypatch.setattr(pipeline, "adjust_loop", skewed)
    result = run(tee, tmp_path / "out", client)
    assert all(s.ok for s in result.spec.seams)
    assert any(r["source"] == "pair_fix" and r["applied"] for r in result.seam_log)
    assert all(path.exists() and path.stat().st_size > 0 for path in result.files.values())
    logged = json.loads(result.files["seam_log"].read_text())
    assert any(r["source"] == "pair_fix" for r in logged)
    truth = json.loads((tee / "truth.json").read_text())["measurements"]
    for k in MEASUREMENT_KEYS:
        assert result.measurements[k] == pytest.approx(truth[k], abs=2.0), k
    assert json.loads(result.files["measurements"].read_text()) == result.measurements


def test_req_sew_03_cli_prints_one_line_per_edge_changed_by_the_pair_fix(tmp_path, monkeypatch, capsys):
    """REQ-SEW-03: a changed measured edge is shown with from_mm and to_mm next to the seam result."""
    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)
    monkeypatch.setenv("SEWTRACE_OFFLINE", "0")
    real = pipeline.adjust_loop

    def skewed(client, spec):
        back = spec.panel("back")
        longer = set_edge_length(back, "side", seams.edge_length(back, "side") + 2.0)
        return real(client, spec.model_copy(update={"panels": [longer if p.name == "back" else p for p in spec.panels]}))

    monkeypatch.setattr(pipeline, "adjust_loop", skewed)
    real_run = pipeline.run
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache"))
    monkeypatch.setattr(pipeline, "run", lambda d, o: real_run(d, o, client))
    cli.main(["run", str(tee), "--out", str(tmp_path / "out")])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("seam fix:")]
    assert len(lines) == 2  # front.side and back.side
    assert all("mm ->" in ln and "front.side~back.side" in ln for ln in lines)
    assert any(ln.startswith("seam fix: back.side ") for ln in lines)


def test_req_sew_04_cli_prints_failure_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)
    monkeypatch.setenv("SEWTRACE_OFFLINE", "0")
    _unfixable(monkeypatch)
    real_run = pipeline.run
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache"))
    monkeypatch.setattr(pipeline, "run", lambda d, o: real_run(d, o, client))
    with pytest.raises(SystemExit) as exit_:
        cli.main(["run", str(tee), "--out", str(tmp_path / "out")])
    assert exit_.value.code == 1
    assert "front.shoulder~back.shoulder" in capsys.readouterr().err
    assert not (tmp_path / "out" / "pattern.pdf").exists()


def test_req_seg_02_pipeline_asks_for_spec_prompts(tmp_path):
    """REQ-SEG-02: front and back ask for "t-shirt", the sleeve photo for "sleeve"."""
    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache"))
    asked = []
    real = client.segment
    client.segment = lambda photo, prompt: (asked.append((Path(photo).name, prompt)), real(photo, prompt))[1]

    run(tee, tmp_path / "out", client)

    assert asked == [("front.jpg", "t-shirt"), ("back.jpg", "t-shirt"), ("sleeve.jpg", "sleeve")]


def _client(tmp_path) -> MetaClient:
    return MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache"))


@pytest.mark.parametrize("damage", ["missing", "not_an_image"])
def test_req_cap_01_missing_photo_names_view_and_path_before_any_call(tmp_path, damage):
    """REQ-CAP-01: a missing or unreadable photo gives a clear error with the view and the
    path, and no model call is made."""
    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)
    if damage == "missing":
        (tee / "back.jpg").unlink()
    else:
        (tee / "back.jpg").write_bytes(b"this is not a jpeg")
    client = _client(tmp_path)
    with pytest.raises(pipeline.PhotoMissing, match=r"back photo .*back\.jpg is missing or is not an image") as err:
        run(tee, tmp_path / "out", client)
    assert str(tee / "back.jpg") in str(err.value)
    assert client.calls == 0


def test_req_cap_02_marker_not_found_names_photo_before_any_call(tmp_path):
    """REQ-CAP-02: a photo with no marker is rejected with its name; nothing is paid for first."""
    import cv2
    import numpy as np

    from sewtrace.vision.marker import MarkerNotFound

    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)
    cv2.imwrite(str(tee / "sleeve.jpg"), np.full((600, 600, 3), 120, np.uint8))
    client = _client(tmp_path)
    with pytest.raises(MarkerNotFound, match=r"sleeve photo .*sleeve\.jpg"):
        run(tee, tmp_path / "out", client)
    assert client.calls == 0


def test_req_cap_01_cli_missing_photo_prints_error_and_exits_nonzero(tmp_path, capsys):
    """REQ-CAP-01: the CLI shows the message, not a traceback."""
    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)
    (tee / "front.jpg").unlink()
    with pytest.raises(SystemExit) as exit_:
        cli.main(["run", str(tee), "--out", str(tmp_path / "out")])
    assert exit_.value.code == 1
    assert "front photo" in capsys.readouterr().err


def test_req_geo_01_contour_failure_names_the_photo(tmp_path, monkeypatch):
    """REQ-GEO-01: a joint-finder ValueError becomes ContourFailed with the view and path."""
    from sewtrace.vision import contour
    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)

    def no_joints(outline):
        raise ValueError("found no underarm corners")

    monkeypatch.setattr(contour, "remove_sleeves", no_joints)
    with pytest.raises(pipeline.ContourFailed, match=r"front photo .*front\.jpg: found no underarm corners"):
        run(tee, tmp_path / "out", _client(tmp_path))


def test_req_geo_01_unmirrored_corners_say_to_lay_the_sleeves_the_same_way(tmp_path, monkeypatch, caplog):
    """REQ-GEO-01: CornersNotMirrored becomes ContourFailed that keeps its advice, ends with
    "Take the front photo again." and does not add the generic whole t-shirt advice. Numbers are logged."""
    from sewtrace.vision import contour
    tee = tmp_path / "tee1"
    synth.make_tee(synth.TEES[0], tee, seed=0)

    def refuse(outline):
        raise contour.CornersNotMirrored(
            "the inner sleeve cuff corners do not mirror: the closest pair is 62 mm off (limit 25 mm). "
            "Lay both sleeves the same way", "cuff_top", 62.0, 25.0)

    monkeypatch.setattr(contour, "remove_sleeves", refuse)
    with caplog.at_level("WARNING"), pytest.raises(pipeline.ContourFailed) as exc:
        run(tee, tmp_path / "out", _client(tmp_path))
    text = str(exc.value)
    assert "front photo" in text and "62 mm off (limit 25 mm)" in text and text.endswith("Take the front photo again.")
    assert "flat and in view" not in text
    assert "stage cuff_top, offset 62.0 mm, limit 25.0 mm" in caplog.text


@pytest.mark.parametrize("error, text", [
    ("BudgetExceeded", "over the limit"),
    ("CacheMiss", "offline mode: no cached answer for step X; run once online"),
    ("ApiTimeout", "took more"),
    ("NotImplementedError", "not wired"),
    ("ContourFailed", "front photo f.jpg"),
    ("ApiRequestRejected", "rejected: HTTP 400"),
])
def test_req_api_08_cli_prints_clean_error_not_traceback(tmp_path, monkeypatch, capsys, error, text):
    """REQ-API-08: API, budget, offline and contour errors give `error: run: ...` and exit 1."""
    from sewtrace.api import client as api
    errors = {"BudgetExceeded": api.BudgetExceeded, "CacheMiss": api.CacheMiss, "ApiTimeout": api.ApiTimeout,
              "NotImplementedError": NotImplementedError, "ContourFailed": pipeline.ContourFailed,
              "ApiRequestRejected": api.ApiRequestRejected}

    def boom(d, o):
        raise errors[error](text)

    monkeypatch.setattr(pipeline, "run", boom)
    with pytest.raises(SystemExit) as exit_:
        cli.main(["run", str(tmp_path), "--out", str(tmp_path / "o")])
    assert exit_.value.code == 1
    assert capsys.readouterr().err.startswith(f"error: run: {text}")


def test_req_api_08_cache_miss_message_form(tmp_path):
    from sewtrace.api.client import CacheMiss
    client = MetaClient(Settings(api_key="", mock=False, cache_dir=tmp_path, offline=True))
    with pytest.raises(CacheMiss, match=r"^offline mode: no cached answer for reasoning call .*; run once online"):
        client.reason("p")
