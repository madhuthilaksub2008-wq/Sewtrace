import json
from pathlib import Path

import pytest

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings
from sewtrace.mock import synth
from sewtrace import cli, pipeline
from sewtrace.pipeline import run
from sewtrace.reason import seams
from sewtrace.reason.seams import SeamCheckFailed
from perturb import set_edge_length
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
