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
from sewtrace.types import MEASUREMENT_KEYS


def test_photos_to_pattern_end_to_end(tmp_path):
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
    """Model answers do nothing and the code fix is a no-op, so seams stay broken."""
    real = pipeline.adjust_loop

    def broken(client, spec):
        front = spec.panel("front")
        longer = seams.set_edge_length(front, "armhole", seams.edge_length(front, "armhole") + 16)
        spec = spec.model_copy(update={"panels": [longer if p.name == "front" else p for p in spec.panels]})
        monkeypatch.setattr(seams, "MOCK_FIX_FRACTION", 0.0)
        monkeypatch.setattr(seams, "fallback_fix", lambda s, passes=3: (s, []))
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
    with pytest.raises(SeamCheckFailed, match=r"front\.side~back\.side.*\+") as err:
        run(tee, out, client)
    assert len(err.value.failed) == 2 and "sleeve.cap" in str(err.value)
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
    assert "front.side~back.side" in capsys.readouterr().err
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
