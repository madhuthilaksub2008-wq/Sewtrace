"""T29, T30: demo bundles and the demo page. All mock, no network, no API key."""

import dataclasses
import json
import socket
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sewtrace import cli, demo, pipeline
from sewtrace.api.client import LEDGER_COLUMNS, ApiError, MetaClient
from sewtrace.config import Settings
from sewtrace.mock import proof as mock_proof
from sewtrace.mock import synth
from sewtrace.web.app import create_app

SECRET = "sk-test-secret-0123456789"
LEDGER_MARK = "ledger-row-must-not-travel"


@pytest.fixture(scope="module")
def tee(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("synth") / "tee1"
    synth.make_tee(synth.TEES[0], folder, seed=0)
    mock_proof.make_proof(folder, scale=1.01)  # paper cut 1 % too big, as in the web sample
    return folder


@pytest.fixture(scope="module")
def recorded(tee, tmp_path_factory) -> dict:
    """One recording, shared. The client holds a key and the ledger holds a row: neither may reach the bundle."""
    work = tmp_path_factory.mktemp("rec")
    ledger = work / "ledger.csv"
    ledger.write_text(",".join(LEDGER_COLUMNS) + f"\n1,{LEDGER_MARK},0,0,0,0,0.000000,0\n")
    client = MetaClient(Settings(api_key=SECRET, mock=True, cache_dir=work / "cache", ledger_path=ledger))
    folder = demo.record(tee, "tee1", client=client, demo_dir=work / "demo", eval_csv=None)
    return {"dir": folder, "root": work / "demo", "client": client, "work": work}


def _copy(recorded, tmp_path) -> Path:
    import shutil
    root = tmp_path / "demo"
    shutil.copytree(recorded["dir"], root / "tee1")
    return root


def test_req_api_08_record_then_play_round_trip_with_zero_calls(recorded, monkeypatch):
    """REQ-API-08, C-02: a recorded bundle plays with no model call, no network and no cache folder."""
    assert recorded["client"].calls > 0  # the recording itself made (mock) calls
    manifest = json.loads((recorded["dir"] / "manifest.json").read_text())
    assert manifest["mode"] == "mock" and manifest["has_proof"] and manifest["model_calls"] == recorded["client"].calls

    def forbidden(*args, **kwargs):
        raise AssertionError("the demo must not call a model or open a socket")

    monkeypatch.setattr(MetaClient, "segment", forbidden)
    monkeypatch.setattr(MetaClient, "reason", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    bundle = demo.load_bundle("tee1", recorded["root"])
    data = demo.payload(bundle)
    assert set(data["views"]) == {"front", "back", "sleeve"}
    assert data["proof"]["result"]["mean_mm"] == pytest.approx(1.3, abs=0.6)  # paper 1 % too big
    assert {p["name"] for p in data["placement"]["panels"]} == {"front", "back", "sleeve", "neckband"}
    assert "tee1" in demo.summary(bundle)


def test_req_api_08_pipeline_replays_offline_from_the_bundle_cache_with_zero_calls(recorded, tmp_path):
    """REQ-API-08: the bundle is a garment folder, and its cache folder answers every call offline."""
    bundle = recorded["dir"]
    client = MetaClient(Settings(api_key="", mock=True, offline=True, cache_dir=bundle / "cache",
                                 ledger_path=tmp_path / "ledger.csv"))
    result = pipeline.run(bundle, tmp_path / "out", client)
    assert client.calls == 0  # nothing missed the cache
    recorded_measurements = json.loads((bundle / "measurements.json").read_text())
    assert result.measurements == pytest.approx(recorded_measurements)
    assert all(s.ok for s in result.spec.seams)


def test_req_api_08_pipeline_replay_names_a_cache_miss_offline(recorded, tmp_path):
    """Without the cache folder the same replay stops with a clear miss, and still calls nothing."""
    from sewtrace.api.client import CacheMiss
    client = MetaClient(Settings(api_key="", mock=True, offline=True, cache_dir=tmp_path / "empty",
                                 ledger_path=tmp_path / "ledger.csv"))
    with pytest.raises(CacheMiss):
        pipeline.run(recorded["dir"], tmp_path / "out", client)
    assert client.calls == 0


def test_c07_bundle_has_no_key_and_no_ledger(recorded):
    """C-07, REQ-API-04: no key, no ledger and no absolute path in any file of the bundle."""
    files = [p for p in recorded["dir"].rglob("*") if p.is_file()]
    assert files
    assert not [p for p in files if "ledger" in p.name.lower()]
    for path in files:
        data = path.read_bytes()
        assert SECRET.encode() not in data, path.name
        assert LEDGER_MARK.encode() not in data, path.name
        assert str(recorded["work"]).encode() not in data, path.name
        assert b"META_API_KEY" not in data, path.name


def test_manifest_lists_every_file_and_the_cache_keys_used(recorded):
    manifest = demo.Manifest.model_validate_json((recorded["dir"] / "manifest.json").read_text())
    on_disk = {str(p.relative_to(recorded["dir"])) for p in recorded["dir"].rglob("*") if p.is_file()}
    assert set(manifest.files) == on_disk - {"manifest.json"}
    cached = {p.name.split(".")[0] for p in (recorded["dir"] / "cache").iterdir()}
    assert set(manifest.cache_keys) == cached and len(cached) == 6  # 3 SAM, 2 proof SAM, 1 Muse


@pytest.mark.parametrize("missing", ["spec.json", "masks.json", "placement.json", "proof_result.json",
                                     "front.jpg", "seam_log.json", "manifest.json"])
def test_missing_file_gives_a_clear_error_naming_it(recorded, tmp_path, missing):
    """T30: a bundle with a missing file fails with a message that names the file and says what to do."""
    root = _copy(recorded, tmp_path)
    (root / "tee1" / missing).unlink()
    with pytest.raises(demo.DemoError, match=missing.replace(".", r"\.")) as err:
        demo.load_bundle("tee1", root)
    assert "record" in str(err.value).lower() and str(tmp_path) not in str(err.value)


def test_missing_cache_file_is_named_too(recorded, tmp_path):
    root = _copy(recorded, tmp_path)
    victim = sorted((root / "tee1" / "cache").iterdir())[0]
    victim.unlink()
    with pytest.raises(demo.DemoError, match=victim.name):
        demo.load_bundle("tee1", root)


def test_unknown_or_unsafe_name_is_refused(recorded):
    with pytest.raises(demo.DemoError, match="not found"):
        demo.load_bundle("nope", recorded["root"])
    with pytest.raises(demo.DemoError, match="not allowed"):
        demo.load_bundle("../etc", recorded["root"])


def test_req_api_04_real_mode_record_refuses_without_yes_spend(tee, tmp_path, capsys):
    """REQ-API-04: REAL mode prints the worst case and makes no call, and writes no bundle, without --yes-spend."""
    def transport(*args, **kwargs):
        raise AssertionError("a paid call was made")

    settings = Settings(api_key=SECRET, mock=False, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.csv")
    client = MetaClient(settings, transport=transport, stream_transport=transport)
    with pytest.raises(ApiError, match="--yes-spend"):
        demo.record(tee, "real1", client=client, demo_dir=tmp_path / "demo")
    out = capsys.readouterr().out
    assert "Worst case" in out and "REAL" in out
    assert client.calls == 0 and not (tmp_path / "demo").exists()


def test_worst_case_covers_sam_and_spark_calls(tee, tmp_path):
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path))
    with_proof, without = demo.worst_case_usd(client, True), demo.worst_case_usd(client, False)
    cfg = client.config
    assert with_proof - without == pytest.approx(2 * cfg.models.sam.price_per_image_usd)
    assert without > 3 * cfg.models.sam.price_per_image_usd


def test_cli_real_mode_without_yes_spend_exits_with_the_reason(tee, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("META_API_KEY", SECRET)
    monkeypatch.setenv("SEWTRACE_MOCK", "0")
    monkeypatch.delenv("SEWTRACE_OFFLINE", raising=False)
    monkeypatch.setenv("SEWTRACE_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["demo", "record", str(tee), "--demo-dir", str(tmp_path / "demo")])
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "--yes-spend" in err and SECRET not in err
    assert not (tmp_path / "demo").exists()


def test_cli_record_and_play_check_round_trip(tee, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SEWTRACE_MOCK", "1")
    monkeypatch.setenv("SEWTRACE_CACHE_DIR", str(tmp_path / "cache"))
    root = tmp_path / "demo"
    given = ["--demo-dir", str(root)]
    cli.main(["demo", "record", str(tee), "--name", "cli1", *given, "--eval-csv", str(tmp_path / "none.csv")])
    assert (root / "cli1" / "manifest.json").is_file()
    with pytest.raises(SystemExit):
        cli.main(["demo", "record", str(tee), "--name", "cli1", *given])
    assert "already exists" in capsys.readouterr().err
    cli.main(["demo", "play", "cli1", "--check", *given])
    assert "demo 'cli1'" in capsys.readouterr().out
    (root / "cli1" / "spec.json").unlink()
    with pytest.raises(SystemExit):
        cli.main(["demo", "play", "cli1", "--check", *given])
    assert "spec.json" in capsys.readouterr().err


def test_force_replaces_a_bundle_and_a_failed_record_leaves_no_folder(tee, tmp_path):
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "l.csv"))
    root = tmp_path / "demo"
    demo.record(tee, "a", client=client, demo_dir=root, eval_csv=None)
    (root / "a" / "marker.txt").write_text("old")
    with pytest.raises(demo.DemoError, match="--force"):
        demo.record(tee, "a", client=client, demo_dir=root, eval_csv=None)
    assert (root / "a" / "marker.txt").exists()  # the refusal changed nothing
    demo.record(tee, "a", client=client, demo_dir=root, eval_csv=None, force=True)
    assert not (root / "a" / "marker.txt").exists()
    broken = tmp_path / "broken"
    broken.mkdir()
    with pytest.raises(pipeline.PhotoMissing):
        demo.record(broken, "b", client=client, demo_dir=root, eval_csv=None)
    assert sorted(p.name for p in root.iterdir()) == ["a"]  # no half-written bundle, no temp folder


def test_req_geo_placement_fits_each_panel_to_its_photo_outline(recorded):
    """REQ-GEO-01: the lift starts from where the panel lies on the photo (real mm outlines, via the homography)."""
    bundle = demo.load_bundle("tee1", recorded["root"])
    for panel in bundle.placement.panels:
        if panel.photo_view is None:
            assert panel.name == "neckband" and panel.photo_px is None
            continue
        assert panel.fit_residual_mm < 3.0, panel.name  # mean distance to the mask outline, mm
        view = bundle.masks.views[panel.photo_view]
        outline = np.asarray(view.outline_px)
        px = np.asarray(panel.photo_px)
        assert len(px) == len(panel.stitch_mm) == len(panel.layout_mm)
        assert (px.min(0) >= outline.min(0) - 60).all() and (px.max(0) <= outline.max(0) + 60).all()
    sheet_w, sheet_h = bundle.placement.sheet_width_mm, bundle.placement.sheet_height_mm
    for panel in bundle.placement.panels:
        cut = np.asarray(panel.cut_layout_mm)
        assert cut.min() >= 0 and cut[:, 0].max() <= sheet_w and cut[:, 1].max() <= sheet_h


def test_mask_outlines_are_stored_in_px_and_mm_and_agree_with_the_homography(recorded):
    """C-04: the px outline and the mm outline are the same points under the stored homography."""
    bundle = demo.load_bundle("tee1", recorded["root"])
    for name, view in bundle.masks.views.items():
        h = np.asarray(view.mm_to_px)
        mm = np.c_[np.asarray(view.outline_mm), np.ones(len(view.outline_mm))] @ h.T
        px = mm[:, :2] / mm[:, 2:3]
        assert np.abs(px - np.asarray(view.outline_px)).max() < 0.05, name
        image = cv2.imread(str(recorded["dir"] / view.photo))
        assert (view.width_px, view.height_px) == (image.shape[1], image.shape[0])
    assert bundle.masks.proof_paper is not None and bundle.masks.proof_garment is not None


def test_req_sew_05_seam_log_reaches_the_page_unchanged(recorded):
    """REQ-SEW-05: the page animates the recorded checks: iteration, source, applied, diffs, failing."""
    data = demo.payload(demo.load_bundle("tee1", recorded["root"]))
    log = json.loads((recorded["dir"] / "seam_log.json").read_text())
    assert data["seam_log"] == log and data["max_adjust_calls"] == 3
    assert {s["name"] for s in data["seams"]} == set(log[0]["diffs"])


def _app(recorded, tmp_path, offline=True) -> TestClient:
    settings = Settings(api_key="", mock=False, offline=offline, cache_dir=tmp_path / "cache",
                        ledger_path=tmp_path / "ledger.csv")
    app = create_app(settings, jobs_dir=tmp_path / "jobs", sample_root=tmp_path / "synth", demo_root=recorded["root"])
    return TestClient(app)


def test_req_api_08_web_demo_endpoint_returns_the_bundle_in_offline_mode(recorded, tmp_path, monkeypatch):
    """REQ-API-08: the demo page and its data work offline with no key and no model call."""
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setattr(MetaClient, "segment", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model call")))
    monkeypatch.setattr(MetaClient, "reason", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model call")))
    client = _app(recorded, tmp_path)
    assert client.get("/api/status").json()["mode"] == "OFFLINE"
    assert client.get("/api/demo").json() == {"bundles": ["tee1"]}
    data = client.get("/api/demo/tee1").json()
    assert data["name"] == "tee1" and data["mode"] == "mock"
    assert len(data["measurements"]) == 6 and set(data["measurement_keys"]) == set(data["measurements"])
    assert data["proof"]["result"]["distances_mm"] and len(data["proof"]["points_px"]) == len(data["proof"]["result"]["points_mm"])
    panels = {p["name"]: p for p in data["placement"]["panels"]}
    assert panels["front"]["photo_view"] == "front" and panels["neckband"]["photo_view"] is None
    for key in ("front", "back", "sleeve", "proof", "preview"):
        res = client.get(data["urls"][key])
        assert res.status_code == 200 and res.headers["content-type"].startswith("image/"), key


def test_web_demo_page_has_the_controls_and_the_six_steps(recorded, tmp_path):
    """REQ-SEW-05, SPEC 5: the page has play, pause, step and speed, and draws the six demo steps from data."""
    html = _app(recorded, tmp_path).get("/demo").text
    for ident in ("demoPlay", "demoStep", "demoBack", "demoRestart", "demoSpeed", "demoSeek", "demoStage"):
        assert f'id="{ident}"' in html, ident
    for step in ("masks", "measure", "lift", "seams", "proof", "end"):
        assert f'"{step}"' in html, step
    assert "/api/demo" in html and "Model calls" in html
    assert 'href="/demo"' in _app(recorded, tmp_path).get("/").text  # linked from the test page


def test_web_demo_files_are_whitelisted(recorded, tmp_path):
    client = _app(recorded, tmp_path)
    for file in ("manifest.json", "spec.json", "../manifest.json", "ledger.csv", "cache"):
        assert client.get(f"/api/demo/tee1/files/{file}").status_code == 404, file
    assert client.get("/api/demo/..%2f..%2fetc/files/front.jpg").status_code == 404
    assert client.get("/api/demo/..%2f..%2fetc").status_code == 404


def test_web_demo_missing_file_gives_a_404_that_names_it(recorded, tmp_path):
    root = _copy(recorded, tmp_path)
    (root / "tee1" / "placement.json").unlink()
    app = create_app(Settings(api_key="", mock=False, offline=True, cache_dir=tmp_path / "cache",
                              ledger_path=tmp_path / "ledger.csv"),
                     jobs_dir=tmp_path / "jobs", sample_root=tmp_path / "synth", demo_root=root)
    res = TestClient(app).get("/api/demo/tee1")
    assert res.status_code == 404
    message = res.json()["error"]["message"]
    assert "placement.json" in message and str(tmp_path) not in message


def test_eval_table_is_read_from_the_last_eval_results(tee, tmp_path):
    """SPEC 15: the end card shows the six-measurement error table from the last eval."""
    csv_file = tmp_path / "results.csv"
    cols = ["garment", "run", *[f"err_{k}" for k in demo.MEASUREMENT_KEYS], "mean_abs_err", "seams_ok", "model_calls", "error"]
    rows = [["tee1", "1", "0.1", "-0.2", "0.3", "0.0", "0.5", "-0.1", "0.2", "True", "0", ""],
            ["tee2", "1", "", "", "", "", "", "", "", "False", "0", "SeamCheckFailed: x"]]
    csv_file.write_text("\n".join(",".join(r) for r in [cols, *rows]) + "\n")
    table = demo.read_eval(csv_file)
    assert [r.garment for r in table.rows] == ["tee1", "tee2"]
    assert table.rows[0].errors_mm["body_length"] == pytest.approx(-0.2)
    assert table.rows[1].errors_mm["chest_width"] is None and table.rows[1].error.startswith("SeamCheckFailed")
    assert table.mean_abs_err_mm == pytest.approx(0.2)
    assert demo.read_eval(tmp_path / "missing.csv") is None


def _dense(corners: np.ndarray, closed: bool) -> np.ndarray:
    ring = np.vstack([corners, corners[:1]]) if closed else corners
    return np.vstack([np.linspace(a, b, 60, endpoint=False) for a, b in zip(ring[:-1], ring[1:])])


def test_fit_rigid_recovers_a_known_placement():
    """The placement fit finds a known rotation and shift of a half outline on a whole outline."""
    half = _dense(np.array([[0, 0], [100, 0], [100, 300], [0, 300]], float)[:3], closed=False)  # fold edge left out
    whole = _dense(np.array([[-100, 0], [100, 0], [100, 300], [-100, 300]], float), closed=True)
    theta = np.radians(5.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = whole @ rot.T + [500.0, 200.0]
    found, shift, residual = demo.fit_rigid(half, target, np.array([510.0, 195.0]))
    assert residual < 1.0 and found == pytest.approx(theta, abs=np.radians(1.0))
    assert shift == pytest.approx([500.0, 200.0], abs=2.0)


def _jpeg_with_exif(tags: list[int], order: str = "little") -> bytes:
    """SOI, one APP1 EXIF segment whose first IFD holds the given tags, EOI. Enough for the GPS check."""
    entries = b"".join(t.to_bytes(2, order) + (4).to_bytes(2, order) + (1).to_bytes(4, order) + bytes(4) for t in tags)
    tiff = (b"II" if order == "little" else b"MM") + (42).to_bytes(2, order) + (8).to_bytes(4, order) \
        + len(tags).to_bytes(2, order) + entries + bytes(4)
    body = b"Exif\x00\x00" + tiff
    return b"\xff\xd8\xff\xe1" + (len(body) + 2).to_bytes(2, "big") + body + b"\xff\xd9"


@pytest.mark.parametrize("order", ["little", "big"])
def test_gps_exif_is_detected_so_a_bundle_is_not_published_with_a_location(tee, tmp_path, order):
    with_gps, without = tmp_path / "a.jpg", tmp_path / "b.jpg"
    with_gps.write_bytes(_jpeg_with_exif([0x0112, 0x8825], order))
    without.write_bytes(_jpeg_with_exif([0x0112], order))
    assert demo.has_gps_exif(with_gps) and not demo.has_gps_exif(without)
    assert not demo.has_gps_exif(tee / "front.jpg")  # a plain JPEG
