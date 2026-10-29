"""T23: the one-page test UI. All mock, no network, no API key."""

import base64
import dataclasses
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sewtrace import cli, pipeline
from sewtrace.api.client import ApiError, BudgetExceeded, CacheMiss, SegmentationFailed
from sewtrace.config import Settings, get_config
from sewtrace.mock import synth
from sewtrace.output.marker_pdf import marker_cells, marker_pdf_bytes, marker_square_pt
from sewtrace.pipeline import ContourFailed, PanelInferenceFailed, PhotoMissing
from sewtrace.reason.seams import SeamCheckFailed
from sewtrace.vision.marker import MarkerNotFound, MarkerRejected, detect_marker
from sewtrace.web.app import WebConfigError, create_app, password_ok
from sewtrace.web.errors import describe_error
from sewtrace.web.jobs import JobStore


@pytest.fixture(scope="session")
def sample_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth")
    synth.make_tee(synth.TEES[0], root / "tee1", seed=0)
    return root


def make_client(tmp_path, sample_root, *, mock=True, password="", api_key="", web=None, offline=False):
    base = Settings(api_key=api_key, mock=mock, cache_dir=tmp_path / "cache", offline=offline,
                    ledger_path=tmp_path / "ledger.csv")
    if web is not None:
        config = base.config.model_copy(update={"web": base.config.web.model_copy(update=web)})
        base = dataclasses.replace(base, config=config)
    app = create_app(base, password=password, jobs_dir=tmp_path / "jobs", sample_root=sample_root)
    return TestClient(app)


def wait_job(client, job_id, timeout_s=90.0) -> dict:
    end = time.time() + timeout_s
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def run_sample(client) -> dict:
    res = client.post("/api/run", data={"sample": "1"})
    assert res.status_code == 202, res.text
    return wait_job(client, res.json()["id"])


def photo_bytes(sample_root, view) -> bytes:
    return (sample_root / "tee1" / f"{view}.jpg").read_bytes()


@pytest.fixture
def client(tmp_path, sample_root):
    return make_client(tmp_path, sample_root)


def test_req_cap_03_page_shows_capture_rules_and_sleeve_diagram(client):
    """REQ-CAP-03: rules and diagram before upload."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "Put the fold on the left and the cap at the top" in html
    assert "<svg" in html and "FOLD" in html and "CAP at the top" in html
    assert "100%" in html and "100 mm" in html and "/marker.pdf" in html
    assert html.index("Take the photos") < html.index("Run with my photos")


def test_req_cap_03_marker_pdf_is_100_mm_and_a_valid_marker():
    """REQ-SCL-01, SPEC 9.1: the black square is 100 mm on A4 and is ArUco DICT_4X4_50 id 0."""
    side = 100 * 72 / 25.4
    x, y, w, h = marker_square_pt()
    assert w == pytest.approx(side, abs=1e-9) and h == pytest.approx(side, abs=1e-9)
    data = marker_pdf_bytes()
    assert data.startswith(b"%PDF")
    rects = [tuple(float(v) for v in r) for r in re.findall(rb"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re", data)]
    assert any(abs(rw - side) < 1e-3 for *_, rw, _ in rects)  # the full-width top border row
    assert min(r[0] for r in rects) == pytest.approx(x, abs=1e-3)
    assert max(r[0] + r[2] for r in rects) == pytest.approx(x + side, abs=1e-3)
    assert max(r[1] + r[3] for r in rects) == pytest.approx(y + side, abs=1e-3)
    assert min(r[1] for r in rects) == pytest.approx(y, abs=1e-3)
    # The cell grid, drawn big on white, is found as marker id 0.
    cells = np.array(marker_cells(), np.uint8)
    img = np.pad(np.where(cells, 0, 255).astype(np.uint8).repeat(40, 0).repeat(40, 1), 80, constant_values=255)
    corners = detect_marker(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    assert np.ptp(corners[:, 0]) == pytest.approx(40 * len(cells), abs=3)


def test_req_cap_03_marker_pdf_endpoint(client):
    res = client.get("/marker.pdf")
    assert res.status_code == 200 and res.headers["content-type"] == "application/pdf"
    assert res.content == marker_pdf_bytes()


def test_req_sew_05_sample_run_end_to_end(client):
    """REQ-CAP-03, REQ-SEW-05, REQ-OUT-01: sample tee gives measurements, seams, preview and downloads."""
    job = run_sample(client)
    assert job["state"] == "done", job
    r, jid = job["result"], job["id"]
    assert r["synthetic"] and len(r["measurements"]) == 6
    assert all(abs(m["error_mm"]) < 2.0 for m in r["measurements"])
    assert r["seams"] and all(s["ok"] for s in r["seams"])
    assert r["seam_log"][0]["iteration"] == 0 and "diffs" in r["seam_log"][0]
    assert isinstance(r["model_calls"], int)
    preview = client.get(f"/api/jobs/{jid}/files/{r['preview']}")
    assert preview.status_code == 200 and preview.content[:8] == b"\x89PNG\r\n\x1a\n"
    for p in r["photos"]:
        img = client.get(f"/api/jobs/{jid}/files/{p['file']}")
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    pdf = client.get(f"/api/jobs/{jid}/files/pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert "attachment" in pdf.headers["content-disposition"]
    for name in ("svg", "dxf", "spec"):
        assert client.get(f"/api/jobs/{jid}/files/{name}").status_code == 200
    assert client.get(f"/api/jobs/{jid}/files/../../etc/passwd").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    status = client.get("/api/status").json()
    assert status["has_pattern"] and status["latest"]["run"] == jid


def test_req_cap_01_upload_with_missing_photo_names_it(client, sample_root):
    """REQ-CAP-01: a missing photo gives a clear error, not a traceback."""
    res = client.post("/api/run", files={"front": ("front.jpg", photo_bytes(sample_root, "front"), "image/jpeg"),
                                         "sleeve": ("sleeve.jpg", photo_bytes(sample_root, "sleeve"), "image/jpeg")})
    assert res.status_code == 400
    err = res.json()["error"]
    assert err["kind"] == "PhotoMissing" and "back photo" in err["message"]
    assert "Traceback" not in res.text


def test_mock_mode_refuses_real_photos_with_a_clear_message(client, sample_root):
    """Mock SAM has no mask for an upload: say so instead of failing."""
    files = {v: (f"{v}.jpg", photo_bytes(sample_root, v), "image/jpeg") for v in ("front", "back", "sleeve")}
    res = client.post("/api/run", files=files)
    assert res.status_code == 400
    message = res.json()["error"]["message"]
    assert "Real photos need REAL mode: set META_API_KEY and SEWTRACE_MOCK=0" in message
    assert ".[real]" in message
    res = client.post("/api/proof", files={"photo": ("p.jpg", photo_bytes(sample_root, "front"), "image/jpeg")})
    assert res.status_code == 400 and res.json()["error"]["kind"] == "MockMode"
    assert client.get("/api/status").json()["uploads_enabled"] is False


def test_req_api_04_second_concurrent_run_is_rejected(client, monkeypatch):
    """One pipeline run at a time: a second run is refused while one is active."""
    started, release = threading.Event(), threading.Event()
    real_run = pipeline.run

    def slow_run(*args, **kwargs):
        started.set()
        assert release.wait(30)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(pipeline, "run", slow_run)
    first = client.post("/api/run", data={"sample": "1"})
    assert first.status_code == 202
    assert started.wait(30)
    assert client.get("/api/status").json()["busy"] is True
    second = client.post("/api/run", data={"sample": "1"})
    assert second.status_code == 409 and "already active" in second.json()["error"]["message"]
    assert client.post("/api/proof", data={"sample": "1"}).status_code == 409
    release.set()
    assert wait_job(client, first.json()["id"])["state"] == "done"
    assert client.post("/api/run", data={"sample": "1"}).status_code == 202  # free again
    wait_job(client, client.get("/api/status").json()["latest"]["run"])


def test_real_mode_without_password_refuses_to_start(tmp_path, sample_root, monkeypatch, capsys):
    """Real mode spends money: no password, no server."""
    monkeypatch.delenv("SEWTRACE_WEB_PASSWORD", raising=False)
    with pytest.raises(WebConfigError, match="SEWTRACE_WEB_PASSWORD"):
        make_client(tmp_path, sample_root, mock=False, api_key="k")
    assert make_client(tmp_path, sample_root, mock=False, api_key="k", password="pw")  # starts with a password
    assert make_client(tmp_path, sample_root, mock=True)  # mock mode: password is optional
    monkeypatch.setenv("META_API_KEY", "dummy-key-never-sent")
    monkeypatch.setenv("SEWTRACE_MOCK", "0")
    monkeypatch.delenv("SEWTRACE_OFFLINE", raising=False)
    with pytest.raises(SystemExit) as stop:
        cli.main(["serve"])
    assert stop.value.code == 1
    assert "SEWTRACE_WEB_PASSWORD" in capsys.readouterr().err


def test_password_is_required_and_checked_when_set(tmp_path, sample_root):
    """HTTP Basic over the whole app; /healthz stays open for the host's health check."""
    client = make_client(tmp_path, sample_root, password="s3cret é")

    def auth(user, pw):
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}

    for path in ("/", "/api/status", "/marker.pdf", "/api/jobs/x"):
        res = client.get(path)
        assert res.status_code == 401 and "Basic" in res.headers["www-authenticate"]
    assert client.post("/api/run", data={"sample": "1"}).status_code == 401
    assert client.get("/", headers=auth("bala", "wrong")).status_code == 401
    assert client.get("/", headers=auth("bala", "s3cret")).status_code == 401
    assert client.get("/", headers={"Authorization": "Basic !!!"}).status_code == 401
    assert client.get("/", headers=auth("bala", "s3cret é")).status_code == 200
    assert client.get("/healthz").status_code == 200
    assert not password_ok(None, "x") and not password_ok("Bearer abc", "x")


def test_upload_size_and_type_limits(tmp_path, sample_root):
    """Per-photo size limit from config [web], and only real images are accepted."""
    client = make_client(tmp_path, sample_root, web={"max_upload_mb": 0.05})
    noise = np.random.default_rng(0).integers(0, 255, (250, 250, 3), dtype=np.uint8)
    jpg = cv2.imencode(".jpg", noise)[1].tobytes()
    assert 0.05 * 1024 * 1024 < len(jpg) < 150_000
    small = cv2.imencode(".jpg", np.full((40, 40, 3), 90, np.uint8))[1].tobytes()
    ok = ("a.jpg", small, "image/jpeg")
    big = client.post("/api/run", files={"front": ("f.jpg", jpg, "image/jpeg"), "back": ok, "sleeve": ok})
    assert big.status_code == 413 and "front photo" in big.json()["error"]["message"]
    huge = client.post("/api/run", files={"front": ("f.jpg", b"0" * 400_000, "image/jpeg")})
    assert huge.status_code == 413
    text = client.post("/api/run", files={"front": ("f.jpg", b"not an image", "image/jpeg"), "back": ok, "sleeve": ok})
    assert text.status_code == 415 and text.json()["error"]["kind"] == "PhotoNotImage"
    fake = client.post("/api/run", files={"front": ok, "back": ("b.jpg", b"\xff\xd8\xff" + b"0" * 50, "image/jpeg"), "sleeve": ok})
    assert fake.status_code == 415 and "back photo" in fake.json()["error"]["message"]
    assert get_config().web.max_upload_mb == 20


def test_req_ver_03_sample_proof_returns_gap_stats_and_overlay(client):
    """REQ-VER-01, REQ-VER-03: mean/max/p95, percent within tolerance, neck line, overlay image."""
    early = client.post("/api/proof", data={"sample": "1"})
    assert wait_job(client, early.json()["id"])["error"]["message"].startswith("proof: there is no pattern yet")
    assert run_sample(client)["state"] == "done"
    res = client.post("/api/proof", data={"sample": "1"})
    assert res.status_code == 202
    job = wait_job(client, res.json()["id"])
    assert job["state"] == "done", job
    r = job["result"]
    assert 0 < r["mean_mm"] < r["tol_mm"] and r["mean_mm"] <= r["p95_mm"] <= r["max_mm"]
    assert r["pct_within"] > 50 and r["tol_mm"] == get_config().proof.gap_tolerance_mm
    assert r["neck"] is not None and r["neck"]["points"] > 0
    overlay = client.get(f"/api/jobs/{job['id']}/files/{r['overlay']}")
    assert overlay.status_code == 200 and overlay.content[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("err", [
    PhotoMissing("front photo f.jpg is missing. Put the file there."),
    MarkerNotFound("sleeve photo: the whole marker is not in the photo."),
    MarkerRejected("back photo: marker error 2.00 mm."),
    SegmentationFailed("front.jpg: score too low."),
    ContourFailed("front photo: no sleeve."),
    SeamCheckFailed([], []),
    PanelInferenceFailed("panel inference: no depth."),
    BudgetExceeded("ledger is over the limit."),
    CacheMiss("not in the cache."),
    ApiError("timeout."),
    FileNotFoundError("mock SAM has no 'sleeve' mask for x.jpg"),
    ValueError("proof: no paper found."),
    RuntimeError("secret internal detail"),
])
def test_error_mapping_never_shows_a_traceback(err):
    """Every pipeline error becomes a short message with a title; unknown errors stay generic."""
    out = describe_error(err, scrub=["/srv/jobs/abc/"])
    assert out["kind"] == type(err).__name__ and out["title"] and out["message"]
    assert "Traceback" not in out["message"] and "File \"" not in out["message"]
    if isinstance(err, RuntimeError) and type(err) is RuntimeError:
        assert "secret" not in out["message"] and out["title"] == "Unexpected error"
    else:
        assert str(err).split(".")[0][:10] in out["message"]


def test_failed_jobs_show_the_step_error_and_scrub_server_paths(client, monkeypatch):
    def fail(garment_dir, *args, **kwargs):
        raise MarkerNotFound(f"front photo {garment_dir}/front.jpg: the whole marker is not in the photo.")

    monkeypatch.setattr(pipeline, "run", fail)
    res = client.post("/api/run", data={"sample": "1"})
    job = wait_job(client, res.json()["id"])
    assert job["state"] == "error" and job["error"]["title"] == "Marker not found"
    assert "front photo" in job["error"]["message"] and "Traceback" not in str(job)
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: 1 / 0)
    job = wait_job(client, client.post("/api/run", data={"sample": "1"}).json()["id"])
    assert job["error"]["title"] == "Unexpected error" and "division" not in job["error"]["message"]


def test_status_bar_and_no_key_leak(tmp_path, sample_root):
    """Status shows mode, reasoning model and ledger vs limit, never the key."""
    key = "sk-test-secret-key-123"
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("time,model,input_tokens,cached_tokens,output_tokens,images,cost_usd,estimated\n"
                      "t,m,0,0,0,1,1.25,False\n")
    client = make_client(tmp_path, sample_root, api_key=key)
    status = client.get("/api/status").json()
    cfg = get_config()
    assert status["mode"] == "MOCK" and status["reasoning_model"] == cfg.reasoning_model.id
    assert status["ledger_usd"] == pytest.approx(1.25) and status["budget_limit_usd"] == cfg.api.budget_limit_usd
    job = run_sample(client)
    assert key not in client.get("/api/status").text and key not in str(job) and key not in client.get("/").text
    assert make_client(tmp_path, sample_root, offline=True, mock=False).get("/api/status").json()["mode"] == "OFFLINE"
    assert make_client(tmp_path, sample_root, mock=False, password="x", api_key="k").get("/healthz").status_code == 200


def test_job_folders_are_deleted_after_the_retention_time(tmp_path):
    """Uploads do not stay on the server."""
    store = JobStore(tmp_path / "jobs", retention_min=1)
    done = threading.Event()
    job = store.start("run", lambda j: done.set() or {}, {"front.jpg": b"x"})
    assert done.wait(10)
    while store.busy:
        time.sleep(0.01)
    assert (job.dir / "front.jpg").exists()
    store.purge()
    assert job.dir.exists()  # too young
    job.created -= 3600
    store.purge()
    assert not job.dir.exists() and store.get(job.id) is None and "run" not in store.latest


def test_uploads_are_stored_per_job_as_jpeg(tmp_path, sample_root, monkeypatch):
    """Outside mock mode an upload reaches the pipeline as <job folder>/<view>.jpg; a PNG is converted."""
    from sewtrace.web import service

    seen = {}

    def fake(ctx, job, garment_dir, sample):
        seen.update({p.name: p.read_bytes() for p in garment_dir.glob("*.jpg")}, dir=garment_dir, sample=sample)
        return {"ok": True}

    monkeypatch.setattr(service, "run_pattern", fake)
    client = make_client(tmp_path, sample_root, mock=False, offline=True)
    png = cv2.imencode(".png", np.full((30, 30, 3), 120, np.uint8))[1].tobytes()
    jpg = photo_bytes(sample_root, "front")
    files = {"front": ("a.jpg", jpg, "image/jpeg"), "back": ("b.png", png, "image/png"), "sleeve": ("c.jpg", jpg, "image/jpeg")}
    res = client.post("/api/run", files=files)
    assert res.status_code == 202 and wait_job(client, res.json()["id"])["state"] == "done"
    assert seen["sample"] is False and str(tmp_path / "jobs") in str(seen["dir"])
    assert seen["front.jpg"] == jpg and seen["back.jpg"][:3] == b"\xff\xd8\xff"
