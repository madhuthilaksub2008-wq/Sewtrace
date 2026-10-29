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


CSRF = {"X-Sewtrace": "1"}  # the header the page sends on every POST


def make_client(tmp_path, sample_root, *, mock=True, password="", api_key="", web=None, offline=False, csrf=True):
    base = Settings(api_key=api_key, mock=mock, cache_dir=tmp_path / "cache", offline=offline,
                    ledger_path=tmp_path / "ledger.csv")
    web = {"auth_fail_delay_s": 0.0, **(web or {})}  # tests do not wait; one test checks the delay
    config = base.config.model_copy(update={"web": base.config.web.model_copy(update=web)})
    base = dataclasses.replace(base, config=config)
    app = create_app(base, password=password, jobs_dir=tmp_path / "jobs", sample_root=sample_root)
    return TestClient(app, headers=CSRF if csrf else None)


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


def test_c_01_second_concurrent_run_is_rejected(client, monkeypatch):
    """C-01: one pipeline run at a time. Two runs at once would read the same ledger total, and
    both could pass the budget check. (No REQ covers the web lock; C-01 is the credit it protects.)"""
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
    client = make_client(tmp_path, sample_root, web={"max_upload_mb": 0.05}, mock=False, offline=True)  # MOCK refuses photos before decoding
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
    assert make_client(tmp_path, sample_root, password="x").head("/healthz").status_code == 200


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


# ---- review fixes ---------------------------------------------------------------------------

def test_req_api_04_real_mode_in_the_image_refuses_a_ledger_folder_that_is_not_a_mount(
        tmp_path, sample_root, monkeypatch, caplog):
    """REQ-API-04, C-01: on a host without a disk the ledger would restart at $0. The image sets
    SEWTRACE_REQUIRE_LEDGER_MOUNT=1; REAL mode then needs the ledger folder to be a mount point."""
    monkeypatch.setenv("SEWTRACE_REQUIRE_LEDGER_MOUNT", "1")
    mounted: list[bool] = [False]
    seen: list[str] = []
    monkeypatch.setattr("os.path.ismount", lambda p: seen.append(str(p)) or mounted[0])
    with pytest.raises(WebConfigError) as err:
        make_client(tmp_path, sample_root, mock=False, api_key="k", password="pw")
    assert str(tmp_path) in str(err.value) and "persistent volume" in str(err.value)
    assert seen == [str(tmp_path)]  # the folder of the ledger file
    mounted[0] = True
    (tmp_path / "ledger.csv").write_text("time,model,input_tokens,cached_tokens,output_tokens,images,cost_usd,estimated\n"
                                         "t,m,0,0,0,1,2.5,0\n")
    with caplog.at_level("INFO", logger="sewtrace.web.app"):
        assert make_client(tmp_path, sample_root, mock=False, api_key="k", password="pw")
    line = next(r.getMessage() for r in caplog.records if "ledger" in r.getMessage())
    assert str(tmp_path / "ledger.csv") in line and "$2.5000" in line and "pw" not in line


def test_req_api_04_mount_check_does_not_apply_to_mock_offline_or_local(tmp_path, sample_root, monkeypatch):
    """MOCK and OFFLINE spend nothing; without the env (local ``sewtrace serve``) nothing changes."""
    monkeypatch.setattr("os.path.ismount", lambda p: False)
    monkeypatch.setenv("SEWTRACE_REQUIRE_LEDGER_MOUNT", "1")
    assert make_client(tmp_path, sample_root, mock=True)
    assert make_client(tmp_path, sample_root, mock=False, offline=True)
    monkeypatch.delenv("SEWTRACE_REQUIRE_LEDGER_MOUNT")
    assert make_client(tmp_path, sample_root, mock=False, api_key="k", password="pw")


def test_req_api_04_serve_refuses_an_unmounted_ledger_folder(monkeypatch, capsys):
    """The same check runs for ``sewtrace serve`` in REAL mode inside the image."""
    monkeypatch.setenv("META_API_KEY", "dummy-key-never-sent")
    monkeypatch.setenv("SEWTRACE_MOCK", "0")
    monkeypatch.setenv("SEWTRACE_WEB_PASSWORD", "pw")
    monkeypatch.setenv("SEWTRACE_REQUIRE_LEDGER_MOUNT", "1")
    monkeypatch.delenv("SEWTRACE_OFFLINE", raising=False)
    monkeypatch.setattr("os.path.ismount", lambda p: False)
    with pytest.raises(SystemExit) as stop:
        cli.main(["serve"])
    assert stop.value.code == 1 and "persistent volume" in capsys.readouterr().err


def test_req_api_02_image_sets_the_cache_folder_inside_the_one_volume():
    """REQ-API-02, REQ-API-08: one volume at /app/data holds ledger and cache."""
    root = Path(__file__).resolve().parent.parent
    docker = (root / "Dockerfile").read_text()
    assert "SEWTRACE_CACHE_DIR=/app/data/cache" in docker and "SEWTRACE_REQUIRE_LEDGER_MOUNT=1" in docker
    entry = (root / "docker-entrypoint.sh").read_text()
    assert "mkdir -p" in entry and "chown -R sewtrace:sewtrace /app/data" in entry


def test_proof_names_its_pattern_and_a_failed_run_clears_it(client, monkeypatch):
    """A proof is never checked against a pattern of an earlier run without saying so."""
    assert run_sample(client)["state"] == "done"
    proof = wait_job(client, client.post("/api/proof", data={"sample": "1"}).json()["id"])
    assert proof["result"]["spec_source"] == "sample"
    ctx = client.app.state.ctx
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: 1 / 0)
    assert wait_job(client, client.post("/api/run", data={"sample": "1"}).json()["id"])["state"] == "error"
    assert ctx.last_spec is None and client.get("/api/status").json()["has_pattern"] is False
    late = wait_job(client, client.post("/api/proof", data={"sample": "1"}).json()["id"])
    assert late["state"] == "error" and "no pattern yet" in late["error"]["message"]


def test_post_without_the_sewtrace_header_is_refused(tmp_path, sample_root):
    """A cross-site form cannot set a custom header, so it cannot start a run with cached credentials."""
    client = make_client(tmp_path, sample_root, csrf=False)
    for path in ("/api/run", "/api/proof"):
        res = client.post(path, data={"sample": "1"})
        assert res.status_code == 403 and res.json()["error"]["kind"] == "Forbidden"
    assert client.post("/api/run", data={"sample": "1"}, headers={"X-Sewtrace": "0"}).status_code == 403
    assert client.get("/api/status").status_code == 200  # reads need no header
    assert client.post("/api/run", data={"sample": "1"}, headers=CSRF).status_code == 202
    assert "X-Sewtrace" in (Path(__file__).parent.parent / "sewtrace/web/index.html").read_text()
    wait_job(client, client.get("/api/status").json()["latest"]["run"])


def test_post_without_content_length_gets_411(client):
    res = client.post("/api/run", content=iter([b"sample=1"]),
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert res.status_code == 411 and res.json()["error"]["kind"] == "LengthRequired"


def _png_header(width: int, height: int) -> bytes:
    """A PNG signature and IHDR that declare a size. The rest is missing, so it cannot decode."""
    import struct
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)


def test_photo_pixel_cap_is_read_from_the_header_before_decoding(tmp_path, sample_root, monkeypatch):
    """A small file can declare a gigapixel image. Refuse it by its header, never call imdecode."""
    from sewtrace.web import imagesize

    def no_decode(*a, **k):
        raise AssertionError("decoded an image above the pixel cap")

    monkeypatch.setattr(cv2, "imdecode", no_decode)
    client = make_client(tmp_path, sample_root, mock=False, offline=True)
    huge = ("f.png", _png_header(32000, 32000), "image/png")  # about 1 gigapixel
    res = client.post("/api/run", files={"front": huge, "back": huge, "sleeve": huge})
    assert res.status_code == 413 and res.json()["error"]["kind"] == "PhotoTooManyPixels"
    assert "front photo" in res.json()["error"]["message"]
    limit = get_config().web.max_pixels
    assert limit == 50_000_000
    assert imagesize.image_size(_png_header(7000, 7000)) == (7000, 7000) and 7000 * 7000 < limit  # 49 MP
    assert imagesize.image_size(_png_header(8000, 8000)) == (8000, 8000) and 8000 * 8000 > limit
    # JPEG: the frame header comes after an EXIF segment, which is skipped by its length.
    exif = b"\xff\xe1" + (2 + 6).to_bytes(2, "big") + b"Exif\0\0"
    sof = b"\xff\xc2\x00\x0b\x08" + (6000).to_bytes(2, "big") + (8000).to_bytes(2, "big") + b"\x01\x01\x11\x00"
    assert imagesize.image_size(b"\xff\xd8" + exif + sof) == (8000, 6000)
    assert imagesize.image_size(b"\xff\xd8\xff\xe0\x00\x02") is None and imagesize.image_size(b"junk") is None
    monkeypatch.undo()
    jpg = photo_bytes(sample_root, "front")
    w, h = imagesize.image_size(jpg)
    small = make_client(tmp_path / "small", sample_root, web={"max_pixels": w * h - 1}, mock=False, offline=True)
    res = small.post("/api/proof", files={"photo": ("p.jpg", jpg, "image/jpeg")})
    assert res.status_code == 413
    ok = make_client(tmp_path / "ok", sample_root, web={"max_pixels": w * h}, mock=False, offline=True)
    assert ok.post("/api/proof", files={"photo": ("p.jpg", jpg, "image/jpeg")}).status_code == 202
    wait_job(ok, ok.get("/api/status").json()["latest"]["proof"])


def test_error_messages_hide_server_folders(client, monkeypatch):
    """The ledger, cache, repo root and sample folders never reach the page."""
    from sewtrace.config import ROOT
    ctx = client.app.state.ctx
    secrets = [str(ctx.client.ledger_path), str(ctx.client.cache_dir), str(ROOT / "data" / "synth" / "tee1"),
               str(ROOT / "sewtrace" / "x.py")]

    def fail(*a, **k):
        raise ApiError("failed with " + " and ".join(secrets))

    monkeypatch.setattr(pipeline, "run", fail)
    job = wait_job(client, client.post("/api/run", data={"sample": "1"}).json()["id"])
    message = job["error"]["message"]
    assert job["state"] == "error" and message.startswith("failed with")
    for text in secrets + [str(ROOT), str(ctx.client.ledger_path.parent)]:
        assert text not in message


def test_wrong_password_waits_before_the_401(tmp_path, sample_root):
    """A failed Basic auth waits auth_fail_delay_s. No header at all (the first browser prompt) does not."""
    client = make_client(tmp_path, sample_root, password="pw", web={"auth_fail_delay_s": 0.3})
    t0 = time.monotonic()
    assert client.get("/").status_code == 401
    assert time.monotonic() - t0 < 0.25
    t0 = time.monotonic()
    wrong = {"Authorization": "Basic " + base64.b64encode(b"u:nope").decode()}
    assert client.get("/", headers=wrong).status_code == 401
    assert time.monotonic() - t0 >= 0.3
    right = {"Authorization": "Basic " + base64.b64encode(b"u:pw").decode()}
    t0 = time.monotonic()
    assert client.get("/", headers=right).status_code == 200
    assert time.monotonic() - t0 < 0.25


@pytest.mark.parametrize("name", ["..%2F..%2Fetc%2Fpasswd", "%2e%2e%2f%2e%2e%2fetc%2fpasswd", "%2e%2e%5c%2e%2e%5cetc",
                                  "..%252f..%252fetc%252fpasswd", "mask_front%00.jpg", "%2Fetc%2Fpasswd"])
def test_file_route_refuses_encoded_traversal(client, name):
    """The route serves only names in the job's file table. httpx would flatten literal ``..``, so
    the dot segments are percent-encoded and sent as they are."""
    jid = run_sample(client)["id"]
    res = client.get(f"/api/jobs/{jid}/files/{name}")
    assert res.status_code == 404 and "root:" not in res.text
    assert client.get(f"/api/jobs/{jid}/files/mask_front").status_code == 200  # the same route works for a real name


def test_overlays_make_no_model_call(client, monkeypatch):
    """The mask overlays re-use the cached masks: client.calls does not move after pipeline.run."""
    seen = {}
    real_run = pipeline.run

    def spy(*args, **kwargs):
        result = real_run(*args, **kwargs)
        seen["after_pipeline"] = client.app.state.ctx.client.calls
        return result

    monkeypatch.setattr(pipeline, "run", spy)
    job = run_sample(client)
    assert job["state"] == "done" and "after_pipeline" in seen
    assert client.app.state.ctx.client.calls == seen["after_pipeline"] == job["result"]["model_calls"]


def test_job_store_deletes_stale_folders_of_earlier_processes(tmp_path):
    root = tmp_path / "jobs"
    old, young = root / "old", root / "young"
    for folder in (old, young):
        (folder).mkdir(parents=True)
        (folder / "front.jpg").write_bytes(b"x")
    stamp = time.time() - 3 * 3600
    import os
    os.utime(old, (stamp, stamp))
    JobStore(root, retention_min=120)
    assert not old.exists() and young.exists()


# ---- review of 0b697b9 ----------------------------------------------------------------------

def _sof(width: int, height: int, marker: int = 0xC0) -> bytes:
    return bytes([0xFF, marker, 0, 11, 8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x01\x01\x11\x00"


def _crafted_jpeg() -> bytes:
    """SOI, FF 00 with a length field, an APP1 whose payload holds a fake 10x10 SOF, then the real
    9000x9000 SOF. libjpeg skips FF 00 as garbage and the APP1 as a segment, so it decodes the real
    frame. A parser that reads FF 00 as a segment with length 6 lands on the fake SOF."""
    fake = _sof(10, 10)
    app1 = b"\xff\xe1" + (2 + len(fake)).to_bytes(2, "big") + fake
    return b"\xff\xd8\xff\x00\x00\x06" + app1 + _sof(9000, 9000)


def test_req_api_08_crafted_jpeg_with_ff00_is_refused_without_decoding(tmp_path, sample_root, monkeypatch):
    """Review minor 1: libjpeg scans past FF 00, so the parser must not read it as a segment."""
    from sewtrace.web import imagesize

    def no_decode(*a, **k):
        raise AssertionError("decoded a photo whose header cannot be read safely")

    monkeypatch.setattr(cv2, "imdecode", no_decode)
    crafted = _crafted_jpeg()
    assert imagesize.image_size(crafted) is None
    client = make_client(tmp_path, sample_root, mock=False, offline=True)
    ok = ("a.jpg", crafted, "image/jpeg")
    res = client.post("/api/run", files={"front": ok, "back": ok, "sleeve": ok})
    assert res.status_code == 415 and res.json()["error"]["kind"] == "PhotoNotImage"
    assert client.post("/api/proof", files={"photo": ok}).status_code == 415


def test_jpeg_header_edge_cases_fail_closed():
    """FF 00 and reserved codes are refused; fill bytes and standalone markers are skipped."""
    from sewtrace.web.imagesize import image_size
    soi, sof = b"\xff\xd8", _sof(800, 600)
    com = b"\xff\xfe\x00\x04hi"
    assert image_size(soi + com + sof) == (800, 600)
    assert image_size(soi + b"\xff\xff\xff" + com + b"\xff\xff" + sof) == (800, 600)  # fill bytes
    assert image_size(soi + b"\xff\x01\xff\xd0\xff\xd7" + sof) == (800, 600)  # TEM, RST0, RST7 stand alone
    for code in (0x00, 0x02, 0x7F, 0xBF, 0xD8, 0xD9, 0xDA):
        assert image_size(soi + bytes([0xFF, code, 0, 2]) + sof) is None, hex(code)
    assert image_size(soi + b"\xff\xff") is None  # fill bytes up to the end
    assert image_size(soi + b"\xff\xfe\x00\x01" + sof) is None  # impossible length
    assert image_size(soi + b"\xff\xc0\x00\x04\x08\x00") is None  # SOF too short
    assert image_size(soi + _sof(800, 600, 0xC4)) is None  # DHT is not a frame; ends without SOF
    assert image_size(soi + _sof(800, 600, 0xC2)) == (800, 600)


def test_req_api_08_mock_refuses_real_photos_before_reading_them(tmp_path, sample_root, monkeypatch):
    """Review minor 1c: in MOCK mode a photo is never decoded, not even to be refused."""
    monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: (_ for _ in ()).throw(AssertionError("decoded")))
    client = make_client(tmp_path, sample_root)
    jpg = ("a.jpg", b"\xff\xd8\xff" + b"0" * 50, "image/jpeg")
    res = client.post("/api/run", files={"front": jpg, "back": jpg, "sleeve": jpg})
    assert res.status_code == 400 and res.json()["error"]["kind"] == "MockMode"
    assert client.post("/api/proof", files={"photo": jpg}).json()["error"]["kind"] == "MockMode"
    missing = client.post("/api/run", files={"front": jpg})  # missing and oversize checks still come first
    assert missing.status_code == 400 and missing.json()["error"]["kind"] == "PhotoMissing"


def test_opencv_pixel_cap_env_applies_to_imdecode_and_is_set_by_the_app():
    """Review minor 1b: OPENCV_IO_MAX_IMAGE_PIXELS stops cv2.imdecode in this OpenCV version.
    A subprocess is needed, because OpenCV reads the variable once per process."""
    import os
    import subprocess
    import sys
    code = ("import cv2, numpy as np\n"
            "j = cv2.imencode('.jpg', np.zeros((100, 100, 3), np.uint8))[1]\n"
            "try:\n    cv2.imdecode(j, 1)\nexcept cv2.error:\n    print('refused')\n"
            "else:\n    print('decoded')\n")
    env = {**os.environ, "OPENCV_IO_MAX_IMAGE_PIXELS": "5000"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "refused"
    env["OPENCV_IO_MAX_IMAGE_PIXELS"] = "20000"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "decoded"


def test_req_api_08_app_sets_the_opencv_cap_from_config_and_the_image_matches(tmp_path, sample_root, monkeypatch):
    monkeypatch.delenv("OPENCV_IO_MAX_IMAGE_PIXELS", raising=False)
    make_client(tmp_path, sample_root, web={"max_pixels": 12_345_678})
    import os
    assert os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] == "12345678"
    monkeypatch.delenv("OPENCV_IO_MAX_IMAGE_PIXELS")
    docker = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text()
    assert f"OPENCV_IO_MAX_IMAGE_PIXELS={get_config().web.max_pixels}" in docker


def test_req_api_04_client_in_real_mode_refuses_an_unmounted_ledger_folder(tmp_path, monkeypatch):
    """Review minor 2: `docker exec sewtrace run` makes a MetaClient in REAL mode without the web app."""
    from sewtrace.api.client import LedgerMountError, MetaClient
    monkeypatch.setenv("SEWTRACE_REQUIRE_LEDGER_MOUNT", "1")
    monkeypatch.setattr("os.path.ismount", lambda p: False)
    real = Settings(api_key="k", mock=False, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.csv")
    with pytest.raises(LedgerMountError) as err:
        MetaClient(real)
    assert str(tmp_path) in str(err.value) and "persistent volume" in str(err.value)
    for mock, offline in ((True, False), (False, True)):  # these spend nothing
        assert MetaClient(dataclasses.replace(real, mock=mock, offline=offline))
    monkeypatch.delenv("SEWTRACE_REQUIRE_LEDGER_MOUNT")
    assert MetaClient(real)
    monkeypatch.setenv("SEWTRACE_REQUIRE_LEDGER_MOUNT", "1")
    monkeypatch.setattr("os.path.ismount", lambda p: True)
    assert MetaClient(real)


def test_req_api_04_cli_run_in_real_mode_stops_on_an_unmounted_ledger_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("META_API_KEY", "dummy-key-never-sent")
    monkeypatch.setenv("SEWTRACE_MOCK", "0")
    monkeypatch.setenv("SEWTRACE_REQUIRE_LEDGER_MOUNT", "1")
    monkeypatch.delenv("SEWTRACE_OFFLINE", raising=False)
    monkeypatch.setattr("os.path.ismount", lambda p: False)
    with pytest.raises(SystemExit) as stop:
        cli.main(["run", str(tmp_path)])
    assert stop.value.code == 1 and "persistent volume" in capsys.readouterr().err


def test_req_api_08_muse_cache_key_uses_its_own_jpeg_quality(tmp_path):
    """Nit: tuning SAM upload quality must not change Muse keys."""
    from sewtrace.api.client import MetaClient
    cfg = get_config()
    assert cfg.reasoning.upload_jpeg_quality == cfg.sam.upload_jpeg_quality == 95
    other = cfg.model_copy(update={"sam": cfg.sam.model_copy(update={"upload_jpeg_quality": 80})})
    base = Settings(api_key="", mock=False, offline=True, cache_dir=tmp_path / "c", ledger_path=tmp_path / "l.csv")
    a = MetaClient(base)
    b = MetaClient(dataclasses.replace(base, config=other))
    assert a.config.reasoning.upload_jpeg_quality == b.config.reasoning.upload_jpeg_quality == 95
