"""FastAPI app: one page and a small JSON API (SPEC 6).

Access: when SEWTRACE_WEB_PASSWORD is set, every route except /healthz needs HTTP Basic auth
(any user name, that password, constant-time compare). REAL mode (paid calls) refuses to start
without the password. In the Docker image (SEWTRACE_REQUIRE_LEDGER_MOUNT=1) REAL mode also refuses
to start unless the ledger folder is a mount point, because a ledger on a disk that is wiped on
restart resets the budget guard (REQ-API-04, C-01). Every POST needs the header X-Sewtrace: 1, which a
cross-site form cannot send. Only one pipeline run is active at a time.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import mimetypes
import os
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.datastructures import UploadFile

from sewtrace.api.client import LedgerMountError, MetaClient
from sewtrace.config import ROOT, Settings
from sewtrace.output.marker_pdf import marker_pdf_bytes
from sewtrace.web import service
from sewtrace.web.imagesize import image_size
from sewtrace.web.jobs import Busy, JobStore

log = logging.getLogger(__name__)

PASSWORD_ENV = "SEWTRACE_WEB_PASSWORD"
MAX_PIXELS_ENV = "OPENCV_IO_MAX_IMAGE_PIXELS"
CSRF_HEADER = "x-sewtrace"
PAGE = Path(__file__).with_name("index.html")
DOWNLOAD_NAMES = {"pdf", "svg", "dxf", "spec"}
MULTIPART_SLACK_BYTES = 64 * 1024  # form overhead on top of the photo limits
REAL_MODE_MESSAGE = (
    "Real photos need REAL mode: set META_API_KEY and SEWTRACE_MOCK=0 on the server, "
    "and install the .[real] extra. Mock mode works with the sample tee only."
)


class WebConfigError(RuntimeError):
    """The server is set up in a way that can spend money without a password."""


def mode_label(settings: Settings) -> str:
    return "OFFLINE" if settings.offline else "MOCK" if settings.mock else "REAL"


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def password_ok(header: str | None, password: str) -> bool:
    """True for ``Authorization: Basic base64(user:password)`` with the right password.
    Compares fixed-length digests in constant time."""
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        _, _, given = base64.b64decode(header[6:].strip(), validate=True).decode().partition(":")
    except ValueError:  # bad base64 or bad UTF-8
        return False
    return hmac.compare_digest(_digest(given), _digest(password))


def _error(status: int, kind: str, title: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"kind": kind, "title": title, "message": message}}, status_code=status)


async def _take_photo(upload: UploadFile | str | None, label: str, max_bytes: int) -> tuple[bytes | None, JSONResponse | None]:
    """The raw bytes of one uploaded photo, or the error response for a missing or oversize file.
    Nothing is decoded here."""
    if not isinstance(upload, UploadFile) or not upload.filename:
        return None, _error(400, "PhotoMissing", "A photo is missing",
                            f"The {label} photo is missing. Choose all three photos, or use the sample tee.")
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        return None, _error(413, "PhotoTooLarge", "A photo is too large",
                            f"The {label} photo is larger than {max_bytes / 1024 / 1024:g} MB. Use a smaller file.")
    return data, None


def _check_photo(data: bytes, label: str, max_pixels: int) -> tuple[bytes | None, JSONResponse | None]:
    """One photo as JPEG bytes, or the error response. A JPEG is kept as sent (EXIF stays); a PNG
    becomes a JPEG, because the pipeline reads ``<view>.jpg``. The header size is checked before
    the decode, and a header that cannot be read safely is refused without a decode."""
    size = image_size(data)
    if size is not None and size[0] * size[1] > max_pixels:
        return None, _error(413, "PhotoTooManyPixels", "A photo has too many pixels",
                            f"The {label} photo is {size[0] * size[1] / 1e6:.0f} MP. The limit is "
                            f"{max_pixels / 1e6:g} MP. Use a smaller photo.")
    image = None
    if size is not None:
        try:
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:  # OpenCV's own pixel cap (OPENCV_IO_MAX_IMAGE_PIXELS) refuses the frame
            image = None
    if image is None:
        return None, _error(415, "PhotoNotImage", "A photo is not an image",
                            f"The {label} photo is not a JPEG or PNG image. Choose a photo file.")
    if data[:3] == b"\xff\xd8\xff":
        return data, None
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), None


def create_app(
    settings: Settings | None = None,
    password: str | None = None,
    jobs_dir: Path | None = None,
    sample_root: Path | None = None,
) -> FastAPI:
    """Build the app. Raises WebConfigError in REAL mode without a password."""
    settings = settings or Settings.load()
    password = os.environ.get(PASSWORD_ENV, "") if password is None else password
    if not settings.mock and not settings.offline and not password:
        raise WebConfigError(
            f"REAL mode spends money, so the web page needs a password. Set {PASSWORD_ENV} and start again, "
            "or use mock mode (SEWTRACE_MOCK=1)."
        )
    web = settings.config.web
    max_bytes, max_pixels = int(web.max_upload_mb * 1024 * 1024), web.max_pixels
    # Defence in depth: OpenCV refuses a frame above this size even if a header trick fools the
    # check in imagesize.py. OpenCV reads the variable once, at the first decode of the process.
    os.environ[MAX_PIXELS_ENV] = str(max_pixels)
    try:
        client = MetaClient(settings)  # REAL mode in the image needs the ledger on a mounted volume
    except LedgerMountError as err:
        raise WebConfigError(str(err)) from err
    log.info("ledger %s, total $%.4f of $%.2f, mode %s, cache %s", client.ledger_path, client.ledger_total_usd(),
             settings.config.api.budget_limit_usd, mode_label(settings), client.cache_dir)
    sample_root = sample_root or ROOT / "data" / "synth"
    jobs_dir = jobs_dir or Path(tempfile.gettempdir()) / "sewtrace-web"
    sample_dir = service.ensure_sample(sample_root)
    # Server folders and the key never reach the page, even inside an error message.
    scrub = (settings.api_key, str(ROOT), str(client.cache_dir), str(client.ledger_path), str(sample_root), str(sample_dir))
    store = JobStore(jobs_dir, web.job_retention_min, scrub=scrub)
    ctx = service.Context(client)
    mock_only_sample = settings.mock

    app = FastAPI(title="Sewtrace test page", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.ctx = store, ctx

    @app.middleware("http")
    async def guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path == "/healthz":
            return await call_next(request)
        given = request.headers.get("authorization")
        if password and not password_ok(given, password):
            if given:  # a wrong guess waits; the first prompt of a browser does not
                await asyncio.sleep(web.auth_fail_delay_s)
            return Response("Password needed.", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Sewtrace"'})
        if request.method == "POST":
            if request.headers.get(CSRF_HEADER) != "1":
                return _error(403, "Forbidden", "Request refused", "Start runs from the Sewtrace page.")
            length = request.headers.get("content-length", "")
            if not length.isdigit():
                return _error(411, "LengthRequired", "The upload has no size", "Send the request with a Content-Length.")
            if int(length) > 3 * max_bytes + MULTIPART_SLACK_BYTES:
                return _error(413, "PhotoTooLarge", "The upload is too large",
                              f"Each photo may be at most {web.max_upload_mb:g} MB.")
        return await call_next(request)

    @app.api_route("/healthz", methods=["GET", "HEAD"])
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(PAGE, media_type="text/html")

    @app.get("/marker.pdf")
    def marker() -> Response:
        return Response(marker_pdf_bytes(), media_type="application/pdf",
                        headers={"Content-Disposition": 'inline; filename="sewtrace-marker.pdf"'})

    @app.get("/api/status")
    def status() -> dict:
        store.purge()
        return {
            "mode": mode_label(settings),
            "reasoning_model": settings.config.reasoning_model.id,
            "ledger_usd": client.ledger_total_usd(),
            "budget_limit_usd": settings.config.api.budget_limit_usd,
            "busy": store.busy,
            "max_upload_mb": web.max_upload_mb,
            "uploads_enabled": not mock_only_sample,
            "latest": dict(store.latest),
            "has_pattern": ctx.last_spec is not None,
        }

    def start(kind: str, work: Callable, uploads: dict[str, bytes] | None = None) -> JSONResponse:
        try:
            job = store.start(kind, work, uploads)
        except Busy as err:
            return _error(409, "Busy", "A run is active", str(err))
        return JSONResponse({"id": job.id}, status_code=202)

    @app.post("/api/run")
    async def run(request: Request) -> JSONResponse:
        form = await request.form()
        if form.get("sample"):
            return start("run", lambda job: service.run_pattern(ctx, job, sample_dir, sample=True))
        raw: dict[str, bytes] = {}
        for view in service.VIEWS:
            data, err = await _take_photo(form.get(view), view, max_bytes)
            if err is not None:
                return err
            raw[view] = data  # type: ignore[assignment]
        if mock_only_sample:  # before any decode: MOCK mode never works on real photos
            return _error(400, "MockMode", "Real photos need REAL mode", REAL_MODE_MESSAGE)
        uploads: dict[str, bytes] = {}
        for view, data in raw.items():
            checked, err = _check_photo(data, view, max_pixels)
            if err is not None:
                return err
            uploads[f"{view}.jpg"] = checked  # type: ignore[assignment]
        return start("run", lambda job: service.run_pattern(ctx, job, job.dir, sample=False), uploads)

    @app.post("/api/proof")
    async def proof(request: Request) -> JSONResponse:
        form = await request.form()
        if form.get("sample"):
            return start("proof", lambda job: service.run_proof(ctx, job, None))
        data, err = await _take_photo(form.get("photo"), "proof", max_bytes)
        if err is not None:
            return err
        if mock_only_sample:  # before any decode
            return _error(400, "MockMode", "Real photos need REAL mode", REAL_MODE_MESSAGE)
        data, err = _check_photo(data, "proof", max_pixels)  # type: ignore[arg-type]
        if err is not None:
            return err
        return start("proof", lambda job: service.run_proof(ctx, job, job.dir / "proof.jpg"), {"proof.jpg": data})  # type: ignore[dict-item]

    @app.get("/api/jobs/{job_id}")
    def job_state(job_id: str) -> JSONResponse:
        job = store.get(job_id)
        if job is None:
            return _error(404, "JobGone", "Job not found", "This job is unknown or was deleted. Run again.")
        return JSONResponse(job.public())

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str) -> Response:
        job = store.get(job_id)
        path = job.files.get(name) if job else None
        if path is None or not path.is_file():
            return _error(404, "FileGone", "File not found", "This file is unknown or was deleted. Run again.")
        media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if name in DOWNLOAD_NAMES:
            return FileResponse(path, media_type=media, filename=path.name)
        return FileResponse(path, media_type=media)

    return app
