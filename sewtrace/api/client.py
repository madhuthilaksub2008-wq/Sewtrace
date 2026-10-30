"""Meta Model API client: SAM 3.1 segmentation and Muse reasoning.

This is the only module that talks HTTP (REQ-API-01). Every answer is cached
by SHA-256 of mode + model ID + prompt + images + all request parameters
(REQ-API-02), so the same request is never paid for twice and a mock answer is
never served as a real one. Every paid call goes through ``_paid``: budget
check, retry on HTTP 429 and 5xx, timeout, then one ledger row (REQ-API-03,
04, 06, 07). Model IDs, prices and limits come from config.toml (REQ-API-05).

The ledger is data/ledger.csv (``api.ledger_path``), outside the cache folder, so
clearing the cache or changing SEWTRACE_CACHE_DIR cannot reset the total. Real
answers are cached as ``real-<key>.*``. A paid call is refused when the ledger
file is missing while the cache holds such a file (REQ-API-04, C-01).

Mock mode (no API key, or SEWTRACE_MOCK=1):
  * segment() returns the ground-truth region mask written next to a synthetic
    photo (``<photo>.masks/<prompt>.png``), lightly jittered to look like a
    model edge. Without a sidecar it falls back to a backdrop colour threshold.
  * reason() returns the caller's ``mock`` callable, so each module supplies a
    deterministic stand-in for the model's answer.

Offline mode (SEWTRACE_OFFLINE=1 or ``api.offline`` in config): a cache miss
raises CacheMiss. Neither the mock nor the network is used (REQ-API-08).

Real endpoints follow Meta's cookbook (07_sam/01_api_basics, 01_api_fundamentals):
  * SAM 3.1 uses the Responses API (POST /responses, stream=true). The streamed
    segmentation grammar is decoded with Meta's ``meta_sam_parser``, imported only on
    that path (``pip install -e '.[real]'``). The raw events are cached before decoding,
    so a decoding bug on Day 1 never costs a second paid call.
  * Muse uses Chat Completions with a json_schema response_format. Stable instructions
    go in a developer message first, images and changing data in the user message, so
    the prompt cache applies.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import http.client
import json
import logging
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing, contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from sewtrace.config import ROOT, ModelConfig, Settings, resolve_ledger_path

log = logging.getLogger(__name__)

LEDGER_COLUMNS = ["time", "model", "input_tokens", "cached_tokens", "output_tokens", "images", "cost_usd", "estimated"]
CHARS_PER_TOKEN = 4  # rough token count for the pre-call estimate only

# (url, headers, json body, timeout_s) -> parsed JSON response. Raises HttpError
# for a non-2xx status and TimeoutError when the call is too slow. The streaming transport
# (SAM, Responses API) has the same shape and returns {"events": [<one dict per SSE event>]}.
Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]

# Bump when ``sam_results_from`` or the mask decoding changes. It is part of the key of the decoded
# mask only, so the stored raw events are reused and a decoder fix needs no new paid call.
SAM_DECODE_VERSION = 4  # 4: downsized rasters use INTER_AREA; 3: raster resized to its bounds, confidence from the record

PARSER_MISSING = (
    "SAM 3.1 output is decoded with Meta's meta_sam_parser, which is not installed. "
    "No call was made. Run: pip install -e '.[real]'"
)


# Failures of the connection itself. Only these become ApiTransportError, which callers may retry.
# OSError covers ConnectionError, socket.timeout, ssl.SSLError and urllib.error.URLError.
# Anything else (TypeError, KeyError, KeyboardInterrupt, CancelledError) keeps its type.
TRANSPORT_ERRORS = (OSError, http.client.HTTPException, json.JSONDecodeError)


class ApiError(RuntimeError):
    """A Meta API call failed after the allowed tries."""


class StoredAnswerRejected(ApiError):
    """Replay of a stored paid answer that was cut off or not JSON. ``base_message`` is the text
    of the live failure; callers that put the error into a prompt use it, so the prompt (and its
    cache key) is the same for the live run and every replay."""

    def __init__(self, base_message: str, stored_name: str) -> None:
        super().__init__(f"{base_message} (stored paid answer {stored_name}; delete it to pay for a new attempt)")
        self.base_message = base_message


class LedgerMountError(ApiError):
    """REAL mode in the image, and the ledger folder is not a mounted volume."""


REQUIRE_MOUNT_ENV = "SEWTRACE_REQUIRE_LEDGER_MOUNT"  # set to 1 in the Docker image


class ApiRequestRejected(ApiError):
    """The server refused the request itself: any HTTP status except 429 and 5xx (400 bad parameter,
    401 or 403 bad key, 404 wrong model ID, 422 bad schema). A retry or a fallback cannot help,
    so callers stop the run."""


class ApiTimeout(ApiError):
    """The call did not finish within ``api.timeout_s``."""


class ApiTransportError(ApiError):
    """The connection broke after the request may have been sent, so the call may be billed."""


class ApiNotSent(ApiError):
    """The request never left this machine (DNS failure, connection refused, TLS handshake
    failure). Nothing can be billed, so ``_paid`` writes no ledger row. A transport must raise
    this only when it is sure; every other failure after the call started is billed."""


class BudgetExceeded(ApiError):
    """The next paid call would take the ledger over ``api.budget_limit_usd``."""


class PolicyViolation(ApiError):
    """The call would break a data rule (real photo to the contributor endpoint). Never retried."""


class CacheMiss(ApiError):
    """Offline mode and the answer is not in the cache."""


class SegmentationFailed(ApiError):
    """SAM found no mask that scores high enough (REQ-SEG-04)."""


class SamResult(BaseModel):
    """One mask SAM returned, full image size, with its confidence if the API gave one.

    ``score`` is the mask record's ``confidence`` (0..1). It is None when SAM sent none (or an
    out-of-range value): the parser keeps None and does not mean zero."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    mask: np.ndarray  # bool, H x W
    score: float | None = None


class SamRecordInfo(BaseModel):
    """One SAM mask record for the Day 1 report. ``bounds`` and ``box_in_photo`` are
    (left, top, right, bottom); ``bounds`` is in SAM's frame, ``box_in_photo`` in the original photo."""

    model_config = ConfigDict(frozen=True)

    object_id: str
    confidence: float | None
    box_confidence: float | None  # the box token's ``c``; not used for the score (REQ-SEG-03 asks for the mask's)
    bounds: tuple[float, float, float, float]
    raster_wh: tuple[int, int]
    box_in_photo: tuple[float, float, float, float] | None


class SamInspection(BaseModel):
    """Stored SAM answer in plain numbers. Sizes are (width, height) in px; ``source_wh`` is None
    when SAM's box lines carry no readable size."""

    model_config = ConfigDict(frozen=True)

    image_wh: tuple[int, int]
    sent_wh: tuple[int, int]
    source_wh: tuple[int, int] | None
    reported_wh: list[tuple[int, int]]  # every distinct size in the box lines; more than one means they disagree
    records: list[SamRecordInfo]


class HttpError(Exception):
    def __init__(self, status: int, body: str = ""):
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status


class Usage:
    """Billable units of one call."""

    def __init__(self, input_tokens: int = 0, cached_tokens: int = 0, output_tokens: int = 0, images: int = 0):
        self.input_tokens, self.cached_tokens = input_tokens, cached_tokens
        self.output_tokens, self.images = output_tokens, images


def cost_usd(model: ModelConfig, usage: Usage) -> float:
    if model.price_per_image_usd is not None:
        return usage.images * model.price_per_image_usd
    assert model.input_per_m_usd is not None and model.cached_input_per_m_usd is not None
    assert model.output_per_m_usd is not None
    fresh = usage.input_tokens - usage.cached_tokens
    return (fresh * model.input_per_m_usd + usage.cached_tokens * model.cached_input_per_m_usd
            + usage.output_tokens * model.output_per_m_usd) / 1e6


@contextmanager
def _open(url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float) -> Iterator[Any]:
    """POST ``body`` as JSON and yield the response. Maps urllib errors to our error types."""
    request = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 (https URL from config)
            yield response
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode(errors="replace")) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise TimeoutError(str(e)) from e
        if isinstance(e.reason, ConnectionError) and not isinstance(e.reason, ConnectionRefusedError):
            raise ApiTransportError(f"connection lost: {e.reason}") from e
        raise ApiNotSent(f"network error: {e.reason}") from e  # DNS, refused or handshake: never sent
    except ConnectionError as e:  # reset while reading the answer: the server got the request
        raise ApiTransportError(f"connection lost: {e}") from e
    except (http.client.HTTPException, ssl.SSLError) as e:
        # IncompleteRead, a bad status line or a TLS error while reading: the request was sent.
        raise ApiTransportError(f"connection broke while reading the answer: {e!r}") from e


def urllib_transport(url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    with _open(url, headers, body, timeout_s) as response:
        raw = response.read()
    try:
        return json.loads(raw)
    except ValueError as e:  # the server answered, so the call may be billed
        raise ApiTransportError(f"the answer is not JSON: {raw[:100]!r}") from e


def urllib_stream_transport(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Read a server-sent-events stream to its end (REQ-API-07).

    ``timeout_s`` is the socket timeout of the connect and of each read. A total deadline of
    ``timeout_s`` is also checked each time a line arrives, so a stream that keeps trickling stops.
    The deadline is NOT enforced on a silent socket: urllib has no public way to lower the socket
    timeout of an open response, so the last read may wait up to ``timeout_s`` more. The worst
    case is therefore just under 2 x ``timeout_s``."""
    deadline = clock() + timeout_s
    with _open(url, {**headers, "Accept": "text/event-stream"}, body, timeout_s) as response:
        return {"events": list(parse_sse(_lines_before(response, deadline, clock, timeout_s)))}


def _lines_before(
    lines: Iterable[bytes], deadline: float, clock: Callable[[], float], timeout_s: float,
) -> Iterator[str]:
    for raw in lines:
        if clock() > deadline:
            raise TimeoutError(f"stream still open after {timeout_s:g} s")
        yield raw.decode("utf-8", errors="replace").rstrip("\r\n")


def parse_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """JSON payloads of ``data:`` lines. Comments, ``event:`` lines and ``[DONE]`` are ignored:
    every Responses event carries its own ``type`` field."""
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ApiError(f"stream event is not JSON: {payload[:100]!r}") from e
        if isinstance(event, dict):
            yield event


class MetaClient:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: Transport = urllib_transport,
        stream_transport: Transport = urllib_stream_transport,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings or Settings.load()
        self.config = self.settings.config
        self.cache_dir = self.settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.settings.ledger_path or resolve_ledger_path(self.config)
        self._require_ledger_mount()
        self.calls = 0  # real or mocked calls that missed the cache
        self.cache_files_used: set[Path] = set()  # every cache file looked up; `demo record` copies the existing ones
        self._transport = transport
        self._stream_transport = stream_transport
        self._sleep = sleep

    def _require_ledger_mount(self) -> None:
        """In the image, REAL mode needs the ledger folder on a mounted volume. A folder on the
        container disk is wiped on restart, and the cache with it, so the budget guard would start
        again at $0 without any other check noticing (REQ-API-04, C-01). This sits in the client,
        not in the web app, so ``docker exec sewtrace run`` is covered too."""
        settings = self.settings
        if settings.mock or settings.offline or os.environ.get(REQUIRE_MOUNT_ENV) != "1":
            return
        folder = self.ledger_path.parent
        if not os.path.ismount(folder):
            raise LedgerMountError(
                f"REAL mode spends money, and the ledger folder {folder} is not a mounted volume, so the spend "
                f"total would restart at $0 after a restart or a redeploy. Attach a persistent volume at {folder} "
                "and start again, or use mock mode (SEWTRACE_MOCK=1). See docs/DEPLOY.md, 'The ledger volume'."
            )

    # ---- public API -------------------------------------------------------

    def segment(self, image_path: str | Path, prompt: str) -> np.ndarray:
        """Boolean HxW mask of the text ``prompt`` (SAM takes text prompts only, REQ-SEG-01).

        With several masks, the highest score wins (REQ-SEG-03). If that score is below
        ``sam.min_score``, raises SegmentationFailed (REQ-SEG-04). The best mask and its
        score are cached together, so the threshold is checked on every call and a low
        score is never paid for twice.

        The score is the mask record's ``confidence``. When SAM sends none for some mask (stand-in,
        see ``select_best``), the largest mask wins and the min_score check is skipped with a
        warning. It is never faked as 1.0."""
        image_path = Path(image_path)
        key = self._sam_key(image_path, prompt, decoded=True)
        cached, cached_score = self._cache_file(key, ".png"), self._cache_file(key, ".score.json")
        if cached.exists() and cached_score.exists():
            stored = json.loads(cached_score.read_text())["score"]
            best = SamResult(
                mask=cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE) > 127,
                score=None if stored is None else float(stored),
            )
        else:
            # Offline can still re-decode stored raw events (no network); otherwise it is a miss.
            replay = (self.settings.offline and not self.settings.mock
                      and self._cache_file(self._sam_key(image_path, prompt, decoded=False), ".events.json").exists())
            if not replay:
                self._require_online(f"segment {image_path.name!r} with prompt {prompt!r}")
            if not replay:  # a replay makes no call, so eval must not count one
                self.calls += 1
            if self.settings.mock:
                results = _mock_segment(image_path, prompt, self.config.sam.prompts.garment)
            else:
                results = self._call_sam(image_path, prompt)
            if not results:
                raise SegmentationFailed(f"{image_path}: SAM returned no mask for the prompt {prompt!r}.")
            best = select_best(results)
            cv2.imwrite(str(cached), best.mask.astype(np.uint8) * 255)
            cached_score.write_text(json.dumps({"score": best.score}))
        min_score = self.config.sam.min_score
        if best.score is None:
            log.warning(
                "segment %s %r: SAM gave no score, so the min_score check (%g) is skipped. "
                "Check the mask by eye. (SAM sent no mask confidence.)",
                image_path.name, prompt, min_score,
            )
        elif best.score < min_score:
            raise SegmentationFailed(
                f"{image_path}: best SAM score {best.score:.2f} for the prompt {prompt!r} is below "
                f"the minimum {min_score:g}. Take the photo again on a plain, contrasting background."
            )
        return best.mask

    def reason(
        self,
        prompt: str,
        image_paths: Sequence[str | Path] = (),
        mock: Callable[[], dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        instructions: str | None = None,
    ) -> dict[str, Any]:
        """Structured JSON answer from the reasoning model.

        ``instructions`` is the stable part of the prompt. It is sent first, as a developer
        message, so the provider's prompt cache can serve it. ``prompt`` (the changing part)
        and the images go in the user message."""
        reasoning = self.config.reasoning
        model = self.config.reasoning_model
        params = {
            "reasoning_effort": reasoning.effort,
            "max_tokens": reasoning.max_tokens,
            "schema_sha256": _sha256_json(schema) if schema is not None else None,
        }
        if instructions is not None:  # absent when unused, so keys of calls without it are unchanged
            params["instructions_sha256"] = hashlib.sha256(instructions.encode()).hexdigest()
        paths = [Path(p) for p in image_paths]
        self._require_synthetic_for_contributor(paths)
        images: list[np.ndarray] = []
        digests: list[str] | None = None
        if paths and not self.settings.mock:
            # Like SAM: the key follows the decoded pixels that are sent, not the file bytes.
            images = [_read_pixels(p) for p in paths]
            digests = [hashlib.sha256(i.tobytes()).hexdigest() for i in images]
            params["upload_jpeg_quality"] = self.config.reasoning.upload_jpeg_quality
            params["max_upload_px"] = self.config.reasoning.max_upload_px
            params["shapes"] = [list(i.shape) for i in images]
        key = self._key("reason", model.id, prompt, paths, params, image_digests=digests)
        cached = self._cache_file(key, ".json")
        if cached.exists():
            return json.loads(cached.read_text())
        rejected = self._cache_file(key, ".rejected.json")
        if rejected.exists():
            # A paid answer that was cut off or not JSON. Judged by the same code as live, so it fails
            # the same way and the caller's retry or fallback runs without a second payment.
            try:
                return _answer_from_choice(model.id, json.loads(rejected.read_text()), reasoning)
            except ApiError as e:
                # Same text on every replay, so the retry prompt that embeds it keeps one cache key.
                raise StoredAnswerRejected(str(e), rejected.name) from e
        self._require_online(f"reasoning call to {model.id}")
        self.calls += 1
        if self.settings.mock:
            if mock is None:
                raise RuntimeError("reason() called in mock mode without a mock answer")
            answer = mock()
        else:
            answer = self._call_reasoning(model, prompt, images, schema, instructions, rejected_file=rejected)
        cached.write_text(json.dumps(answer))
        return answer

    def ledger_total_usd(self) -> float:
        if not self.ledger_path.exists():
            return 0.0
        with self.ledger_path.open(newline="") as f:
            return sum(float(row["cost_usd"]) for row in csv.DictReader(f))

    # ---- real endpoints ---------------------------------------------------

    def _call_sam(self, image_path: Path, prompt: str) -> list[SamResult]:
        """SAM 3.1 through the Responses API, as in the cookbook (07_sam/01_api_basics).
        Text prompt only (REQ-SEG-01): the request has no box or point field.

        The parser is checked before the paid call, and the raw events are cached right
        after it, so a decoding problem never makes a second paid call necessary."""
        parser, image, result = self._sam_parsed(image_path, prompt)
        sent_hw = _fitted_hw(*image.shape[:2], self.config.sam.max_upload_px)
        return sam_results_from(result, parser.decode_mask_to_raster, sent_hw, image_path.name, prompt,
                                box_tol_px=self.config.sam.box_overhang_tol_px, full_hw=image.shape[:2])

    def sam_boxes(self, image_path: str | Path, prompt: str) -> tuple[tuple[int, int], list[tuple[str, float, float, float, float]]]:
        """``((width, height), [(object_id, left, top, right, bottom), ...])`` of the cached SAM
        answer, in the pixel frame of the decoded photo (the original, not the downscaled upload).
        Uses the same frame mapping as ``segment``. Never calls the API: it needs the stored raw
        events (Day 1 uses it to check the box frame by eye)."""
        inspection = self.sam_inspect(image_path, prompt, raise_on_bad_frame=True)
        return inspection.image_wh, [(r.object_id, *r.box_in_photo) for r in inspection.records if r.box_in_photo]

    def sam_inspect(self, image_path: str | Path, prompt: str, *, raise_on_bad_frame: bool = False) -> SamInspection:
        """What SAM returned for a stored answer: per record confidence, bounds, raster size, and the
        source size SAM reported against the size we sent. Never calls the API.

        Records are collected without the frame check, so a frame SAM got wrong still shows its facts
        (``box_in_photo`` is None then). ``sam_boxes`` raises for such a frame."""
        image_path = Path(image_path)
        if not self._events_file(image_path, prompt).exists():
            raise CacheMiss(f"no stored SAM answer for {image_path.name!r} with prompt {prompt!r}; segment it first.")
        _, image, result = self._sam_parsed(image_path, prompt)
        sent_hw = _fitted_hw(*image.shape[:2], self.config.sam.max_upload_px)
        try:
            frame_h, frame_w = sam_frame_hw(result, sent_hw, image_path.name, prompt)
            sx, sy = image.shape[1] / frame_w, image.shape[0] / frame_h
        except ApiError:
            if raise_on_bad_frame:
                raise
            sx = sy = None
        boxes = {str(r.object_id): r for r in result.records if r.kind == "box"}
        records: list[SamRecordInfo] = []
        for r in (r for r in result.records if r.kind == "mask"):
            b = r.bounds
            box = boxes.get(str(r.object_id))
            records.append(SamRecordInfo(
                object_id=str(r.object_id), confidence=r.confidence,
                box_confidence=None if box is None else box.confidence,
                bounds=(float(b.left), float(b.top), float(b.right), float(b.bottom)),
                raster_wh=(int(r.mask.width), int(r.mask.height)),
                box_in_photo=None if box is None or sx is None or sy is None else (float(box.left) * sx, float(box.top) * sy,
                                                       float(box.right) * sx, float(box.bottom) * sy),
            ))
        raw_output = getattr(result, "raw_output", None)
        reported = sam_source_hw(raw_output)
        return SamInspection(
            image_wh=(image.shape[1], image.shape[0]), sent_wh=(sent_hw[1], sent_hw[0]),
            source_wh=None if reported is None else (reported[1], reported[0]),
            reported_wh=[(w, h) for h, w in sorted(_sam_source_sizes(raw_output))], records=records)

    def _sam_parsed(self, image_path: Path, prompt: str) -> tuple[Any, np.ndarray, Any]:
        parser = _load_sam_parser()
        image = _read_pixels(image_path)
        events_file = self._events_file(image_path, prompt)
        if events_file.exists():
            events = json.loads(events_file.read_text())
        else:
            quality, max_px = self.config.sam.upload_jpeg_quality, self.config.sam.max_upload_px
            body = sam_request_body(self.config.models.sam.id, prompt, _jpeg_data_url(image, quality, max_px))
            response = self._paid(self.config.models.sam, "/responses", body, Usage(images=1),
                                  transport=self._stream_transport, usage_from=_sam_usage)
            events = response["events"]
            if not any(e.get("type") == "response.completed" for e in events):
                # A cut-off or failed stream must not be replayed forever from the cache.
                kinds = sorted({str(e.get("type")) for e in events if "fail" in str(e.get("type"))
                                or "incomplete" in str(e.get("type"))}) or ["none"]
                raise ApiError(
                    f"{image_path.name}: the SAM stream for the prompt {prompt!r} ended without "
                    f"response.completed (error events: {', '.join(kinds)}). The call was billed at the "
                    "usage the ledger shows and the answer was not cached. Run the command again to "
                    "pay for a new attempt, or check the Meta API status first."
                )
            events_file.write_text(json.dumps(events))
        return parser, image, _parse_sam_events(parser, events)

    def _events_file(self, image_path: Path, prompt: str) -> Path:
        return self._cache_file(self._sam_key(image_path, prompt, decoded=False), ".events.json")

    def _sam_key(self, image_path: Path, prompt: str, *, decoded: bool) -> str:
        """Key of the raw events (``decoded=False``) or of the decoded mask. Real calls hash the
        decoded pixels that are sent plus the upload quality; mock calls hash the file bytes. Only
        the decoded mask depends on ``SAM_DECODE_VERSION``."""
        model = self.config.models.sam.id
        if self.settings.mock:
            return self._key("sam", model, prompt, [image_path], {})
        params: dict[str, Any] = {"upload_jpeg_quality": self.config.sam.upload_jpeg_quality,
                                  "max_upload_px": self.config.sam.max_upload_px}
        if decoded:
            params["decode_version"] = SAM_DECODE_VERSION
        pixels = _read_pixels(image_path)
        params["shape"] = list(pixels.shape)  # same bytes, different shape must not collide
        return self._key("sam", model, prompt, [image_path], params,
                         image_digests=[hashlib.sha256(pixels.tobytes()).hexdigest()])

    def _call_reasoning(
        self, model: ModelConfig, prompt: str, images: Sequence[np.ndarray], schema: dict[str, Any] | None,
        instructions: str | None = None, rejected_file: Path | None = None,
    ) -> dict[str, Any]:
        """Muse chat completion, request shape as in the cookbook (01_api_fundamentals).
        ``images`` are decoded pixels, downscaled and re-encoded here so no EXIF (GPS, camera) leaves the
        server. A paid answer that is cut off or not JSON is stored in ``rejected_file`` before the
        error is raised, so a rerun does not pay for it again (REQ-API-02)."""
        reasoning = self.config.reasoning
        quality, max_px = reasoning.upload_jpeg_quality, reasoning.max_upload_px
        body = reasoning_request_body(
            model.id, prompt, [_jpeg_data_url(i, quality, max_px) for i in images], schema, instructions,
            effort=reasoning.effort, max_tokens=reasoning.max_tokens,
            max_tokens_param=reasoning.max_tokens_param, prompt_cache_key=reasoning.prompt_cache_key,
        )
        estimate = Usage(
            input_tokens=(len(prompt) + len(instructions or "")) // CHARS_PER_TOKEN
            + reasoning.est_image_tokens * len(images),
            output_tokens=reasoning.max_tokens,  # worst case: the model uses the whole limit
            images=len(images),
        )
        response = self._paid(model, "/chat/completions", body, estimate)
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        try:
            return _answer_from_choice(model.id, choice, reasoning)
        except ApiError:
            if rejected_file is not None:
                rejected_file.write_text(json.dumps({
                    "finish_reason": choice.get("finish_reason"),
                    "message": {"content": (choice.get("message") or {}).get("content")}}))
            raise

    def _require_synthetic_for_contributor(self, paths: Sequence[Path]) -> None:
        """SPEC 4.3 item 5: the contributor endpoint trains on the data it gets, so a real
        photo must never reach it. A photo counts as synthetic only if its folder holds the
        truth.json that ``sewtrace synth`` writes (it has ``"synthetic": true``)."""
        if self.config.reasoning.model != "spark_contributor":
            return
        for path in paths:
            if not _is_synthetic_photo(path):
                raise PolicyViolation(
                    f"{path} is not a synthetic photo, and {self.config.reasoning_model.id} trains on "
                    "its data. Set reasoning.model to spark or wonder in config.toml for real photos."
                )

    def _paid(
        self, model: ModelConfig, path: str, body: dict[str, Any], estimate: Usage,
        transport: Transport | None = None,
        usage_from: Callable[[dict[str, Any], Usage], Usage] | None = None,
    ) -> dict[str, Any]:
        """The one wrapper for every paid call: budget check, retry, timeout, ledger.
        ``transport`` and ``usage_from`` default to the chat-completions ones; SAM streams."""
        transport = transport or self._transport
        usage_from = usage_from or _usage_from
        api = self.config.api
        if not self.settings.api_key:
            raise ApiError("META_API_KEY is not set. Set it in the environment, or use mock mode.")
        self._require_ledger()
        total, projected = self.ledger_total_usd(), cost_usd(model, estimate)
        if total + projected > api.budget_limit_usd:
            raise BudgetExceeded(
                f"ledger ${total:.4f} + estimate ${projected:.4f} is over the limit "
                f"${api.budget_limit_usd:.2f}. No call was made. Use offline mode (SPEC 8.3)."
            )
        headers = {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"}
        url = api.base_url.rstrip("/") + path
        response: dict[str, Any] | None = None
        for attempt in range(1, api.max_tries + 1):
            try:
                response = transport(url, headers, body, api.timeout_s)
                break
            except TimeoutError as e:
                # The server probably kept working and bills it: log the worst case (REQ-API-04).
                self._append_ledger(model, estimate, estimated=True)
                raise ApiTimeout(
                    f"{model.id} call took more than {api.timeout_s:g} s. "
                    "The ledger holds the worst-case cost of this call."
                ) from e
            except ApiNotSent:
                raise  # never left this machine: nothing to bill
            except ApiTransportError:
                self._append_ledger(model, estimate, estimated=True)
                raise
            except HttpError as e:
                if e.status == 429 or e.status >= 500:
                    if attempt == api.max_tries:
                        raise ApiError(f"{model.id} call failed after {attempt} tries: {e}") from e
                    self._sleep(api.backoff_s * 2 ** (attempt - 1))
                    continue
                raise ApiRequestRejected(
                    f"{model.id} call was rejected: {e}. Check the model ID, the request parameters and "
                    "META_API_KEY in config.toml and the environment. Nothing was retried.") from e
            except BaseException as e:  # noqa: BLE001 (any failure after sending may be billed)
                # E.g. a stream event that is not JSON, Ctrl-C or a cancelled task during a long
                # stream, or an error nobody mapped. Bill the worst case, then re-raise: never leave
                # a possibly paid call without a row (REQ-API-04). One row per failure.
                self._append_ledger(model, estimate, estimated=True)
                if isinstance(e, ApiError) or not isinstance(e, TRANSPORT_ERRORS):
                    raise  # interrupts and code bugs keep their type, so callers do not retry them
                raise ApiTransportError(
                    f"{model.id} call failed after the request was sent: {e!r}. "
                    "The ledger holds the worst-case cost of this call.") from e
        assert response is not None
        try:
            usage = usage_from(response, estimate)
        except Exception:  # noqa: BLE001
            # The money is spent whatever the answer looks like: bill the estimate, never skip the row.
            log.exception("%s: usage missing or malformed in the answer; the ledger holds the estimate", model.id)
            usage = estimate
        # ``usage_from`` returns the estimate object itself when the answer reports no usage.
        self._append_ledger(model, usage, estimated=usage is estimate)
        return response

    def _require_ledger(self) -> None:
        """Rule: no ledger file but a real answer in the cache means the ledger was lost,
        and the total would restart at $0. Refuse the paid call (REQ-API-04, C-01).

        The repo cache folder is checked too when the ledger is the default one, so a changed
        SEWTRACE_CACHE_DIR cannot hide the real answers. A ledger path set by a test is not
        tied to the repo cache."""
        if self.ledger_path.exists():
            return
        folders = [self.cache_dir]
        if self.ledger_path == resolve_ledger_path(self.config):
            folders.append(ROOT / "cache")
        for folder in folders:
            if any(folder.glob("real-*")):
                raise ApiError(
                    f"The ledger {self.ledger_path} is missing, but {folder} holds real answers, so "
                    "money was spent before. No call was made. Restore the ledger file from a backup. "
                    "If you are sure nothing was spent, create it with the header line: "
                    + ",".join(LEDGER_COLUMNS)
                )

    def _append_ledger(self, model: ModelConfig, usage: Usage, estimated: bool = False) -> None:
        new = not self.ledger_path.exists()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LEDGER_COLUMNS)
            w.writerow([int(time.time()), model.id, usage.input_tokens, usage.cached_tokens,
                        usage.output_tokens, usage.images, f"{cost_usd(model, usage):.6f}", int(estimated)])

    # ---- helpers ----------------------------------------------------------

    def _require_online(self, what: str) -> None:
        if self.settings.offline:
            raise CacheMiss(f"offline mode: no cached answer for {what}; run once online to fill the cache.")

    def _cache_file(self, key: str, suffix: str) -> Path:
        """Real answers are named ``real-<key>`` so the ledger guard can tell them from mock ones."""
        path = self.cache_dir / f"{'' if self.settings.mock else 'real-'}{key}{suffix}"
        self.cache_files_used.add(path)
        return path

    def _key(
        self, kind: str, model: str, prompt: str, images: Sequence[Path], params: dict[str, Any],
        image_digests: Sequence[str] | None = None,
    ) -> str:
        """SHA-256 of mode, model, prompt, image content and the request parameters that change the
        answer (REQ-API-02). ``max_tokens_param`` (only the wire name of the limit, whose value is
        in the key) and ``prompt_cache_key`` (only routes the provider's prompt cache) are left
        out on purpose, so renaming them does not pay for the same answer again.

        Image content is the file bytes. For real SAM and Muse calls it is ``image_digests``: the
        hash of the decoded pixels that are sent, so the key follows what the model sees (EXIF
        rotation applied)."""
        digests = list(image_digests) if image_digests is not None else [
            hashlib.sha256(p.read_bytes()).hexdigest() for p in images]
        return _sha256_json({
            "mode": "mock" if self.settings.mock else "real",
            "kind": kind,
            "model": model,
            "prompt": prompt,
            "images": digests,
            "params": params,
        })


def _answer_from_choice(model_id: str, choice: dict[str, Any], reasoning: Any) -> dict[str, Any]:
    """The JSON answer of one chat choice, or ApiError. The one place that judges a raw answer, for
    the live call and for the replay of a stored rejected one."""
    if choice.get("finish_reason") == "length":
        raise ApiError(
            f"{model_id} answer was cut off at {reasoning.max_tokens_param}={reasoning.max_tokens} "
            "(reasoning tokens count against it). Raise reasoning.max_tokens in config.toml."
        )
    try:
        return json.loads(choice["message"]["content"])
    except (KeyError, IndexError):
        raise ApiError(f"{model_id} answer has no message content (empty choices or no message).") from None
    except (json.JSONDecodeError, TypeError) as e:
        raise ApiError(f"{model_id} answer is not valid JSON: {e}") from e


def _read_pixels(path: Path) -> np.ndarray:
    """The pixels the pipeline works on. ``cv2.imread`` applies the EXIF orientation, so this is
    the frame every later step (marker, contour, mm scale) uses."""
    image = cv2.imread(str(path))
    if image is None:
        raise ApiError(f"{path} is not a readable image (unknown image type or damaged file). Use a .jpg or .png photo.")
    return image


def _fit_long_side(image: np.ndarray, max_px: int) -> np.ndarray:
    """``image`` shrunk with INTER_AREA so its long side is at most ``max_px``; the same array if it
    already fits. Both sides are rounded, so a caller maps coordinates back with the per-axis
    factors ``original / sent``, not one scale."""
    height, width = image.shape[:2]
    sent_h, sent_w = _fitted_hw(height, width, max_px)
    if (sent_h, sent_w) == (height, width):
        return image
    return cv2.resize(image, (sent_w, sent_h), interpolation=cv2.INTER_AREA)


def _fitted_hw(height: int, width: int, max_px: int) -> tuple[int, int]:
    """``(height, width)`` that ``_fit_long_side`` sends; one rounding rule for both."""
    long_side = max(height, width)
    if long_side <= max_px:
        return height, width
    scale = max_px / long_side
    return max(1, round(height * scale)), max(1, round(width * scale))


def _jpeg_data_url(image: np.ndarray, quality: int, max_px: int) -> str:
    """Downscale (long side ``max_px``, so the image token count matches SPEC 8.1) and re-encode the
    decoded pixels for SAM and Muse. Encoding drops EXIF (GPS, camera), so a model cannot rotate the
    photo behind our back. SAM box coordinates are in the frame of the downscaled image."""
    ok, buffer = cv2.imencode(".jpg", _fit_long_side(image, max_px), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ApiError("Could not re-encode the photo for the model.")
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode()}"


def sam_request_body(model_id: str, prompt: str, image_data_url: str) -> dict[str, Any]:
    """Responses API body, as in the cookbook's ``response_input``: one user message with a
    short noun phrase and one image. Text prompts only (REQ-SEG-01)."""
    phrase = prompt.strip()
    if not phrase:
        raise ValueError("The SAM prompt must be a short noun phrase.")
    return {
        "model": model_id,
        "stream": True,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": phrase},
                {"type": "input_image", "image_url": image_data_url},
            ],
        }],
    }


def with_output_text_done(events: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """The API streams ``response.content_part.done`` but not ``response.output_text.done``,
    which the parser needs to close a lane. Add it from the content part. Same logic as the
    cookbook's ``stream_events``."""
    done: set[tuple[Any, Any, Any]] = set()
    for event in events:
        lane = (event.get("item_id"), event.get("output_index"), event.get("content_index"))
        if event.get("type") == "response.output_text.done":
            if lane not in done:
                done.add(lane)
                yield event
            continue
        yield event
        part = event.get("part")
        if (event.get("type") == "response.content_part.done" and lane not in done
                and isinstance(part, dict) and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)):
            done.add(lane)
            yield {"type": "response.output_text.done", "item_id": event.get("item_id"),
                   "output_index": event.get("output_index"), "content_index": event.get("content_index"),
                   "text": part["text"]}


def _load_sam_parser() -> Any:
    try:
        import meta_sam_parser  # lazy: only the real SAM path needs it (mock mode and tests do not)
    except ImportError as e:
        raise ApiError(PARSER_MISSING) from e
    return meta_sam_parser


def _run_sync(make_coroutine: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Run a coroutine to completion from sync code, also when the caller sits inside a running event
    loop (a FastAPI async handler). ``asyncio.run`` refuses there, so the coroutine gets a fresh loop
    in a worker thread. The work is CPU only (the stored events are replayed), so blocking the caller
    for its short duration is fine."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coroutine())
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make_coroutine())).result()


def _parse_sam_events(parser: Any, events: list[dict[str, Any]]) -> Any:
    """Run the official parser over the stored events and check the outcome."""

    async def run() -> Any:
        async def source() -> AsyncIterator[dict[str, Any]]:
            for event in with_output_text_done(events):
                yield event

        # aclosing: the parser returns on the terminal event while the generator is suspended.
        async with aclosing(source()) as stream:
            return await parser.parse_responses_stream(stream, parser.image_segmentation_format()).final_result()

    result = _run_sync(run)
    if result.outcome.status != "completed":
        reason = getattr(result.outcome, "reason", "unknown")
        raise ApiError(f"SAM answer was incomplete: {reason} ({getattr(result.outcome, 'detail', None)})")
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise ApiError("SAM answer was rejected by the parser: "
                       + "; ".join(f"line {d.line}: {d.message}" for d in errors))
    return result


# One box token of the wire grammar, e.g. ``<|box;x1=1;y1=2;x2=3;y2=4;w=1024;h=710;c=0.9|>``. ``w`` and ``h``
# are the size of the frame SAM processed (meta_sam_parser reads them as source_width and source_height
# but does not keep them on the record).
_BOX_TOKEN = re.compile(r"<\|box;([^|]*)\|>")
_UINT = re.compile(r"[0-9]+")


def _sam_source_sizes(raw_output: object) -> set[tuple[int, int]]:
    """Every distinct ``(height, width)`` found in the box tokens of ``raw_output``. Never raises."""
    if not isinstance(raw_output, str):
        return set()
    sizes: set[tuple[int, int]] = set()
    for token in _BOX_TOKEN.finditer(raw_output):
        fields = {k.strip(): v.strip() for k, _, v in (f.partition("=") for f in token.group(1).split(";"))}
        w, h = fields.get("w", ""), fields.get("h", "")
        if _UINT.fullmatch(w) and _UINT.fullmatch(h) and int(w) > 0 and int(h) > 0:
            sizes.add((int(h), int(w)))
    return sizes


def sam_source_hw(raw_output: object) -> tuple[int, int] | None:
    """``(height, width)`` of the frame SAM reports in its box lines, or None when the text has no box
    token, a token without a valid ``w`` and ``h``, or boxes that disagree. Never raises."""
    sizes = _sam_source_sizes(raw_output)
    return next(iter(sizes)) if len(sizes) == 1 else None


# Retaking the photo does not help: EXIF is stripped before the upload and a transposed frame is server behaviour.
_SAM_FRAME_ADVICE = ("The sizes are above. This is a problem in the SAM answer, not in the photo. "
                     "Keep the stored answer in the cache folder and report these sizes.")


def sam_frame_hw(result: Any, sent_hw: tuple[int, int], photo: str, prompt: str) -> tuple[int, int]:
    """``(height, width)`` of the frame the SAM coordinates are in: the size SAM reports, else the
    size we sent.

    A reported size with the sent aspect (within rounding) is a uniform server resize and is used.
    Anything else (transposed, cropped, padded) is an error: the parser only checks bounds against the
    reported frame, so mapping it with separate x and y scales would stretch the mask by tens of mm.
    Boxes that report different sizes are an error too, since no single frame fits them."""
    sizes = _sam_source_sizes(getattr(result, "raw_output", None))
    sent_h, sent_w = sent_hw
    if len(sizes) > 1:
        listed = ", ".join(f"{w} x {h}" for h, w in sorted(sizes))
        raise ApiError(f"{photo}: SAM reported different frame sizes for the prompt {prompt!r} ({listed} px). "
                       "No single frame fits the masks, so no mask was used. " + _SAM_FRAME_ADVICE)
    if not sizes:
        log.info("SAM did not report a readable source size, so the sent size %d x %d px is assumed.", sent_w, sent_h)
        return sent_hw
    (rep_h, rep_w), = sizes
    if (rep_h, rep_w) == sent_hw:
        return sent_hw
    if abs(rep_w * sent_h - rep_h * sent_w) > max(sent_h, sent_w):
        raise ApiError(
            f"{photo}: SAM worked on a {rep_w} x {rep_h} px frame for the prompt {prompt!r}, but "
            f"{sent_w} x {sent_h} px was sent. The mask and the photo use different pixel frames "
            "(rotated, cropped or padded on the server). The mask would land in the wrong place, so no "
            "mask was used. " + _SAM_FRAME_ADVICE)
    log.warning("SAM worked on a %d x %d px frame, but %d x %d px was sent (resized on the server). "
                "Coordinates are mapped with the size SAM reported.", rep_w, rep_h, sent_w, sent_h)
    return rep_h, rep_w


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bool ``mask`` resized to ``width`` x ``height``: INTER_AREA when it shrinks (no aliasing of thin
    parts), INTER_LINEAR otherwise; then 0.5 keeps a smooth edge. A mixed resize (one axis shrinks, one
    grows) uses INTER_LINEAR and may alias slightly on the shrinking axis."""
    shrinking = width <= mask.shape[1] and height <= mask.shape[0]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=interpolation) > 0.5


def _raster_of(mask: Any, decode_mask: Callable[[Any], bytes], photo: str, object_id: str) -> np.ndarray:
    """Bool ``height x width`` raster of one mask record. The parser returns ``width * height`` bytes,
    row-major, one per pixel (0 or 1)."""
    mw, mh = int(mask.width), int(mask.height)
    raw = decode_mask(mask)
    if len(raw) != mw * mh:
        raise ApiError(f"{photo}: SAM mask {object_id} decodes to {len(raw)} values, expected {mw} x {mh}.")
    return np.frombuffer(raw, dtype=np.uint8).reshape(mh, mw) > 0


def sam_results_from(
    result: Any, decode_mask: Callable[[Any], bytes], image_hw: tuple[int, int], photo: str, prompt: str,
    *, box_tol_px: float, full_hw: tuple[int, int] | None = None,
) -> list[SamResult]:
    """Parsed segmentation records -> full-image boolean masks (meta_sam_parser 0.1.0 contract).

    Each mask record holds a raster of its own size (``mask.width`` x ``mask.height``) and half-open
    ``bounds`` in the frame SAM processed. The raster is resized to the bounds and placed at
    (left, top). Its score is the record's ``confidence`` (None when SAM sent none).

    ``image_hw`` is the shape of the pixels that were sent to SAM. The coordinate frame is the size SAM
    reports in its box lines (``sam_frame_hw``), which is ``image_hw`` unless the server resized the photo.
    Photos are downscaled before the upload (``sam.max_upload_px``), so pass the original shape as
    ``full_hw``: masks are scaled back to that frame, which keeps the mm geometry unchanged. Bounds that
    pass the frame by more than ``box_tol_px`` mean the frames differ (EXIF rotation): that is an error,
    never a clip, since it would misplace the mask in mm silently."""
    height, width = sam_frame_hw(result, image_hw, photo, prompt)
    full_h, full_w = full_hw or image_hw
    sx, sy = full_w / width, full_h / height
    tol_px = box_tol_px * max(sx, sy)
    box_confidence = {str(r.object_id): r.confidence for r in result.records if r.kind == "box"}
    out: list[SamResult] = []
    for record in (r for r in result.records if r.kind == "mask"):
        if record.confidence is None and box_confidence.get(str(record.object_id)) is not None:
            log.warning("%s: SAM mask %s for %r has no confidence, but its box has %g. The box value is not "
                        "used as the score (REQ-SEG-03); the min_score check will be skipped. Run 'sewtrace "
                        "day1' to see both values.", photo, record.object_id, prompt, box_confidence[str(record.object_id)])
        raster = _raster_of(record.mask, decode_mask, photo, str(record.object_id))
        b = record.bounds
        left, top, right, bottom = round(b.left), round(b.top), round(b.right), round(b.bottom)
        if right <= left or bottom <= top:
            raise ApiError(f"{photo}: SAM mask {record.object_id} for {prompt!r} has empty bounds "
                           f"({left}, {top}) to ({right}, {bottom}).")
        if left * sx < -tol_px or top * sy < -tol_px or right * sx > full_w + tol_px or bottom * sy > full_h + tol_px:
            raise ApiError(
                f"{photo}: SAM mask {record.object_id} for {prompt!r} is ({left * sx:g}, {top * sy:g}) to "
                f"({right * sx:g}, {bottom * sy:g}), but the image is {full_w} x {full_h} px. The mask and the "
                "photo use different pixel frames (EXIF rotation or a resize on the server). The mask "
                "would land in the wrong place, so no mask was used. " + _SAM_FRAME_ADVICE)
        bw, bh = right - left, bottom - top
        if raster.shape != (bh, bw):  # the raster has its own size; linear then 0.5 keeps a smooth edge
            raster = _resize_mask(raster, bw, bh)
        full = np.zeros((height, width), bool)
        x0, y0, x1, y1 = max(left, 0), max(top, 0), min(right, width), min(bottom, height)
        if x1 > x0 and y1 > y0:
            full[y0:y1, x0:x1] = raster[y0 - top : y1 - top, x0 - left : x1 - left]
        if full.any():
            if (full_h, full_w) != (height, width):
                full = _resize_mask(full, full_w, full_h)
            out.append(SamResult(mask=full, score=record.confidence))
    return out


def select_best(results: Sequence[SamResult]) -> SamResult:
    """Highest score wins (REQ-SEG-03). If any mask has no score, no score is trusted: the
    largest mask wins and the winner's score is None (stand-in, see MetaClient.segment)."""
    if all(r.score is not None for r in results):
        return max(results, key=lambda r: r.score or 0.0)
    best = max(results, key=lambda r: int(r.mask.sum()))
    return best.model_copy(update={"score": None})


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic JSON schema -> the plain shape of the cookbook example: ``$ref`` inlined, no
    ``$defs``, ``additionalProperties: false`` on every object. Constraints such as ``minimum``
    stay, since pydantic checks the answer again (UNCONFIRMED: Day 1 shows whether the API
    accepts them)."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = defs[node["$ref"].removeprefix("#/$defs/")]
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return walk({**target, **extra})
        out = {k: walk(v) for k, v in node.items() if k != "$defs"}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out

    return walk(schema)


def reasoning_request_body(
    model_id: str, prompt: str, image_data_urls: Sequence[str], schema: dict[str, Any] | None,
    instructions: str | None, *, effort: str, max_tokens: int, max_tokens_param: str, prompt_cache_key: str,
) -> dict[str, Any]:
    """Chat Completions body as in the cookbook. Stable instructions first as a developer
    message; images only in the user message (a developer message with an image gets HTTP 400)."""
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "developer", "content": instructions})
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in image_data_urls]
    messages.append({"role": "user", "content": content})
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "reasoning_effort": effort,
        max_tokens_param: max_tokens,
        "prompt_cache_key": prompt_cache_key,
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema", "json_schema": {"name": "answer", "schema": strict_json_schema(schema)},
        }
    return body


def _is_synthetic_photo(path: Path) -> bool:
    truth = path.parent / "truth.json"
    try:
        return json.loads(truth.read_text()).get("synthetic") is True
    except (OSError, ValueError, AttributeError):
        return False


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _usage_from(response: dict[str, Any], estimate: Usage) -> Usage:
    """Billed usage from the response. If the API omits it, bill the estimate (never log $0)."""
    u = response.get("usage")
    if not u:
        return estimate
    details = u.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=int(u.get("prompt_tokens", 0)),
        cached_tokens=int(details.get("cached_tokens", 0)),
        output_tokens=int(u.get("completion_tokens", 0)),
        images=estimate.images,
    )


def _sam_usage(response: dict[str, Any], estimate: Usage) -> Usage:
    """SAM is billed per image. Token counts come from the ``response.completed`` event if it
    has them (UNCONFIRMED: Day 1 shows where the API puts the image token count)."""
    for event in reversed(response.get("events", [])):
        if event.get("type") == "response.completed":
            u = (event.get("response") or {}).get("usage")
            if not u:
                return estimate  # no usage reported: the row is flagged as estimated
            details = u.get("input_tokens_details") or {}
            return Usage(input_tokens=int(u.get("input_tokens", 0)), cached_tokens=int(details.get("cached_tokens", 0)),
                         output_tokens=int(u.get("output_tokens", 0)), images=estimate.images)
    return estimate


def _mock_segment(image_path: Path, prompt: str, garment_prompt: str) -> list[SamResult]:
    """The mock is always sure of its one mask: score 1.0."""
    sidecar = image_path.with_suffix(".masks") / f"{prompt}.png"
    if sidecar.exists():
        mask = cv2.imread(str(sidecar), cv2.IMREAD_GRAYSCALE) > 127
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        return [SamResult(mask=_jitter_edges(mask, seed=seed), score=1.0)]
    if prompt != garment_prompt:
        raise FileNotFoundError(f"mock SAM has no '{prompt}' mask for {image_path}")
    return [SamResult(mask=_threshold_largest(cv2.imread(str(image_path))), score=1.0)]


def _jitter_edges(mask: np.ndarray, seed: int) -> np.ndarray:
    """Roughen edges by about a pixel, the way a real mask boundary wobbles."""
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(0, 1, mask.shape).astype(np.float32), (0, 0), 3)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)
    return (soft + 0.15 * noise) > 0.5


def _threshold_largest(image: np.ndarray) -> np.ndarray:
    """Garment = pixels that differ from the border (backdrop) colour; largest blob."""
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    backdrop = np.median(border, axis=0)
    dist = np.linalg.norm(image.astype(float) - backdrop, axis=2)
    fg = (dist > 60).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return np.zeros(fg.shape, bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest
