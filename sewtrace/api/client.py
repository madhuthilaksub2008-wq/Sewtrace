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
import json
import logging
import mimetypes
import socket
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
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

PARSER_MISSING = (
    "SAM 3.1 output is decoded with Meta's meta_sam_parser, which is not installed. "
    "No call was made. Run: pip install -e '.[real]'"
)


class ApiError(RuntimeError):
    """A Meta API call failed after the allowed tries."""


class ApiRequestRejected(ApiError):
    """The server refused the request itself: any HTTP status except 429 and 5xx (400 bad parameter,
    401 or 403 bad key, 404 wrong model ID, 422 bad schema). A retry or a fallback cannot help,
    so callers stop the run."""


class ApiTimeout(ApiError):
    """The call did not finish within ``api.timeout_s``."""


class ApiTransportError(ApiError):
    """The connection broke after the request may have been sent, so the call may be billed."""


class BudgetExceeded(ApiError):
    """The next paid call would take the ledger over ``api.budget_limit_usd``."""


class PolicyViolation(ApiError):
    """The call would break a data rule (real photo to the contributor endpoint). Never retried."""


class CacheMiss(ApiError):
    """Offline mode and the answer is not in the cache."""


class SegmentationFailed(ApiError):
    """SAM found no mask that scores high enough (REQ-SEG-04)."""


class SamResult(BaseModel):
    """One mask SAM returned, full image size, with its score if the API gave one.

    The cookbook shows no score on SAM records (SPEC conflict with REQ-SEG-03 and 04).
    ``score`` is None when the parsed record has no score or confidence attribute."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    mask: np.ndarray  # bool, H x W
    score: float | None = None


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
        raise ApiError(f"network error: {e.reason}") from e  # DNS or refused: never sent
    except ConnectionError as e:  # reset while reading the answer: the server got the request
        raise ApiTransportError(f"connection lost: {e}") from e


def urllib_transport(url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    with _open(url, headers, body, timeout_s) as response:
        return json.loads(response.read())


def urllib_stream_transport(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Read a server-sent-events stream to its end. ``timeout_s`` is the limit for the whole
    call, not for one read: a stream that keeps trickling must still stop (REQ-API-07)."""
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
        self.calls = 0  # real or mocked calls that missed the cache
        self._transport = transport
        self._stream_transport = stream_transport
        self._sleep = sleep

    # ---- public API -------------------------------------------------------

    def segment(self, image_path: str | Path, prompt: str) -> np.ndarray:
        """Boolean HxW mask of the text ``prompt`` (SAM takes text prompts only, REQ-SEG-01).

        With several masks, the highest score wins (REQ-SEG-03). If that score is below
        ``sam.min_score``, raises SegmentationFailed (REQ-SEG-04). The best mask and its
        score are cached together, so the threshold is checked on every call and a low
        score is never paid for twice.

        SPEC CONFLICT, for the user to decide on Day 1: the cookbook shows no score on SAM
        records. When the masks carry no score, the largest mask wins and the min_score check
        is skipped with a warning. It is never faked as 1.0."""
        image_path = Path(image_path)
        model = self.config.models.sam
        key = self._key("sam", model.id, prompt, [image_path], {})
        cached, cached_score = self._cache_file(key, ".png"), self._cache_file(key, ".score.json")
        if cached.exists() and cached_score.exists():
            stored = json.loads(cached_score.read_text())["score"]
            best = SamResult(
                mask=cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE) > 127,
                score=None if stored is None else float(stored),
            )
        else:
            self._require_online(f"segment {image_path.name!r} with prompt {prompt!r}")
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
                "Check the mask by eye. (SPEC conflict with REQ-SEG-04, Day 1.)",
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
        key = self._key("reason", model.id, prompt, paths, params)
        cached = self._cache_file(key, ".json")
        if cached.exists():
            return json.loads(cached.read_text())
        self._require_online(f"reasoning call to {model.id}")
        self.calls += 1
        if self.settings.mock:
            if mock is None:
                raise RuntimeError("reason() called in mock mode without a mock answer")
            answer = mock()
        else:
            answer = self._call_reasoning(model, prompt, paths, schema, instructions)
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
        model = self.config.models.sam
        parser = _load_sam_parser()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ApiError(f"{image_path} is not a readable image. Check the photo path.")
        events_file = self._cache_file(self._key("sam", model.id, prompt, [image_path], {}), ".events.json")
        if events_file.exists():
            events = json.loads(events_file.read_text())
        else:
            body = sam_request_body(model.id, prompt, _data_url(image_path))
            response = self._paid(model, "/responses", body, Usage(images=1),
                                  transport=self._stream_transport, usage_from=_sam_usage)
            events = response["events"]
            events_file.write_text(json.dumps(events))
        result = _parse_sam_events(parser, events)
        return sam_results_from(result, parser.decode_mask_to_raster, image.shape[:2], image_path.name, prompt)

    def _call_reasoning(
        self, model: ModelConfig, prompt: str, image_paths: Sequence[Path], schema: dict[str, Any] | None,
        instructions: str | None = None,
    ) -> dict[str, Any]:
        """Muse chat completion, request shape as in the cookbook (01_api_fundamentals)."""
        reasoning = self.config.reasoning
        body = reasoning_request_body(
            model.id, prompt, [_data_url(p) for p in image_paths], schema, instructions,
            effort=reasoning.effort, max_tokens=reasoning.max_tokens,
            max_tokens_param=reasoning.max_tokens_param, prompt_cache_key=reasoning.prompt_cache_key,
        )
        estimate = Usage(
            input_tokens=(len(prompt) + len(instructions or "")) // CHARS_PER_TOKEN
            + reasoning.est_image_tokens * len(image_paths),
            output_tokens=reasoning.max_tokens,  # worst case: the model uses the whole limit
            images=len(image_paths),
        )
        response = self._paid(model, "/chat/completions", body, estimate)
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ApiError(
                f"{model.id} answer was cut off at {reasoning.max_tokens_param}={reasoning.max_tokens} "
                "(reasoning tokens count against it). Raise reasoning.max_tokens in config.toml."
            )
        try:
            return json.loads(choice["message"]["content"])
        except json.JSONDecodeError as e:
            raise ApiError(f"{model.id} answer is not valid JSON: {e}") from e

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
        assert response is not None
        self._append_ledger(model, usage_from(response, estimate))
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
        return self.cache_dir / f"{'' if self.settings.mock else 'real-'}{key}{suffix}"

    def _key(self, kind: str, model: str, prompt: str, images: Sequence[Path], params: dict[str, Any]) -> str:
        """SHA-256 of mode, model, prompt, image bytes and every request parameter (REQ-API-02)."""
        return _sha256_json({
            "mode": "mock" if self.settings.mock else "real",
            "kind": kind,
            "model": model,
            "prompt": prompt,
            "images": [hashlib.sha256(p.read_bytes()).hexdigest() for p in images],
            "params": params,
        })


def _data_url(path: Path) -> str:
    """The cookbook sends images as ``data:<mime>;base64,...`` for both SAM and Muse."""
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None or not mime.startswith("image/"):
        raise ApiError(f"Cannot tell the image type of {path.name}. Use a .jpg or .png photo.")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


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


def _parse_sam_events(parser: Any, events: list[dict[str, Any]]) -> Any:
    """Run the official parser over the stored events and check the outcome."""

    async def run() -> Any:
        async def source() -> AsyncIterator[dict[str, Any]]:
            for event in with_output_text_done(events):
                yield event

        # aclosing: the parser returns on the terminal event while the generator is suspended.
        async with aclosing(source()) as stream:
            return await parser.parse_responses_stream(stream, parser.image_segmentation_format()).final_result()

    result = asyncio.run(run())
    if result.outcome.status != "completed":
        reason = getattr(result.outcome, "reason", "unknown")
        raise ApiError(f"SAM answer was incomplete: {reason} ({getattr(result.outcome, 'detail', None)})")
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise ApiError("SAM answer was rejected by the parser: "
                       + "; ".join(f"line {d.line}: {d.message}" for d in errors))
    return result


def _record_score(*records: Any) -> float | None:
    """A score the parser exposes on a record, if any. Attribute names are a guess (the cookbook
    shows none), so this stays None unless a numeric ``score`` or ``confidence`` exists."""
    for record in records:
        for name in ("score", "confidence"):
            value = getattr(record, name, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def sam_results_from(
    result: Any, decode_mask: Callable[[Any], Any], image_hw: tuple[int, int], photo: str, prompt: str,
) -> list[SamResult]:
    """Parsed segmentation records -> full-image boolean masks.

    UNCONFIRMED (Day 1, REQ-API-09): in the cookbook example each mask is as large as its box
    (174 x 203 for box 751..925 x 290..493), so a mask is read as a raster local to its box and
    placed at the box top-left. A mask the size of the whole image is used as it is. Any other
    size is an error, not a guess. The raster is read as row-major, any non-zero value is
    foreground."""
    height, width = image_hw
    boxes = {r.object_id: r for r in result.records if r.kind == "box"}
    out: list[SamResult] = []
    for record in (r for r in result.records if r.kind == "mask"):
        mw, mh = int(record.mask.width), int(record.mask.height)
        flat = np.asarray(decode_mask(record.mask))
        if flat.size != mw * mh:
            raise ApiError(f"{photo}: SAM mask {record.object_id} decodes to {flat.size} values, "
                           f"expected {mw} x {mh}.")
        raster = flat.reshape(mh, mw) > 0
        box = boxes.get(record.object_id)
        full = np.zeros((height, width), bool)
        if (mw, mh) == (width, height):
            full = raster
        elif box is not None and (mw, mh) == (int(box.right - box.left), int(box.bottom - box.top)):
            left, top = int(box.left), int(box.top)
            x0, y0, x1, y1 = max(left, 0), max(top, 0), min(left + mw, width), min(top + mh, height)
            if x1 > x0 and y1 > y0:
                full[y0:y1, x0:x1] = raster[y0 - top : y1 - top, x0 - left : x1 - left]
        else:
            raise ApiError(
                f"{photo}: SAM mask {record.object_id} for {prompt!r} is {mw} x {mh} px. That is neither "
                f"the image ({width} x {height}) nor its box. The mask layout is unconfirmed (Day 1): "
                "check a real response and fix sam_results_from.")
        if full.any():
            out.append(SamResult(mask=full, score=_record_score(record, box)))
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
            u = (event.get("response") or {}).get("usage") or {}
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
