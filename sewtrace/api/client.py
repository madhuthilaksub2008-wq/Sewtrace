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
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from sewtrace.config import ModelConfig, Settings, resolve_ledger_path

LEDGER_COLUMNS = ["time", "model", "input_tokens", "cached_tokens", "output_tokens", "images", "cost_usd", "estimated"]
CHARS_PER_TOKEN = 4  # rough token count for the pre-call estimate only

# (url, headers, json body, timeout_s) -> parsed JSON response. Raises HttpError
# for a non-2xx status and TimeoutError when the call is too slow.
Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class ApiError(RuntimeError):
    """A Meta API call failed after the allowed tries."""


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
    """One mask SAM returned, with its score. The real response shape is unconfirmed
    (Day 1 test, REQ-API-09); _call_sam turns it into these and nothing else changes."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    mask: np.ndarray  # bool, H x W
    score: float


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


def urllib_transport(url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 (https URL from config)
            return json.loads(response.read())
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


class MetaClient:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings or Settings.load()
        self.config = self.settings.config
        self.cache_dir = self.settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.settings.ledger_path or resolve_ledger_path(self.config)
        self.calls = 0  # real or mocked calls that missed the cache
        self._transport = transport
        self._sleep = sleep

    # ---- public API -------------------------------------------------------

    def segment(self, image_path: str | Path, prompt: str) -> np.ndarray:
        """Boolean HxW mask of the text ``prompt`` (SAM takes text prompts only, REQ-SEG-01).

        With several masks, the highest score wins (REQ-SEG-03). If that score is below
        ``sam.min_score``, raises SegmentationFailed (REQ-SEG-04). The best mask and its
        score are cached together, so the threshold is checked on every call and a low
        score is never paid for twice."""
        image_path = Path(image_path)
        model = self.config.models.sam
        key = self._key("sam", model.id, prompt, [image_path], {})
        cached, cached_score = self._cache_file(key, ".png"), self._cache_file(key, ".score.json")
        if cached.exists() and cached_score.exists():
            best = SamResult(
                mask=cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE) > 127,
                score=float(json.loads(cached_score.read_text())["score"]),
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
            best = max(results, key=lambda r: r.score)
            cv2.imwrite(str(cached), best.mask.astype(np.uint8) * 255)
            cached_score.write_text(json.dumps({"score": best.score}))
        min_score = self.config.sam.min_score
        if best.score < min_score:
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
    ) -> dict[str, Any]:
        """Structured JSON answer from the reasoning model."""
        reasoning = self.config.reasoning
        model = self.config.reasoning_model
        params = {
            "reasoning_effort": reasoning.effort,
            "max_tokens": reasoning.max_tokens,
            "schema_sha256": _sha256_json(schema) if schema is not None else None,
        }
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
            answer = self._call_reasoning(model, prompt, paths, schema)
        cached.write_text(json.dumps(answer))
        return answer

    def ledger_total_usd(self) -> float:
        if not self.ledger_path.exists():
            return 0.0
        with self.ledger_path.open(newline="") as f:
            return sum(float(row["cost_usd"]) for row in csv.DictReader(f))

    # ---- real endpoints ---------------------------------------------------

    def _call_sam(self, image_path: Path, prompt: str) -> list[SamResult]:
        # UNCONFIRMED: the SAM 3.1 request and response shape is not known until
        # the Day 1 test (REQ-API-09). When it is, build the request here and send
        # it through self._paid(self.config.models.sam, ...). Nothing else changes.
        raise NotImplementedError(
            "Real SAM 3.1 call not wired yet (Day 1 test, REQ-API-09). Run with SEWTRACE_MOCK=1."
        )

    def _call_reasoning(
        self, model: ModelConfig, prompt: str, image_paths: Sequence[Path], schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Muse chat completion (OpenAI-compatible, SPEC 4.2). Parameter names follow the
        OpenAI Chat Completions API; confirm them in the Day 1 test (REQ-API-09)."""
        reasoning = self.config.reasoning
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            b64 = base64.b64encode(path.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        body: dict[str, Any] = {
            "model": model.id,
            "messages": [{"role": "user", "content": content}],
            "reasoning_effort": reasoning.effort,
            "max_tokens": reasoning.max_tokens,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema", "json_schema": {"name": "answer", "schema": schema},
            }
        estimate = Usage(
            input_tokens=len(prompt) // CHARS_PER_TOKEN + reasoning.est_image_tokens * len(image_paths),
            output_tokens=reasoning.max_tokens,  # worst case: the model uses the whole limit
            images=len(image_paths),
        )
        response = self._paid(model, "/chat/completions", body, estimate)
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ApiError(
                f"{model.id} answer was cut off at max_tokens={reasoning.max_tokens} "
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

    def _paid(self, model: ModelConfig, path: str, body: dict[str, Any], estimate: Usage) -> dict[str, Any]:
        """The one wrapper for every paid call: budget check, retry, timeout, ledger."""
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
                response = self._transport(url, headers, body, api.timeout_s)
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
                if not (e.status == 429 or e.status >= 500):
                    raise ApiError(f"{model.id} call failed: {e}") from e
                if attempt == api.max_tries:
                    raise ApiError(f"{model.id} call failed after {attempt} tries: {e}") from e
                self._sleep(api.backoff_s * 2 ** (attempt - 1))
        assert response is not None
        self._append_ledger(model, _usage_from(response, estimate))
        return response

    def _require_ledger(self) -> None:
        """Rule: no ledger file but a real answer in the cache means the ledger was lost,
        and the total would restart at $0. Refuse the paid call (REQ-API-04, C-01)."""
        if self.ledger_path.exists():
            return
        if any(self.cache_dir.glob("real-*")):
            raise ApiError(
                f"The ledger {self.ledger_path} is missing, but {self.cache_dir} holds real answers, so "
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
