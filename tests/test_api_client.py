"""MetaClient: config, cache keys, budget guard, retry, timeout, ledger, offline mode.

Every test runs with a fake transport. No test touches the network or needs a key.
"""

import ast
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from sewtrace.api.client import (
    ApiError,
    ApiNotSent,
    ApiRequestRejected,
    ApiTimeout,
    BudgetExceeded,
    CacheMiss,
    HttpError,
    MetaClient,
    SamResult,
    SegmentationFailed,
    Usage,
)
from sewtrace.config import AppConfig, Settings, load_config

KEY = "test-key-not-real"


def _answer(content: dict | None = None, finish: str = "stop", usage: dict | None = None) -> dict:
    return {
        "choices": [{"finish_reason": finish, "message": {"content": json.dumps(content or {"ok": 1})}}],
        "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 2000,
                           "prompt_tokens_details": {"cached_tokens": 400}},
    }


class FakeTransport:
    """Returns queued responses in order; an Exception in the queue is raised."""

    def __init__(self, *queue):
        self.queue = list(queue)
        self.calls: list[tuple[str, dict, dict, float]] = []

    def __call__(self, url, headers, body, timeout_s):
        self.calls.append((url, headers, body, timeout_s))
        item = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def _config(**api) -> AppConfig:
    cfg = load_config()
    return cfg.model_copy(update={"api": cfg.api.model_copy(update=api)})


def _client(tmp_path, transport, *, mock=False, offline=False, config=None, key=KEY):
    sleeps: list[float] = []
    settings = Settings(api_key=key, mock=mock, cache_dir=tmp_path, offline=offline,
                        ledger_path=tmp_path / "ledger" / "ledger.csv",  # never the real data/ledger.csv
                        config=config or load_config())
    return MetaClient(settings, transport=transport, sleep=sleeps.append), sleeps


@pytest.fixture
def photo(tmp_path) -> Path:
    path = tmp_path / "p.jpg"
    cv2.imwrite(str(path), np.full((40, 40, 3), 200, np.uint8))  # real SAM keys hash the decoded pixels
    return path


def _rows(client) -> list[dict]:
    with client.ledger_path.open(newline="") as f:
        return list(csv.DictReader(f))


# ---- config (REQ-API-05) ---------------------------------------------------------

def test_req_api_05_config_holds_models_prices_and_effort():
    cfg = load_config()
    assert cfg.api.base_url == "https://api.meta.ai/v1"
    assert cfg.models.wonder.id == "muse-wonder-1.0"
    assert cfg.models.spark.output_per_m_usd == 4.25
    assert cfg.models.sam.price_per_image_usd == pytest.approx(0.0025)
    assert cfg.reasoning_model.id == "muse-spark-1.3"
    assert cfg.reasoning.effort == "high"
    assert cfg.api.budget_limit_usd == 140.0 and cfg.api.max_tries == 3


def test_req_api_05_effort_none_is_rejected():
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.reasoning.model_validate({**cfg.reasoning.model_dump(), "effort": "none"})
    for ok in ("minimal", "low", "medium", "high", "xhigh", "max"):
        cfg.reasoning.model_validate({**cfg.reasoning.model_dump(), "effort": ok})


def test_req_api_05_model_change_needs_no_code_change(tmp_path):
    cfg = load_config()
    cfg = cfg.model_copy(update={"reasoning": cfg.reasoning.model_copy(update={"model": "wonder"})})
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path, transport, config=cfg)
    client.reason("p")
    assert transport.calls[0][2]["model"] == "muse-wonder-1.0"


# ---- call shape, ledger (REQ-API-03) ---------------------------------------------

def test_req_api_03_ledger_row_has_tokens_images_and_cost(tmp_path, photo):
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path, transport)
    assert client.reason("p", [photo, photo]) == {"ok": 1}
    url, headers, body, _ = transport.calls[0]
    assert url == "https://api.meta.ai/v1/chat/completions"
    assert body["reasoning_effort"] == "high" and body["max_tokens"] == 32000
    (row,) = _rows(client)
    assert row["model"] == "muse-spark-1.3"
    assert (row["input_tokens"], row["cached_tokens"], row["output_tokens"], row["images"]) == (
        "1000", "400", "2000", "2")
    expected = (600 * 1.25 + 400 * 0.15 + 2000 * 4.25) / 1e6
    assert float(row["cost_usd"]) == pytest.approx(expected, abs=1e-6)
    assert client.ledger_total_usd() == pytest.approx(expected, abs=1e-6)


def test_req_api_03_missing_usage_bills_the_estimate_not_zero(tmp_path):
    response = _answer()
    del response["usage"]
    client, _ = _client(tmp_path, FakeTransport(response))
    client.reason("p")
    assert client.ledger_total_usd() > 0


def test_req_api_03_sam_price_is_per_image(tmp_path):
    client, _ = _client(tmp_path, FakeTransport({}))
    client._paid(client.config.models.sam, "/x", {}, Usage(images=1))
    (row,) = _rows(client)
    assert row["model"] == "sam-3.1" and float(row["cost_usd"]) == pytest.approx(0.0025)


def test_c07_api_key_is_not_written_to_cache_or_ledger(tmp_path):
    """C-07: the key never reaches a cache file or the ledger."""
    client, _ = _client(tmp_path, FakeTransport(_answer()))
    client.reason("p")
    assert all(KEY not in p.read_text() for p in tmp_path.iterdir() if p.is_file())


def test_req_rsn_06_truncated_answer_is_an_error_and_not_cached_as_an_answer(tmp_path):
    """Changed for the task "cache rejected answers" (REQ-API-02): the paid, cut-off answer is kept
    under a ``.rejected.json`` marker only, so a rerun fails the same way without paying again.
    It is never stored where reason() looks for a good answer."""
    transport = FakeTransport(_answer(finish="length"))
    client, _ = _client(tmp_path, transport)
    with pytest.raises(ApiError, match="max_tokens"):
        client.reason("p")
    assert [f.name for f in tmp_path.glob("*.json") if not f.name.endswith(".rejected.json")] == []
    calls = len(transport.calls)
    with pytest.raises(ApiError, match="max_tokens"):
        client.reason("p")
    assert len(transport.calls) == calls


# ---- budget guard (REQ-API-04) ---------------------------------------------------

def test_req_api_04_low_limit_blocks_call_before_network(tmp_path):
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path, transport, config=_config(budget_limit_usd=0.01))
    with pytest.raises(BudgetExceeded):
        client.reason("p")  # worst case 32,000 output tokens is about $0.136
    assert transport.calls == []


def test_req_api_04_ledger_total_counts_toward_limit(tmp_path):
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path, transport, config=_config(budget_limit_usd=0.20))
    client.reason("first")  # logs about $0.0094
    client.reason("second")  # still fits: 0.0094 + 0.136 < 0.20
    spent = client.ledger_total_usd()
    tight, _ = _client(tmp_path, transport, config=_config(budget_limit_usd=spent + 0.05))
    n = len(transport.calls)
    with pytest.raises(BudgetExceeded):
        tight.reason("third")
    assert len(transport.calls) == n


def test_req_api_04_cache_hit_is_free_even_over_budget(tmp_path):
    client, _ = _client(tmp_path, FakeTransport(_answer()))
    client.reason("p")
    over, _ = _client(tmp_path, FakeTransport(), config=_config(budget_limit_usd=0.001))
    assert over.reason("p") == {"ok": 1}


# ---- retry (REQ-API-06) ----------------------------------------------------------

def test_req_api_06_retries_429_then_succeeds(tmp_path):
    transport = FakeTransport(HttpError(429), _answer())
    client, sleeps = _client(tmp_path, transport)
    assert client.reason("p") == {"ok": 1}
    assert len(transport.calls) == 2
    assert sleeps == [2.0]
    assert len(_rows(client)) == 1  # the failed try is not logged as a paid call


def test_req_api_06_three_5xx_then_failure(tmp_path):
    transport = FakeTransport(HttpError(503))
    client, sleeps = _client(tmp_path, transport)
    with pytest.raises(ApiError, match="3 tries"):
        client.reason("p")
    assert len(transport.calls) == 3
    assert sleeps == [2.0, 4.0]  # backoff doubles; no sleep after the last try
    assert not client.ledger_path.exists()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_req_api_06_other_4xx_is_rejected_not_retried(tmp_path, status):
    """REQ-API-06, REQ-RSN-03: a refused request is ApiRequestRejected, so callers stop instead of falling back."""
    transport = FakeTransport(HttpError(status, "bad effort"))
    client, sleeps = _client(tmp_path, transport)
    with pytest.raises(ApiRequestRejected, match=str(status)):
        client.reason("p")
    assert len(transport.calls) == 1 and sleeps == []
    assert not client.ledger_path.exists()


def test_req_api_06_429_and_5xx_stay_plain_api_errors(tmp_path):
    client, _ = _client(tmp_path, FakeTransport(HttpError(429)))
    with pytest.raises(ApiError) as exc:
        client.reason("p")
    assert not isinstance(exc.value, ApiRequestRejected)


# ---- timeout (REQ-API-07) --------------------------------------------------------

def test_req_api_07_slow_call_times_out_with_configured_timeout(tmp_path):
    def slow(url, headers, body, timeout_s):
        latency_s = 5.0  # simulated server time, no real sleep
        if latency_s > timeout_s:
            raise TimeoutError("timed out")
        return _answer()

    client, _ = _client(tmp_path, slow, config=_config(timeout_s=1.5))
    with pytest.raises(ApiTimeout, match="1.5"):
        client.reason("p")
    assert [r["estimated"] for r in _rows(client)] == ["1"]  # billed at the worst case, see below


def test_req_api_07_every_call_gets_the_timeout(tmp_path):
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path, transport)
    client.reason("p")
    assert transport.calls[0][3] == client.config.api.timeout_s > 0


# ---- cache keys (REQ-API-02) -----------------------------------------------------

def test_req_api_02_mock_and_real_do_not_share_cache(tmp_path, photo):
    mock_client, _ = _client(tmp_path, FakeTransport(), mock=True, key="")
    assert mock_client.reason("p", [photo], mock=lambda: {"from": "mock"}) == {"from": "mock"}
    transport = FakeTransport(_answer({"from": "real"}))
    real_client, _ = _client(tmp_path, transport)
    assert real_client.reason("p", [photo]) == {"from": "real"}
    assert len(transport.calls) == 1
    # And the other way round: a real answer is never replayed as a mock one.
    again, _ = _client(tmp_path, FakeTransport(), mock=True, key="")
    assert again.reason("p", [photo], mock=lambda: {"from": "mock2"}) == {"from": "mock"}


def test_req_api_02_second_identical_call_makes_no_request(tmp_path, photo):
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path, transport)
    client.reason("p", [photo])
    client.reason("p", [photo])
    assert len(transport.calls) == 1


@pytest.mark.parametrize("change", ["prompt", "image", "effort", "max_tokens", "model", "schema"])
def test_req_api_02_changed_parameter_misses_cache(tmp_path, photo, change):
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path, transport)
    client.reason("p", [photo], schema={"type": "object"})
    cfg, prompt, schema = client.config, "p", {"type": "object"}
    if change == "prompt":
        prompt = "p2"
    elif change == "image":
        cv2.imwrite(str(photo), np.full((40, 40, 3), 90, np.uint8))  # other pixels (real keys hash pixels)
    elif change == "schema":
        schema = {"type": "array"}
    elif change == "effort":
        cfg = cfg.model_copy(update={"reasoning": cfg.reasoning.model_copy(update={"effort": "low"})})
    elif change == "max_tokens":
        cfg = cfg.model_copy(update={"reasoning": cfg.reasoning.model_copy(update={"max_tokens": 16000})})
    elif change == "model":
        cfg = cfg.model_copy(update={"reasoning": cfg.reasoning.model_copy(update={"model": "wonder"})})
    other, _ = _client(tmp_path, transport, config=cfg)
    other.reason(prompt, [photo], schema=schema)
    assert len(transport.calls) == 2


def test_req_api_02_sam_mock_mask_is_not_served_to_real_mode(tmp_path, photo):
    img = np.full((40, 40, 3), 255, np.uint8)
    img[10:30, 10:30] = 0
    cv2.imwrite(str(photo), img)
    mock_client, _ = _client(tmp_path, FakeTransport(), mock=True, key="")
    mock_client.segment(photo, "t-shirt")
    mock_client.segment(photo, "t-shirt")
    assert mock_client.calls == 1
    real_client, _ = _client(tmp_path, FakeTransport())
    asked = []  # the real SAM call is reached: a miss, not the cached mock mask
    real_client._call_sam = lambda image, prompt: asked.append(prompt) or [SamResult(mask=_blob(2), score=0.9)]
    assert not np.array_equal(real_client.segment(photo, "t-shirt"), mock_client.segment(photo, "t-shirt"))
    assert asked == ["t-shirt"]


# ---- offline mode (REQ-API-08) ---------------------------------------------------

def test_req_api_08_offline_miss_raises_and_calls_nothing(tmp_path, photo):
    transport = FakeTransport(_answer())
    mock_called = []
    for mock in (False, True):
        client, _ = _client(tmp_path, transport, offline=True, mock=mock)
        with pytest.raises(CacheMiss):
            client.reason("p", [photo], mock=lambda: mock_called.append(1) or {})
        with pytest.raises(CacheMiss):
            client.segment(photo, "t-shirt")
    assert transport.calls == [] and mock_called == []


def test_req_api_08_offline_replays_real_answer_without_key(tmp_path, photo):
    online, _ = _client(tmp_path, FakeTransport(_answer({"v": 7})))
    online.reason("p", [photo])
    transport = FakeTransport()
    offline, _ = _client(tmp_path, transport, offline=True, key="")
    assert offline.reason("p", [photo]) == {"v": 7}
    assert transport.calls == []


def test_req_api_08_settings_load_reads_offline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SEWTRACE_OFFLINE", "1")
    monkeypatch.setenv("SEWTRACE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("SEWTRACE_MOCK", raising=False)
    s = Settings.load()
    assert s.offline and not s.mock  # offline replays real answers unless mock is asked for


def test_no_key_without_offline_means_mock(monkeypatch, tmp_path):
    """C-08: no key without offline means mock."""
    monkeypatch.delenv("SEWTRACE_OFFLINE", raising=False)
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SEWTRACE_CACHE_DIR", str(tmp_path))
    s = Settings.load()
    assert s.mock and not s.offline


def test_real_call_without_key_fails_loudly(tmp_path):
    """C-07: real call without key fails loudly."""
    client, _ = _client(tmp_path, FakeTransport(_answer()), key="")
    with pytest.raises(ApiError, match="META_API_KEY"):
        client.reason("p")


# ---- one HTTP client (REQ-API-01) ------------------------------------------------

BANNED = ("httpx", "requests", "openai", "urllib.request", "http.client", "aiohttp", "urllib3")


def _imports(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def test_req_api_01_only_the_client_imports_an_http_client():
    package = Path(__file__).resolve().parent.parent / "sewtrace"
    offenders = []
    for path in package.rglob("*.py"):
        if path == package / "api" / "client.py":
            continue
        for name in _imports(ast.parse(path.read_text())):
            if any(name == b or name.startswith(b + ".") for b in BANNED):
                offenders.append(f"{path.relative_to(package)} imports {name}")
    assert offenders == []


# ---- SAM scores (REQ-SEG-03, REQ-SEG-04) ---------------------------------------------

def _sam_client(tmp_path, results, **kwargs):
    """A real-mode client whose SAM call returns ``results`` (the response shape is unconfirmed)."""
    client, _ = _client(tmp_path, FakeTransport(), **kwargs)
    asked = []
    client._call_sam = lambda image, prompt: asked.append((image, prompt)) or results
    return client, asked


def _blob(x0: int) -> np.ndarray:
    mask = np.zeros((40, 40), bool)
    mask[10:30, x0 : x0 + 10] = True
    return mask


def test_req_seg_03_highest_score_mask_wins(tmp_path, photo):
    """REQ-SEG-03: SAM returns two masks; the one with the higher score is used."""
    low, high = SamResult(mask=_blob(2), score=0.6), SamResult(mask=_blob(20), score=0.9)
    for order in ((low, high), (high, low)):
        client, _ = _sam_client(tmp_path / f"c{id(order)}", list(order))
        assert np.array_equal(client.segment(photo, "t-shirt"), high.mask)


def test_req_seg_04_low_score_stops_and_names_photo_and_prompt(tmp_path, photo):
    """REQ-SEG-04: best score below sam.min_score raises, naming the photo and the prompt."""
    min_score = load_config().sam.min_score
    client, asked = _sam_client(tmp_path, [SamResult(mask=_blob(2), score=min_score - 0.01),
                                           SamResult(mask=_blob(20), score=0.1)])
    with pytest.raises(SegmentationFailed, match=r"p\.jpg.*'sleeve'.*below"):
        client.segment(photo, "sleeve")
    # The score is cached with the mask: asking again fails again without a second paid call.
    with pytest.raises(SegmentationFailed):
        client.segment(photo, "sleeve")
    assert len(asked) == 1


def test_req_seg_04_score_at_threshold_passes_and_empty_result_fails(tmp_path, photo):
    """REQ-SEG-04: a score equal to the threshold is accepted; no mask at all is an error."""
    min_score = load_config().sam.min_score
    client, _ = _sam_client(tmp_path / "ok", [SamResult(mask=_blob(2), score=min_score)])
    assert client.segment(photo, "t-shirt").any()
    empty, _ = _sam_client(tmp_path / "empty", [])
    with pytest.raises(SegmentationFailed, match="no mask"):
        empty.segment(photo, "t-shirt")


def test_req_seg_03_mock_masks_score_one(tmp_path, photo):
    """REQ-SEG-03: the mock returns one mask with score 1.0 (sidecar or threshold)."""
    from sewtrace.api.client import _mock_segment

    img = np.full((40, 40, 3), 255, np.uint8)
    img[10:30, 10:30] = 0
    cv2.imwrite(str(photo), img)
    results = _mock_segment(photo, "t-shirt", "t-shirt")
    assert len(results) == 1 and results[0].score == 1.0


# ---- ledger outside the cache (REQ-API-04, C-01) ---------------------------------

def test_req_api_04_default_ledger_is_outside_cache_and_cache_dir_env(monkeypatch, tmp_path):
    """REQ-API-04: the ledger path comes from config, under the repo root, and ignores SEWTRACE_CACHE_DIR."""
    from sewtrace.config import ROOT
    monkeypatch.setenv("SEWTRACE_CACHE_DIR", str(tmp_path / "other_cache"))
    monkeypatch.delenv("META_API_KEY", raising=False)
    s = Settings.load()
    assert s.ledger_path == ROOT / load_config().api.ledger_path == ROOT / "data" / "ledger.csv"
    assert s.ledger_path.parent != s.cache_dir
    assert MetaClient(s).ledger_path == s.ledger_path


def test_req_api_04_new_cache_dir_keeps_the_total(tmp_path):
    """REQ-API-04: a different cache folder with the same ledger still counts the spend."""
    first, _ = _client(tmp_path / "a", FakeTransport(_answer()))
    first.reason("p")
    spent = first.ledger_total_usd()
    settings = Settings(api_key=KEY, mock=False, cache_dir=tmp_path / "b", ledger_path=first.ledger_path,
                        config=load_config())
    assert MetaClient(settings).ledger_total_usd() == pytest.approx(spent) and spent > 0


def test_req_api_04_missing_ledger_with_real_cache_refuses_paid_call(tmp_path):
    """REQ-API-04, C-01: no ledger file but a real cached answer means the ledger was lost."""
    first, _ = _client(tmp_path, FakeTransport(_answer()))
    first.reason("p")
    first.ledger_path.unlink()
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path, transport)
    with pytest.raises(ApiError, match="ledger .* is missing"):
        client.reason("other prompt")
    assert transport.calls == []
    assert client.reason("p") == {"ok": 1}  # a cached answer costs nothing, so it still works


def test_req_api_04_missing_ledger_with_empty_or_mock_cache_is_allowed(tmp_path):
    """REQ-API-04: a first real call (or one after mock runs only) creates the ledger."""
    mock, _ = _client(tmp_path, FakeTransport(), mock=True)
    mock.reason("p", mock=lambda: {"m": 1})
    client, _ = _client(tmp_path, FakeTransport(_answer()))
    client.reason("q")
    assert client.ledger_path.exists()


def test_req_api_04_missing_ledger_with_real_answers_in_repo_cache_refuses_paid_call(tmp_path, monkeypatch):
    """REQ-API-04, C-01: a different SEWTRACE_CACHE_DIR must not hide real answers in ROOT/cache."""
    import sewtrace.api.client as client_module
    import sewtrace.config as config_module

    repo = tmp_path / "repo"
    (repo / "cache").mkdir(parents=True)
    (repo / "cache" / "real-abc.json").write_text("{}")
    monkeypatch.setattr(config_module, "ROOT", repo)
    monkeypatch.setattr(client_module, "ROOT", repo)
    transport = FakeTransport(_answer())
    settings = Settings(api_key=KEY, mock=False, cache_dir=tmp_path / "other_cache")  # default ledger path
    client = MetaClient(settings, transport=transport, sleep=lambda s: None)
    assert client.ledger_path == repo / "data" / "ledger.csv"
    with pytest.raises(ApiError, match="ledger .* is missing"):
        client.reason("p")
    assert transport.calls == []
    # A ledger path chosen by a test is not tied to the repo cache.
    own = Settings(api_key=KEY, mock=False, cache_dir=tmp_path / "c2", ledger_path=tmp_path / "l" / "ledger.csv")
    assert MetaClient(own, transport=FakeTransport(_answer()), sleep=lambda s: None).reason("p") == {"ok": 1}


# ---- timeout bills the worst case (REQ-API-03, REQ-API-04, REQ-API-07) -----------

def test_req_api_07_timeout_logs_worst_case_estimate(tmp_path):
    """REQ-API-07, REQ-API-04: the server probably bills a timed out call; log the estimate."""
    client, _ = _client(tmp_path, FakeTransport(TimeoutError("slow")))
    with pytest.raises(ApiTimeout):
        client.reason("p")
    [row] = _rows(client)
    cfg = client.config
    assert int(row["output_tokens"]) == cfg.reasoning.max_tokens and row["estimated"] == "1"
    assert float(row["cost_usd"]) == pytest.approx(
        cfg.reasoning.max_tokens * cfg.reasoning_model.output_per_m_usd / 1e6, rel=0.01)


def test_req_api_07_lost_connection_logs_estimate_but_dns_failure_does_not(tmp_path):
    """REQ-API-04: a reset after sending is billed at the estimate; a call never sent is not."""
    from sewtrace.api.client import ApiTransportError
    client, _ = _client(tmp_path, FakeTransport(ApiTransportError("reset")))
    with pytest.raises(ApiTransportError):
        client.reason("p")
    assert _rows(client)[0]["estimated"] == "1"
    other, _ = _client(tmp_path / "x", FakeTransport(ApiNotSent("dns")))
    with pytest.raises(ApiNotSent):
        other.reason("p")
    assert not other.ledger_path.exists()


def test_req_api_07_real_responses_are_not_flagged_estimated(tmp_path):
    client, _ = _client(tmp_path, FakeTransport(_answer()))
    client.reason("p")
    assert _rows(client)[0]["estimated"] == "0"


def test_req_api_07_timeout_is_600_s_and_effort_unchanged():
    cfg = load_config()
    assert cfg.api.timeout_s == 600.0
    assert cfg.reasoning.effort == "high" and cfg.reasoning.max_tokens == 32000


# ---- contributor endpoint: synthetic photos only (SPEC 4.3 item 5) ---------------

def _contributor_config():
    cfg = load_config()
    return cfg.model_copy(update={"reasoning": cfg.reasoning.model_copy(update={"model": "spark_contributor"})})


def test_spec_4_3_contributor_refuses_real_photo(tmp_path, photo):
    """SPEC 4.3 item 5: a photo with no synthetic truth.json never goes to the contributor model."""
    from sewtrace.api.client import PolicyViolation
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path, transport, config=_contributor_config())
    with pytest.raises(PolicyViolation, match="not a synthetic photo"):
        client.reason("p", [photo])
    assert transport.calls == []
    (photo.parent / "truth.json").write_text(json.dumps({"measurements": {}}))  # no synthetic flag
    with pytest.raises(PolicyViolation):
        client.reason("p", [photo])


def test_spec_4_3_contributor_accepts_synthetic_photos_and_text_only(tmp_path, photo):
    from sewtrace.mock import synth
    tee = tmp_path / "tee"
    synth.make_tee(synth.TEES[0], tee, seed=0)
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path / "c", transport, config=_contributor_config())
    assert client.reason("p", [tee / "front.jpg"]) == {"ok": 1}
    assert transport.calls[0][2]["model"] == "muse-spark-1.3-contributor"
    assert client.reason("text only") == {"ok": 1}  # no images: nothing to protect


# ---- Muse gets re-encoded pixels, not the upload (privacy, REQ-API-02) -----------------------

def _jpeg_with_exif(path: Path, color: int) -> bytes:
    """A JPEG with an APP1 (EXIF) segment that holds a fake GPS tag, as a phone writes it."""
    ok, buf = cv2.imencode(".jpg", np.full((40, 40, 3), color, np.uint8))
    payload = b"Exif\0\0" + b"GPSLatitude=12.34;Make=SecretPhone"
    data = bytes(buf)[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + bytes(buf)[2:]
    path.write_bytes(data)
    return data


def test_req_api_02_muse_images_are_reencoded_without_exif(tmp_path):
    """The bytes sent to Muse hold no APP1/EXIF segment, like the SAM upload; the cache key follows the pixels."""
    import base64
    photo = tmp_path / "gps.jpg"
    original = _jpeg_with_exif(photo, 200)
    assert b"GPSLatitude" in original
    transport = FakeTransport(_answer())
    client, _ = _client(tmp_path, transport)
    client.reason("p", [photo])
    content = transport.calls[0][2]["messages"][-1]["content"]
    (url,) = [c["image_url"]["url"] for c in content if c.get("type") == "image_url"]
    assert url.startswith("data:image/jpeg;base64,")
    sent = base64.b64decode(url.split(",", 1)[1])
    assert b"GPSLatitude" not in sent and b"SecretPhone" not in sent and b"Exif" not in sent
    assert b"\xff\xe1" not in sent[:200]  # no APP1 marker among the first segments
    assert cv2.imdecode(np.frombuffer(sent, np.uint8), cv2.IMREAD_COLOR).shape == (40, 40, 3)
    # Same pixels in a file without EXIF: same key, so no second paid call.
    plain = tmp_path / "plain.jpg"
    cv2.imwrite(str(plain), cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR))
    client.reason("p", [plain])
    assert len(transport.calls) == 1
    # Different pixels: a new key.
    other = tmp_path / "other.jpg"
    _jpeg_with_exif(other, 90)
    client.reason("p", [other])
    assert len(transport.calls) == 2
