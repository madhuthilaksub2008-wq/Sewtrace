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
    ApiTimeout,
    BudgetExceeded,
    CacheMiss,
    HttpError,
    MetaClient,
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
                        config=config or load_config())
    return MetaClient(settings, transport=transport, sleep=sleeps.append), sleeps


@pytest.fixture
def photo(tmp_path) -> Path:
    path = tmp_path / "p.jpg"
    path.write_bytes(b"not really a jpeg")
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
    client, _ = _client(tmp_path, FakeTransport(_answer()))
    client.reason("p")
    assert all(KEY not in p.read_text() for p in tmp_path.iterdir() if p.is_file())


def test_req_rsn_06_truncated_answer_is_an_error_and_not_cached(tmp_path):
    client, _ = _client(tmp_path, FakeTransport(_answer(finish="length")))
    with pytest.raises(ApiError, match="max_tokens"):
        client.reason("p")
    assert not list(tmp_path.glob("*.json"))


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


def test_req_api_06_other_4xx_is_not_retried(tmp_path):
    transport = FakeTransport(HttpError(400, "bad effort"))
    client, sleeps = _client(tmp_path, transport)
    with pytest.raises(ApiError, match="400"):
        client.reason("p")
    assert len(transport.calls) == 1 and sleeps == []


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
    assert not client.ledger_path.exists()


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
        photo.write_bytes(b"other bytes")
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
    with pytest.raises(NotImplementedError):  # a miss, not the cached mock mask
        real_client.segment(photo, "t-shirt")


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
    monkeypatch.delenv("SEWTRACE_OFFLINE", raising=False)
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SEWTRACE_CACHE_DIR", str(tmp_path))
    s = Settings.load()
    assert s.mock and not s.offline


def test_real_call_without_key_fails_loudly(tmp_path):
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
