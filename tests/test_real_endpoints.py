"""Real endpoint wiring against Meta's cookbook: request bodies, SSE, SAM decoding, Day 1 command.

No test touches the network, needs a key or needs the ``meta_sam_parser`` package. A fake parser
module with the cookbook's attribute names stands in for it.
"""

import json
import sys
import tomllib
import types
import urllib.request
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from sewtrace import cli
from sewtrace.api import client as client_mod
from sewtrace.api.client import (
    ApiError,
    ApiTimeout,
    MetaClient,
    SamResult,
    SegmentationFailed,
    parse_sse,
    reasoning_request_body,
    sam_request_body,
    sam_results_from,
    select_best,
    strict_json_schema,
    urllib_stream_transport,
    with_output_text_done,
)
from sewtrace.config import Settings, load_config
from sewtrace.day1 import run_day1
from sewtrace.reason.panels import PanelInference
from sewtrace.reason.seams import ADJUST_SCHEMA

KEY = "test-key-not-real"
H, W = 60, 80


# ---- fakes -------------------------------------------------------------------------

def _box(oid, left, top, right, bottom):
    return NS(kind="box", object_id=oid, left=left, top=top, right=right, bottom=bottom)


def _mask(oid, w, h, fill=None, **extra):
    return NS(kind="mask", object_id=oid, mask=NS(width=w, height=h, encoding="lossless", data=fill), **extra)


def _result(records, status="completed", diagnostics=()):
    return NS(outcome=NS(status=status), diagnostics=list(diagnostics), records=list(records))


def _decode(mask):
    """Fake of decode_mask_to_raster: a flat row-major list. ``data`` None means all foreground."""
    return mask.data if mask.data is not None else [1] * (mask.width * mask.height)


def _fake_parser(result, seen=None):
    """A module with the cookbook's names. It consumes the events like the real parser."""
    mod = types.ModuleType("meta_sam_parser")

    class Parsed:
        def __init__(self, events):
            self.events = events

        async def final_result(self):
            async for event in self.events:
                if seen is not None:
                    seen.append(event)
            return result

    mod.parse_responses_stream = lambda events, fmt: Parsed(events)
    mod.image_segmentation_format = lambda: "image-format"
    mod.decode_mask_to_raster = _decode
    return mod


EVENTS = [
    {"type": "response.created"},
    {"type": "response.content_part.done", "item_id": "i", "output_index": 0, "content_index": 0,
     "part": {"type": "output_text", "text": "grammar"}},
    {"type": "response.completed", "response": {"usage": {"input_tokens": 1234, "output_tokens": 7}}},
]


class StreamTransport:
    def __init__(self, events=EVENTS):
        self.events, self.calls = events, []

    def __call__(self, url, headers, body, timeout_s):
        self.calls.append((url, headers, body, timeout_s))
        return {"events": self.events}


class ChatTransport:
    def __init__(self, content=None):
        self.content, self.calls = content or {"view": "front", "neck_visible": True}, []

    def __call__(self, url, headers, body, timeout_s):
        self.calls.append((url, headers, body, timeout_s))
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(self.content)}}],
                "usage": {"prompt_tokens": 1700, "completion_tokens": 900,
                          "prompt_tokens_details": {"cached_tokens": 0}}}


@pytest.fixture
def photo(tmp_path) -> Path:
    path = tmp_path / "p.jpg"
    cv2.imwrite(str(path), np.full((H, W, 3), 200, np.uint8))
    return path


def _client(tmp_path, *, stream=None, chat=None, config=None):
    settings = Settings(api_key=KEY, mock=False, cache_dir=tmp_path, offline=False,
                        ledger_path=tmp_path / "ledger" / "ledger.csv", config=config or load_config())
    return MetaClient(settings, transport=chat or ChatTransport(), stream_transport=stream or StreamTransport(),
                      sleep=lambda s: None)


# ---- SAM request (REQ-SEG-01) -------------------------------------------------------

def test_req_seg_01_request_body_is_the_cookbook_responses_shape():
    body = sam_request_body("sam-3.1", "  t-shirt ", "data:image/jpeg;base64,AAAA")
    assert body == {
        "model": "sam-3.1",
        "stream": True,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "t-shirt"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA"},
        ]}],
    }


def test_req_seg_01_text_prompt_only_no_box_or_point(tmp_path, photo, monkeypatch):
    stream = StreamTransport()
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    _client(tmp_path, stream=stream).segment(photo, "t-shirt")
    url, headers, body, timeout = stream.calls[0]
    text = json.dumps(body).lower()
    assert url.endswith("/responses") and not url.endswith("/chat/completions")
    for banned in ("box", "point", "bbox", "mask_input"):
        assert banned not in text
    assert body["model"] == load_config().models.sam.id and timeout == load_config().api.timeout_s
    assert body["input"][0]["content"][0] == {"type": "input_text", "text": "t-shirt"}
    assert body["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")


# ---- SSE (REQ-API-07) -----------------------------------------------------------------

def test_req_api_07_parse_sse_keeps_data_lines_only():
    lines = ["event: response.created", 'data: {"type": "response.created"}', "", ": keep-alive",
             'data: {"type": "x", "n": 1}', "data: [DONE]"]
    assert list(parse_sse(lines)) == [{"type": "response.created"}, {"type": "x", "n": 1}]
    with pytest.raises(ApiError, match="not JSON"):
        list(parse_sse(["data: {oops"]))


class _FakeResponse:
    def __init__(self, lines, on_line=None):
        self.lines, self.on_line = lines, on_line

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for line in self.lines:
            if self.on_line:
                self.on_line()
            yield line


def test_req_api_07_stream_transport_reads_events_and_asks_for_sse(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["accept"], seen["timeout"] = request.get_header("Accept"), timeout
        seen["body"] = json.loads(request.data)
        return _FakeResponse([b'data: {"type": "a"}\r\n', b"\r\n", b'data: {"type": "b"}\n'])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = urllib_stream_transport("https://x/responses", {"Authorization": "Bearer k"}, {"stream": True}, 12.0)
    assert out == {"events": [{"type": "a"}, {"type": "b"}]}
    assert seen == {"accept": "text/event-stream", "timeout": 12.0, "body": {"stream": True}}


def test_req_api_07_stream_total_time_is_limited(monkeypatch):
    """A stream that keeps sending must still stop at the timeout, not only on a silent socket."""
    now = [0.0]
    monkeypatch.setattr(urllib.request, "urlopen", lambda r, timeout: _FakeResponse(
        [b'data: {"type": "a"}\n'] * 5, on_line=lambda: now.__setitem__(0, now[0] + 4.0)))
    with pytest.raises(TimeoutError):
        urllib_stream_transport("https://x", {}, {}, 10.0, clock=lambda: now[0])


def test_req_api_07_stream_timeout_logs_worst_case_and_raises(tmp_path, photo, monkeypatch):
    class Slow:
        def __call__(self, *a):
            raise TimeoutError("slow")

    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([])))
    client = _client(tmp_path, stream=Slow())
    with pytest.raises(ApiTimeout):
        client.segment(photo, "t-shirt")
    assert client.ledger_total_usd() == pytest.approx(load_config().models.sam.price_per_image_usd)


def test_req_seg_01_output_text_done_is_synthesized_once():
    out = list(with_output_text_done(EVENTS))
    types_ = [e["type"] for e in out]
    assert types_ == ["response.created", "response.content_part.done", "response.output_text.done",
                      "response.completed"]
    assert out[2] == {"type": "response.output_text.done", "item_id": "i", "output_index": 0,
                      "content_index": 0, "text": "grammar"}
    # The API may send it one day: do not double it.
    with_real = EVENTS[:2] + [{"type": "response.output_text.done", "item_id": "i", "output_index": 0,
                               "content_index": 0, "text": "grammar"}]
    assert [e["type"] for e in with_output_text_done(with_real)].count("response.output_text.done") == 1
    # A second lane gets its own event.
    other = {**EVENTS[1], "content_index": 1}
    assert [e["type"] for e in with_output_text_done([EVENTS[1], other])].count("response.output_text.done") == 2


# ---- SAM decoding (REQ-SEG-01, REQ-SEG-03) ------------------------------------------------

def test_req_seg_01_box_local_mask_is_placed_at_the_box_top_left():
    """UNCONFIRMED layout (Day 1): the mask is as large as its box and sits at the box top-left."""
    data = [1, 0, 0, 1, 1, 1]  # 3 wide, 2 high: [[1,0,0],[1,1,1]]
    result = _result([_box("0", 10, 5, 13, 7), _mask("0", 3, 2, data)])
    (res,) = sam_results_from(result, _decode, (H, W), "p.jpg", "t-shirt")
    expected = np.zeros((H, W), bool)
    expected[5, 10] = True
    expected[6, 10:13] = True
    assert res.mask.shape == (H, W) and np.array_equal(res.mask, expected)
    assert res.score is None


def test_req_seg_01_cookbook_sizes_box_width_is_right_minus_left():
    result = _result([_box("0", 751, 290, 925, 493), _mask("0", 174, 203)])
    (res,) = sam_results_from(result, _decode, (600, 1000), "ducks.jpg", "duck")
    assert res.mask.sum() == 174 * 203
    ys, xs = np.nonzero(res.mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (751, 924, 290, 492)


def test_req_seg_01_full_image_mask_is_used_as_is_and_box_past_edge_is_clipped():
    full = _result([_mask("0", W, H)])
    assert sam_results_from(full, _decode, (H, W), "p", "x")[0].mask.all()
    clipped = _result([_box("0", W - 2, H - 1, W + 3, H + 2), _mask("0", 5, 3)])
    (res,) = sam_results_from(clipped, _decode, (H, W), "p", "x")
    assert res.mask.sum() == 2 and res.mask[H - 1, W - 2:].all()


def test_req_seg_01_unexpected_mask_size_is_an_error_not_a_guess():
    with pytest.raises(ApiError, match="neither the image"):
        sam_results_from(_result([_box("0", 0, 0, 10, 10), _mask("0", 7, 7)]), _decode, (H, W), "p.jpg", "t-shirt")
    with pytest.raises(ApiError, match="decodes to"):
        sam_results_from(_result([_mask("0", W, H, [1, 1])]), _decode, (H, W), "p.jpg", "t-shirt")


def test_req_seg_03_score_attribute_is_used_when_the_record_has_one():
    result = _result([_box("0", 0, 0, 5, 5), _mask("0", 5, 5, score=0.4),
                      _box("1", 20, 20, 30, 30), _mask("1", 10, 10, score=0.9)])
    results = sam_results_from(result, _decode, (H, W), "p", "x")
    assert [r.score for r in results] == [0.4, 0.9]
    assert select_best(results).score == 0.9  # the higher score wins although its mask is bigger anyway
    small_but_sure = [SamResult(mask=np.ones((2, 2), bool), score=0.9), SamResult(mask=np.ones((9, 9), bool), score=0.5)]
    assert select_best(small_but_sure).score == 0.9


def test_req_seg_03_no_score_stand_in_picks_largest_mask_and_does_not_fake_a_score():
    small, big = np.zeros((H, W), bool), np.zeros((H, W), bool)
    small[0:3, 0:3] = True
    big[10:30, 10:30] = True
    best = select_best([SamResult(mask=small), SamResult(mask=big)])
    assert np.array_equal(best.mask, big) and best.score is None
    # One missing score is enough to distrust the others.
    mixed = select_best([SamResult(mask=small, score=0.99), SamResult(mask=big)])
    assert np.array_equal(mixed.mask, big) and mixed.score is None


def test_req_seg_04_missing_score_skips_threshold_with_a_warning(tmp_path, photo, monkeypatch, caplog):
    records = [_box("0", 0, 0, 5, 5), _mask("0", 5, 5), _box("1", 10, 10, 30, 30), _mask("1", 20, 20)]
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result(records)))
    client = _client(tmp_path)
    with caplog.at_level("WARNING", logger="sewtrace.api.client"):
        mask = client.segment(photo, "t-shirt")
    assert mask.sum() == 400 and "no score" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="sewtrace.api.client"):  # cache hit: the warning repeats
        assert np.array_equal(client.segment(photo, "t-shirt"), mask)
    assert "no score" in caplog.text
    assert len(client._stream_transport.calls) == 1


def test_req_seg_04_score_below_threshold_still_stops_when_the_parser_gives_one(tmp_path, photo, monkeypatch):
    records = [_box("0", 0, 0, 5, 5), _mask("0", 5, 5, score=0.1)]
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result(records)))
    with pytest.raises(SegmentationFailed, match="below"):
        _client(tmp_path).segment(photo, "t-shirt")


def test_req_seg_04_empty_valid_result_is_no_mask(tmp_path, photo, monkeypatch):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([])))
    with pytest.raises(SegmentationFailed, match="no mask"):
        _client(tmp_path).segment(photo, "t-shirt")


@pytest.mark.parametrize("result, match", [
    (_result([], status="incomplete"), "incomplete"),
    (_result([], diagnostics=[NS(severity="error", line=3, message="bad record")]), "line 3: bad record"),
])
def test_req_seg_01_incomplete_or_rejected_output_is_an_error(tmp_path, photo, monkeypatch, result, match):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(result))
    with pytest.raises(ApiError, match=match):
        _client(tmp_path).segment(photo, "t-shirt")


def test_req_seg_01_parser_gets_events_with_output_text_done_and_usage_reaches_the_ledger(tmp_path, photo, monkeypatch):
    seen: list[dict] = []
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)]), seen))
    client = _client(tmp_path)
    client.segment(photo, "t-shirt")
    assert "response.output_text.done" in [e["type"] for e in seen]
    (row,) = [r.split(",") for r in client.ledger_path.read_text().splitlines()[1:]]
    assert row[1] == "sam-3.1" and row[2] == "1234" and row[5] == "1"  # model, input tokens, images
    assert float(row[6]) == pytest.approx(load_config().models.sam.price_per_image_usd)


def test_req_seg_01_missing_parser_names_the_fix_and_makes_no_call(tmp_path, photo, monkeypatch):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", None)  # import raises ImportError
    client = _client(tmp_path)
    with pytest.raises(ApiError, match=r"pip install -e '\.\[real\]'"):
        client.segment(photo, "t-shirt")
    assert client._stream_transport.calls == [] and not client.ledger_path.exists()


def test_req_api_02_raw_events_are_cached_so_a_decoding_error_is_not_paid_twice(tmp_path, photo, monkeypatch):
    bad = _result([_box("0", 0, 0, 10, 10), _mask("0", 7, 7)])
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(bad))
    client = _client(tmp_path)
    with pytest.raises(ApiError, match="neither the image"):
        client.segment(photo, "t-shirt")
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    assert client.segment(photo, "t-shirt").all()  # decoded from the stored events
    assert len(client._stream_transport.calls) == 1
    assert len(client.ledger_path.read_text().splitlines()) == 2  # header + one paid call


def test_real_sam_module_is_not_imported_at_load():
    """The parser is a lazy import: mock mode and the tests do not need it."""
    src = Path(client_mod.__file__).read_text().splitlines()
    top = [ln for ln in src if ln.startswith(("import meta_sam_parser", "from meta_sam_parser"))]
    assert top == []


def test_pyproject_has_real_extra_with_the_parser():
    root = Path(__file__).resolve().parent.parent
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert extras["real"] == ["meta_sam_parser"]
    assert "meta_sam_parser" not in "".join(tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"])


# ---- Muse request (REQ-RSN-01, REQ-RSN-02) -----------------------------------------------------

def _chat_body(tmp_path, photo, *, config=None, **kwargs):
    chat = ChatTransport()
    client = _client(tmp_path, chat=chat, config=config)
    client.reason("volatile data", [photo, photo], schema=PanelInference.model_json_schema(), **kwargs)
    url, headers, body, timeout = chat.calls[0]
    return url, body


def test_req_rsn_02_body_matches_the_cookbook_chat_shape(tmp_path, photo):
    cfg = load_config()
    url, body = _chat_body(tmp_path, photo, instructions="stable rules")
    assert url.endswith("/chat/completions")
    assert body["model"] == cfg.reasoning_model.id
    assert body["reasoning_effort"] == cfg.reasoning.effort
    assert body["max_tokens"] == cfg.reasoning.max_tokens and "max_completion_tokens" not in body
    assert body["prompt_cache_key"] == cfg.reasoning.prompt_cache_key
    dev, user = body["messages"]
    assert dev == {"role": "developer", "content": "stable rules"}  # text only, first
    assert user["role"] == "user"
    kinds = [part["type"] for part in user["content"]]
    assert kinds == ["text", "image_url", "image_url"]  # images only in the user message
    assert user["content"][0]["text"] == "volatile data"
    assert user["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_req_rsn_02_without_instructions_there_is_only_a_user_message(tmp_path, photo):
    _, body = _chat_body(tmp_path, photo)
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_req_rsn_06_token_limit_parameter_name_comes_from_config(tmp_path, photo):
    cfg = load_config()
    renamed = cfg.model_copy(update={"reasoning": cfg.reasoning.model_copy(
        update={"max_tokens_param": "max_completion_tokens", "max_tokens": 5000})})
    _, body = _chat_body(tmp_path, photo, config=renamed)
    assert body["max_completion_tokens"] == 5000 and "max_tokens" not in body


def test_req_rsn_02_response_format_is_inlined_strict_json_schema(tmp_path, photo):
    _, body = _chat_body(tmp_path, photo)
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["name"] == "answer"
    schema = fmt["json_schema"]["schema"]
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text
    assert schema["additionalProperties"] is False
    cap = schema["properties"]["sleeve_cap"]
    assert cap["type"] == "object" and cap["additionalProperties"] is False
    assert cap["required"] == ["crown_run_mm", "underarm_run_mm"]
    assert cap["description"]  # sibling keys of the $ref survive
    # The seam-loop schema is inlined the same way.
    assert "$ref" not in json.dumps(strict_json_schema(ADJUST_SCHEMA))


def test_req_rsn_02_strict_schema_does_not_mutate_its_input():
    schema = PanelInference.model_json_schema()
    before = json.dumps(schema, sort_keys=True)
    strict_json_schema(schema)
    assert json.dumps(schema, sort_keys=True) == before


def test_req_api_02_instructions_are_part_of_the_cache_key(tmp_path, photo):
    chat = ChatTransport()
    client = _client(tmp_path, chat=chat)
    client.reason("p", [photo], instructions="rules A")
    client.reason("p", [photo], instructions="rules A")
    client.reason("p", [photo], instructions="rules B")
    assert len(chat.calls) == 2


def test_req_rsn_01_unknown_image_type_fails_before_any_call(tmp_path):
    bad = tmp_path / "photo.xyz"
    bad.write_bytes(b"x")
    chat = ChatTransport()
    with pytest.raises(ApiError, match="image type"):
        _client(tmp_path, chat=chat).reason("p", [bad])
    assert chat.calls == []


def test_req_rsn_02_png_photo_gets_png_mime():
    body = reasoning_request_body("m", "p", ["data:image/png;base64,AA"], None, None, effort="high",
                                  max_tokens=10, max_tokens_param="max_tokens", prompt_cache_key="k")
    assert "response_format" not in body
    assert body["messages"][0]["content"][1]["image_url"]["url"] == "data:image/png;base64,AA"


# ---- Day 1 command (REQ-API-09) ---------------------------------------------------------------

def test_req_api_09_day1_refuses_without_yes_spend_and_prints_the_estimate(tmp_path, photo, capsys):
    client = _client(tmp_path)
    with pytest.raises(ApiError, match="--yes-spend"):
        run_day1(photo, yes_spend=False, client=client)
    out = capsys.readouterr().out
    assert "worst case" in out and "sam-3.1" in out and "muse-spark-1.3" in out
    assert client._stream_transport.calls == [] and client._transport.calls == []


def test_req_api_09_day1_cli_refuses_without_the_flag(tmp_path, photo, monkeypatch, capsys):
    monkeypatch.delenv("META_API_KEY", raising=False)
    with pytest.raises(SystemExit) as stop:
        cli.main(["day1", str(photo)])
    assert stop.value.code == 1
    assert "--yes-spend" in capsys.readouterr().err


def test_req_api_09_day1_refuses_in_mock_mode_even_with_the_flag(tmp_path, photo):
    mock = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path,
                               ledger_path=tmp_path / "l" / "ledger.csv", config=load_config()))
    with pytest.raises(ApiError, match="Mock or offline"):
        run_day1(photo, yes_spend=True, client=mock)


def test_req_api_09_day1_makes_exactly_one_call_to_each_model_and_reports_the_ledger(tmp_path, photo, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    client = _client(tmp_path)
    run_day1(photo, yes_spend=True, client=client)
    assert len(client._stream_transport.calls) == 1 and len(client._transport.calls) == 1
    out = capsys.readouterr().out
    assert "SAM: model sam-3.1  input tokens 1234" in out
    assert "Spark: model muse-spark-1.3  input tokens 1700" in out
    assert "Ledger total before: $0.000000" in out and "Ledger total after:" in out
    assert "latency" in out and "cost $" in out
    # Run again: both answers come from the cache, nothing more is spent.
    run_day1(photo, yes_spend=True, client=client)
    assert len(client._stream_transport.calls) == 1 and len(client._transport.calls) == 1
    assert "nothing was spent" in capsys.readouterr().out


def test_req_api_09_day1_has_no_loop_over_calls():
    src = (Path(__file__).resolve().parent.parent / "sewtrace" / "day1.py").read_text()
    assert "while " not in src and "for _ in" not in src
