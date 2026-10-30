"""Real endpoint wiring against Meta's cookbook: request bodies, SSE, SAM decoding, Day 1 command.

No test touches the network, needs a key or needs the ``meta_sam_parser`` package. A fake parser
module with the cookbook's attribute names stands in for it.
"""

import asyncio
import http.client
import json
import socket
import ssl
import sys
import tomllib
import types
import urllib.error
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
    ApiNotSent,
    ApiTimeout,
    ApiTransportError,
    BudgetExceeded,
    MetaClient,
    SamResult,
    SegmentationFailed,
    parse_sse,
    reasoning_request_body,
    sam_frame_hw,
    sam_request_body,
    sam_results_from,
    select_best,
    strict_json_schema,
    urllib_stream_transport,
    urllib_transport,
    with_output_text_done,
)
from sewtrace.config import Settings, load_config
from sewtrace.day1 import run_day1
from sewtrace.reason.panels import PanelInference
from sewtrace.reason.seams import ADJUST_SCHEMA

KEY = "test-key-not-real"
TOL = load_config().sam.box_overhang_tol_px
H, W = 60, 80


# ---- fakes -------------------------------------------------------------------------

def _box(oid, left, top, right, bottom, confidence=None):
    """Like meta_sam_parser's SegmentationBoxRecord (half-open box in the frame SAM saw)."""
    return NS(kind="box", object_id=oid, left=left, top=top, right=right, bottom=bottom, confidence=confidence)


def _mask(oid, w, h, fill=None, at=None, confidence=None):
    """Like SegmentationMaskRecord. ``w`` x ``h`` is the RASTER size (mask.width, mask.height); ``at`` is the
    half-open ``bounds`` (left, top, right, bottom) in the SAM frame, default the whole raster at (0, 0)."""
    left, top, right, bottom = at if at is not None else (0, 0, w, h)
    return NS(kind="mask", object_id=oid, mask=NS(width=w, height=h, encoding="lossless", payload="~", data=fill),
              bounds=NS(left=left, top=top, right=right, bottom=bottom), confidence=confidence)


def _raw(w, h, n=1):
    """raw_output text with ``n`` box lines carrying the source size, as the API writes them."""
    return "\n".join(f"<0f>{i}<|box;x1=0;y1=0;x2=1;y2=1;w={w};h={h};c=0.9|><|mask;x=0;y=0;data=1,1,~AA|>"
                      for i in range(n))


def _result(records, status="completed", diagnostics=(), raw_output=""):
    return NS(outcome=NS(status=status), diagnostics=list(diagnostics), records=list(records), raw_output=raw_output)


def _decode(mask):
    """Fake of decode_mask_to_raster: ``bytes`` of width*height, row-major, 0 or 1. ``data`` None means all foreground."""
    return bytes(mask.data) if mask.data is not None else b"\x01" * (mask.width * mask.height)


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


PANEL_ANSWER = {"front_neck_depth_mm": 45.0, "back_neck_depth_mm": 18.0,
                "sleeve_cap": {"crown_run_mm": 30.0, "underarm_run_mm": 20.0}, "grain_line": "vertical"}


class ChatTransport:
    def __init__(self, content=None):
        self.content, self.calls = content or PANEL_ANSWER, []

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

def test_req_seg_01_box_local_mask_is_placed_at_the_bounds_top_left():
    """meta_sam_parser 0.1.0: the raster is box-local and is placed at bounds.left, bounds.top."""
    data = [1, 0, 0, 1, 1, 1]  # 3 wide, 2 high: [[1,0,0],[1,1,1]]
    result = _result([_box("0", 10, 5, 13, 7), _mask("0", 3, 2, data, at=(10, 5, 13, 7))])
    (res,) = sam_results_from(result, _decode, (H, W), "p.jpg", "t-shirt", box_tol_px=TOL)
    expected = np.zeros((H, W), bool)
    expected[5, 10] = True
    expected[6, 10:13] = True
    assert res.mask.shape == (H, W) and np.array_equal(res.mask, expected)
    assert res.score is None


def test_req_seg_01_decode_returns_bytes_that_are_read_row_major():
    """Day 1 bug: np.asarray(bytes) is one 0-d value. The raster must be frombuffer(...).reshape(h, w)."""
    data = bytes([1, 0, 0, 0, 0, 1])  # 3 x 2
    (res,) = sam_results_from(_result([_mask("0", 3, 2, data, at=(0, 0, 3, 2))]), _decode, (H, W), "p", "x", box_tol_px=TOL)
    assert res.mask[0, 0] and res.mask[1, 2] and res.mask.sum() == 2


def test_req_seg_01_raster_is_resized_to_its_bounds():
    """The raster has its own size. Day 1: 'decodes to 1 values, expected 1024 x 710'. Here 4 x 2 covers 8 x 4."""
    data = [1, 1, 0, 0, 1, 1, 0, 0]  # left half foreground
    result = _result([_mask("0", 4, 2, data, at=(10, 20, 18, 24))])
    (res,) = sam_results_from(result, _decode, (H, W), "p", "x", box_tol_px=TOL)
    assert res.mask[20:24, 10:14].all() and not res.mask[20:24, 14:18].any() and res.mask.sum() == 16


def test_req_seg_01_cookbook_sizes_bounds_width_is_right_minus_left():
    result = _result([_box("0", 751, 290, 925, 493), _mask("0", 174, 203, at=(751, 290, 925, 493))])
    (res,) = sam_results_from(result, _decode, (600, 1000), "ducks.jpg", "duck", box_tol_px=TOL)
    assert res.mask.sum() == 174 * 203
    ys, xs = np.nonzero(res.mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (751, 924, 290, 492)


def test_req_seg_01_bounds_within_one_px_of_the_edge_are_clipped():
    edge = _result([_mask("0", 3, 2, at=(W - 2, H - 1, W + 1, H + 1))])
    (res,) = sam_results_from(edge, _decode, (H, W), "p", "x", box_tol_px=TOL)
    assert res.mask.sum() == 2 and res.mask[H - 1, W - 2:].all()


def test_req_seg_01_bounds_past_the_image_by_more_than_one_px_are_an_error_not_a_clip():
    past = _result([_mask("0", 5, 3, at=(W - 2, H - 1, W + 3, H + 2))])
    with pytest.raises(ApiError, match=r"p\.jpg.*different pixel frames"):
        sam_results_from(past, _decode, (H, W), "p.jpg", "x", box_tol_px=TOL)
    negative = _result([_mask("0", 5, 2, at=(-3, 0, 2, 2))])
    with pytest.raises(ApiError, match="different pixel frames"):
        sam_results_from(negative, _decode, (H, W), "p.jpg", "x", box_tol_px=TOL)


def test_req_seg_01_wrong_decoded_length_and_empty_bounds_are_errors():
    with pytest.raises(ApiError, match=r"p\.jpg.*decodes to 2 values, expected 80 x 60"):
        sam_results_from(_result([_mask("0", W, H, [1, 1])]), _decode, (H, W), "p.jpg", "t-shirt", box_tol_px=TOL)
    with pytest.raises(ApiError, match="empty bounds"):
        sam_results_from(_result([_mask("0", 3, 3, at=(5, 5, 5, 8))]), _decode, (H, W), "p.jpg", "t-shirt", box_tol_px=TOL)


def test_req_seg_01_source_size_is_read_from_the_box_lines_and_never_crashes():
    from sewtrace.api.client import sam_source_hw
    assert sam_source_hw(_raw(1024, 710, n=2)) == (710, 1024)
    assert sam_source_hw(_raw(1024, 710) + "\n" + _raw(500, 400)) is None  # boxes disagree
    for bad in ("", "no boxes here", "<|box;w=abc;h=5|>", "<|box;w=0;h=5|>", "<|box;h=5|>", "<|box;w=5|", None, 7):
        assert sam_source_hw(bad) is None


def test_req_seg_01_server_side_resize_is_mapped_with_the_reported_source_size(caplog):
    """We sent 80 x 60 but SAM worked on 160 x 120: bounds are in the 160 x 120 frame."""
    data = [1] * 20 * 10
    result = _result([_mask("0", 20, 10, data, at=(40, 20, 60, 30))], raw_output=_raw(160, 120))
    with caplog.at_level("WARNING", logger="sewtrace.api.client"):
        (res,) = sam_results_from(result, _decode, (H, W), "p", "x", box_tol_px=TOL)
    assert "resized on the server" in caplog.text
    ys, xs = np.nonzero(res.mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (20, 29, 10, 14)  # halved into the 80 x 60 frame


def test_req_seg_01_unreadable_source_size_keeps_the_sent_size():
    result = _result([_mask("0", 4, 4, at=(1, 1, 5, 5))], raw_output="garbage <|box;w=|>")
    (res,) = sam_results_from(result, _decode, (H, W), "p", "x", box_tol_px=TOL)
    assert res.mask.sum() == 16


def test_req_seg_03_confidence_of_the_mask_record_is_the_score():
    result = _result([_mask("0", 5, 5, confidence=0.4), _mask("1", 10, 10, at=(20, 20, 30, 30), confidence=0.9)])
    results = sam_results_from(result, _decode, (H, W), "p", "x", box_tol_px=TOL)
    assert [r.score for r in results] == [0.4, 0.9]
    assert select_best(results).score == 0.9
    small_but_sure = [SamResult(mask=np.ones((2, 2), bool), score=0.9), SamResult(mask=np.ones((9, 9), bool), score=0.5)]
    assert select_best(small_but_sure).score == 0.9
    # The box confidence is not the mask confidence (REQ-SEG-03 asks for the mask).
    boxed = _result([_box("0", 0, 0, 5, 5, confidence=0.99), _mask("0", 5, 5)])
    assert sam_results_from(boxed, _decode, (H, W), "p", "x", box_tol_px=TOL)[0].score is None


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
    records = [_mask("0", 5, 5), _mask("1", 20, 20, at=(10, 10, 30, 30))]
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
    records = [_mask("0", 5, 5, confidence=0.1)]
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result(records)))
    with pytest.raises(SegmentationFailed, match=r"p\.jpg.*'t-shirt'.*below"):
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
    bad = _result([_mask("0", 7, 7, [1, 1])])
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(bad))
    client = _client(tmp_path)
    with pytest.raises(ApiError, match="decodes to 2 values"):
        client.segment(photo, "t-shirt")
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    assert client.segment(photo, "t-shirt").all()  # decoded from the stored events
    assert len(client._stream_transport.calls) == 1
    assert len(client.ledger_path.read_text().splitlines()) == 2  # header + one paid call


def test_real_sam_module_is_not_imported_at_load():
    """C-08: The parser is a lazy import: mock mode and the tests do not need it."""
    src = Path(client_mod.__file__).read_text().splitlines()
    top = [ln for ln in src if ln.startswith(("import meta_sam_parser", "from meta_sam_parser"))]
    assert top == []


def test_pyproject_has_real_extra_with_the_parser():
    """C-08."""
    root = Path(__file__).resolve().parent.parent
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert extras["real"] == ["meta_sam_parser==0.1.0"]
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
    run_day1(photo, yes_spend=True, client=client, out_dir=tmp_path / "o")
    assert len(client._stream_transport.calls) == 1 and len(client._transport.calls) == 1
    out = capsys.readouterr().out
    assert "SAM: model sam-3.1  input tokens 1234" in out
    assert "Spark: model muse-spark-1.3  input tokens 1700" in out
    assert "Ledger total before: $0.000000" in out and "Ledger total after:" in out
    assert "latency" in out and "cost $" in out
    # Run again: both answers come from the cache, nothing more is spent.
    run_day1(photo, yes_spend=True, client=client, out_dir=tmp_path / "o")
    assert len(client._stream_transport.calls) == 1 and len(client._transport.calls) == 1
    assert "nothing was spent" in capsys.readouterr().out


def test_req_api_09_day1_sends_the_production_schema_and_prints_pass(tmp_path, photo, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    client = _client(tmp_path)
    run_day1(photo, yes_spend=True, client=client, out_dir=tmp_path / "o")
    _, _, body, _ = client._transport.calls[0]
    sent = body["response_format"]["json_schema"]["schema"]
    assert sent == strict_json_schema(PanelInference.model_json_schema())
    for field in ("front_neck_depth_mm", "back_neck_depth_mm", "sleeve_cap", "grain_line"):
        assert field in body["messages"][-1]["content"][0]["text"]
    assert "Spark schema check: PASS" in capsys.readouterr().out


def test_req_api_09_day1_prints_fail_when_the_answer_does_not_validate(tmp_path, photo, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    client = _client(tmp_path, chat=ChatTransport({"view": "front"}))
    with pytest.raises(ApiError, match="PanelInference"):
        run_day1(photo, yes_spend=True, client=client, out_dir=tmp_path / "o")
    assert "Spark schema check: FAIL" in capsys.readouterr().out


def test_req_seg_01_day1_prints_boxes_next_to_image_size_and_writes_overlay(tmp_path, photo, monkeypatch, capsys):
    records = [_box("0", 10, 5, 30, 25, confidence=0.5), _mask("0", 10, 10, at=(10, 5, 30, 25), confidence=0.875)]
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result(records, raw_output=_raw(W, H))))
    run_day1(photo, yes_spend=True, client=_client(tmp_path), out_dir=tmp_path / "o")
    out = capsys.readouterr().out
    assert f"image {W} x {H} px" in out and "box 0: left 10  top 5  right 30  bottom 25" in out
    assert "record 0: mask confidence 0.875  box confidence 0.500  bounds 10,5 to 30,25  raster 10 x 10" in out
    assert f"sent to SAM {W} x {H} px; SAM reported source {W} x {H} px" in out and "DIFFERS" not in out
    overlay = cv2.imread(str(tmp_path / "o" / "p.sam_overlay.png"))
    assert overlay.shape == (H, W, 3)
    assert overlay[15, 20, 2] > overlay[15, 20, 0]  # mask area is shaded red


def test_req_api_09_day1_prints_approximate_image_tokens(tmp_path, photo, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    run_day1(photo, yes_spend=True, client=_client(tmp_path), out_dir=tmp_path / "o")
    out = capsys.readouterr().out
    # ChatTransport bills 1700 input tokens; the day1 prompt is about 50 estimated text tokens.
    assert "Spark: image tokens APPROXIMATELY 16" in out and "SAM: image tokens APPROXIMATELY 123" in out


def test_req_api_09_day1_has_no_loop_over_calls():
    src = (Path(__file__).resolve().parent.parent / "sewtrace" / "day1.py").read_text()
    assert "while " not in src and "for _ in" not in src


# ---- SAM input frame, cache keys, stream outcome (REQ-SEG, REQ-API-02, REQ-API-04) ------------------

def _decode_data_url(url: str) -> np.ndarray:
    import base64
    raw = base64.b64decode(url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def test_req_seg_01_sam_gets_the_decoded_pixels_reencoded_not_the_file_bytes(tmp_path, monkeypatch):
    image = np.zeros((H, W, 3), np.uint8)  # smooth, like a photo: noise is the worst case for JPEG
    image[..., 0], image[..., 1], image[..., 2] = np.arange(W) * 3, np.arange(H)[:, None] * 4, 90
    path = tmp_path / "q.png"
    cv2.imwrite(str(path), image)
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    stream = StreamTransport()
    _client(tmp_path / "c", stream=stream).segment(path, "t-shirt")
    url = stream.calls[0][2]["input"][0]["content"][1]["image_url"]
    assert url.startswith("data:image/jpeg;base64,")  # re-encoded: the PNG file was not forwarded
    sent = _decode_data_url(url)
    assert sent.shape == image.shape and np.abs(sent.astype(int) - image).mean() < 3  # JPEG q95 noise only


def test_req_api_02_sam_key_follows_the_pixels_not_the_file_bytes(tmp_path, monkeypatch):
    image = np.random.default_rng(2).integers(0, 255, (H, W, 3), dtype=np.uint8)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(a), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(str(b), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert a.read_bytes() != b.read_bytes()
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    client = _client(tmp_path / "c")
    client.segment(a, "t-shirt")
    client.segment(b, "t-shirt")  # same pixels: cache hit
    assert len(client._stream_transport.calls) == 1
    other = tmp_path / "o.png"
    cv2.imwrite(str(other), 255 - image)
    client.segment(other, "t-shirt")
    assert len(client._stream_transport.calls) == 2


def test_req_seg_01_unreadable_photo_is_an_error_before_any_call(tmp_path, monkeypatch):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([])))
    client = _client(tmp_path)
    with pytest.raises(ApiError, match="not a readable image"):
        client.segment(bad, "t-shirt")
    assert client._stream_transport.calls == [] and not client.ledger_path.exists()


def test_req_seg_01_box_outside_the_sent_frame_stops_the_run(tmp_path, photo, monkeypatch):
    records = [_mask("0", W + 40, H + 40)]
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result(records)))
    with pytest.raises(ApiError, match="different pixel frames"):
        _client(tmp_path).segment(photo, "t-shirt")


@pytest.mark.parametrize("tail", [
    [{"type": "response.failed", "response": {"error": {"message": "overloaded"}}}],
    [{"type": "response.created"}],
])
def test_req_api_02_stream_without_completed_is_billed_but_not_cached(tmp_path, photo, monkeypatch, tail):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    stream = StreamTransport(tail)
    client = _client(tmp_path, stream=stream)
    with pytest.raises(ApiError, match=r"p\.jpg.*'t-shirt'.*without response\.completed.*Run the command again"):
        client.segment(photo, "t-shirt")
    assert not list(tmp_path.glob("*.events.json"))
    assert len(client.ledger_path.read_text().splitlines()) == 2  # header + the billed attempt
    stream.events = EVENTS  # the retry is a new, real attempt
    assert client.segment(photo, "t-shirt").all()
    assert len(stream.calls) == 2


def test_req_api_02_decoder_version_reuses_events_and_redecodes(tmp_path, photo, monkeypatch):
    seen: list[dict] = []
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", 10, 10, [1, 1, 1])]), seen))
    client = _client(tmp_path)
    with pytest.raises(ApiError, match="decodes to 3 values"):
        client.segment(photo, "t-shirt")  # a short raster: the decoder rejects it
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    assert client.segment(photo, "t-shirt").all()
    monkeypatch.setattr(client_mod, "SAM_DECODE_VERSION", client_mod.SAM_DECODE_VERSION + 1)
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H, [1, 0] * (W * H // 2))])))
    redone = client.segment(photo, "t-shirt")  # new decoder version: decoded again from the same events
    assert 0 < redone.sum() < W * H
    assert len(client._stream_transport.calls) == 1


def test_req_api_08_offline_redecodes_stored_events_and_misses_cleanly_without_them(tmp_path, photo, monkeypatch):
    from sewtrace.api.client import CacheMiss
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", 10, 10, [1, 1, 1])])))
    online = _client(tmp_path)
    with pytest.raises(ApiError):
        online.segment(photo, "t-shirt")  # events cached, mask not decodable yet
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))

    def offline_client(folder):
        settings = Settings(api_key="", mock=False, cache_dir=folder, offline=True,
                            ledger_path=tmp_path / "ledger" / "ledger.csv", config=load_config())
        return MetaClient(settings, transport=ChatTransport(), stream_transport=StreamTransport([]), sleep=lambda s: None)

    assert offline_client(tmp_path).segment(photo, "t-shirt").all()
    with pytest.raises(CacheMiss):
        offline_client(tmp_path / "empty").segment(photo, "t-shirt")
    monkeypatch.setitem(sys.modules, "meta_sam_parser", None)
    (tmp_path / "x").mkdir()
    for f in tmp_path.glob("real-*.png"):
        f.unlink()
    for f in tmp_path.glob("real-*.score.json"):
        f.unlink()
    with pytest.raises(ApiError, match=r"pip install -e '\.\[real\]'"):
        offline_client(tmp_path).segment(photo, "t-shirt")


# ---- every failure after sending is ledgered (REQ-API-04, C-01) ---------------------------------

class Raises:
    def __init__(self, error):
        self.error, self.calls = error, []

    def __call__(self, *args):
        self.calls.append(args)
        raise self.error


MID_STREAM_ERRORS = [
    lambda: ApiError("stream event is not JSON: 'x'"),
    lambda: http.client.IncompleteRead(b"par", 10),
    lambda: ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
    lambda: json.JSONDecodeError("bad body", "x", 0),
]


def _one_estimated_row(client, model_id, **expected):
    lines = client.ledger_path.read_text().splitlines()
    assert len(lines) == 2, lines  # header + exactly one row
    row = dict(zip(lines[0].split(","), lines[1].split(",")))
    assert row["model"] == model_id and row["estimated"] == "1"
    for key, value in expected.items():
        assert row[key] == str(value)


@pytest.mark.parametrize("make_error", MID_STREAM_ERRORS)
def test_req_api_04_failure_in_the_sam_stream_logs_one_estimated_row(tmp_path, photo, monkeypatch, make_error):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([])))
    client = _client(tmp_path, stream=Raises(make_error()))
    with pytest.raises(ApiError):
        client.segment(photo, "t-shirt")
    _one_estimated_row(client, load_config().models.sam.id, images=1)
    assert not list(tmp_path.glob("*.events.json"))


@pytest.mark.parametrize("make_error", MID_STREAM_ERRORS)
def test_req_api_04_failure_in_a_chat_answer_logs_one_estimated_row(tmp_path, photo, make_error):
    client = _client(tmp_path, chat=Raises(make_error()))
    with pytest.raises(ApiError):
        client.reason("p", [photo])
    _one_estimated_row(client, load_config().reasoning_model.id,
                       output_tokens=load_config().reasoning.max_tokens)


def test_req_api_04_failure_before_sending_logs_nothing(tmp_path, photo, monkeypatch):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([])))
    client = _client(tmp_path, stream=Raises(ApiNotSent("dns")), chat=Raises(ApiNotSent("refused")))
    with pytest.raises(ApiNotSent):
        client.segment(photo, "t-shirt")
    with pytest.raises(ApiNotSent):
        client.reason("p", [photo])
    assert not client.ledger_path.exists()
    # The budget stop comes before the request: no row, no transport call.
    tight = load_config()
    tight = tight.model_copy(update={"api": tight.api.model_copy(update={"budget_limit_usd": 0.0001})})
    chat = Raises(ApiError("must not be called"))
    broke = _client(tmp_path / "b", chat=chat, config=tight)
    with pytest.raises(BudgetExceeded):
        broke.reason("p", [photo])
    assert chat.calls == [] and not broke.ledger_path.exists()


@pytest.mark.parametrize("error", [KeyboardInterrupt(), asyncio.CancelledError()])
def test_req_api_04_interrupt_after_sending_logs_one_estimated_row_and_propagates(tmp_path, photo, monkeypatch, error):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([])))
    client = _client(tmp_path, stream=Raises(error))
    with pytest.raises(type(error)) as caught:
        client.segment(photo, "t-shirt")
    assert caught.value is error  # re-raised unchanged
    _one_estimated_row(client, load_config().models.sam.id, images=1)


@pytest.mark.parametrize("error", [TypeError("code bug"), KeyError("k")])
def test_req_api_04_code_bug_after_sending_is_billed_but_not_made_retryable(tmp_path, photo, error):
    """Only OSError, HTTPException and JSONDecodeError count as transport errors (TRANSPORT_ERRORS)."""
    client = _client(tmp_path, chat=Raises(error))
    with pytest.raises(type(error)) as caught:
        client.reason("p", [photo])
    assert caught.value is error and not isinstance(error, ApiError)
    _one_estimated_row(client, load_config().reasoning_model.id)


def _ledger_row(client):
    lines = client.ledger_path.read_text().splitlines()
    assert len(lines) == 2, lines
    return dict(zip(lines[0].split(","), lines[1].split(",")))


@pytest.mark.parametrize("events", [
    [{"type": "response.created"}, {"type": "response.completed", "response": {}}],
    [{"type": "response.created"}, {"type": "response.completed", "response": {"usage": None}}],
])
def test_req_api_04_sam_answer_without_usage_is_flagged_estimated(tmp_path, photo, monkeypatch, events):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    client = _client(tmp_path, stream=StreamTransport(events))
    client.segment(photo, "t-shirt")
    row = _ledger_row(client)
    assert row["estimated"] == "1" and row["images"] == "1" and float(row["cost_usd"]) > 0


def test_req_api_04_usage_parse_failure_is_flagged_estimated(tmp_path, photo, monkeypatch):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    bad = [{"type": "response.completed", "response": {"usage": {"input_tokens": "many"}}}]
    client = _client(tmp_path, stream=StreamTransport(bad))
    client.segment(photo, "t-shirt")
    assert _ledger_row(client)["estimated"] == "1"


def test_req_api_04_reported_usage_is_not_flagged_estimated(tmp_path, photo, monkeypatch):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    client = _client(tmp_path)
    client.segment(photo, "t-shirt")
    assert _ledger_row(client)["estimated"] == "0"


def test_req_api_09_day1_says_image_tokens_unknown_when_usage_is_estimated(tmp_path, photo, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    no_usage = [{"type": "response.completed", "response": {}}]
    run_day1(photo, yes_spend=True, client=_client(tmp_path, stream=StreamTransport(no_usage)),
             out_dir=tmp_path / "o")
    out = capsys.readouterr().out
    assert "SAM: image tokens unknown (no usage reported)" in out
    assert "SAM: image tokens APPROXIMATELY" not in out and "Spark: image tokens APPROXIMATELY" in out


def test_req_api_08_offline_replay_of_stored_sam_events_is_not_counted_as_a_call(tmp_path, photo, monkeypatch):
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    online = _client(tmp_path)
    online.segment(photo, "t-shirt")
    assert online.calls == 1
    for f in tmp_path.glob("real-*.png"):  # drop the decoded mask, keep the raw events
        f.unlink()
    for f in tmp_path.glob("real-*.score.json"):
        f.unlink()
    settings = Settings(api_key="", mock=False, cache_dir=tmp_path, offline=True,
                        ledger_path=online.ledger_path, config=load_config())
    offline = MetaClient(settings)
    offline.segment(photo, "t-shirt")
    assert offline.calls == 0


class _Body:
    """A response whose body fails or is garbage. Works for ``read`` and for line iteration."""

    def __init__(self, error=None, data=b""):
        self.error, self.data = error, data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        if self.error:
            raise self.error
        return self.data

    def __iter__(self):
        yield b'data: {"type": "a"}\n'
        if self.error:
            raise self.error
        yield b"data: {oops\n"


@pytest.mark.parametrize("error", [http.client.IncompleteRead(b"x", 5), ssl.SSLEOFError(8, "EOF"),
                                   http.client.RemoteDisconnected("closed")])
@pytest.mark.parametrize("transport", [urllib_transport, urllib_stream_transport])
def test_req_api_04_urllib_transports_map_broken_reads_to_transport_error(monkeypatch, transport, error):
    monkeypatch.setattr(urllib.request, "urlopen", lambda r, timeout: _Body(error))
    with pytest.raises(ApiTransportError):
        transport("https://x", {}, {}, 5.0)


def test_req_api_04_urllib_bad_json_and_never_sent_errors(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda r, timeout: _Body(data=b"<html>gateway</html>"))
    with pytest.raises(ApiTransportError, match="not JSON"):
        urllib_transport("https://x", {}, {}, 5.0)
    with pytest.raises(ApiError, match="not JSON"):  # stream: parse_sse error, billed by _paid
        urllib_stream_transport("https://x", {}, {}, 5.0)

    def refuse(request, timeout):
        raise urllib.error.URLError(socket.gaierror("no such host"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    for transport in (urllib_transport, urllib_stream_transport):
        with pytest.raises(ApiNotSent):
            transport("https://x", {}, {}, 5.0)


# ---- SAM frame safety (REQ-SEG-01), review of 79246e8 --------------------------------------

def _one_mask_result(raw):
    return _result([_mask("0", 4, 4, at=(1, 1, 5, 5))], raw_output=raw)


def test_req_seg_01_transposed_reported_frame_is_an_error_not_a_stretch():
    """Sent 80 x 60, SAM says 60 x 80: separate x and y scales would stretch the mask by 1.33 and 0.75."""
    with pytest.raises(ApiError, match=r"p\.jpg.*'x'.*different pixel frames.*day1"):
        sam_results_from(_one_mask_result(_raw(H, W)), _decode, (H, W), "p.jpg", "x", box_tol_px=TOL)


def test_req_seg_01_cropped_reported_frame_is_an_error():
    with pytest.raises(ApiError, match="different pixel frames"):
        sam_results_from(_one_mask_result(_raw(W, H - 10)), _decode, (H, W), "p.jpg", "x", box_tol_px=TOL)


def test_req_seg_01_uniform_half_size_frame_is_mapped(caplog):
    """SAM worked on 40 x 30 (same aspect as the 80 x 60 sent): bounds double into the sent frame."""
    result = _result([_mask("0", 10, 5, at=(10, 5, 20, 10))], raw_output=_raw(W // 2, H // 2))
    with caplog.at_level("WARNING", logger="sewtrace.api.client"):
        (res,) = sam_results_from(result, _decode, (H, W), "p", "x", box_tol_px=TOL)
    assert "resized on the server" in caplog.text
    ys, xs = np.nonzero(res.mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (20, 39, 10, 19)


def test_req_seg_01_rounding_of_the_reported_size_is_accepted():
    sent = (1841, 2048)  # H, W
    result = _result([_mask("0", 4, 4, at=(1, 1, 5, 5))], raw_output=_raw(1024, 921))  # 920.5 rounded up
    assert sam_frame_hw(result, sent) == (921, 1024)


def test_req_seg_01_equal_reported_frame_is_a_no_op(caplog):
    with caplog.at_level("INFO", logger="sewtrace.api.client"):
        assert sam_frame_hw(_one_mask_result(_raw(W, H)), (H, W)) == (H, W)
    assert not caplog.records


def test_req_seg_01_transposed_real_sizes_name_the_photo():
    """The review case: sent 2048 x 1841, reported 1841 x 2048."""
    with pytest.raises(ApiError, match=r"a\.jpg.*1841 x 2048.*2048 x 1841"):
        sam_frame_hw(_one_mask_result(_raw(1841, 2048)), (1841, 2048), "a.jpg", "t-shirt")


def test_req_seg_01_disagreeing_box_sizes_are_an_error_but_unreadable_is_not(caplog):
    both = _raw(W, H) + "\n" + _raw(W // 2, H // 2)
    with pytest.raises(ApiError, match=r"p\.jpg.*different frame sizes"):
        sam_frame_hw(_one_mask_result(both), (H, W), "p.jpg", "x")
    assert sam_frame_hw(_one_mask_result("no boxes"), (H, W)) == (H, W)


def test_req_seg_01_raster_downsize_uses_area_and_upsize_uses_linear():
    # 6 x 6 raster to 2 x 2 bounds: each 3 x 3 block is 6/9 foreground. INTER_AREA keeps it (INTER_LINEAR
    # would sample the empty middle column).
    down = [1, 0, 1, 1, 0, 1] * 6
    (res,) = sam_results_from(_result([_mask("0", 6, 6, down, at=(0, 0, 2, 2))]), _decode, (H, W), "p", "x", box_tol_px=TOL)
    assert res.mask.sum() == 4
    # 2 x 1 raster [1, 0] to 4 x 1 bounds: linear gives 1, .75, .25, 0.
    (up,) = sam_results_from(_result([_mask("0", 2, 1, [1, 0], at=(0, 0, 4, 1))]), _decode, (H, W), "p", "x", box_tol_px=TOL)
    assert up.mask[0, :4].tolist() == [True, True, False, False] and up.mask.sum() == 2


def test_req_seg_04_box_confidence_without_mask_confidence_warns_and_is_not_the_score(caplog):
    result = _result([_box("0", 0, 0, 5, 5, confidence=0.93), _mask("0", 5, 5)])
    with caplog.at_level("WARNING", logger="sewtrace.api.client"):
        (res,) = sam_results_from(result, _decode, (H, W), "p.jpg", "x", box_tol_px=TOL)
    assert res.score is None
    assert "p.jpg" in caplog.text and "box has 0.93" in caplog.text and "not used as the score" in caplog.text


def test_req_api_02_raw_events_key_is_unchanged_and_only_the_decoded_key_moved(tmp_path):
    """Day 1 paid for a SAM answer stored under the raw key: it must still be found. Values were computed
    with the code before the frame fix (SAM_DECODE_VERSION 3)."""
    image = np.random.default_rng(7).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), image)
    client = _client(tmp_path)
    assert client._sam_key(path, "t-shirt", decoded=False) == "9266f858babaa1dd1646868da73f42f25411827278ea8b3ffe7d33ebc5e5101e"
    assert client._sam_key(path, "t-shirt", decoded=True) != "8ff2b54ab437db8bcee5f1271b775895727b65ffc2ed5b232882844db64acc3c"
