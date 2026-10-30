"""Photo downscaling before SAM and Muse, cached rejected answers, SAM inside a running event loop.

No network, no key. Helpers (fake parser, fake transports) come from test_real_endpoints.
"""

import asyncio
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_panels import meas, measured, photo_set
from test_real_endpoints import (
    TOL,
    H,
    StreamTransport,
    W,
    _box,
    _chat_body,
    _client,
    _decode,
    _decode_data_url,
    _fake_parser,
    _mask,
    _result,
    photo,  # noqa: F401  (fixture)
)

from sewtrace.api import client as client_mod
from sewtrace.api.client import ApiError, CacheMiss, MetaClient, sam_results_from
from sewtrace.config import Settings, load_config
from sewtrace.mock import synth
from sewtrace.reason.panels import draft_spec, infer_panels


def _big_photo(tmp_path, w=4000, h=3000, name="big.jpg") -> Path:
    path = tmp_path / name
    image = np.zeros((h, w, 3), np.uint8)
    image[..., 0], image[..., 1] = np.arange(w)[None, :] // 16, np.arange(h)[:, None] // 12
    cv2.imwrite(str(path), image)
    return path


def _sam_cfg(max_px):
    cfg = load_config()
    return cfg.model_copy(update={"sam": cfg.sam.model_copy(update={"max_upload_px": max_px})})


def _muse_cfg(max_px):
    cfg = load_config()
    return cfg.model_copy(update={"reasoning": cfg.reasoning.model_copy(update={"max_upload_px": max_px})})


def _sent_image(stream: StreamTransport) -> np.ndarray:
    return _decode_data_url(stream.calls[0][2]["input"][0]["content"][1]["image_url"])


# ---- downscaling (REQ-API-02, SPEC 8.1, 13) -----------------------------------------------------

def test_req_api_02_sam_upload_is_downscaled_and_a_box_local_mask_lands_in_the_original_frame(tmp_path, monkeypatch):
    """4000 x 3000 photo, upload 2000 x 1500 (factor 2). The mask is box-local in the SENT frame."""
    photo_path = _big_photo(tmp_path)
    mw, mh = 300, 200  # sent-frame box (500, 400) to (800, 600)
    data = np.zeros((mh, mw), np.uint8)
    data[:100, :] = 1  # top half of the box
    result = _result([_mask("0", mw, mh, data.ravel().tolist(), at=(500, 400, 800, 600))])
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(result))
    stream = StreamTransport()
    mask = _client(tmp_path / "c", stream=stream, config=_sam_cfg(2000)).segment(photo_path, "t-shirt")
    assert _sent_image(stream).shape[:2] == (1500, 2000)
    assert mask.shape == (3000, 4000)
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max()) == pytest.approx((1000, 1599), abs=1)
    assert (ys.min(), ys.max()) == pytest.approx((800, 999), abs=1)


def test_req_api_02_sam_box_check_runs_in_the_original_frame_after_rescaling(tmp_path, monkeypatch):
    photo_path = _big_photo(tmp_path)
    # The box ends 30 sent px (60 original px) past the frame: a frame mismatch, still an error.
    records = [_mask("0", 2030, 1500)]
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result(records)))
    with pytest.raises(ApiError, match="different pixel frames"):
        _client(tmp_path / "c", config=_sam_cfg(2000)).segment(photo_path, "t-shirt")


def test_req_api_02_one_sent_pixel_of_box_rounding_is_tolerated(tmp_path):
    """Tolerance is box_overhang_tol_px SENT pixels: 2 original px at a factor of 2, not 1."""
    kwargs = {"box_tol_px": TOL, "full_hw": (3000, 4000)}
    ok = _result([_mask("0", 2000, 1500, at=(0, 0, 2001, 1500))])
    (res,) = sam_results_from(ok, _decode, (1500, 2000), "p", "x", **kwargs)
    assert res.mask.shape == (3000, 4000) and res.mask.all()
    past = _result([_mask("0", 2003, 1500)])
    with pytest.raises(ApiError, match="different pixel frames"):
        sam_results_from(past, _decode, (1500, 2000), "p", "x", **kwargs)


def test_req_api_02_sam_boxes_are_reported_in_the_original_frame(tmp_path, monkeypatch):
    photo_path = _big_photo(tmp_path)
    result = _result([_box("0", 500, 400, 800, 600), _mask("0", 300, 200, at=(500, 400, 800, 600))])
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(result))
    client = _client(tmp_path / "c", config=_sam_cfg(2000))
    client.segment(photo_path, "t-shirt")
    assert client.sam_boxes(photo_path, "t-shirt") == ((4000, 3000), [("0", 1000.0, 800.0, 1600.0, 1200.0)])


def test_req_api_02_small_photo_is_not_resized(tmp_path, photo, monkeypatch):  # noqa: F811
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    stream = StreamTransport()
    _client(tmp_path / "c", stream=stream).segment(photo, "t-shirt")
    assert _sent_image(stream).shape[:2] == (H, W)


def test_req_api_02_muse_images_have_a_long_side_of_at_most_the_config_value(tmp_path):
    photo_path = _big_photo(tmp_path)
    _, body = _chat_body(tmp_path / "c", photo_path, config=_muse_cfg(1000))
    urls = [p["image_url"]["url"] for p in body["messages"][-1]["content"] if p["type"] == "image_url"]
    assert len(urls) == 2
    for url in urls:
        assert _decode_data_url(url).shape[:2] == (750, 1000)
    assert load_config().reasoning.max_upload_px == 1568 and load_config().sam.max_upload_px == 2048


def test_req_api_02_cache_keys_change_with_max_upload_px(tmp_path):
    photo_path = _big_photo(tmp_path)
    a, b = _client(tmp_path / "a", config=_sam_cfg(2000)), _client(tmp_path / "b", config=_sam_cfg(1000))
    for decoded in (False, True):
        assert a._sam_key(photo_path, "t-shirt", decoded=decoded) != b._sam_key(photo_path, "t-shirt", decoded=decoded)
    shared = tmp_path / "shared"
    first, second = _client(shared, config=_muse_cfg(1500)), _client(shared, config=_muse_cfg(900))
    first.reason("p", [photo_path])
    second.reason("p", [photo_path])
    assert len(first._transport.calls) == 1 and len(second._transport.calls) == 1  # no collision
    first.reason("p", [photo_path])
    assert len(first._transport.calls) == 1  # same value: hit


def test_req_api_02_mock_mode_ignores_max_upload_px(tmp_path):
    d = tmp_path / "tee"
    synth.make_tee(synth.TEES[0], d, seed=0)
    mask = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "c")).segment(d / "front.jpg", "t-shirt")
    assert mask.shape == cv2.imread(str(d / "front.jpg")).shape[:2]


def test_req_api_02_downscale_keeps_mm_measurements_within_half_a_mm(tmp_path, capsys):
    """Simulate the upload at the default sam.max_upload_px: shrink each sidecar mask (INTER_AREA), let
    sam_results_from scale it back to the photo frame, measure both. The synthetic photos are 2489 px
    wide, so the factor is 1.22. A factor of 2 on them (0.8 mm per sent pixel, far coarser than a
    12 MP phone photo) moved the shoulder by 0.62 mm: binary-mask quantisation, not a mapping bias."""
    from test_contour import extract, truth_homography

    from sewtrace.types import MEASUREMENT_KEYS
    from sewtrace.vision.measure import measure_tee

    d = tmp_path / "tee"
    truth = synth.make_tee(synth.TEES[1], d, seed=1)
    full_parts, small_parts = [], []
    for view in ("front", "back", "sleeve"):
        prompt = "sleeve" if view == "sleeve" else "t-shirt"
        mask = cv2.imread(str(d / f"{view}.masks" / f"{prompt}.png"), cv2.IMREAD_GRAYSCALE) > 127
        h, w = mask.shape
        scale = load_config().sam.max_upload_px / max(h, w)
        sh, sw = round(h * scale), round(w * scale)
        small = cv2.resize(mask.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_AREA) > 127
        (res,) = sam_results_from(_result([_mask("0", sw, sh, small.ravel().astype(int).tolist())]), _decode,
                                  (sh, sw), view, prompt, box_tol_px=TOL, full_hw=(h, w))
        assert res.mask.shape == mask.shape
        hom = truth_homography(truth, view)
        full_parts.extend(extract(mask, hom, view))
        small_parts.extend(extract(res.mask, hom, view))
    want, got = measure_tee(*full_parts), measure_tee(*small_parts)
    errors = {k: abs(got[k] - want[k]) for k in MEASUREMENT_KEYS}
    with capsys.disabled():
        print("\ndownscale (default max_upload_px) measurement change (mm):", {k: round(v, 3) for k, v in errors.items()})
    assert max(errors.values()) <= 0.5, errors


# ---- rejected answers are cached (REQ-API-02) ---------------------------------------------------

class Scripted:
    """Chat transport that answers with one scripted choice per call (the last one repeats)."""

    def __init__(self, *choices):
        self.choices, self.calls = list(choices), []

    def __call__(self, url, headers, body, timeout_s):
        self.calls.append(body)
        choice = self.choices[min(len(self.calls), len(self.choices)) - 1]
        return {"choices": [choice], "usage": {"prompt_tokens": 1700, "completion_tokens": 900,
                                                "prompt_tokens_details": {"cached_tokens": 0}}}


TRUNCATED = {"finish_reason": "length", "message": {"content": '{"front_neck_depth_mm": 4'}}
NOT_JSON = {"finish_reason": "stop", "message": {"content": "sorry, I cannot"}}


@pytest.mark.parametrize(("choice", "text"), [(TRUNCATED, "cut off"), (NOT_JSON, "not valid JSON")])
def test_req_api_02_truncated_or_unparsable_answer_is_paid_once(tmp_path, photo, choice, text):  # noqa: F811
    chat = Scripted(choice)
    client = _client(tmp_path, chat=chat)
    with pytest.raises(ApiError, match=text):
        client.reason("p", [photo])
    paid = client.ledger_total_usd()
    assert paid > 0 and len(chat.calls) == 1
    again = _client(tmp_path, chat=chat)  # a rerun
    with pytest.raises(ApiError, match=text):  # judged again and rejected again, as live
        again.reason("p", [photo])
    assert len(chat.calls) == 1 and again.calls == 0 and again.ledger_total_usd() == paid
    with pytest.raises(ApiError):  # a different prompt is a new question and a new call
        _client(tmp_path, chat=chat).reason("p2", [photo])
    assert len(chat.calls) == 2


def test_req_api_02_rejected_answer_is_never_stored_as_a_good_one(tmp_path, photo):  # noqa: F811
    client = _client(tmp_path, chat=Scripted(TRUNCATED))
    with pytest.raises(ApiError):
        client.reason("p", [photo])
    files = [f.name for f in tmp_path.glob("real-*.json")]
    assert files and all(f.endswith(".rejected.json") for f in files)


def test_req_api_08_rejected_answer_replays_offline(tmp_path, photo):  # noqa: F811
    online = _client(tmp_path, chat=Scripted(TRUNCATED))
    with pytest.raises(ApiError):
        online.reason("p", [photo])
    settings = Settings(api_key="", mock=False, cache_dir=tmp_path, offline=True,
                        ledger_path=online.ledger_path, config=load_config())
    with pytest.raises(ApiError, match="cut off") as caught:
        MetaClient(settings).reason("p", [photo])
    assert not isinstance(caught.value, CacheMiss)


def test_req_api_02_schema_rejected_answer_is_replayed_not_repaid(tmp_path, photo):  # noqa: F811
    """A valid JSON answer with bad values was cached already. Keep it so: infer_panels rejects it."""
    bad = {"finish_reason": "stop", "message": {"content": json.dumps({"front_neck_depth_mm": -5})}}
    chat = Scripted(bad)
    first = _client(tmp_path, chat=chat).reason("p", [photo])
    second = _client(tmp_path, chat=chat).reason("p", [photo])
    assert first == second == {"front_neck_depth_mm": -5} and len(chat.calls) == 1


def test_req_api_02_infer_panels_truncated_twice_then_rerun_pays_nothing(tmp_path):
    """Attempt 1 and the retry (a different prompt) are two paid calls. A rerun pays for none."""
    d = tmp_path / "tee"
    synth.make_tee(synth.TEES[1], d, seed=1)
    draft = draft_spec(*measured(d))
    chat = Scripted(TRUNCATED)
    history: list[dict] = []
    infer_panels(_client(tmp_path / "c", chat=chat), photo_set(d), draft, meas(d), history=history)
    assert len(chat.calls) == 2 and [h["result"] for h in history] == ["retry", "fallback"]
    rerun = _client(tmp_path / "c", chat=chat)
    history = []
    infer_panels(rerun, photo_set(d), draft, meas(d), history=history)
    assert len(chat.calls) == 2 and rerun.calls == 0
    assert [h["result"] for h in history] == ["retry", "fallback"] and not any(h["new_call"] for h in history)


# ---- SAM path inside a running event loop (REQ-SEG-01) ------------------------------------------

def test_req_seg_01_segment_works_from_inside_a_running_event_loop(tmp_path, photo, monkeypatch):  # noqa: F811
    monkeypatch.setitem(sys.modules, "meta_sam_parser", _fake_parser(_result([_mask("0", W, H)])))
    client = _client(tmp_path)

    async def handler():  # like a FastAPI async endpoint
        return client.segment(photo, "t-shirt")

    mask = asyncio.run(handler())
    assert mask.shape == (H, W) and mask.all()

    async def answer():
        return 7

    assert client_mod._run_sync(answer) == 7  # a plain sync caller still works
