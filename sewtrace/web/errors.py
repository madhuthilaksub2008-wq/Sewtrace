"""Pipeline errors to a short message for the page. The page never shows a traceback."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sewtrace.api.client import ApiError, BudgetExceeded, CacheMiss, SegmentationFailed
from sewtrace.pipeline import ContourFailed, PanelInferenceFailed, PhotoMissing
from sewtrace.reason.seams import SeamCheckFailed, is_measured_pair
from sewtrace.vision.marker import MarkerNotFound, MarkerRejected

log = logging.getLogger(__name__)

# First match wins, so subclasses come before ApiError.
_LABELS: tuple[tuple[type[BaseException], str], ...] = (
    (PhotoMissing, "A photo is missing or is not an image"),
    (MarkerNotFound, "Marker not found"),
    (MarkerRejected, "Marker rejected"),
    (SegmentationFailed, "Segmentation failed"),
    (ContourFailed, "Outline failed"),
    (SeamCheckFailed, "Seam check failed (no pattern was exported)"),
    (PanelInferenceFailed, "Panel inference failed"),
    (BudgetExceeded, "Budget limit reached (no call was made)"),
    (CacheMiss, "Offline mode: the answer is not in the cache"),
    (ApiError, "Meta API error"),
    (NotImplementedError, "Not available on this server"),
    (FileNotFoundError, "Mock mode cannot read this photo"),
    (ValueError, "Check failed"),
)


def describe_error(err: BaseException, scrub: Sequence[str] = ()) -> dict:
    """``{"kind", "title", "message"}`` for any exception. ``scrub`` strings (server folders,
    the API key) are removed from the message. An unknown error is logged with its traceback
    on the server and shown without one."""
    for cls, title in _LABELS:
        if isinstance(err, cls):
            message = str(err)
            break
    else:
        log.exception("unexpected error in a web job", exc_info=err)
        title, message = "Unexpected error", "Something failed on the server. Read the server log, then run again."
    for text in sorted(scrub, key=len, reverse=True):  # a long path before the folder that holds it
        if text:
            message = message.replace(text, "")
    out = {"kind": type(err).__name__, "title": title, "message": message}
    if isinstance(err, SeamCheckFailed):
        out["seam_log"] = err.seam_log
        # Shoulder or side pairs too far apart to pick one length: what each photo gave, so the page
        # can ask for the user's own tape measurement.
        out["review"] = [
            {"edge": s.a.partition(".")[2], "front_mm": round(s.len_a_mm, 2), "back_mm": round(s.len_b_mm, 2),
             "used_mm": None, "source": "none"}
            for s in err.failed if is_measured_pair(s)
        ]
    return out
