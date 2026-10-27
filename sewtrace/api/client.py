"""Meta Model API client: SAM 3.1 segmentation and Muse reasoning.

Every call is cached by sha256(model + prompt + image bytes), so the same
photo is never paid for twice (credit rule, parent page section 11). Real
calls append a row to cache/spend.csv.

Mock mode (no API key, or SEWTRACE_MOCK=1):
  * segment() returns the ground-truth region mask written next to a synthetic
    photo (``<photo>.masks/<prompt>.png``), lightly jittered to look like a
    model edge. Without a sidecar it falls back to a backdrop colour threshold.
  * reason() returns the caller's ``mock`` callable, so each module supplies a
    deterministic stand-in for the model's answer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sewtrace.config import Settings

SAM_MODEL = "sam-3.1"
REASONING_MODELS = {"spark": "muse-spark-1.3", "wonder": "muse-wonder"}


class MetaClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.cache_dir = self.settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0  # real or mocked calls that missed the cache

    # ---- public API -------------------------------------------------------

    def segment(self, image_path: str | Path, prompt: str) -> np.ndarray:
        """Boolean HxW mask of ``prompt`` (e.g. "garment", "body", "sleeve", "paper")."""
        image_path = Path(image_path)
        key = self._key(SAM_MODEL, prompt, [image_path])
        cached = self.cache_dir / f"{key}.png"
        if cached.exists():
            return cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE) > 127
        self.calls += 1
        if self.settings.mock:
            mask = _mock_segment(image_path, prompt)
        else:
            mask = self._call_sam(image_path, prompt)
        cv2.imwrite(str(cached), mask.astype(np.uint8) * 255)
        return mask

    def reason(
        self,
        prompt: str,
        image_paths: Sequence[str | Path] = (),
        mock: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Structured JSON answer from the reasoning model."""
        model = REASONING_MODELS[self.settings.reasoning_model]
        key = self._key(model, prompt, [Path(p) for p in image_paths])
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text())
        self.calls += 1
        if self.settings.mock:
            if mock is None:
                raise RuntimeError("reason() called in mock mode without a mock answer")
            answer = mock()
        else:
            answer = self._call_reasoning(model, prompt, image_paths)
        cached.write_text(json.dumps(answer))
        return answer

    # ---- real endpoints (Day 1 tests T8 / T9) -----------------------------

    def _call_sam(self, image_path: Path, prompt: str) -> np.ndarray:
        raise NotImplementedError(
            "Real SAM 3.1 call not wired yet (Day 1 test T8). Run with SEWTRACE_MOCK=1."
        )

    def _call_reasoning(self, model: str, prompt: str, image_paths: Sequence) -> dict:
        raise NotImplementedError(
            "Real Muse call not wired yet (Day 1 test T9). Run with SEWTRACE_MOCK=1."
        )

    def log_spend(self, model: str, input_tokens: int, output_tokens: int, usd: float) -> None:
        path = self.cache_dir / "spend.csv"
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "model", "input_tokens", "output_tokens", "usd"])
            w.writerow([int(time.time()), model, input_tokens, output_tokens, f"{usd:.4f}"])

    # ---- helpers ----------------------------------------------------------

    def _key(self, model: str, prompt: str, images: Sequence[Path]) -> str:
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(b"\0")
        h.update(prompt.encode())
        for p in images:
            h.update(b"\0")
            h.update(p.read_bytes())
        return h.hexdigest()


def _mock_segment(image_path: Path, prompt: str) -> np.ndarray:
    sidecar = image_path.with_suffix(".masks") / f"{prompt}.png"
    if sidecar.exists():
        mask = cv2.imread(str(sidecar), cv2.IMREAD_GRAYSCALE) > 127
        return _jitter_edges(mask, seed=int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16))
    if prompt != "garment":
        raise FileNotFoundError(f"mock SAM has no '{prompt}' mask for {image_path}")
    return _threshold_largest(cv2.imread(str(image_path)))


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
