"""Settings: the one config file (config.toml) parsed into pydantic models, plus
environment values (API key, mock and offline switches, cache folder)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

# Printed reference marker (ArUco DICT_4X4_50, id 0). Side length of the black square.
MARKER_DICT = "DICT_4X4_50"
MARKER_ID = 0
MARKER_SIZE_MM = 100.0

# SPEC 4.3 item 6. "none" is absent on purpose: the API answers HTTP 400.
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiConfig(_Strict):
    base_url: str
    timeout_s: float = Field(gt=0)
    max_tries: int = Field(ge=1)
    backoff_s: float = Field(ge=0)
    budget_limit_usd: float = Field(gt=0)
    ledger_path: Path
    offline: bool = False


class ModelConfig(_Strict):
    """One Meta model: API ID and price. Token models and per-image models differ."""

    id: str
    price_per_image_usd: float | None = Field(default=None, ge=0)
    input_per_m_usd: float | None = Field(default=None, ge=0)
    cached_input_per_m_usd: float | None = Field(default=None, ge=0)
    output_per_m_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _has_one_price_kind(self) -> ModelConfig:
        per_image = self.price_per_image_usd is not None
        per_token = None not in (self.input_per_m_usd, self.cached_input_per_m_usd, self.output_per_m_usd)
        if per_image == per_token:
            raise ValueError(f"model {self.id}: give a per-image price or all three token prices")
        return self


class ModelsConfig(_Strict):
    sam: ModelConfig
    spark: ModelConfig
    spark_contributor: ModelConfig
    wonder: ModelConfig
    image: ModelConfig


class ReasoningConfig(_Strict):
    model: Literal["spark", "spark_contributor", "wonder"]
    effort: ReasoningEffort
    max_tokens: int = Field(gt=0)
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    prompt_cache_key: str = Field(default="sewtrace-v1", min_length=1)
    est_image_tokens: int = Field(ge=0)
    upload_jpeg_quality: int = Field(default=95, ge=1, le=100)  # own value, so tuning SAM does not change Muse cache keys
    max_upload_px: int = Field(default=1568, ge=64)  # long side of a photo sent to Muse; larger photos are downscaled


class SamPrompts(_Strict):
    garment: str
    sleeve: str
    paper: str


class SamConfig(_Strict):
    min_score: float = Field(ge=0, le=1)
    upload_jpeg_quality: int = Field(ge=1, le=100)  # SAM gets the pixels the pipeline reads, re-encoded
    max_upload_px: int = Field(default=2048, ge=64)  # long side of a photo sent to SAM; larger photos are downscaled
    box_overhang_tol_px: float = Field(ge=0)  # a SAM box may pass the image edge by this much (rounding)
    prompts: SamPrompts


class ArmholeConfig(_Strict):
    shoulder_handle: float = Field(gt=0, lt=1)
    underarm_handle: float = Field(gt=0, lt=1)


class MaskConfig(_Strict):
    cleanup_mm: float = Field(gt=0)


class JointsConfig(_Strict):
    corner_window_mm: float = Field(gt=0)
    underarm_min_deg: float = Field(gt=0, lt=180)
    corner_min_deg: float = Field(gt=0, lt=180)
    pair_tol_mm: float = Field(gt=0)
    shoulder_symmetry_mm: float = Field(gt=0)
    shoulder_plateau_mm: float = Field(gt=0)
    rounding_deg: float = Field(gt=0, lt=180)
    rounding_margin_mm: float = Field(ge=0)


class ContourConfig(_Strict):
    armhole: ArmholeConfig
    mask: MaskConfig
    joints: JointsConfig


class InferenceConfig(_Strict):
    measured_tol_mm: float = Field(gt=0)
    front_neck_depth_min_mm: float = Field(gt=0)
    front_neck_depth_max_mm: float = Field(gt=0)
    back_neck_depth_min_mm: float = Field(gt=0)
    back_neck_depth_max_mm: float = Field(gt=0)
    front_minus_back_min_mm: float = Field(gt=0)
    fallback_front_minus_back_mm: float = Field(gt=0)

    @model_validator(mode="after")
    def _ranges(self) -> InferenceConfig:
        if (self.front_neck_depth_min_mm > self.front_neck_depth_max_mm
                or self.back_neck_depth_min_mm > self.back_neck_depth_max_mm):
            raise ValueError("a neck depth minimum is more than its maximum")
        if self.fallback_front_minus_back_mm < self.front_minus_back_min_mm:
            raise ValueError("fallback_front_minus_back_mm is less than front_minus_back_min_mm")
        return self


class SeamConfig(_Strict):
    shoulder_tol_mm: float = Field(ge=0)
    side_tol_mm: float = Field(ge=0)
    underarm_tol_mm: float = Field(ge=0)
    cap_ease_min_mm: float
    cap_ease_max_mm: float
    max_adjust_calls: int = Field(ge=0)
    measured_pair_max_fix_mm: float = Field(ge=0)

    @model_validator(mode="after")
    def _ease_range(self) -> SeamConfig:
        if self.cap_ease_min_mm > self.cap_ease_max_mm:
            raise ValueError("cap_ease_min_mm is more than cap_ease_max_mm")
        return self


class PatternConfig(_Strict):
    seam_allowance_mm: float = Field(ge=0)
    neckband_allowance_mm: float = Field(ge=0)
    chest_drop_mm: float = Field(ge=0)


class ProofConfig(_Strict):
    gap_tolerance_mm: float = Field(gt=0)
    fold_straight_tol_mm: float = Field(gt=0)
    fold_smooth_mm: float = Field(gt=0)
    fold_min_inset_mm: float = Field(gt=0)
    fold_exclude_mm: float = Field(gt=0)
    centre_margin_mm: float = Field(ge=0)
    neck_gap_tolerance_mm: float = Field(gt=0)


class MeasureConfig(_Strict):
    neck_fit_mm: float = Field(gt=0)
    neck_search_mm: float = Field(gt=0)


class MarkerConfig(_Strict):
    tolerance_mm: float = Field(gt=0)
    min_grid_found: float = Field(gt=0, le=1)
    error_percentile: float = Field(gt=0, le=100)


class GroundTruthConfig(_Strict):
    repeat_tol_mm: float = Field(gt=0)


class WebConfig(_Strict):
    """The test web page (sewtrace serve). Not a pipeline value."""

    max_upload_mb: float = Field(gt=0)  # per photo
    job_retention_min: float = Field(gt=0)  # a finished job folder is deleted after this age
    port: int = Field(ge=1, le=65535)
    max_pixels: int = Field(gt=0)  # per photo, from the file header
    auth_fail_delay_s: float = Field(ge=0)  # wait before a 401 for a wrong password


class AppConfig(_Strict):
    api: ApiConfig
    models: ModelsConfig
    reasoning: ReasoningConfig
    sam: SamConfig
    contour: ContourConfig
    inference: InferenceConfig
    seam: SeamConfig
    pattern: PatternConfig
    proof: ProofConfig
    measure: MeasureConfig
    marker: MarkerConfig
    ground_truth: GroundTruthConfig
    web: WebConfig

    @property
    def reasoning_model(self) -> ModelConfig:
        return getattr(self.models, self.reasoning.model)


def load_config(path: str | Path | None = None) -> AppConfig:
    path = Path(path or os.environ.get("SEWTRACE_CONFIG") or DEFAULT_CONFIG_PATH)
    with path.open("rb") as f:
        return AppConfig.model_validate(tomllib.load(f))


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Config for pure functions that have no Settings in reach."""
    return load_config()


def resolve_ledger_path(config: AppConfig) -> Path:
    """The ledger lives under the repo root, never under SEWTRACE_CACHE_DIR (REQ-API-04).
    No env override: an override is one more way to start a second ledger at $0."""
    return ROOT / config.api.ledger_path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    api_key: str
    mock: bool
    cache_dir: Path
    offline: bool = False
    ledger_path: Path | None = None  # None: api.ledger_path in config, under the repo root
    config: AppConfig = field(default_factory=get_config)

    @classmethod
    def load(cls) -> Settings:
        _load_dotenv(ROOT / ".env")
        config = load_config()
        api_key = os.environ.get("META_API_KEY", "")
        offline = config.api.offline or os.environ.get("SEWTRACE_OFFLINE") == "1"
        mock_env = os.environ.get("SEWTRACE_MOCK")
        if offline:
            # Offline replays whatever the cache holds. Real answers are replayed
            # unless mock is asked for by name.
            mock = mock_env == "1"
        else:
            # No key means mocked, whatever SEWTRACE_MOCK says.
            mock = (mock_env or "1") == "1" or not api_key
        return cls(
            api_key=api_key,
            mock=mock,
            cache_dir=Path(os.environ.get("SEWTRACE_CACHE_DIR", ROOT / "cache")),
            offline=offline,
            ledger_path=resolve_ledger_path(config),
            config=config,
        )
