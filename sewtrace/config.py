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
    est_image_tokens: int = Field(ge=0)


class SamPrompts(_Strict):
    garment: str
    sleeve: str
    paper: str


class SamConfig(_Strict):
    min_score: float = Field(ge=0, le=1)
    prompts: SamPrompts


class ArmholeConfig(_Strict):
    shoulder_handle: float = Field(gt=0, lt=1)
    underarm_handle: float = Field(gt=0, lt=1)


class ContourConfig(_Strict):
    armhole: ArmholeConfig


class SeamConfig(_Strict):
    shoulder_tol_mm: float = Field(ge=0)
    side_tol_mm: float = Field(ge=0)
    underarm_tol_mm: float = Field(ge=0)
    cap_ease_min_mm: float
    cap_ease_max_mm: float
    max_adjust_calls: int = Field(ge=0)

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


class MarkerConfig(_Strict):
    tolerance_mm: float = Field(gt=0)


class AppConfig(_Strict):
    api: ApiConfig
    models: ModelsConfig
    reasoning: ReasoningConfig
    sam: SamConfig
    contour: ContourConfig
    seam: SeamConfig
    pattern: PatternConfig
    proof: ProofConfig
    marker: MarkerConfig

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
            config=config,
        )
