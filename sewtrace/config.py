"""Settings from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Printed reference marker (ArUco DICT_4X4_50, id 0). Side length of the black square.
MARKER_DICT = "DICT_4X4_50"
MARKER_ID = 0
MARKER_SIZE_MM = 100.0

# Sewability tolerance (parent page section 1, step 5).
SEAM_TOLERANCE_MM = 3.0
MAX_ADJUST_CALLS = 3


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
    api_base_url: str
    mock: bool
    reasoning_model: str
    cache_dir: Path

    @classmethod
    def load(cls) -> "Settings":
        _load_dotenv(ROOT / ".env")
        api_key = os.environ.get("META_API_KEY", "")
        return cls(
            api_key=api_key,
            api_base_url=os.environ.get("META_API_BASE_URL", ""),
            # No key means mocked, whatever SEWTRACE_MOCK says.
            mock=os.environ.get("SEWTRACE_MOCK", "1") == "1" or not api_key,
            reasoning_model=os.environ.get("SEWTRACE_REASONING_MODEL", "spark"),
            cache_dir=Path(os.environ.get("SEWTRACE_CACHE_DIR", ROOT / "cache")),
        )
