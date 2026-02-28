"""Desktop client configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DesktopSettings:
    base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
