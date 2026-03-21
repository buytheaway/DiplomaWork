from __future__ import annotations

import os
from dataclasses import dataclass

# На Windows localhost иногда резолвится медленно или в IPv6,
# поэтому по умолчанию используем 127.0.0.1
_DEFAULT_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class DesktopSettings:
    base_url: str = os.getenv("API_BASE_URL", _DEFAULT_URL)
    request_timeout_sec: int = int(os.getenv("API_TIMEOUT_SEC", "15"))
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    camera_preview_interval_ms: int = int(os.getenv("CAMERA_PREVIEW_INTERVAL_MS", "33"))
    live_scan_interval_ms: int = int(os.getenv("LIVE_SCAN_INTERVAL_MS", "1200"))
