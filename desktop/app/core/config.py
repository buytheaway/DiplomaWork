from __future__ import annotations

import os
from dataclasses import dataclass

# On Windows "localhost" may resolve slowly or prefer IPv6,
# so the desktop defaults to 127.0.0.1.
_DEFAULT_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class DesktopSettings:
    base_url: str = os.getenv("API_BASE_URL", _DEFAULT_URL)
    api_key: str = os.getenv("API_KEY", "")
    admin_api_key: str = os.getenv("ADMIN_API_KEY", "")
    request_timeout_sec: int = int(os.getenv("API_TIMEOUT_SEC", "15"))
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    camera_frame_width: int = int(os.getenv("CAMERA_FRAME_WIDTH", "1280"))
    camera_frame_height: int = int(os.getenv("CAMERA_FRAME_HEIGHT", "720"))
    camera_preview_interval_ms: int = int(os.getenv("CAMERA_PREVIEW_INTERVAL_MS", "33"))
    live_scan_interval_ms: int = int(os.getenv("LIVE_SCAN_INTERVAL_MS", "150"))
    live_max_width: int = int(os.getenv("LIVE_MAX_WIDTH", "640"))
    live_jpeg_quality: int = int(os.getenv("LIVE_JPEG_QUALITY", "72"))
    custom_live_max_width: int = int(os.getenv("CUSTOM_LIVE_MAX_WIDTH", "960"))
    custom_live_jpeg_quality: int = int(os.getenv("CUSTOM_LIVE_JPEG_QUALITY", "82"))
