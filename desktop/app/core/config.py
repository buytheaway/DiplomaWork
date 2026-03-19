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
