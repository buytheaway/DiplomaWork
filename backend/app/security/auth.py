from __future__ import annotations

import secrets
from typing import Literal

from app.core.config import Settings

ActorRole = Literal["operator", "admin"]


def _matches_secret(candidate: str, expected: str) -> bool:
    return bool(candidate and expected) and secrets.compare_digest(candidate, expected)


def classify_api_key(settings: Settings, api_key: str) -> ActorRole | None:
    if _matches_secret(api_key, settings.admin_api_key):
        return "admin"
    if _matches_secret(api_key, settings.api_key):
        return "operator"
    return None
