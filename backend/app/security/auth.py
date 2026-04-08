from __future__ import annotations

from typing import Literal

from app.core.config import Settings

ActorRole = Literal["operator", "admin"]


def classify_api_key(settings: Settings, api_key: str) -> ActorRole | None:
    if settings.admin_api_key and api_key == settings.admin_api_key:
        return "admin"
    if settings.api_key and api_key == settings.api_key:
        return "operator"
    return None

