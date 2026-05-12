from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from time import monotonic
from typing import Literal

from fastapi import Request

from app.core.config import Settings
from app.security.auth import classify_api_key

RateLimitCategory = Literal["search", "enroll", "admin"]
WINDOW_SECONDS = 60.0


@dataclass(frozen=True)
class RateLimitRule:
    category: RateLimitCategory
    limit: int


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    category: RateLimitCategory
    limit: int
    remaining: int
    retry_after_seconds: int


class InMemoryRateLimiter:
    def __init__(
        self,
        *,
        window_seconds: float = WINDOW_SECONDS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._window_seconds = window_seconds
        self._clock = clock
        self._events: dict[tuple[RateLimitCategory, str], deque[float]] = defaultdict(deque)

    def check(
        self,
        *,
        identity: str,
        category: RateLimitCategory,
        limit: int,
    ) -> RateLimitDecision:
        if limit <= 0:
            return RateLimitDecision(
                allowed=True,
                category=category,
                limit=limit,
                remaining=0,
                retry_after_seconds=0,
            )

        now = self._clock()
        bucket = self._events[(category, identity)]
        cutoff = now - self._window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= limit:
            retry_after = max(1, ceil(bucket[0] + self._window_seconds - now))
            return RateLimitDecision(
                allowed=False,
                category=category,
                limit=limit,
                remaining=0,
                retry_after_seconds=retry_after,
            )

        bucket.append(now)
        return RateLimitDecision(
            allowed=True,
            category=category,
            limit=limit,
            remaining=max(0, limit - len(bucket)),
            retry_after_seconds=0,
        )


def rate_limit_rule_for_request(settings: Settings, method: str, path: str) -> RateLimitRule | None:
    prefix = settings.api_v1_prefix.rstrip("/")
    normalized_path = path.rstrip("/") or "/"
    method = method.upper()

    if method == "POST" and normalized_path == f"{prefix}/search":
        return RateLimitRule("search", settings.rate_limit_search_per_min)
    if method == "POST" and normalized_path == f"{prefix}/enroll":
        return RateLimitRule("enroll", settings.rate_limit_enroll_per_min)
    if method == "POST" and normalized_path == f"{prefix}/index/rebuild":
        return RateLimitRule("admin", settings.rate_limit_admin_per_min)
    if method == "DELETE" and normalized_path.startswith(f"{prefix}/persons/"):
        return RateLimitRule("admin", settings.rate_limit_admin_per_min)
    return None


def rate_limit_identity(settings: Settings, request: Request) -> str:
    api_key = request.headers.get("X-API-Key", "")
    if classify_api_key(settings, api_key) is not None:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        return f"api:{digest}"

    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host or 'unknown'}"
