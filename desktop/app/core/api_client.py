from __future__ import annotations

import json
from typing import Any

import requests

from app.core.config import DesktopSettings


class ApiError(Exception):
    def __init__(self, status_code: int | None, body: str | None) -> None:
        self.status_code = status_code
        self.body = body or ""
        message = f"HTTP {status_code}" if status_code is not None else "HTTP error"
        if self.body:
            message = f"{message}: {self.body}"
        super().__init__(message)


def format_api_error(exc: Exception) -> str:
    if isinstance(exc, ApiError):
        return str(exc)
    # Бэкенд не запущен или недоступен
    if isinstance(exc, requests.ConnectionError):
        return "Не удалось подключиться к бэкенду. Убедитесь что сервер запущен."
    if isinstance(exc, requests.Timeout):
        return "Таймаут запроса к бэкенду."
    if isinstance(exc, requests.RequestException) and exc.response is not None:
        body = exc.response.text or ""
        return f"HTTP {exc.response.status_code}: {body}".strip()
    return str(exc)


class ApiClient:
    def __init__(self, settings: DesktopSettings | None = None) -> None:
        self.settings = settings or DesktopSettings()
        self.base_url = self.settings.base_url.rstrip("/")
        self.timeout = self.settings.request_timeout_sec

    def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        response = requests.request(method, url, timeout=self.timeout, **kwargs)
        if not response.ok:
            raise ApiError(response.status_code, response.text)
        return response.json()

    # ── enroll / search ──────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", f"{self.base_url}/v1/health")

    def enroll(
        self,
        image_path: str,
        label: str | None,
        pipeline: str = "pretrained",
    ) -> dict[str, Any]:
        with open(image_path, "rb") as handle:
            files = {"file": handle}
            data = {"pipeline": pipeline}
            if label:
                data["label"] = label
            return self._request_json("POST", f"{self.base_url}/v1/enroll", files=files, data=data)

    def search(self, image_path: str, k: int, pipeline: str = "pretrained") -> dict[str, Any]:
        with open(image_path, "rb") as handle:
            files = {"file": handle}
            return self._request_json(
                "POST",
                f"{self.base_url}/v1/search",
                params={"k": k, "pipeline": pipeline},
                files=files,
            )

    def search_compare(self, image_path: str, k: int) -> dict[str, Any]:
        with open(image_path, "rb") as handle:
            files = {"file": handle}
            return self._request_json(
                "POST",
                f"{self.base_url}/v1/search/compare",
                params={"k": k},
                files=files,
            )

    # ── index ────────────────────────────────────────────────────────────

    def index_stats(self, pipeline: str = "pretrained") -> dict[str, Any]:
        return self._request_json(
            "GET",
            f"{self.base_url}/v1/index/stats",
            params={"pipeline": pipeline},
        )

    def rebuild_index(
        self,
        index_type: str,
        params: dict[str, Any],
        pipeline: str = "pretrained",
    ) -> dict[str, Any]:
        payload = {"index_type": index_type, "params": params, "pipeline": pipeline}
        return self._request_json(
            "POST",
            f"{self.base_url}/v1/index/rebuild",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )

    # ── persons ──────────────────────────────────────────────────────────

    def get_person(self, person_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"{self.base_url}/v1/persons/{person_id}")

    def list_persons(self, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        return self._request_json(
            "GET", f"{self.base_url}/v1/persons",
            params={"limit": limit, "offset": offset},
        )

    def delete_person(self, person_id: str) -> dict[str, Any]:
        return self._request_json("DELETE", f"{self.base_url}/v1/persons/{person_id}")
