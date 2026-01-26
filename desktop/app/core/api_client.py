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
    if isinstance(exc, requests.RequestException) and exc.response is not None:
        body = exc.response.text or ""
        return f"HTTP {exc.response.status_code}: {body}".strip()
    return str(exc)


class ApiClient:
    def __init__(self, settings: DesktopSettings | None = None) -> None:
        self.settings = settings or DesktopSettings()
        self.base_url = self.settings.base_url.rstrip("/")
        self.timeout = 15

    def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        response = requests.request(method, url, timeout=self.timeout, **kwargs)
        if not response.ok:
            raise ApiError(response.status_code, response.text)
        return response.json()

    def enroll(self, image_path: str, label: str | None) -> dict[str, Any]:
        with open(image_path, "rb") as handle:
            files = {"file": handle}
            data = {"label": label} if label else {}
            return self._request_json("POST", f"{self.base_url}/v1/enroll", files=files, data=data)

    def search(self, image_path: str, k: int) -> dict[str, Any]:
        with open(image_path, "rb") as handle:
            files = {"file": handle}
            return self._request_json(
                "POST",
                f"{self.base_url}/v1/search",
                params={"k": k},
                files=files,
            )

    def index_stats(self) -> dict[str, Any]:
        return self._request_json("GET", f"{self.base_url}/v1/index/stats")

    def rebuild_index(self, index_type: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = {"index_type": index_type, "params": params}
        return self._request_json(
            "POST",
            f"{self.base_url}/v1/index/rebuild",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
