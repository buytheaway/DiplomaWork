from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

from app.core.config import DesktopSettings


class ApiError(Exception):
    def __init__(self, status_code: int | None, body: str | None) -> None:
        self.status_code = status_code
        self.body = body or ""
        super().__init__(_build_api_error_message(status_code, self.body))


def _extract_error_detail(body: str | None) -> str:
    if not body:
        return ""

    raw = body.strip()
    if not raw:
        return ""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail.strip()
        if isinstance(detail, list):
            parts: list[str] = []
            for item in detail[:3]:
                if not isinstance(item, dict):
                    continue
                loc = ".".join(str(part) for part in item.get("loc", []) if part != "body")
                msg = str(item.get("msg") or "").strip()
                if loc and msg:
                    parts.append(f"{loc}: {msg}")
                elif msg:
                    parts.append(msg)
            if parts:
                return "; ".join(parts)
    return raw


def _friendly_http_error(status_code: int | None, detail: str) -> str:
    if status_code is None:
        return detail or "HTTP error"

    normalized = detail.lower()
    if status_code in {401, 403}:
        if status_code == 403:
            return "This action requires the admin API key."
        return "Request was rejected. Check the API key used by the desktop and backend."
    if status_code == 400:
        if "invalid image" in normalized:
            return "The selected file is not a valid image."
        if "index update failed" in normalized:
            return "The search index could not be updated. Open Logs and rebuild the index."
    if status_code == 422:
        if "no face" in normalized:
            return "No face was detected. Use a clearer image or move closer to the camera."
        if "multiple faces" in normalized:
            return "Multiple faces were detected. Use a single-face image for enroll."
    if status_code >= 500:
        if "index" in normalized or "snapshot" in normalized or "map.json" in normalized:
            return "The search index is not ready. Open Logs and rebuild the index."
        return "The backend failed to process the request. Check the backend logs and try again."
    return detail or f"HTTP {status_code}"


def _build_api_error_message(status_code: int | None, body: str | None) -> str:
    detail = _extract_error_detail(body)
    friendly = _friendly_http_error(status_code, detail)
    if friendly:
        return friendly
    return f"HTTP {status_code}" if status_code is not None else "HTTP error"


def format_api_error(exc: Exception) -> str:
    if isinstance(exc, ApiError):
        return str(exc)
    if isinstance(exc, requests.ConnectionError):
        return "Could not reach the backend. Start it first and verify API_BASE_URL."
    if isinstance(exc, requests.Timeout):
        return "The backend did not respond in time."
    if isinstance(exc, requests.RequestException) and exc.response is not None:
        return _build_api_error_message(exc.response.status_code, exc.response.text)
    return str(exc)


class ApiClient:
    def __init__(self, settings: DesktopSettings | None = None) -> None:
        self.settings = settings or DesktopSettings()
        self.base_url = self.settings.base_url.rstrip("/")
        self.timeout = self.settings.request_timeout_sec

    def _build_headers(
        self,
        headers: dict[str, str] | None = None,
        *,
        admin: bool = False,
    ) -> dict[str, str]:
        merged = dict(headers or {})
        api_key = self.settings.admin_api_key if admin and self.settings.admin_api_key else self.settings.api_key
        if api_key:
            merged.setdefault("X-API-Key", api_key)
        return merged

    def _request_json(self, method: str, url: str, **kwargs: Any) -> Any:
        admin = bool(kwargs.pop("admin", False))
        headers = self._build_headers(kwargs.pop("headers", None), admin=admin)
        response = requests.request(
            method,
            url,
            timeout=self.timeout,
            headers=headers,
            **kwargs,
        )
        if not response.ok:
            raise ApiError(response.status_code, response.text)
        return response.json()

    @contextmanager
    def _image_file(self, image_source: str | bytes, filename: str = "capture.jpg"):
        if isinstance(image_source, str):
            with open(image_source, "rb") as handle:
                yield {"file": (Path(image_source).name, handle)}
            return
        yield {"file": (filename, image_source, "image/jpeg")}

    # enroll / search

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", f"{self.base_url}/v1/health")

    def enroll(
        self,
        image_path: str | bytes,
        label: str | None,
        pipeline: str = "both",
    ) -> dict[str, Any]:
        with self._image_file(image_path) as files:
            data = {"pipeline": pipeline}
            if label:
                data["label"] = label
            return self._request_json("POST", f"{self.base_url}/v1/enroll", files=files, data=data)

    def search(
        self,
        image_path: str | bytes,
        k: int,
        pipeline: str = "pretrained",
        *,
        source: str | None = None,
        multi_face: bool | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"k": k, "pipeline": pipeline}
        if source:
            params["source"] = source
        if multi_face is not None:
            params["multi_face"] = str(bool(multi_face)).lower()
        with self._image_file(image_path) as files:
            return self._request_json(
                "POST",
                f"{self.base_url}/v1/search",
                params=params,
                files=files,
            )

    # index

    def index_stats(self, pipeline: str = "pretrained") -> dict[str, Any]:
        return self._request_json(
            "GET",
            f"{self.base_url}/v1/index/stats",
            params={"pipeline": pipeline},
        )

    def database_stats(self) -> dict[str, Any]:
        return self._request_json("GET", f"{self.base_url}/v1/database/stats")

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
            admin=True,
        )

    # persons

    def get_person(self, person_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"{self.base_url}/v1/persons/{person_id}")

    def list_persons(
        self,
        limit: int = 200,
        offset: int = 0,
        q: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if q:
            params["q"] = q
        payload = self._request_json(
            "GET", f"{self.base_url}/v1/persons",
            params=params,
        )
        if isinstance(payload, list):
            return {
                "items": payload,
                "total": len(payload),
                "limit": limit,
                "offset": offset,
            }
        if isinstance(payload, dict):
            items = payload.get("items", [])
            return {
                "items": items if isinstance(items, list) else [],
                "total": int(payload.get("total", 0) or 0),
                "limit": int(payload.get("limit", limit) or limit),
                "offset": int(payload.get("offset", offset) or offset),
            }
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    def delete_person(self, person_id: str) -> dict[str, Any]:
        return self._request_json(
            "DELETE",
            f"{self.base_url}/v1/persons/{person_id}",
            admin=True,
        )
