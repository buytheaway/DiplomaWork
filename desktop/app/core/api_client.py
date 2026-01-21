import json
from typing import Any

import requests

from app.core.config import DesktopSettings


class ApiClient:
    def __init__(self, settings: DesktopSettings | None = None) -> None:
        self.settings = settings or DesktopSettings()
        self.base_url = self.settings.base_url.rstrip("/")
        self.timeout = 15

    def enroll(self, image_path: str, label: str | None) -> dict[str, Any]:
        with open(image_path, "rb") as handle:
            files = {"file": handle}
            data = {"label": label} if label else {}
            response = requests.post(
                f"{self.base_url}/v1/enroll", files=files, data=data, timeout=self.timeout
            )
        response.raise_for_status()
        return response.json()

    def search(self, image_path: str, k: int) -> dict[str, Any]:
        with open(image_path, "rb") as handle:
            files = {"file": handle}
            response = requests.post(
                f"{self.base_url}/v1/search",
                params={"k": k},
                files=files,
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response.json()

    def index_stats(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/v1/index/stats", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def rebuild_index(self, index_type: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = {"index_type": index_type, "params": params}
        response = requests.post(
            f"{self.base_url}/v1/index/rebuild",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
