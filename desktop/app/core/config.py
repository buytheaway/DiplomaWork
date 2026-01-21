from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DesktopSettings:
    base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
