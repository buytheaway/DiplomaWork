from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class ActivityEvent:
    timestamp: datetime
    category: str
    message: str
    severity: str = "INFO"
    details: str = ""
    meta: dict[str, str] = field(default_factory=dict)


_EVENTS: deque[ActivityEvent] = deque(maxlen=250)


def record_event(
    category: str,
    message: str,
    *,
    severity: str = "INFO",
    details: str = "",
    meta: dict[str, str] | None = None,
) -> None:
    _EVENTS.appendleft(
        ActivityEvent(
            timestamp=datetime.now(),
            category=category,
            message=message,
            severity=severity.upper(),
            details=details,
            meta=meta or {},
        )
    )


def recent_events(limit: int = 50) -> list[ActivityEvent]:
    return list(_EVENTS)[:limit]


def export_events_csv(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "severity", "category", "message", "details", "meta"])
        for event in recent_events(limit=250):
            writer.writerow(
                [
                    event.timestamp.isoformat(timespec="seconds"),
                    event.severity,
                    event.category,
                    event.message,
                    event.details,
                    "; ".join(f"{key}={value}" for key, value in event.meta.items()),
                ]
            )
    return out_path
