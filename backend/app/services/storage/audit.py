from __future__ import annotations

import logging

from fastapi import Request
from sqlalchemy.orm import Session

from app.services.storage.repositories import AuditLogRepo

logger = logging.getLogger(__name__)


def record_audit_event(
    db: Session,
    request: Request,
    *,
    event_type: str,
    status_code: int,
    details: dict,
) -> None:
    try:
        actor_role = getattr(request.state, "actor_role", "system")
        AuditLogRepo(db).create(
            event_type=event_type,
            actor_role=actor_role,
            route=request.url.path,
            status_code=status_code,
            details=details,
        )
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("Failed to write audit event %s: %s", event_type, exc)
