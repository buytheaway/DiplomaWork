"""Person response schema."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class EmbeddingSummary(BaseModel):
    id: str
    model: str
    dim: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_uuid(cls, v: object) -> str:
        return str(v) if isinstance(v, uuid.UUID) else v  # type: ignore[return-value]


class PersonListItem(BaseModel):
    # Легкая модель для списка (without embeddings)
    id: str
    label: str | None
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_uuid(cls, v: object) -> str:
        return str(v) if isinstance(v, uuid.UUID) else v  # type: ignore[return-value]


class PersonResponse(BaseModel):
    id: str
    label: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    embeddings: list[EmbeddingSummary]

    model_config = ConfigDict(from_attributes=True)

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_uuid(cls, v: object) -> str:
        return str(v) if isinstance(v, uuid.UUID) else v  # type: ignore[return-value]
