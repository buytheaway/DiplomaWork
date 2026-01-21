from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EmbeddingSummary(BaseModel):
    id: str
    model: str
    dim: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PersonResponse(BaseModel):
    id: str
    label: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    embeddings: list[EmbeddingSummary]

    model_config = ConfigDict(from_attributes=True)
