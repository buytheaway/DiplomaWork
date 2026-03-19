"""Enrol response schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EnrollmentItem(BaseModel):
    pipeline: str
    embedding_id: str
    model: str
    dim: int


class EnrollResponse(BaseModel):
    person_id: str
    embedding_id: str
    faces_detected: int
    model: str
    dim: int
    pipeline: str | None = None
    available_pipelines: list[str] = Field(default_factory=list)
    enrollments: list[EnrollmentItem] = Field(default_factory=list)
