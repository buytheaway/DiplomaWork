"""Enrol response schema."""

from __future__ import annotations

from pydantic import BaseModel


class EnrollResponse(BaseModel):
    person_id: str
    embedding_id: str
    faces_detected: int
    model: str
    dim: int
