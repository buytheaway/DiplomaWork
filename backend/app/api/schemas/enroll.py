from pydantic import BaseModel


class EnrollResponse(BaseModel):
    person_id: str
    embedding_id: str
    faces_detected: int
    model: str
    dim: int
