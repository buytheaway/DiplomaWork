from pydantic import BaseModel


class SearchResult(BaseModel):
    person_id: str
    embedding_id: str
    score: float
    distance: float
    label: str | None = None


class SearchResponse(BaseModel):
    k: int
    model: str
    results: list[SearchResult]
