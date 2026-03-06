"""Person management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.schemas.persons import PersonListItem, PersonResponse
from app.services.storage.repositories import PersonRepo

router = APIRouter()


@router.get("/persons", response_model=list[PersonListItem])
def list_persons(
    limit: int = 200,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[PersonListItem]:
    repo = PersonRepo(db)
    persons = repo.list_active(limit=limit, offset=offset)
    return [PersonListItem.model_validate(p) for p in persons]


@router.get("/persons/{person_id}", response_model=PersonResponse)
def get_person(person_id: str, db: Session = Depends(get_db)) -> PersonResponse:
    repo = PersonRepo(db)
    person = repo.get_with_embeddings(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return PersonResponse.model_validate(person)


@router.delete("/persons/{person_id}")
def delete_person(person_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    repo = PersonRepo(db)
    deleted = repo.soft_delete(person_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Person not found")
    db.commit()
    return {"status": "deleted", "person_id": person_id}
