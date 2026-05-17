"""Person management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.api.schemas.persons import PersonListItem, PersonListResponse, PersonResponse
from app.services.storage.audit import record_audit_event
from app.services.storage.repositories import PersonRepo

router = APIRouter()


@router.get("/persons", response_model=PersonListResponse)
def list_persons(
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
    q: str | None = Query(None, max_length=100),
    db: Session = Depends(get_db),
) -> PersonListResponse:
    repo = PersonRepo(db)
    persons = repo.list_active(limit=limit, offset=offset, q=q)
    return PersonListResponse(
        items=[PersonListItem.model_validate(p) for p in persons],
        total=repo.count_active(q=q),
        limit=limit,
        offset=offset,
    )


@router.get("/persons/{person_id}", response_model=PersonResponse)
def get_person(person_id: str, db: Session = Depends(get_db)) -> PersonResponse:
    repo = PersonRepo(db)
    person = repo.get_with_embeddings(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return PersonResponse.model_validate(person)


@router.delete("/persons/{person_id}")
def delete_person(
    request: Request,
    person_id: str,
    db: Session = Depends(get_db),
    _admin: str = Depends(require_admin),
) -> dict[str, object]:
    repo = PersonRepo(db)
    person = repo.get_with_embeddings(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    active_pipelines = {emb.pipeline for emb in person.embeddings if emb.is_active}
    repo.soft_delete(person_id)
    db.commit()
    record_audit_event(
        db,
        request,
        event_type="delete_person",
        status_code=200,
        details={
            "person_id": person_id,
            "affected_pipelines": sorted(active_pipelines),
            "index_update": "deferred",
        },
    )
    return {
        "status": "deleted",
        "person_id": person_id,
        "affected_pipelines": sorted(active_pipelines),
        "index_update": "deferred",
    }
