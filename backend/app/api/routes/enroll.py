from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_extractor, get_index_manager
from app.api.schemas.enroll import EnrollResponse
from app.core.config import settings
from app.services.embeddings.interface import (
    EmbeddingExtractor,
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.services.storage.repositories import EmbeddingRepo, PersonRepo

router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(
    file: UploadFile = File(...),
    label: str | None = Form(None),
    db: Session = Depends(get_db),
    extractor: EmbeddingExtractor = Depends(get_extractor),
    index_manager=Depends(get_index_manager),
) -> EnrollResponse:
    image_bytes = await file.read()
    try:
        embedding = extractor.extract_embedding(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (NoFaceDetectedError, MultipleFacesDetectedError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Embedding extraction failed") from exc

    person_repo = PersonRepo(db)
    embedding_repo = EmbeddingRepo(db)

    person = person_repo.create(label=label)
    db.flush()
    embedding_row = embedding_repo.create(
        person_id=person.id,
        model=extractor.model_name,
        dim=int(embedding.shape[0]),
        vector=embedding.tobytes(),
    )
    db.commit()
    db.refresh(person)
    db.refresh(embedding_row)

    try:
        index_manager.add_embedding(str(embedding_row.id), embedding)
        if settings.auto_save_index:
            index_manager.save_snapshot(db)
            db.commit()
    except Exception as exc:
        embedding_repo.deactivate(embedding_row.id)
        db.commit()
        raise HTTPException(status_code=500, detail="Index update failed") from exc

    return EnrollResponse(
        person_id=str(person.id),
        embedding_id=str(embedding_row.id),
        faces_detected=1,
        model=extractor.model_name,
        dim=int(embedding.shape[0]),
    )
