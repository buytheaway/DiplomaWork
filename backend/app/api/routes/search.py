from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_extractor, get_index_manager
from app.api.schemas.search import SearchResponse, SearchResult
from app.services.embeddings.interface import (
    EmbeddingExtractor,
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.services.storage.repositories import EmbeddingRepo

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=100),
    db: Session = Depends(get_db),
    extractor: EmbeddingExtractor = Depends(get_extractor),
    index_manager=Depends(get_index_manager),
) -> SearchResponse:
    image_bytes = await file.read()
    try:
        embedding = extractor.extract_embedding(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (NoFaceDetectedError, MultipleFacesDetectedError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Embedding extraction failed") from exc

    if index_manager.count() == 0:
        return SearchResponse(k=k, model=extractor.model_name, results=[])

    matches = index_manager.search(embedding, k=k)
    embedding_ids = [match.embedding_id for match in matches]

    repo = EmbeddingRepo(db)
    rows = repo.get_embeddings_with_person(embedding_ids)
    lookup = {str(row.id): row for row in rows}

    results: list[SearchResult] = []
    for match in matches:
        row = lookup.get(match.embedding_id)
        if row is None or row.person is None:
            continue
        results.append(
            SearchResult(
                person_id=str(row.person.id),
                embedding_id=match.embedding_id,
                score=match.score,
                distance=match.distance,
                label=row.person.label,
            )
        )

    return SearchResponse(k=k, model=extractor.model_name, results=results)
