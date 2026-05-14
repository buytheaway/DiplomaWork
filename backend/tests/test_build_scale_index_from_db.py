from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.db.models import Base, Embedding, IndexSnapshot, Person
from app.services.index.index_manager import IndexManager
from scripts import build_scale_index_from_db as builder
from scripts.build_scale_index_from_db import BuildConfig, build_scale_index


def _sqlite_url(tmp_path) -> str:
    return f"sqlite+pysqlite:///{(tmp_path / 'scale_index.db').as_posix()}"


def _create_schema(database_url: str):
    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
    return engine


def _normalized(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def _insert_person_with_embeddings(
    engine,
    *,
    label: str,
    vectors: list[np.ndarray | bytes],
    pipeline: str = "pretrained",
    model_name: str = "dummy",
    status: str = "active",
    is_active: bool = True,
    dim: int = 4,
) -> uuid.UUID:
    session_factory = sessionmaker(bind=engine, future=True)
    person_id = uuid.uuid4()
    now = datetime.now(UTC)
    with session_factory() as db:
        person = Person(
            id=person_id,
            label=label,
            status=status,
            created_at=now,
            updated_at=now,
        )
        db.add(person)
        db.flush()
        for vector in vectors:
            payload = vector if isinstance(vector, bytes) else vector.astype(np.float32).tobytes()
            db.add(
                Embedding(
                    id=uuid.uuid4(),
                    person_id=person_id,
                    pipeline=pipeline,
                    model=model_name,
                    dim=dim,
                    vector=payload,
                    created_at=now,
                    is_active=is_active,
                )
            )
        db.commit()
    return person_id


def test_build_scale_index_dry_run_writes_no_files_or_snapshot(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)
    _insert_person_with_embeddings(
        engine,
        label="Dataset Identity 000000001",
        vectors=[_normalized([1.0, 0.0, 0.0, 0.0])],
    )
    index_path = tmp_path / "dry.faiss"

    result = build_scale_index(
        BuildConfig(
            database_url=database_url,
            index_type="flat",
            index_path=index_path,
            dry_run=True,
        )
    )

    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db:
        snapshot_count = db.execute(select(func.count()).select_from(IndexSnapshot)).scalar_one()

    assert result.dry_run is True
    assert result.selected_embeddings == 1
    assert result.vectors_added == 0
    assert not index_path.exists()
    assert not index_path.with_suffix(".faiss.map.json").exists()
    assert snapshot_count == 0


def test_build_scale_index_flat_writes_backend_compatible_snapshot(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)
    for person_index in range(3):
        vectors = [
            _normalized([1.0, person_index, sample_index, 0.0])
            for sample_index in range(4)
        ]
        _insert_person_with_embeddings(
            engine,
            label=f"Dataset Identity {person_index + 1:09d}",
            vectors=vectors,
        )
    index_path = tmp_path / "scale.faiss"

    result = build_scale_index(
        BuildConfig(
            database_url=database_url,
            index_type="flat",
            index_path=index_path,
            batch_size=5,
        )
    )

    map_path = index_path.with_suffix(".faiss.map.json")
    assert result.vectors_added == 12
    assert result.skipped_malformed == 0
    assert result.snapshot_id is not None
    assert index_path.read_bytes().startswith(b"ENC1")
    assert map_path.read_bytes().startswith(b"ENC1")

    settings = get_settings().model_copy(
        update={
            "index_path": str(index_path),
            "embedding_dim": 4,
            "index_type": "flat",
        }
    )
    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db:
        reloaded = IndexManager(settings)
        assert reloaded.load_latest_snapshot(db) is True
        assert reloaded.count() == 12
        results = reloaded.search(_normalized([1.0, 0.0, 0.0, 0.0]), k=1)

    assert results
    assert result.output_path == str(index_path.resolve())


def test_build_scale_index_skips_malformed_vectors(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)
    _insert_person_with_embeddings(
        engine,
        label="Dataset Identity 000000001",
        vectors=[
            _normalized([1.0, 0.0, 0.0, 0.0]),
            b"bad",
            _normalized([0.0, 1.0, 0.0, 0.0]),
        ],
    )

    result = build_scale_index(
        BuildConfig(
            database_url=database_url,
            index_type="flat",
            index_path=tmp_path / "malformed.faiss",
        )
    )

    assert result.selected_embeddings == 3
    assert result.vectors_added == 2
    assert result.skipped_malformed == 1


def test_build_scale_index_filters_pipeline_model_and_active_rows(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)
    vector = _normalized([1.0, 0.0, 0.0, 0.0])
    _insert_person_with_embeddings(engine, label="keep", vectors=[vector])
    _insert_person_with_embeddings(
        engine,
        label="wrong-pipeline",
        vectors=[vector],
        pipeline="custom",
    )
    _insert_person_with_embeddings(
        engine,
        label="wrong-model",
        vectors=[vector],
        model_name="other",
    )
    _insert_person_with_embeddings(
        engine,
        label="inactive-embedding",
        vectors=[vector],
        is_active=False,
    )
    _insert_person_with_embeddings(
        engine,
        label="deleted-person",
        vectors=[vector],
        status="deleted",
    )

    result = build_scale_index(
        BuildConfig(
            database_url=database_url,
            index_type="flat",
            index_path=tmp_path / "filtered.faiss",
        )
    )

    assert result.selected_embeddings == 1
    assert result.vectors_added == 1


def test_build_scale_index_requires_yes_for_large_matching_count(tmp_path, monkeypatch):
    database_url = _sqlite_url(tmp_path)
    index_path = tmp_path / "too_large.faiss"
    monkeypatch.setattr(builder, "count_matching_embeddings", lambda _db, _config: 100_001)
    monkeypatch.setattr(builder, "selected_embedding_dim", lambda _db, _config: 4)

    with pytest.raises(ValueError, match="Pass --yes"):
        build_scale_index(
            BuildConfig(
                database_url=database_url,
                index_type="flat",
                index_path=index_path,
                limit=1000,
            )
        )

    assert not index_path.exists()
