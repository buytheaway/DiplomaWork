from __future__ import annotations

import uuid
from collections.abc import Collection, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, joinedload

from app.db.models import AuditLog, Embedding, IndexSnapshot, Person
from app.security.crypto import encrypt_embedding_payload


class PersonRepo:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, label: str | None) -> Person:
        person = Person(label=label)
        self.db.add(person)
        return person

    def get(self, person_id: str) -> Person | None:
        return self.db.get(Person, person_id)

    def get_by_label(self, label: str) -> Person | None:
        stmt = select(Person).where(Person.label == label)
        return self.db.execute(stmt).scalars().first()

    def get_active_by_label(self, label: str) -> Person | None:
        stmt = (
            select(Person)
            .where(Person.label == label, Person.status == "active")
            .order_by(Person.created_at.desc())
        )
        return self.db.execute(stmt).scalars().first()

    def get_with_embeddings(self, person_id: str) -> Person | None:
        stmt = select(Person).where(Person.id == person_id).options(joinedload(Person.embeddings))
        return self.db.execute(stmt).scalars().first()

    def list_active(self, limit: int = 200, offset: int = 0) -> list[Person]:
        stmt = (
            select(Person)
            .where(Person.status == "active")
            .order_by(Person.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.execute(stmt).scalars().all())

    def count_active(self) -> int:
        stmt = select(func.count()).select_from(Person).where(Person.status == "active")
        return int(self.db.execute(stmt).scalar_one())

    def soft_delete(self, person_id: str) -> bool:
        person = self.get(person_id)
        if person is None:
            return False
        person.status = "deleted"
        for emb in person.embeddings:
            emb.is_active = False
        return True

    def soft_delete_duplicate_labels(self, label: str, keep_person_id: str) -> set[str]:
        stmt = (
            select(Person)
            .where(
                Person.label == label,
                Person.status == "active",
                Person.id != keep_person_id,
            )
            .options(joinedload(Person.embeddings))
        )
        duplicates = self.db.execute(stmt).unique().scalars().all()
        affected_pipelines: set[str] = set()
        for person in duplicates:
            person.status = "deleted"
            for emb in person.embeddings:
                if emb.is_active:
                    emb.is_active = False
                    affected_pipelines.add(emb.pipeline)
        return affected_pipelines


def _matches_snapshot_path(candidate: str, base_path: str) -> bool:
    candidate_path = Path(candidate)
    base = Path(base_path)
    if candidate_path == base:
        return True
    return (
        candidate_path.parent == base.parent
        and candidate_path.suffix == base.suffix
        and candidate_path.name.startswith(f"{base.stem}.")
    )


class EmbeddingRepo:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        person_id: str,
        pipeline: str,
        model: str,
        dim: int,
        vector: bytes,
    ) -> Embedding:
        embedding = Embedding(
            person_id=person_id,
            pipeline=pipeline,
            model=model,
            dim=dim,
            vector=encrypt_embedding_payload(vector),
        )
        self.db.add(embedding)
        return embedding

    def deactivate(self, embedding_id: str) -> None:
        embedding = self.db.get(Embedding, embedding_id)
        if embedding is not None:
            embedding.is_active = False

    def deactivate_active_for_person(
        self,
        person_id: str,
        pipelines: Collection[str] | None = None,
    ) -> set[str]:
        stmt = select(Embedding).where(
            Embedding.person_id == person_id,
            Embedding.is_active.is_(True),
        )
        if pipelines:
            stmt = stmt.where(Embedding.pipeline.in_(list(pipelines)))

        affected_pipelines: set[str] = set()
        for embedding in self.db.execute(stmt).scalars().all():
            embedding.is_active = False
            affected_pipelines.add(embedding.pipeline)
        return affected_pipelines

    def get_active_embeddings(
        self,
        model: str | None = None,
        pipeline: str | None = None,
    ) -> list[Embedding]:
        stmt = select(Embedding).where(Embedding.is_active.is_(True))
        if model is not None:
            stmt = stmt.where(Embedding.model == model)
        if pipeline is not None:
            stmt = stmt.where(Embedding.pipeline == pipeline)
        return list(self.db.execute(stmt).scalars().all())

    def get_embeddings_with_person(
        self,
        embedding_ids: Iterable[str],
        pipeline: str | None = None,
    ) -> list[Embedding]:
        if not embedding_ids:
            return []
        normalized_ids: list[uuid.UUID] = []
        for value in embedding_ids:
            if isinstance(value, uuid.UUID):
                normalized_ids.append(value)
            else:
                normalized_ids.append(uuid.UUID(str(value)))
        stmt = (
            select(Embedding)
            .where(Embedding.id.in_(normalized_ids))
            .options(joinedload(Embedding.person))
        )
        if pipeline is not None:
            stmt = stmt.where(Embedding.pipeline == pipeline)
        return list(self.db.execute(stmt).scalars().all())


class IndexSnapshotRepo:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, index_type: str, params: dict, path: str, embeddings_count: int) -> IndexSnapshot:
        snapshot = IndexSnapshot(
            index_type=index_type,
            params=params,
            path=path,
            embeddings_count=embeddings_count,
        )
        self.db.add(snapshot)
        return snapshot

    def delete(self, snapshot: IndexSnapshot) -> None:
        self.db.delete(snapshot)

    def get_latest(self) -> IndexSnapshot | None:
        stmt = select(IndexSnapshot).order_by(IndexSnapshot.created_at.desc())
        return self.db.execute(stmt).scalars().first()

    def get_latest_for_path(self, path: str) -> IndexSnapshot | None:
        snapshots = self.list_latest_for_path(path)
        return snapshots[0] if snapshots else None

    def list_latest_for_path(self, path: str) -> list[IndexSnapshot]:
        stmt = (
            select(IndexSnapshot)
            .order_by(IndexSnapshot.created_at.desc())
        )
        snapshots = self.db.execute(stmt).scalars().all()
        return [
            snapshot
            for snapshot in snapshots
            if _matches_snapshot_path(snapshot.path, path)
        ]


class AuditLogRepo:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        *,
        event_type: str,
        actor_role: str,
        route: str,
        status_code: int,
        details: dict,
    ) -> AuditLog:
        entry = AuditLog(
            event_type=event_type,
            actor_role=actor_role,
            route=route,
            status_code=status_code,
            details=details,
        )
        self.db.add(entry)
        return entry

    def prune_older_than(self, retention_days: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        stmt = delete(AuditLog).where(AuditLog.created_at < cutoff)
        result = self.db.execute(stmt)
        return int(result.rowcount or 0)
