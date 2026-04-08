from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable
import uuid

from sqlalchemy import delete, select
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

    def soft_delete(self, person_id: str) -> bool:
        person = self.get(person_id)
        if person is None:
            return False
        person.status = "deleted"
        for emb in person.embeddings:
            emb.is_active = False
        return True


class EmbeddingRepo:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, person_id: str, model: str, dim: int, vector: bytes) -> Embedding:
        embedding = Embedding(
            person_id=person_id,
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

    def get_active_embeddings(self, model: str | None = None) -> list[Embedding]:
        stmt = select(Embedding).where(Embedding.is_active.is_(True))
        if model is not None:
            stmt = stmt.where(Embedding.model == model)
        return list(self.db.execute(stmt).scalars().all())

    def get_embeddings_with_person(self, embedding_ids: Iterable[str]) -> list[Embedding]:
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

    def get_latest(self) -> IndexSnapshot | None:
        stmt = select(IndexSnapshot).order_by(IndexSnapshot.created_at.desc())
        return self.db.execute(stmt).scalars().first()

    def get_latest_for_path(self, path: str) -> IndexSnapshot | None:
        stmt = (
            select(IndexSnapshot)
            .where(IndexSnapshot.path == path)
            .order_by(IndexSnapshot.created_at.desc())
        )
        return self.db.execute(stmt).scalars().first()


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
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        stmt = delete(AuditLog).where(AuditLog.created_at < cutoff)
        result = self.db.execute(stmt)
        return int(result.rowcount or 0)
