"""Import real-image benchmark embeddings into a separate database.

This script takes the output produced by ``benchmark_real_image_embeddings.py``
and materializes it as backend-compatible ``persons`` and ``embeddings`` rows.
It does not extract images again and does not mix embedding spaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.models import Base, Embedding, Person  # noqa: E402
from app.security.crypto import encrypt_embedding_payload  # noqa: E402

DEFAULT_PIPELINE = "custom"
DEFAULT_MODEL_NAME = "torch_insightface_iresnet100"
DEFAULT_LABEL_PREFIX = "VGGFace2 Identity"
DEFAULT_BATCH_SIZE = 10_000
CONFIRM_THRESHOLD = 100_000
VECTOR_DIM = 512
VECTOR_DTYPE = np.float32
SUPPORTED_MAPPING_FIELDS = ("row_id", "identity", "image_path")


@dataclass(frozen=True)
class ImportConfig:
    database_url: str
    mapping_path: Path
    memmap_path: Path
    pipeline: str = DEFAULT_PIPELINE
    model_name: str = DEFAULT_MODEL_NAME
    label_prefix: str = DEFAULT_LABEL_PREFIX
    batch_size: int = DEFAULT_BATCH_SIZE
    limit: int | None = None
    create_schema: bool = False
    replace_existing: bool = False
    dry_run: bool = False
    yes: bool = False


@dataclass(frozen=True)
class MappingStats:
    rows: int
    unique_identities: int
    max_row_id: int
    examples: list[str]


@dataclass(frozen=True)
class ImportResult:
    persons_inserted: int
    embeddings_inserted: int
    elapsed_s: float
    dry_run: bool


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _resolve_sqlite_path(database: str) -> Path | None:
    url = make_url(database)
    if not url.drivername.startswith("sqlite"):
        return None
    if not url.database or url.database == ":memory:":
        return None
    path = Path(url.database)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def unsafe_database_reason(database_url: str) -> str | None:
    try:
        sqlite_path = _resolve_sqlite_path(database_url)
    except Exception as exc:  # noqa: BLE001
        return f"Invalid database URL: {exc}"
    if sqlite_path is None:
        return None

    repo_root = ROOT.resolve()
    backend_root = BACKEND_ROOT.resolve()
    backend_data = (BACKEND_ROOT / "data").resolve()
    for forbidden_root, label in (
        (backend_root, "backend/"),
        (backend_data, "backend/data/"),
    ):
        try:
            sqlite_path.relative_to(forbidden_root)
        except ValueError:
            continue
        return f"Refusing to import into SQLite database inside {label}: {sqlite_path}"

    try:
        sqlite_path.relative_to(repo_root)
    except ValueError:
        return None
    if sqlite_path.name.startswith("backend_local") or sqlite_path.suffix in {".db", ".sqlite"}:
        return f"Refusing to import into repo-local demo SQLite database: {sqlite_path}"
    return None


def memmap_vector_count(memmap_path: Path) -> int:
    bytes_per_vector = VECTOR_DIM * np.dtype(VECTOR_DTYPE).itemsize
    size = memmap_path.stat().st_size
    if size % bytes_per_vector != 0:
        raise ValueError(
            f"Memmap size is not divisible by {bytes_per_vector} bytes per vector: {memmap_path}"
        )
    return size // bytes_per_vector


def person_uuid(config: ImportConfig, identity: str) -> uuid.UUID:
    source = f"real-image-db-person:{config.pipeline}:{config.model_name}:{config.label_prefix}:{identity}"
    return uuid.uuid5(uuid.NAMESPACE_URL, source)


def embedding_uuid(config: ImportConfig, row_id: int, identity: str) -> uuid.UUID:
    source = f"real-image-db-embedding:{config.pipeline}:{config.model_name}:{row_id}:{identity}"
    return uuid.uuid5(uuid.NAMESPACE_URL, source)


def label_for(config: ImportConfig, identity: str) -> str:
    return f"{config.label_prefix} {identity}"


def iter_mapping_rows(mapping_path: Path, limit: int | None = None):
    emitted = 0
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(SUPPORTED_MAPPING_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Mapping CSV is missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            if limit is not None and emitted >= limit:
                return
            row_id_raw = (row.get("row_id") or "").strip()
            identity = (row.get("identity") or "").strip()
            if not row_id_raw or not identity:
                continue
            yield int(row_id_raw), identity
            emitted += 1


def inspect_mapping(config: ImportConfig) -> MappingStats:
    identities: set[str] = set()
    examples: list[str] = []
    rows = 0
    max_row_id = -1
    for row_id, identity in iter_mapping_rows(config.mapping_path, config.limit):
        rows += 1
        max_row_id = max(max_row_id, row_id)
        identities.add(identity)
        if len(examples) < 5:
            examples.append(label_for(config, identity))
    return MappingStats(
        rows=rows,
        unique_identities=len(identities),
        max_row_id=max_row_id,
        examples=examples,
    )


def validate_config(config: ImportConfig) -> MappingStats:
    reason = unsafe_database_reason(config.database_url)
    if reason is not None:
        raise ValueError(reason)
    if not config.pipeline.strip():
        raise ValueError("--pipeline must not be empty")
    if not config.model_name.strip():
        raise ValueError("--model-name must not be empty")
    if not config.label_prefix.strip():
        raise ValueError("--label-prefix must not be empty")
    if not config.mapping_path.exists():
        raise ValueError(f"Mapping CSV not found: {config.mapping_path}")
    if not config.memmap_path.exists():
        raise ValueError(f"Memmap file not found: {config.memmap_path}")

    vector_count = memmap_vector_count(config.memmap_path)
    stats = inspect_mapping(config)
    if stats.rows == 0:
        raise ValueError("Mapping CSV contains no importable rows")
    if stats.max_row_id >= vector_count:
        raise ValueError(
            f"Mapping row_id {stats.max_row_id} exceeds memmap vector count {vector_count}"
        )
    if stats.rows > CONFIRM_THRESHOLD and not config.yes:
        raise ValueError(
            f"Import would create {stats.rows} embeddings. Pass --yes to confirm "
            "this intentionally targets a separate scale database."
        )
    return stats


def existing_counts(db: Session, config: ImportConfig) -> tuple[int, int]:
    person_filter = Person.label.like(f"{config.label_prefix} %")
    person_ids = (
        select(Person.id)
        .where(person_filter)
        .subquery()
    )
    persons_count = db.execute(select(func.count()).select_from(person_ids)).scalar_one()
    embeddings_count = db.execute(
        select(func.count())
        .select_from(Embedding)
        .where(
            Embedding.person_id.in_(select(person_ids.c.id)),
            Embedding.pipeline == config.pipeline,
            Embedding.model == config.model_name,
        )
    ).scalar_one()
    return int(persons_count), int(embeddings_count)


def delete_existing(db: Session, config: ImportConfig) -> tuple[int, int]:
    person_ids = list(
        db.execute(
            select(Person.id).where(Person.label.like(f"{config.label_prefix} %"))
        ).scalars()
    )
    if not person_ids:
        return 0, 0
    embeddings_deleted = db.execute(
        delete(Embedding).where(
            Embedding.person_id.in_(person_ids),
            Embedding.pipeline == config.pipeline,
            Embedding.model == config.model_name,
        )
    ).rowcount
    persons_deleted = db.execute(delete(Person).where(Person.id.in_(person_ids))).rowcount
    db.commit()
    return int(persons_deleted or 0), int(embeddings_deleted or 0)


def insert_persons(db: Session, config: ImportConfig) -> int:
    identities = sorted({identity for _row_id, identity in iter_mapping_rows(config.mapping_path, config.limit)})
    now = datetime.now(UTC)
    rows = [
        {
            "id": person_uuid(config, identity),
            "label": label_for(config, identity),
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }
        for identity in identities
    ]
    if rows:
        db.execute(Person.__table__.insert(), rows)
        db.commit()
    return len(rows)


def import_embedding_batches(db: Session, config: ImportConfig, *, selected_rows: int) -> int:
    vectors = np.memmap(
        config.memmap_path,
        dtype=VECTOR_DTYPE,
        mode="r",
        shape=(memmap_vector_count(config.memmap_path), VECTOR_DIM),
    )
    inserted = 0
    batch: list[dict] = []
    now = datetime.now(UTC)
    started = time.perf_counter()

    for row_id, identity in iter_mapping_rows(config.mapping_path, config.limit):
        vector = np.asarray(vectors[row_id], dtype=np.float32)
        if vector.shape[0] != VECTOR_DIM or not np.isfinite(vector).all():
            continue
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            continue
        vector_bytes = (vector / norm).astype(np.float32, copy=False).tobytes()
        batch.append(
            {
                "id": embedding_uuid(config, row_id, identity),
                "person_id": person_uuid(config, identity),
                "pipeline": config.pipeline,
                "model": config.model_name,
                "dim": VECTOR_DIM,
                "vector": encrypt_embedding_payload(vector_bytes),
                "created_at": now,
                "is_active": True,
            }
        )

        if len(batch) >= config.batch_size:
            db.execute(Embedding.__table__.insert(), batch)
            db.commit()
            inserted += len(batch)
            batch.clear()
            elapsed = time.perf_counter() - started
            rate = inserted / elapsed if elapsed > 0 else 0.0
            print(
                f"progress embeddings={inserted}/{selected_rows} "
                f"rate={rate:.1f}/s elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if batch:
        db.execute(Embedding.__table__.insert(), batch)
        db.commit()
        inserted += len(batch)
    del vectors
    return inserted


def import_to_database(config: ImportConfig) -> tuple[MappingStats, ImportResult]:
    mapping_stats = validate_config(config)
    started = time.perf_counter()
    engine = create_engine(config.database_url, pool_pre_ping=True, future=True)
    session_factory = sessionmaker(bind=engine, autoflush=False, future=True)

    with session_factory() as db:
        if config.create_schema:
            Base.metadata.create_all(bind=engine)

        persons_existing, embeddings_existing = existing_counts(db, config)
        if persons_existing or embeddings_existing:
            if not config.replace_existing:
                raise RuntimeError(
                    "Existing imported rows were found for this label prefix/pipeline/model: "
                    f"persons={persons_existing} embeddings={embeddings_existing}. "
                    "Pass --replace-existing to delete only these imported rows first."
                )
            persons_deleted, embeddings_deleted = delete_existing(db, config)
            print(f"deleted existing imported rows: persons={persons_deleted} embeddings={embeddings_deleted}")

        if config.dry_run:
            return mapping_stats, ImportResult(0, 0, time.perf_counter() - started, dry_run=True)

        try:
            persons_inserted = insert_persons(db, config)
            embeddings_inserted = import_embedding_batches(
                db,
                config,
                selected_rows=mapping_stats.rows,
            )
        except IntegrityError as exc:
            db.rollback()
            raise RuntimeError(
                "Import hit duplicate primary keys. Use a clean scale DB or --replace-existing."
            ) from exc

    return mapping_stats, ImportResult(
        persons_inserted=persons_inserted,
        embeddings_inserted=embeddings_inserted,
        elapsed_s=time.perf_counter() - started,
        dry_run=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import real-image benchmark memmap/mapping into a separate backend database."
    )
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--mapping-path", required=True)
    parser.add_argument("--memmap-path", required=True)
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--label-prefix", default=DEFAULT_LABEL_PREFIX)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--create-schema", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ImportConfig(
        database_url=args.database_url,
        mapping_path=resolve_path(args.mapping_path),
        memmap_path=resolve_path(args.memmap_path),
        pipeline=args.pipeline,
        model_name=args.model_name,
        label_prefix=args.label_prefix,
        batch_size=args.batch_size,
        limit=args.limit,
        create_schema=args.create_schema,
        replace_existing=args.replace_existing,
        dry_run=args.dry_run,
        yes=args.yes,
    )
    mapping_stats, result = import_to_database(config)
    payload = {
        "dry_run": result.dry_run,
        "mapping_rows": mapping_stats.rows,
        "unique_identities": mapping_stats.unique_identities,
        "max_row_id": mapping_stats.max_row_id,
        "example_labels": mapping_stats.examples,
        "persons_inserted": result.persons_inserted,
        "embeddings_inserted": result.embeddings_inserted,
        "pipeline": config.pipeline,
        "model_name": config.model_name,
        "label_prefix": config.label_prefix,
        "elapsed_s": round(result.elapsed_s, 3),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
