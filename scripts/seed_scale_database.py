from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, delete, func, or_, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.models import Embedding, Person  # noqa: E402

DEFAULT_BATCH_SIZE = 5000
DEFAULT_DIM = 512
DEFAULT_PIPELINE = "pretrained"
DEFAULT_MODEL_NAME = "scale_synthetic_512d"
DEFAULT_LABEL_PREFIX = "Scale Person"
DEFAULT_IDENTITY_LABEL_PREFIX = "Dataset Identity"
DEFAULT_SEED = 42
LARGE_COUNT_THRESHOLD = 10_000
DEFAULT_MODE = "one-to-one"
MODES = ("one-to-one", "identities-with-samples")
LABEL_STYLES = ("numbered", "realistic")
FIRST_NAMES = (
    "Aidar",
    "Aigerim",
    "Miras",
    "Dana",
    "Arman",
    "Aruzhan",
    "Sanjar",
    "Madina",
    "Dias",
    "Kamila",
    "Nursultan",
    "Alina",
)
LAST_NAMES = (
    "Sarsenov",
    "Nurlanova",
    "Tulegenov",
    "Akhmetova",
    "Omarov",
    "Kasenova",
    "Zhaksylykov",
    "Ibragimova",
    "Muratov",
    "Bekturova",
    "Iskakov",
    "Karimova",
)


@dataclass(frozen=True)
class SeedConfig:
    database_url: str | None
    count: int | None = None
    mode: str = DEFAULT_MODE
    identities: int | None = None
    samples_per_identity: int | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    dim: int = DEFAULT_DIM
    pipeline: str = DEFAULT_PIPELINE
    model_name: str = DEFAULT_MODEL_NAME
    label_prefix: str = DEFAULT_LABEL_PREFIX
    identity_label_prefix: str = DEFAULT_IDENTITY_LABEL_PREFIX
    label_style: str = "numbered"
    seed: int = DEFAULT_SEED
    replace_existing: bool = False
    dry_run: bool = False
    yes: bool = False


@dataclass(frozen=True)
class SeedResult:
    inserted_persons: int
    inserted_embeddings: int
    elapsed_s: float
    dry_run: bool


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _resolve_sqlite_path(database: str) -> Path | None:
    url = make_url(database)
    if not url.drivername.startswith("sqlite"):
        return None

    raw_path = url.database
    if raw_path in {None, "", ":memory:"}:
        return None

    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def unsafe_database_reason(database_url: str | None) -> str | None:
    if not database_url:
        return "Pass --database-url for a separate scale database. The app DATABASE_URL is not used."

    try:
        sqlite_path = _resolve_sqlite_path(database_url)
    except Exception as exc:  # noqa: BLE001
        return f"Invalid database URL: {exc}"

    if sqlite_path is None:
        return None

    repo_root = ROOT.resolve()
    backend_root = BACKEND_ROOT.resolve()
    backend_data = (BACKEND_ROOT / "data").resolve()
    try:
        sqlite_path.relative_to(backend_root)
        return f"Refusing to seed SQLite database inside backend/: {sqlite_path}"
    except ValueError:
        pass
    try:
        sqlite_path.relative_to(backend_data)
        return f"Refusing to seed SQLite database inside backend/data/: {sqlite_path}"
    except ValueError:
        pass
    try:
        sqlite_path.relative_to(repo_root)
        if sqlite_path.name.startswith("backend_local") or sqlite_path.suffix == ".db":
            return f"Refusing to seed repo-local SQLite demo database: {sqlite_path}"
    except ValueError:
        pass
    return None


def validate_seed_request(config: SeedConfig) -> None:
    reason = unsafe_database_reason(config.database_url)
    if reason is not None:
        raise ValueError(reason)
    if config.mode not in MODES:
        raise ValueError(f"--mode must be one of: {', '.join(MODES)}")
    total_embeddings = _total_embeddings(config)
    if total_embeddings > LARGE_COUNT_THRESHOLD and not config.yes:
        raise ValueError(
            f"requested embeddings count {total_embeddings} is larger than {LARGE_COUNT_THRESHOLD}. "
            "Pass --yes to confirm this scale seed intentionally targets a separate database."
        )
    if not config.pipeline.strip():
        raise ValueError("--pipeline must not be empty")
    if not config.model_name.strip():
        raise ValueError("--model-name must not be empty")
    if config.mode == "one-to-one":
        if config.count is None:
            raise ValueError("--count is required for --mode one-to-one")
        if not config.label_prefix.strip():
            raise ValueError("--label-prefix must not be empty")
        if config.label_style not in LABEL_STYLES:
            raise ValueError(f"--label-style must be one of: {', '.join(LABEL_STYLES)}")
    else:
        if config.identities is None:
            raise ValueError("--identities is required for --mode identities-with-samples")
        if config.samples_per_identity is None:
            raise ValueError("--samples-per-identity is required for --mode identities-with-samples")
        if not config.identity_label_prefix.strip():
            raise ValueError("--identity-label-prefix must not be empty")


def _total_persons(config: SeedConfig) -> int:
    if config.mode == "one-to-one":
        if config.count is None:
            raise ValueError("--count is required for --mode one-to-one")
        return config.count
    if config.identities is None:
        raise ValueError("--identities is required for --mode identities-with-samples")
    return config.identities


def _total_embeddings(config: SeedConfig) -> int:
    if config.mode == "one-to-one":
        return _total_persons(config)
    if config.samples_per_identity is None:
        raise ValueError("--samples-per-identity is required for --mode identities-with-samples")
    return _total_persons(config) * config.samples_per_identity


def _label_for(config: SeedConfig, index: int, width: int) -> str:
    if config.mode == "identities-with-samples":
        return f"{config.identity_label_prefix} {index:0{width}d}"
    if config.label_style == "numbered":
        return f"{config.label_prefix} {index:0{width}d}"

    first = FIRST_NAMES[(index - 1) % len(FIRST_NAMES)]
    last = LAST_NAMES[((index - 1) // len(FIRST_NAMES) + index - 1) % len(LAST_NAMES)]
    return f"{first} {last} #SCALE-{index:0{width}d}"


def _person_uuid(config: SeedConfig, index: int) -> uuid.UUID:
    if config.mode == "identities-with-samples":
        source = (
            f"{config.pipeline}:{config.model_name}:"
            f"{config.identity_label_prefix}:{config.mode}:{index}"
        )
        return uuid.uuid5(uuid.NAMESPACE_URL, source)
    source = (
        f"{config.pipeline}:{config.model_name}:"
        f"{config.label_prefix}:{config.label_style}:{index}"
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, source)


def _embedding_uuid(config: SeedConfig, index: int, sample_index: int | None = None) -> uuid.UUID:
    if config.mode == "identities-with-samples":
        source = (
            f"embedding:{config.pipeline}:{config.model_name}:"
            f"{config.identity_label_prefix}:{config.mode}:{index}:{sample_index}"
        )
        return uuid.uuid5(uuid.NAMESPACE_URL, source)
    source = (
        f"embedding:{config.pipeline}:{config.model_name}:"
        f"{config.label_prefix}:{config.label_style}:{index}"
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, source)


def _generate_vectors(rng: np.random.RandomState, batch_count: int, dim: int) -> np.ndarray:
    vectors = rng.normal(size=(batch_count, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def _scale_person_stmt(config: SeedConfig):
    if config.mode == "identities-with-samples":
        label_filter = Person.label.like(f"{config.identity_label_prefix} %")
    else:
        label_filter = or_(
            Person.label.like(f"{config.label_prefix} %"),
            Person.label.like("%#SCALE-%"),
        )
    return (
        select(Person.id)
        .where(
            label_filter,
            Person.status == "active",
        )
        .distinct()
    )


def _count_existing_scale_rows(db: Session, config: SeedConfig) -> tuple[int, int]:
    person_ids = _scale_person_stmt(config).subquery()
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


def _delete_existing_scale_rows(db: Session, config: SeedConfig) -> tuple[int, int]:
    person_ids = list(db.execute(_scale_person_stmt(config)).scalars())
    if not person_ids:
        return 0, 0
    embeddings_deleted = db.execute(
        delete(Embedding).where(
            Embedding.person_id.in_(person_ids),
            Embedding.pipeline == config.pipeline,
            Embedding.model == config.model_name,
        )
    ).rowcount
    persons_deleted = db.execute(
        delete(Person).where(Person.id.in_(person_ids))
    ).rowcount
    return int(persons_deleted or 0), int(embeddings_deleted or 0)


def _insert_batch(
    db: Session,
    config: SeedConfig,
    *,
    start_index: int,
    batch_count: int,
    width: int,
    rng: np.random.RandomState,
) -> tuple[int, int]:
    now = datetime.now(UTC)
    vectors = _generate_vectors(rng, batch_count, config.dim)

    persons = []
    embeddings = []
    for row_offset in range(batch_count):
        index = start_index + row_offset
        person_id = _person_uuid(config, index)
        persons.append(
            {
                "id": person_id,
                "label": _label_for(config, index, width),
                "status": "active",
                "created_at": now,
                "updated_at": now,
            }
        )
        embeddings.append(
            {
                "id": _embedding_uuid(config, index),
                "person_id": person_id,
                "pipeline": config.pipeline,
                "model": config.model_name,
                "dim": config.dim,
                "vector": vectors[row_offset].tobytes(),
                "created_at": now,
                "is_active": True,
            }
        )

    db.execute(Person.__table__.insert(), persons)
    db.execute(Embedding.__table__.insert(), embeddings)
    db.commit()
    return len(persons), len(embeddings)


def _insert_identity_person_batch(
    db: Session,
    config: SeedConfig,
    *,
    start_index: int,
    batch_count: int,
    width: int,
) -> int:
    now = datetime.now(UTC)
    persons = []
    for row_offset in range(batch_count):
        index = start_index + row_offset
        persons.append(
            {
                "id": _person_uuid(config, index),
                "label": _label_for(config, index, width),
                "status": "active",
                "created_at": now,
                "updated_at": now,
            }
        )
    db.execute(Person.__table__.insert(), persons)
    db.commit()
    return len(persons)


def _insert_identity_embedding_batch(
    db: Session,
    config: SeedConfig,
    *,
    start_embedding_index: int,
    batch_count: int,
    rng: np.random.RandomState,
) -> int:
    if config.samples_per_identity is None:
        raise ValueError("--samples-per-identity is required for --mode identities-with-samples")

    now = datetime.now(UTC)
    vectors = _generate_vectors(rng, batch_count, config.dim)
    embeddings = []
    for row_offset in range(batch_count):
        embedding_index = start_embedding_index + row_offset
        identity_index = ((embedding_index - 1) // config.samples_per_identity) + 1
        sample_index = ((embedding_index - 1) % config.samples_per_identity) + 1
        embeddings.append(
            {
                "id": _embedding_uuid(config, identity_index, sample_index),
                "person_id": _person_uuid(config, identity_index),
                "pipeline": config.pipeline,
                "model": config.model_name,
                "dim": config.dim,
                "vector": vectors[row_offset].tobytes(),
                "created_at": now,
                "is_active": True,
            }
        )
    db.execute(Embedding.__table__.insert(), embeddings)
    db.commit()
    return len(embeddings)


def _example_labels(config: SeedConfig, limit: int = 3) -> list[str]:
    width = max(9, len(str(_total_persons(config))))
    return [
        _label_for(config, index, width)
        for index in range(1, min(limit, _total_persons(config)) + 1)
    ]


def seed_database(config: SeedConfig) -> SeedResult:
    validate_seed_request(config)
    assert config.database_url is not None

    total_persons = _total_persons(config)
    total_embeddings = _total_embeddings(config)
    started = time.perf_counter()
    engine = create_engine(config.database_url, pool_pre_ping=True, future=True)
    session_factory = sessionmaker(bind=engine, autoflush=False, future=True)

    inserted_persons = 0
    inserted_embeddings = 0
    width = max(9, len(str(total_persons)))
    rng = np.random.RandomState(config.seed)

    with session_factory() as db:
        existing_persons, existing_embeddings = _count_existing_scale_rows(db, config)
        if existing_persons or existing_embeddings:
            if not config.replace_existing:
                raise RuntimeError(
                    "Existing scale records were found for this label prefix / pipeline / model "
                    f"({existing_persons} persons, {existing_embeddings} embeddings). "
                    "Use --replace-existing only for a dedicated scale database."
                )
            if config.dry_run:
                print(
                    "DRY RUN: would delete "
                    f"{existing_persons} persons and {existing_embeddings} embeddings"
                )
            else:
                persons_deleted, embeddings_deleted = _delete_existing_scale_rows(db, config)
                db.commit()
                print(
                    "Deleted existing scale records: "
                    f"persons={persons_deleted} embeddings={embeddings_deleted}"
                )

        if config.dry_run:
            elapsed = time.perf_counter() - started
            print(
                "DRY RUN: would insert "
                f"mode={config.mode} persons={total_persons} embeddings={total_embeddings} "
                f"samples_per_identity={_samples_per_identity(config)} "
                f"dim={config.dim} pipeline={config.pipeline} model={config.model_name}"
            )
            print("Example labels: " + ", ".join(_example_labels(config)))
            return SeedResult(0, 0, elapsed, dry_run=True)

        if config.mode == "one-to-one":
            next_index = 1
            while next_index <= total_persons:
                batch_count = min(config.batch_size, total_persons - next_index + 1)
                persons_count, embeddings_count = _insert_batch(
                    db,
                    config,
                    start_index=next_index,
                    batch_count=batch_count,
                    width=width,
                    rng=rng,
                )
                inserted_persons += persons_count
                inserted_embeddings += embeddings_count
                next_index += batch_count
                if inserted_persons % 50_000 == 0 or inserted_persons == total_persons:
                    print(
                        f"progress inserted_persons={inserted_persons} "
                        f"inserted_embeddings={inserted_embeddings}"
                    )
        else:
            next_person_index = 1
            while next_person_index <= total_persons:
                batch_count = min(config.batch_size, total_persons - next_person_index + 1)
                inserted_persons += _insert_identity_person_batch(
                    db,
                    config,
                    start_index=next_person_index,
                    batch_count=batch_count,
                    width=width,
                )
                next_person_index += batch_count
                if inserted_persons % 50_000 == 0 or inserted_persons == total_persons:
                    print(
                        f"progress inserted_persons={inserted_persons} "
                        f"inserted_embeddings={inserted_embeddings}"
                    )

            next_embedding_index = 1
            while next_embedding_index <= total_embeddings:
                batch_count = min(config.batch_size, total_embeddings - next_embedding_index + 1)
                inserted_embeddings += _insert_identity_embedding_batch(
                    db,
                    config,
                    start_embedding_index=next_embedding_index,
                    batch_count=batch_count,
                    rng=rng,
                )
                next_embedding_index += batch_count
                if inserted_embeddings % 50_000 == 0 or inserted_embeddings == total_embeddings:
                    print(
                        f"progress inserted_persons={inserted_persons} "
                        f"inserted_embeddings={inserted_embeddings}"
                    )

    elapsed = time.perf_counter() - started
    return SeedResult(inserted_persons, inserted_embeddings, elapsed, dry_run=False)


def _samples_per_identity(config: SeedConfig) -> int:
    if config.mode == "one-to-one":
        return 1
    if config.samples_per_identity is None:
        raise ValueError("--samples-per-identity is required for --mode identities-with-samples")
    return config.samples_per_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a separate scale-demo database with synthetic persons and embeddings."
    )
    parser.add_argument("--database-url", default=None, help="Required separate scale DB URL")
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    parser.add_argument(
        "--count",
        type=positive_int,
        default=None,
        help="Persons and embeddings to create in one-to-one mode",
    )
    parser.add_argument("--identities", type=positive_int, default=None)
    parser.add_argument("--samples-per-identity", type=positive_int, default=None)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dim", type=positive_int, default=DEFAULT_DIM)
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--label-prefix", default=DEFAULT_LABEL_PREFIX)
    parser.add_argument("--identity-label-prefix", default=DEFAULT_IDENTITY_LABEL_PREFIX)
    parser.add_argument(
        "--label-style",
        choices=LABEL_STYLES,
        default="numbered",
        help="numbered keeps old labels; realistic uses deterministic synthetic names",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required when generated embeddings count is larger than 10000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SeedConfig(
        database_url=args.database_url,
        count=args.count,
        mode=args.mode,
        identities=args.identities,
        samples_per_identity=args.samples_per_identity,
        batch_size=args.batch_size,
        dim=args.dim,
        pipeline=args.pipeline,
        model_name=args.model_name,
        label_prefix=args.label_prefix,
        identity_label_prefix=args.identity_label_prefix,
        label_style=args.label_style,
        seed=args.seed,
        replace_existing=args.replace_existing,
        dry_run=args.dry_run,
        yes=args.yes,
    )

    try:
        result = seed_database(config)
    except (RuntimeError, SQLAlchemyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    total_persons = _total_persons(config)
    total_embeddings = _total_embeddings(config)
    raw_gib = total_embeddings * config.dim * 4 / (1024**3)
    print(
        "SUMMARY "
        f"mode={config.mode} persons={total_persons} embeddings={total_embeddings} "
        f"samples_per_identity={_samples_per_identity(config)} "
        f"pipeline={config.pipeline} model={config.model_name}"
    )
    print("Example labels: " + ", ".join(_example_labels(config)))
    print(
        "DONE "
        f"inserted_persons={result.inserted_persons} "
        f"inserted_embeddings={result.inserted_embeddings} "
        f"elapsed_s={result.elapsed_s:.3f} dry_run={result.dry_run}"
    )
    print(
        "Expected DB size note: raw vector payload is about "
        f"{raw_gib:.2f} GiB before row, index, and database overhead."
    )


if __name__ == "__main__":
    main()
