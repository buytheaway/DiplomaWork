from __future__ import annotations

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Embedding, Person
from scripts.seed_scale_database import SeedConfig, seed_database, validate_seed_request


def _sqlite_url(tmp_path) -> str:
    return f"sqlite+pysqlite:///{(tmp_path / 'scale_seed.db').as_posix()}"


def _create_schema(database_url: str):
    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
    return engine


def test_seed_scale_database_inserts_persons_and_embeddings(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)

    result = seed_database(
        SeedConfig(
            database_url=database_url,
            count=10,
            batch_size=4,
            dim=512,
            pipeline="pretrained",
            model_name="scale_synthetic_512d",
        )
    )

    assert result.inserted_persons == 10
    assert result.inserted_embeddings == 10

    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db:
        persons_count = db.execute(select(func.count()).select_from(Person)).scalar_one()
        embeddings_count = db.execute(select(func.count()).select_from(Embedding)).scalar_one()
        embedding = db.execute(select(Embedding)).scalars().first()

    assert persons_count == 10
    assert embeddings_count == 10
    assert embedding is not None
    assert embedding.pipeline == "pretrained"
    assert embedding.model == "scale_synthetic_512d"
    assert len(embedding.vector) == 512 * 4


def test_seed_scale_database_dry_run_does_not_write(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)

    result = seed_database(
        SeedConfig(
            database_url=database_url,
            count=10,
            dry_run=True,
        )
    )

    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db:
        persons_count = db.execute(select(func.count()).select_from(Person)).scalar_one()
        embeddings_count = db.execute(select(func.count()).select_from(Embedding)).scalar_one()

    assert result.dry_run is True
    assert result.inserted_persons == 0
    assert result.inserted_embeddings == 0
    assert persons_count == 0
    assert embeddings_count == 0


def test_seed_scale_database_identities_with_samples(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)

    result = seed_database(
        SeedConfig(
            database_url=database_url,
            mode="identities-with-samples",
            identities=3,
            samples_per_identity=4,
            batch_size=5,
            dim=512,
            pipeline="pretrained",
            model_name="scale_synthetic_512d",
        )
    )

    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db:
        labels = list(db.execute(select(Person.label).order_by(Person.label)).scalars())
        persons_count = db.execute(select(func.count()).select_from(Person)).scalar_one()
        embeddings_count = db.execute(select(func.count()).select_from(Embedding)).scalar_one()
        embedding = db.execute(select(Embedding)).scalars().first()
        embeddings_per_person = list(
            db.execute(
                select(Embedding.person_id, func.count())
                .group_by(Embedding.person_id)
                .order_by(Embedding.person_id)
            )
        )

    assert result.inserted_persons == 3
    assert result.inserted_embeddings == 12
    assert persons_count == 3
    assert embeddings_count == 12
    assert labels == [
        "Dataset Identity 000000001",
        "Dataset Identity 000000002",
        "Dataset Identity 000000003",
    ]
    assert embedding is not None
    assert embedding.pipeline == "pretrained"
    assert embedding.model == "scale_synthetic_512d"
    assert len(embedding.vector) == 512 * 4
    assert [count for _, count in embeddings_per_person] == [4, 4, 4]


def test_seed_scale_database_identities_dry_run_does_not_write(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)

    result = seed_database(
        SeedConfig(
            database_url=database_url,
            mode="identities-with-samples",
            identities=3,
            samples_per_identity=4,
            dry_run=True,
        )
    )

    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db:
        persons_count = db.execute(select(func.count()).select_from(Person)).scalar_one()
        embeddings_count = db.execute(select(func.count()).select_from(Embedding)).scalar_one()

    assert result.dry_run is True
    assert result.inserted_persons == 0
    assert result.inserted_embeddings == 0
    assert persons_count == 0
    assert embeddings_count == 0


def test_seed_scale_database_realistic_labels_are_deterministic(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)

    result = seed_database(
        SeedConfig(
            database_url=database_url,
            count=3,
            batch_size=2,
            label_style="realistic",
        )
    )

    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db:
        labels = list(db.execute(select(Person.label)).scalars())

    assert result.inserted_persons == 3
    assert all(label is not None for label in labels)
    assert all("#SCALE-" in str(label) for label in labels)
    assert "Aidar Sarsenov #SCALE-000000001" in labels
    assert "Aigerim Nurlanova #SCALE-000000002" in labels
    assert "Miras Tulegenov #SCALE-000000003" in labels
    assert not all(str(label).startswith("Scale Person ") for label in labels)


def test_seed_scale_database_replace_can_switch_numbered_to_realistic(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)

    seed_database(SeedConfig(database_url=database_url, count=2))
    result = seed_database(
        SeedConfig(
            database_url=database_url,
            count=3,
            label_style="realistic",
            replace_existing=True,
        )
    )

    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db:
        labels = list(db.execute(select(Person.label)).scalars())
        embeddings_count = db.execute(select(func.count()).select_from(Embedding)).scalar_one()

    assert result.inserted_persons == 3
    assert len(labels) == 3
    assert embeddings_count == 3
    assert all("#SCALE-" in str(label) for label in labels)


def test_seed_scale_database_requires_yes_for_large_count(tmp_path):
    config = SeedConfig(
        database_url=_sqlite_url(tmp_path),
        count=10_001,
    )

    with pytest.raises(ValueError, match="Pass --yes"):
        validate_seed_request(config)


def test_seed_scale_database_requires_yes_for_large_identity_sample_count(tmp_path):
    config = SeedConfig(
        database_url=_sqlite_url(tmp_path),
        mode="identities-with-samples",
        identities=101,
        samples_per_identity=100,
    )

    with pytest.raises(ValueError, match="Pass --yes"):
        validate_seed_request(config)


def test_seed_scale_database_replace_identities_does_not_duplicate(tmp_path):
    database_url = _sqlite_url(tmp_path)
    engine = _create_schema(database_url)

    base_config = SeedConfig(
        database_url=database_url,
        mode="identities-with-samples",
        identities=3,
        samples_per_identity=4,
        batch_size=5,
    )
    seed_database(base_config)
    result = seed_database(
        SeedConfig(
            database_url=database_url,
            mode="identities-with-samples",
            identities=3,
            samples_per_identity=4,
            batch_size=5,
            replace_existing=True,
        )
    )

    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as db:
        persons_count = db.execute(select(func.count()).select_from(Person)).scalar_one()
        embeddings_count = db.execute(select(func.count()).select_from(Embedding)).scalar_one()

    assert result.inserted_persons == 3
    assert result.inserted_embeddings == 12
    assert persons_count == 3
    assert embeddings_count == 12
