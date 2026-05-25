"""Build a scale-demo FAISS snapshot from database embeddings.

This is an offline builder for large PostgreSQL scale demonstrations. It avoids
the backend ``/v1/index/rebuild`` path because that path loads all embeddings
through ORM objects and stacks all vectors in memory before building FAISS.

The output is compatible with the backend snapshot loader:
* encrypted ``.faiss`` file;
* encrypted ``.faiss.map.json`` sidecar;
* ``index_snapshots`` database row pointing at the snapshot path.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.models import Embedding, IndexSnapshot, Person  # noqa: E402

DEFAULT_PIPELINE = "pretrained"
DEFAULT_MODEL_NAME = "dummy"
DEFAULT_BATCH_SIZE = 50_000
DEFAULT_TRAIN_SIZE = 200_000
DEFAULT_LIMIT_CONFIRM_THRESHOLD = 100_000
INDEX_TYPES = ("flat", "hnsw", "ivfpq")

_PAYLOAD_PREFIX = b"ENC1"
_NONCE_SIZE = 12
_TEST_KEY = base64.urlsafe_b64encode(b"\x11" * 32).decode("ascii")


@dataclass(frozen=True)
class BuildConfig:
    database_url: str
    pipeline: str = DEFAULT_PIPELINE
    model_name: str = DEFAULT_MODEL_NAME
    index_type: str = "ivfpq"
    index_path: Path | str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    train_size: int = DEFAULT_TRAIN_SIZE
    dry_run: bool = False
    yes: bool = False
    limit: int | None = None
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100
    hnsw_ef_search: int = 64
    ivfpq_nlist: int = 4096
    ivfpq_m: int = 32
    ivfpq_nbits: int = 8
    ivfpq_nprobe: int = 32


@dataclass(frozen=True)
class BuildResult:
    selected_embeddings: int
    vectors_added: int
    skipped_malformed: int
    index_type: str
    params: dict[str, Any]
    dim: int
    train_time_s: float
    add_time_s: float
    total_time_s: float
    index_file_size_mb: float
    map_file_size_mb: float
    snapshot_id: str | None
    output_path: str | None
    dry_run: bool


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _is_testing() -> bool:
    return os.getenv("TESTING", "").strip().lower() in {"1", "true", "yes", "on"}


def _decode_key(raw_key: str, *, field_name: str) -> bytes:
    source = raw_key or (_TEST_KEY if _is_testing() else "")
    if not source:
        raise RuntimeError(
            f"{field_name} must be set, or TESTING=true must be used for a local smoke build"
        )

    padded = source + ("=" * (-len(source) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{field_name} must be a urlsafe base64-encoded 32-byte key") from exc

    if len(decoded) != 32:
        raise RuntimeError(f"{field_name} must decode to exactly 32 bytes")
    return decoded


def _encrypt_snapshot_payload(raw: bytes) -> bytes:
    raw_key = os.getenv("SNAPSHOT_ENCRYPTION_KEY", "")
    key = _decode_key(raw_key, field_name="SNAPSHOT_ENCRYPTION_KEY")
    nonce = os.urandom(_NONCE_SIZE)
    ciphertext = AESGCM(key).encrypt(nonce, raw, b"faiss-snapshot")
    return _PAYLOAD_PREFIX + nonce + ciphertext


def _decrypt_embedding_payload(blob: bytes) -> bytes:
    if not blob.startswith(_PAYLOAD_PREFIX):
        return blob
    raw_key = os.getenv("DATA_ENCRYPTION_KEY", "")
    key = _decode_key(raw_key, field_name="DATA_ENCRYPTION_KEY")
    nonce = blob[len(_PAYLOAD_PREFIX):len(_PAYLOAD_PREFIX) + _NONCE_SIZE]
    ciphertext = blob[len(_PAYLOAD_PREFIX) + _NONCE_SIZE:]
    return AESGCM(key).decrypt(nonce, ciphertext, b"embedding-vector")


def resolve_index_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("--index-path is required")
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    return path_obj.resolve()


def _is_dangerous_index_path(path: Path) -> bool:
    backend_index_dir = (BACKEND_ROOT / "data" / "index").resolve()
    try:
        path.relative_to(backend_index_dir)
    except ValueError:
        return False
    return path.name in {"current.faiss", "pretrained.faiss", "custom.faiss"}


def validate_config(config: BuildConfig) -> Path:
    if not config.database_url.strip():
        raise ValueError("--database-url is required")
    if config.index_type not in INDEX_TYPES:
        raise ValueError(f"--index-type must be one of: {', '.join(INDEX_TYPES)}")
    if not config.pipeline.strip():
        raise ValueError("--pipeline must not be empty")
    if not config.model_name.strip():
        raise ValueError("--model-name must not be empty")
    if config.index_type == "ivfpq" and config.ivfpq_m <= 0:
        raise ValueError("--ivfpq-m must be positive")

    index_path = resolve_index_path(config.index_path)
    if _is_dangerous_index_path(index_path):
        raise ValueError(
            "Refusing to write over an active demo index path. "
            "Use an explicit scale path such as tmp/scale_index/pretrained.faiss."
        )
    return index_path


def _selected_filters(config: BuildConfig):
    return (
        Embedding.is_active.is_(True),
        Embedding.pipeline == config.pipeline,
        Embedding.model == config.model_name,
        Person.status == "active",
    )


def count_matching_embeddings(db: Session, config: BuildConfig) -> int:
    stmt = (
        select(func.count())
        .select_from(Embedding)
        .join(Person, Embedding.person_id == Person.id)
        .where(*_selected_filters(config))
    )
    return int(db.execute(stmt).scalar_one())


def count_selected_embeddings(db: Session, config: BuildConfig) -> int:
    total = count_matching_embeddings(db, config)
    return min(total, config.limit) if config.limit is not None else total


def selected_embedding_dim(db: Session, config: BuildConfig) -> int:
    stmt = (
        select(Embedding.dim)
        .join(Person, Embedding.person_id == Person.id)
        .where(*_selected_filters(config))
        .order_by(Embedding.created_at, Embedding.id)
        .limit(1)
    )
    dim = db.execute(stmt).scalar_one_or_none()
    if dim is None:
        raise RuntimeError("No active embeddings matched pipeline/model filters")
    return int(dim)


def iter_embedding_batches(db: Session, config: BuildConfig):
    emitted = 0
    offset = 0
    while config.limit is None or emitted < config.limit:
        batch_limit = config.batch_size
        if config.limit is not None:
            batch_limit = min(batch_limit, config.limit - emitted)
        if batch_limit <= 0:
            return

        stmt = (
            select(Embedding.id, Embedding.vector, Embedding.dim)
            .join(Person, Embedding.person_id == Person.id)
            .where(*_selected_filters(config))
            .order_by(Embedding.created_at, Embedding.id)
            .limit(batch_limit)
            .offset(offset)
        )
        rows = list(db.execute(stmt).all())
        if not rows:
            return
        yield rows
        emitted += len(rows)
        offset += len(rows)


def decode_vector(vector_blob: bytes, row_dim: int, expected_dim: int) -> np.ndarray | None:
    raw = _decrypt_embedding_payload(vector_blob)
    if row_dim != expected_dim:
        return None
    if len(raw) != expected_dim * np.dtype(np.float32).itemsize:
        return None
    vector = np.frombuffer(raw, dtype=np.float32).astype(np.float32, copy=True)
    if vector.shape[0] != expected_dim:
        return None
    if not np.isfinite(vector).all():
        return None
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)


def collect_training_vectors(
    db: Session,
    config: BuildConfig,
    *,
    dim: int,
) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    skipped = 0
    collected = 0
    train_target = min(config.train_size, count_selected_embeddings(db, config))

    for rows in iter_embedding_batches(db, config):
        vectors: list[np.ndarray] = []
        for _embedding_id, vector_blob, row_dim in rows:
            if collected >= train_target:
                break
            try:
                vector = decode_vector(vector_blob, int(row_dim), dim)
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001
                vector = None
            if vector is None:
                skipped += 1
                continue
            vectors.append(vector)
            collected += 1
        if vectors:
            chunks.append(np.vstack(vectors))
        if collected >= train_target:
            break

    if not chunks:
        raise RuntimeError("No valid vectors were available for IVF-PQ training")
    return np.vstack(chunks), skipped


def build_empty_index(config: BuildConfig, *, dim: int, train_vectors: np.ndarray | None) -> tuple[faiss.Index, dict[str, Any], float]:
    train_time_s = 0.0
    if config.index_type == "flat":
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim)), {}, train_time_s

    if config.index_type == "hnsw":
        base = faiss.IndexHNSWFlat(dim, config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = config.hnsw_ef_construction
        base.hnsw.efSearch = config.hnsw_ef_search
        params = {
            "m": config.hnsw_m,
            "ef_construction": config.hnsw_ef_construction,
            "ef_search": config.hnsw_ef_search,
        }
        return faiss.IndexIDMap2(base), params, train_time_s

    if dim % config.ivfpq_m != 0:
        raise ValueError(f"embedding dim {dim} must be divisible by IVFPQ_M={config.ivfpq_m}")
    if train_vectors is None:
        raise RuntimeError("IVF-PQ requires training vectors")

    effective_nlist = min(config.ivfpq_nlist, len(train_vectors))
    quantizer = faiss.IndexFlatIP(dim)
    base = faiss.IndexIVFPQ(
        quantizer,
        dim,
        effective_nlist,
        config.ivfpq_m,
        config.ivfpq_nbits,
        faiss.METRIC_INNER_PRODUCT,
    )
    started = time.perf_counter()
    base.train(np.ascontiguousarray(train_vectors, dtype=np.float32))
    train_time_s = time.perf_counter() - started
    base.nprobe = min(config.ivfpq_nprobe, effective_nlist)
    params = {
        "nlist": effective_nlist,
        "requested_nlist": config.ivfpq_nlist,
        "m": config.ivfpq_m,
        "nbits": config.ivfpq_nbits,
        "nprobe": int(base.nprobe),
        "train_size": len(train_vectors),
    }
    return faiss.IndexIDMap2(base), params, train_time_s


def add_vectors_from_db(
    db: Session,
    config: BuildConfig,
    *,
    index: faiss.Index,
    dim: int,
) -> tuple[dict[int, str], int, int, float]:
    id_map: dict[int, str] = {}
    next_vector_id = 0
    skipped = 0
    started = time.perf_counter()

    selected_total = count_selected_embeddings(db, config)
    for rows in iter_embedding_batches(db, config):
        vectors: list[np.ndarray] = []
        ids: list[int] = []
        for embedding_id, vector_blob, row_dim in rows:
            try:
                vector = decode_vector(vector_blob, int(row_dim), dim)
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001
                vector = None
            if vector is None:
                skipped += 1
                continue
            vector_id = next_vector_id
            next_vector_id += 1
            vectors.append(vector)
            ids.append(vector_id)
            id_map[vector_id] = str(embedding_id)

        if vectors:
            batch = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
            index.add_with_ids(batch, np.asarray(ids, dtype=np.int64))
            del batch
            elapsed = time.perf_counter() - started
            rate = next_vector_id / elapsed if elapsed > 0 else 0.0
            print(
                f"add progress vectors={next_vector_id}/{selected_total} "
                f"rate={rate:.1f}/s elapsed_s={elapsed:.1f}",
                flush=True,
            )
        del vectors
        gc.collect()

    return id_map, next_vector_id, skipped, time.perf_counter() - started


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_bytes(payload)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_encrypted_snapshot_files(
    index: faiss.Index,
    id_map: dict[int, str],
    *,
    index_path: Path,
) -> tuple[float, float]:
    map_path = index_path.with_suffix(index_path.suffix + ".map.json")
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_index_path = Path(tmpdir) / index_path.name
        raw_map_path = raw_index_path.with_suffix(raw_index_path.suffix + ".map.json")
        faiss.write_index(index, str(raw_index_path))
        raw_map_path.write_text(json.dumps(id_map), encoding="utf-8")
        _write_atomic(index_path, _encrypt_snapshot_payload(raw_index_path.read_bytes()))
        _write_atomic(map_path, _encrypt_snapshot_payload(raw_map_path.read_bytes()))
    return (
        index_path.stat().st_size / 1024 / 1024,
        map_path.stat().st_size / 1024 / 1024,
    )


def create_snapshot_row(
    db: Session,
    *,
    index_type: str,
    params: dict[str, Any],
    index_path: Path,
    vectors_added: int,
) -> str:
    snapshot = IndexSnapshot(
        index_type=index_type,
        params=params,
        path=str(index_path),
        embeddings_count=vectors_added,
        created_at=datetime.now(UTC),
    )
    db.add(snapshot)
    db.commit()
    return str(snapshot.id)


def build_scale_index(config: BuildConfig) -> BuildResult:
    index_path = validate_config(config)
    started = time.perf_counter()
    engine = create_engine(config.database_url, pool_pre_ping=True, future=True)
    session_factory = sessionmaker(bind=engine, autoflush=False, future=True)

    with session_factory() as db:
        matching_count = count_matching_embeddings(db, config)
        selected_count = min(matching_count, config.limit) if config.limit is not None else matching_count
        if matching_count > DEFAULT_LIMIT_CONFIRM_THRESHOLD and not config.yes:
            raise ValueError(
                f"Matched {matching_count} active embeddings. Pass --yes to confirm this offline build."
            )
        if selected_count == 0:
            raise RuntimeError("No active embeddings matched pipeline/model filters")
        dim = selected_embedding_dim(db, config)

        print(
            "Selected embeddings: "
            f"count={selected_count} matched={matching_count} "
            f"pipeline={config.pipeline} model={config.model_name} dim={dim}"
        )

        if config.dry_run:
            return BuildResult(
                selected_embeddings=selected_count,
                vectors_added=0,
                skipped_malformed=0,
                index_type=config.index_type,
                params={},
                dim=dim,
                train_time_s=0.0,
                add_time_s=0.0,
                total_time_s=time.perf_counter() - started,
                index_file_size_mb=0.0,
                map_file_size_mb=0.0,
                snapshot_id=None,
                output_path=None,
                dry_run=True,
            )

        train_vectors = None
        training_skipped = 0
        if config.index_type == "ivfpq":
            train_vectors, training_skipped = collect_training_vectors(db, config, dim=dim)
            print(f"Training vectors collected: {len(train_vectors)} skipped={training_skipped}")

        index, params, train_time_s = build_empty_index(
            config,
            dim=dim,
            train_vectors=train_vectors,
        )
        if train_vectors is not None:
            del train_vectors
            gc.collect()

        id_map, vectors_added, add_skipped, add_time_s = add_vectors_from_db(
            db,
            config,
            index=index,
            dim=dim,
        )
        skipped_malformed = add_skipped

        index_size_mb, map_size_mb = write_encrypted_snapshot_files(
            index,
            id_map,
            index_path=index_path,
        )
        snapshot_id = create_snapshot_row(
            db,
            index_type=config.index_type,
            params=params,
            index_path=index_path,
            vectors_added=vectors_added,
        )

    total_time_s = time.perf_counter() - started
    return BuildResult(
        selected_embeddings=selected_count,
        vectors_added=vectors_added,
        skipped_malformed=skipped_malformed,
        index_type=config.index_type,
        params=params,
        dim=dim,
        train_time_s=train_time_s,
        add_time_s=add_time_s,
        total_time_s=total_time_s,
        index_file_size_mb=index_size_mb,
        map_file_size_mb=map_size_mb,
        snapshot_id=snapshot_id,
        output_path=str(index_path),
        dry_run=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a backend-compatible FAISS scale index from database embeddings."
    )
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--index-type", choices=INDEX_TYPES, default="ivfpq")
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-size", type=positive_int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--hnsw-m", type=positive_int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=positive_int, default=100)
    parser.add_argument("--hnsw-ef-search", type=positive_int, default=64)
    parser.add_argument("--ivfpq-nlist", type=positive_int, default=4096)
    parser.add_argument("--ivfpq-m", type=positive_int, default=32)
    parser.add_argument("--ivfpq-nbits", type=positive_int, default=8)
    parser.add_argument("--ivfpq-nprobe", type=positive_int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        database_url=args.database_url,
        pipeline=args.pipeline,
        model_name=args.model_name,
        index_type=args.index_type,
        index_path=args.index_path,
        batch_size=args.batch_size,
        train_size=args.train_size,
        dry_run=args.dry_run,
        yes=args.yes,
        limit=args.limit,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        ivfpq_nlist=args.ivfpq_nlist,
        ivfpq_m=args.ivfpq_m,
        ivfpq_nbits=args.ivfpq_nbits,
        ivfpq_nprobe=args.ivfpq_nprobe,
    )

    try:
        result = build_scale_index(config)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(str(exc)) from exc

    print("DONE")
    print(f"  selected_embeddings={result.selected_embeddings}")
    print(f"  vectors_added={result.vectors_added}")
    print(f"  skipped_malformed={result.skipped_malformed}")
    print(f"  index_type={result.index_type}")
    print(f"  params={json.dumps(result.params, sort_keys=True)}")
    print(f"  train_time_s={result.train_time_s:.3f}")
    print(f"  add_time_s={result.add_time_s:.3f}")
    print(f"  total_time_s={result.total_time_s:.3f}")
    print(f"  index_file_size_mb={result.index_file_size_mb:.3f}")
    print(f"  map_file_size_mb={result.map_file_size_mb:.3f}")
    print(f"  snapshot_id={result.snapshot_id}")
    print(f"  output_path={result.output_path}")


if __name__ == "__main__":
    main()
