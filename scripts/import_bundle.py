"""Import pre-computed model_bundle embeddings into the backend DB.

The source bundle contains ``train.csv`` and a FAISS index with vectors in
``embedding_idx`` order.  This script imports those vectors as backend
``Embedding`` rows for either the pretrained or custom pipeline and can write
the active FAISS artifact to the matching runtime index path.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

PipelineName = Literal["pretrained", "custom"]
DEFAULT_MODEL_NAMES: dict[PipelineName, str] = {
    "pretrained": "onnx_w600k_r50",
    "custom": "torch_ir50",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("import_bundle")


def default_model_name_for_pipeline(pipeline: PipelineName) -> str:
    return DEFAULT_MODEL_NAMES[pipeline]


def clean_person_label(person_id: str) -> str:
    label = person_id
    if label.startswith("AC_"):
        label = label[3:]
    return label.replace("_", " ")


def resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return ROOT / path_obj


def resolve_target_index_path(
    settings,
    pipeline: PipelineName,
    override: str | None,
) -> Path:
    if override:
        return resolve_repo_path(override)
    configured = (
        settings.custom_index_path
        if pipeline == "custom"
        else settings.pretrained_index_path
    )
    return resolve_repo_path(configured)


def load_person_embeddings(
    csv_path: Path,
    n_vectors: int,
    limit: int | None,
) -> dict[str, list[int]]:
    person_embeddings: dict[str, list[int]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            person_id = row["person_id"]
            emb_idx = int(row["embedding_idx"])
            if emb_idx < n_vectors:
                person_embeddings[person_id].append(emb_idx)

    if limit is not None:
        person_embeddings = dict(list(person_embeddings.items())[:limit])
    return person_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import model_bundle faces into DB")
    parser.add_argument(
        "--csv",
        default=str(ROOT / "deploy" / "model_bundle" / "train.csv"),
        help="Source train.csv path",
    )
    parser.add_argument(
        "--index",
        default=str(ROOT / "deploy" / "model_bundle" / "hnsw" / "faiss.index"),
        help="Source FAISS index path",
    )
    parser.add_argument(
        "--index-path",
        dest="target_index_path",
        default=None,
        help="Target runtime FAISS index path override",
    )
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL")
    parser.add_argument("--limit", type=int, default=None, help="Max persons to import")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    parser.add_argument(
        "--pipeline",
        choices=("pretrained", "custom"),
        default="pretrained",
        help="Target pipeline",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Embedding model name stored in DB; defaults from --pipeline",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Deactivate existing active embeddings for the same person/model before import",
    )
    parser.add_argument(
        "--skip-index-copy",
        action="store_true",
        help="Import DB rows without writing the runtime FAISS index file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline: PipelineName = args.pipeline
    model_name = args.model_name or default_model_name_for_pipeline(pipeline)

    csv_path = resolve_repo_path(args.csv)
    index_path = resolve_repo_path(args.index)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not index_path.exists():
        raise SystemExit(f"FAISS index not found: {index_path}")

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("faiss-cpu is required: pip install faiss-cpu") from exc

    logger.info("Loading source FAISS index from %s ...", index_path)
    faiss_index = faiss.read_index(str(index_path))
    n_vectors = faiss_index.ntotal
    dim = faiss_index.d
    logger.info("Source FAISS index: %d vectors, dim=%d", n_vectors, dim)

    logger.info("Parsing CSV %s ...", csv_path)
    person_embeddings = load_person_embeddings(csv_path, n_vectors, args.limit)
    total_persons = len(person_embeddings)
    total_embeddings = sum(len(indices) for indices in person_embeddings.values())
    logger.info("CSV: %d unique persons, %d embeddings", total_persons, total_embeddings)

    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    os.chdir(str(ROOT / "backend"))
    logger.info("Working directory: %s", os.getcwd())

    from sqlalchemy import func, select  # noqa: E402

    from app.core.config import get_settings  # noqa: E402
    from app.db.models import Embedding  # noqa: E402
    from app.db.session import SessionLocal  # noqa: E402
    from app.services.storage.repositories import EmbeddingRepo, PersonRepo  # noqa: E402

    settings = get_settings()
    target_index_path = resolve_target_index_path(
        settings,
        pipeline,
        args.target_index_path,
    )

    logger.info("Target pipeline: %s", pipeline)
    logger.info("Target model name: %s", model_name)
    logger.info("Target index path: %s", target_index_path)
    logger.info("Database URL scheme: %s", settings.database_url.split(":", 1)[0])

    with SessionLocal() as db:
        existing_active = db.execute(
            select(func.count())
            .select_from(Embedding)
            .where(
                Embedding.model == model_name,
                Embedding.pipeline == pipeline,
                Embedding.is_active.is_(True),
            )
        ).scalar_one()
    logger.info("Existing active embeddings for %s: %d", model_name, existing_active)

    if args.dry_run:
        logger.info("DRY RUN - no database or index writes.")
        return

    if not person_embeddings:
        raise SystemExit("No embeddings found in CSV")

    logger.info("Reconstructing vectors from source FAISS index ...")
    all_indices = sorted({idx for indices in person_embeddings.values() for idx in indices})
    vectors = np.zeros((max(all_indices) + 1, dim), dtype=np.float32)
    for idx in all_indices:
        vectors[idx] = faiss_index.reconstruct(idx)

    inserted_persons = 0
    reused_persons = 0
    inserted_embeddings = 0
    skipped_persons = 0
    skipped_embeddings = 0
    id_map: dict[int, str] = {}

    logger.info("Writing embeddings to database ...")
    with SessionLocal() as db:
        person_repo = PersonRepo(db)
        embedding_repo = EmbeddingRepo(db)

        for person_label, emb_indices in person_embeddings.items():
            clean_label = clean_person_label(person_label)
            person = person_repo.get_by_label(clean_label)
            if person is None:
                person = person_repo.create(label=clean_label)
                db.flush()
                inserted_persons += 1
            else:
                reused_persons += 1

            existing_rows = list(
                db.execute(
                    select(Embedding).where(
                        Embedding.person_id == person.id,
                        Embedding.model == model_name,
                        Embedding.pipeline == pipeline,
                        Embedding.is_active.is_(True),
                    )
                ).scalars()
            )
            if existing_rows:
                if args.replace_existing:
                    for row in existing_rows:
                        row.is_active = False
                    db.flush()
                else:
                    skipped_persons += 1
                    skipped_embeddings += len(emb_indices)
                    continue

            for emb_idx in emb_indices:
                embedding = embedding_repo.create(
                    person_id=person.id,
                    model=model_name,
                    pipeline=pipeline,
                    dim=dim,
                    vector=vectors[emb_idx].tobytes(),
                )
                db.flush()
                id_map[emb_idx] = str(embedding.id)
                inserted_embeddings += 1

            if (inserted_persons + reused_persons) % 50 == 0:
                db.commit()
                logger.info(
                    "  ... processed=%d inserted_embeddings=%d",
                    inserted_persons + reused_persons,
                    inserted_embeddings,
                )

        db.commit()

    logger.info(
        "DONE: inserted_persons=%d reused_persons=%d inserted_embeddings=%d "
        "skipped_persons=%d skipped_embeddings=%d",
        inserted_persons,
        reused_persons,
        inserted_embeddings,
        skipped_persons,
        skipped_embeddings,
    )

    if args.skip_index_copy:
        logger.info("Skipping active FAISS index copy by request.")
        return

    if len(id_map) != n_vectors:
        logger.warning(
            "Not copying source index because id map is partial: %d/%d vectors. "
            "Run with --replace-existing or rebuild the target pipeline index.",
            len(id_map),
            n_vectors,
        )
        return

    target_index_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(index_path), str(target_index_path))
    map_path = target_index_path.with_suffix(target_index_path.suffix + ".map.json")
    with map_path.open("w", encoding="utf-8") as handle:
        json.dump(id_map, handle)
    logger.info("Wrote runtime FAISS index to %s", target_index_path)
    logger.info("Wrote runtime FAISS id map to %s", map_path)

    meta_src = index_path.parent / "meta.json"
    if meta_src.exists():
        meta_dst = target_index_path.with_suffix(target_index_path.suffix + ".meta.json")
        shutil.copy2(str(meta_src), str(meta_dst))
        logger.info("Copied index metadata to %s", meta_dst)


if __name__ == "__main__":
    main()
