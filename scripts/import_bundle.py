"""Import pre-computed embeddings from deploy/model_bundle into the backend DB.

Reads ``train.csv`` for person→embedding mapping and loads the pre-built
FAISS HNSW index.  Creates Person + Embedding records so that the desktop
app shows them as already-enrolled faces.

Usage::

    cd <project-root>
    python scripts/import_bundle.py

Options::

    --csv      Path to train.csv            (default: deploy/model_bundle/train.csv)
    --index    Path to faiss.index          (default: deploy/model_bundle/hnsw/faiss.index)
    --db-url   Override DATABASE_URL        (default: from .env)
    --limit    Max persons to import        (default: all)
    --dry-run  Print stats without writing  (default: False)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("import_bundle")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import model_bundle faces into DB")
    parser.add_argument(
        "--csv",
        default=str(ROOT / "deploy" / "model_bundle" / "train.csv"),
        help="Path to train.csv",
    )
    parser.add_argument(
        "--index",
        default=str(ROOT / "deploy" / "model_bundle" / "hnsw" / "faiss.index"),
        help="Path to faiss.index",
    )
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL")
    parser.add_argument("--limit", type=int, default=None, help="Max persons to import")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    parser.add_argument("--pipeline", default="pretrained", help="Pipeline key for model name")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    index_path = Path(args.index)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not index_path.exists():
        raise SystemExit(f"FAISS index not found: {index_path}")

    # ── Load FAISS index ──────────────────────────────────────────────────
    try:
        import faiss
    except ImportError:
        raise SystemExit("faiss-cpu is required: pip install faiss-cpu")

    logger.info("Loading FAISS index from %s ...", index_path)
    faiss_index = faiss.read_index(str(index_path))
    n_vectors = faiss_index.ntotal
    dim = faiss_index.d
    logger.info("FAISS index: %d vectors, dim=%d", n_vectors, dim)

    # ── Parse CSV and group by person ─────────────────────────────────────
    logger.info("Parsing CSV %s ...", csv_path)
    person_embeddings: dict[str, list[int]] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            person_id = row["person_id"]
            emb_idx = int(row["embedding_idx"])
            if emb_idx < n_vectors:
                person_embeddings[person_id].append(emb_idx)

    total_persons = len(person_embeddings)
    total_embeddings = sum(len(v) for v in person_embeddings.values())
    logger.info("CSV: %d unique persons, %d embeddings", total_persons, total_embeddings)

    if args.limit:
        # Take first N persons
        limited = dict(list(person_embeddings.items())[: args.limit])
        person_embeddings = limited
        logger.info("Limited to %d persons", len(person_embeddings))

    if args.dry_run:
        logger.info("DRY RUN — no database writes.")
        for name, indices in list(person_embeddings.items())[:5]:
            logger.info("  %s: %d embeddings (indices %s...)", name, len(indices), indices[:3])
        return

    # ── Reconstruct vectors from FAISS ────────────────────────────────────
    logger.info("Extracting vectors from FAISS index ...")
    all_indices = sorted({idx for idxs in person_embeddings.values() for idx in idxs})
    vectors = np.zeros((max(all_indices) + 1, dim), dtype=np.float32)
    for idx in all_indices:
        vectors[idx] = faiss_index.reconstruct(idx)

    # ── Write to database ─────────────────────────────────────────────────
    if args.db_url:
        import os
        os.environ["DATABASE_URL"] = args.db_url

    # Change to backend/ so relative SQLite paths resolve the same as the server
    import os
    os.chdir(str(ROOT / "backend"))
    logger.info("Working directory: %s", os.getcwd())

    from app.core.config import get_settings  # noqa: E402
    from app.db.session import SessionLocal  # noqa: E402
    from app.services.storage.repositories import EmbeddingRepo, PersonRepo  # noqa: E402

    settings = get_settings()
    model_name = f"onnx_w600k_r50"  # model used to generate the embeddings

    logger.info("Writing to database (url=%s) ...", settings.database_url[:40])

    inserted_persons = 0
    inserted_embeddings = 0

    with SessionLocal() as db:
        person_repo = PersonRepo(db)
        embedding_repo = EmbeddingRepo(db)

        for person_label, emb_indices in person_embeddings.items():
            # Clean up label: AC_Angela_Phuong_Trinh → Angela Phuong Trinh
            clean_label = person_label
            if clean_label.startswith("AC_"):
                clean_label = clean_label[3:]
            clean_label = clean_label.replace("_", " ")

            # Check if person already exists
            existing = person_repo.get_by_label(clean_label)
            if existing is not None:
                logger.debug("Person %s already exists, skipping", clean_label)
                continue

            person = person_repo.create(label=clean_label)
            db.flush()
            inserted_persons += 1

            for emb_idx in emb_indices:
                vector = vectors[emb_idx]
                embedding_repo.create(
                    person_id=person.id,
                    model=model_name,
                    dim=dim,
                    vector=vector.tobytes(),
                )
                inserted_embeddings += 1

            if inserted_persons % 50 == 0:
                db.commit()
                logger.info("  ... %d persons, %d embeddings", inserted_persons, inserted_embeddings)

        db.commit()

    logger.info(
        "DONE: inserted %d persons, %d embeddings",
        inserted_persons,
        inserted_embeddings,
    )

    # ── Copy FAISS index to custom pipeline path ──────────────────────────
    import shutil

    custom_index_path = Path(settings.custom_index_path)
    pretrained_index_path = Path(settings.pretrained_index_path)

    for target_path in [pretrained_index_path]:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(index_path), str(target_path))
        logger.info("Copied FAISS index to %s", target_path)

    # Also write meta.json next to the index
    import json
    meta_src = index_path.parent / "meta.json"
    if meta_src.exists():
        for target_path in [pretrained_index_path]:
            meta_dst = target_path.parent / "meta.json"
            shutil.copy2(str(meta_src), str(meta_dst))
            logger.info("Copied meta.json to %s", meta_dst)


if __name__ == "__main__":
    main()
