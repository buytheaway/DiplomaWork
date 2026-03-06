"""Batch‑compute face embeddings from a dataset folder and insert into the DB.

Dataset format::

    dataset/<identity>/*.(jpg|jpeg|png|bmp)

Uses whichever ``EMBEDDING_BACKEND`` is configured (dummy / insightface / torch / onnx).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services.embeddings.interface import (  # noqa: E402
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    create_extractor,
)
from app.services.storage.repositories import EmbeddingRepo, PersonRepo  # noqa: E402

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_dataset(dataset_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield ``(label, image_path)`` in deterministic order."""
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label = person_dir.name
        files = sorted(
            (
                p
                for p in person_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda p: p.name,
        )
        for file_path in files:
            yield label, file_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute embeddings from dataset folder")
    parser.add_argument("--dataset", required=True, help="Path to dataset folder")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override SEED from config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("compute_embeddings")

    settings = get_settings()
    seed = args.seed if args.seed is not None else settings.seed
    random.seed(seed)
    logger.info("deterministic_order=true  seed=%d  backend=%s", seed, settings.embedding_backend)

    extractor = create_extractor(settings)
    logger.info("Extractor: model=%s  dim=%d", extractor.model_name, extractor.dim)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset path not found: {dataset_dir}")

    processed = 0
    inserted = 0
    skipped_no_face = 0
    skipped_multi_face = 0
    skipped_invalid = 0
    other_errors = 0
    pending_inserts = 0

    with SessionLocal() as db:
        person_repo = PersonRepo(db)
        embedding_repo = EmbeddingRepo(db)
        person_cache: dict[str, object] = {}

        def commit_batch() -> None:
            nonlocal inserted, pending_inserts
            if pending_inserts == 0:
                return
            try:
                db.commit()
                inserted += pending_inserts
                pending_inserts = 0
            except Exception:
                db.rollback()
                raise

        for label, img_path in iter_dataset(dataset_dir):
            if args.max_images is not None and processed >= args.max_images:
                break

            processed += 1
            try:
                person = person_cache.get(label)
                if person is None:
                    person = person_repo.get_by_label(label)
                    if person is None:
                        person = person_repo.create(label=label)
                        db.flush()
                    person_cache[label] = person

                image_bytes = img_path.read_bytes()
                embedding = extractor.extract_embedding(image_bytes)
                embedding_repo.create(
                    person_id=person.id,
                    model=extractor.model_name,
                    dim=int(embedding.shape[0]),
                    vector=embedding.tobytes(),
                )
                pending_inserts += 1
            except NoFaceDetectedError:
                skipped_no_face += 1
            except MultipleFacesDetectedError:
                skipped_multi_face += 1
            except InvalidImageError:
                skipped_invalid += 1
            except Exception:
                other_errors += 1
                logger.exception("Unexpected error processing %s", img_path)

            if pending_inserts >= args.batch_size:
                try:
                    commit_batch()
                except Exception as exc:
                    other_errors += pending_inserts
                    logger.exception("Batch commit failed: %s", exc)
                    raise SystemExit(1) from exc

            if processed % args.log_every == 0:
                logger.info(
                    "processed=%d inserted=%d skipped_no_face=%d skipped_multi=%d "
                    "skipped_invalid=%d errors=%d",
                    processed,
                    inserted,
                    skipped_no_face,
                    skipped_multi_face,
                    skipped_invalid,
                    other_errors,
                )

        try:
            commit_batch()
        except Exception as exc:
            other_errors += pending_inserts
            logger.exception("Final batch commit failed: %s", exc)
            raise SystemExit(1) from exc

    logger.info(
        "DONE processed=%d inserted=%d skipped_no_face=%d skipped_multi=%d "
        "skipped_invalid=%d errors=%d",
        processed,
        inserted,
        skipped_no_face,
        skipped_multi_face,
        skipped_invalid,
        other_errors,
    )


if __name__ == "__main__":
    main()
