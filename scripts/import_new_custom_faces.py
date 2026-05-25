"""Import source face images into the new custom Torch runtime embedding space.

This script intentionally creates new embedding rows for the current custom
runtime model and never mutates historical embeddings from older custom models.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings, settings  # noqa: E402
from app.db.models import Embedding, Person  # noqa: E402
from app.services.embeddings.interface import (  # noqa: E402
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    create_extractor,
)
from app.services.index.index_manager import IndexManager  # noqa: E402
from app.services.storage.repositories import EmbeddingRepo, PersonRepo  # noqa: E402

EXPECTED_MODEL_NAME = "torch_insightface_iresnet100"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImportItem:
    person_id: str | None
    label: str
    image_path: Path


@dataclass
class ImportStats:
    persons_created: int = 0
    persons_reused: int = 0
    embeddings_created: int = 0
    skipped_no_face: int = 0
    skipped_invalid: int = 0
    skipped_multiple_faces: int = 0
    errors: int = 0
    old_custom_before: int = 0
    old_custom_after: int = 0
    new_custom_before: int = 0
    new_custom_after: int = 0
    rebuild_time_s: float = 0.0
    index_path: str | None = None
    index_size_bytes: int = 0
    index_embeddings_count: int = 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def custom_runtime_settings(base: Settings) -> Settings:
    updates = {
        "embedding_backend": base.custom_backend,
        "index_path": base.custom_index_path,
        "detection_backend": base.custom_detection_backend,
        "allow_center_crop": base.custom_allow_center_crop,
    }
    if base.custom_min_det_score is not None:
        updates["min_det_score"] = base.custom_min_det_score
    if base.custom_face_crop_margin is not None:
        updates["face_crop_margin"] = base.custom_face_crop_margin
    if base.custom_yolo_imgsz is not None:
        updates["yolo_imgsz"] = base.custom_yolo_imgsz
    return base.model_copy(update=updates)


def validate_runtime(custom_settings: Settings) -> None:
    if custom_settings.embedding_backend != "torch":
        raise RuntimeError("CUSTOM_BACKEND must resolve to torch for this importer")
    if custom_settings.torch_model_arch != "insightface_iresnet100":
        raise RuntimeError(
            "TORCH_MODEL_ARCH must be insightface_iresnet100 for the new custom model"
        )
    if custom_settings.torch_preprocess != "runtime_fallback_center_crop":
        raise RuntimeError("TORCH_PREPROCESS must be runtime_fallback_center_crop")
    if custom_settings.torch_tta != "hflip":
        raise RuntimeError("TORCH_TTA must be hflip")


def load_csv_items(csv_path: Path) -> list[ImportItem]:
    items: list[ImportItem] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "image_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            label = (row.get("label") or "").strip()
            image = (row.get("image_path") or "").strip()
            if not label or not image:
                continue
            items.append(
                ImportItem(
                    person_id=(row.get("person_id") or "").strip() or None,
                    label=label,
                    image_path=resolve_path(image),
                )
            )
    return items


def load_folder_items(folder: Path) -> list[ImportItem]:
    items: list[ImportItem] = []
    for person_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
        label = person_dir.name.strip()
        if not label:
            continue
        for image_path in sorted(person_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                items.append(ImportItem(person_id=None, label=label, image_path=image_path.resolve()))
    return items


def get_or_create_person(
    db: Session,
    person_repo: PersonRepo,
    item: ImportItem,
    stats: ImportStats,
) -> Person:
    person: Person | None = None
    if item.person_id:
        try:
            person = person_repo.get(str(uuid.UUID(item.person_id)))
        except ValueError:
            person = None
        if person is not None and person.status != "active":
            person = None

    if person is None:
        person = person_repo.get_active_by_label(item.label)

    if person is not None:
        stats.persons_reused += 1
        return person

    person = person_repo.create(item.label)
    db.flush()
    stats.persons_created += 1
    return person


def count_active_embeddings(db: Session, *, pipeline: str, model: str) -> int:
    stmt = (
        select(func.count())
        .select_from(Embedding)
        .join(Person, Embedding.person_id == Person.id)
        .where(
            Embedding.pipeline == pipeline,
            Embedding.model == model,
            Embedding.is_active.is_(True),
            Person.status == "active",
        )
    )
    return int(db.execute(stmt).scalar_one())


def import_items(
    items: list[ImportItem],
    *,
    custom_settings: Settings,
    batch_commit: int,
    dry_run: bool,
) -> ImportStats:
    validate_runtime(custom_settings)
    extractor = create_extractor(custom_settings)
    if extractor.model_name != EXPECTED_MODEL_NAME:
        raise RuntimeError(
            f"Extractor model mismatch: expected {EXPECTED_MODEL_NAME}, got {extractor.model_name}"
        )

    engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
    session_factory = sessionmaker(bind=engine, autoflush=False, future=True)
    stats = ImportStats()

    with session_factory() as db:
        stats.old_custom_before = count_active_embeddings(
            db, pipeline="custom", model="torch_ir50"
        )
        stats.new_custom_before = count_active_embeddings(
            db, pipeline="custom", model=EXPECTED_MODEL_NAME
        )

        if dry_run:
            stats.old_custom_after = stats.old_custom_before
            stats.new_custom_after = stats.new_custom_before
            return stats

        person_repo = PersonRepo(db)
        embedding_repo = EmbeddingRepo(db)
        pending = 0
        for item in items:
            if not item.image_path.exists():
                stats.skipped_invalid += 1
                continue
            try:
                vector = extractor.extract_embedding(item.image_path.read_bytes())
            except NoFaceDetectedError:
                stats.skipped_no_face += 1
                continue
            except MultipleFacesDetectedError:
                stats.skipped_multiple_faces += 1
                continue
            except InvalidImageError:
                stats.skipped_invalid += 1
                continue
            except Exception:  # noqa: BLE001
                stats.errors += 1
                continue

            person = get_or_create_person(db, person_repo, item, stats)
            embedding_repo.create(
                person_id=person.id,
                pipeline="custom",
                model=EXPECTED_MODEL_NAME,
                dim=int(vector.shape[0]),
                vector=vector.tobytes(),
            )
            stats.embeddings_created += 1
            pending += 1
            if pending >= batch_commit:
                db.commit()
                pending = 0
        if pending:
            db.commit()

        stats.old_custom_after = count_active_embeddings(
            db, pipeline="custom", model="torch_ir50"
        )
        stats.new_custom_after = count_active_embeddings(
            db, pipeline="custom", model=EXPECTED_MODEL_NAME
        )

        if stats.old_custom_before != stats.old_custom_after:
            raise RuntimeError("Safety violation: old torch_ir50 embedding count changed")

        started = time.perf_counter()
        index_manager = IndexManager(
            custom_settings,
            model_name=extractor.model_name,
            pipeline="custom",
            index_path_override=custom_settings.index_path,
        )
        index_stats = index_manager.rebuild(db, custom_settings.index_type, {})
        db.commit()
        stats.rebuild_time_s = time.perf_counter() - started
        stats.index_path = str(index_stats["file_path"])
        stats.index_embeddings_count = int(index_stats["embeddings_count"])
        path = Path(stats.index_path)
        stats.index_size_bytes = path.stat().st_size if path.exists() else 0

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import images into the new custom torch_insightface_iresnet100 embedding space."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", dest="csv_path")
    source.add_argument("--folder", dest="folder_path")
    parser.add_argument("--batch-commit", type=positive_int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    custom_settings = custom_runtime_settings(settings)
    items = (
        load_csv_items(resolve_path(args.csv_path))
        if args.csv_path
        else load_folder_items(resolve_path(args.folder_path))
    )
    if not items:
        raise SystemExit("No importable image rows found")

    started = time.perf_counter()
    stats = import_items(
        items,
        custom_settings=custom_settings,
        batch_commit=args.batch_commit,
        dry_run=args.dry_run,
    )
    elapsed_s = time.perf_counter() - started
    summary = {
        "dry_run": bool(args.dry_run),
        "items_seen": len(items),
        "persons_created": stats.persons_created,
        "persons_reused": stats.persons_reused,
        "embeddings_created": stats.embeddings_created,
        "skipped_no_face": stats.skipped_no_face,
        "skipped_invalid": stats.skipped_invalid,
        "skipped_multiple_faces": stats.skipped_multiple_faces,
        "errors": stats.errors,
        "old_torch_ir50_before": stats.old_custom_before,
        "old_torch_ir50_after": stats.old_custom_after,
        "new_model_before": stats.new_custom_before,
        "new_model_after": stats.new_custom_after,
        "new_index_path": stats.index_path,
        "new_index_size_bytes": stats.index_size_bytes,
        "new_index_embeddings_count": stats.index_embeddings_count,
        "rebuild_time_s": round(stats.rebuild_time_s, 3),
        "total_time_s": round(elapsed_s, 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
