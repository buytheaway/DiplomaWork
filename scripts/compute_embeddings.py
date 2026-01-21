import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.embeddings.insightface_extractor import InsightFaceEmbeddingExtractor
from app.services.storage.repositories import EmbeddingRepo, PersonRepo


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute embeddings from dataset folder")
    parser.add_argument("--dataset", required=True, help="Path to dataset folder")
    args = parser.parse_args()

    settings = get_settings()
    extractor = InsightFaceEmbeddingExtractor(settings)
    dataset_dir = Path(args.dataset)

    with SessionLocal() as db:
        person_repo = PersonRepo(db)
        embedding_repo = EmbeddingRepo(db)

        for person_dir in dataset_dir.iterdir():
            if not person_dir.is_dir():
                continue
            label = person_dir.name
            person = person_repo.get_by_label(label)
            if person is None:
                person = person_repo.create(label=label)
                db.commit()
                db.refresh(person)

            for img_path in person_dir.glob("*.*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                image_bytes = img_path.read_bytes()
                try:
                    embedding = extractor.extract_embedding(image_bytes)
                except Exception:
                    continue
                embedding_repo.create(
                    person_id=person.id,
                    model=extractor.model_name,
                    dim=int(embedding.shape[0]),
                    vector=embedding.tobytes(),
                )
                db.commit()


if __name__ == "__main__":
    main()
