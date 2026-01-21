import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.index.index_manager import IndexManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from DB embeddings")
    parser.add_argument("--index-type", default=None, choices=["flat", "hnsw", "ivfpq"])
    args = parser.parse_args()

    settings = get_settings()
    index_type = args.index_type or settings.index_type

    with SessionLocal() as db:
        manager = IndexManager(settings)
        params = manager.default_params_for(index_type)
        stats = manager.rebuild(db, index_type=index_type, params=params)
        db.commit()

    print(stats)


if __name__ == "__main__":
    main()
