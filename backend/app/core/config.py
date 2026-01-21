from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "Fast Biometric Face Search"
    api_v1_prefix: str = "/v1"
    log_level: str = "INFO"

    database_url: str = Field(..., alias="DATABASE_URL")
    model_name: str = Field("buffalo_l", alias="MODEL_NAME")

    index_type: Literal["flat", "hnsw", "ivfpq"] = Field("hnsw", alias="INDEX_TYPE")
    index_path: str = Field(
        str(BASE_DIR / "backend" / "data" / "index" / "current.faiss"),
        alias="INDEX_PATH",
    )

    hnsw_m: int = Field(32, alias="HNSW_M")
    hnsw_ef_construction: int = Field(200, alias="HNSW_EF_CONSTRUCTION")
    hnsw_ef_search: int = Field(64, alias="HNSW_EF_SEARCH")

    ivfpq_nlist: int = Field(100, alias="IVFPQ_NLIST")
    ivfpq_m: int = Field(16, alias="IVFPQ_M")
    ivfpq_nbits: int = Field(8, alias="IVFPQ_NBITS")

    strict_single_face: bool = Field(True, alias="STRICT_SINGLE_FACE")
    allow_center_crop: bool = Field(False, alias="ALLOW_CENTER_CROP")
    min_det_score: float = Field(0.5, alias="MIN_DET_SCORE")

    auto_save_index: bool = Field(True, alias="AUTO_SAVE_INDEX")

    seed: int = Field(42, alias="SEED")
    testing: bool = Field(False, alias="TESTING")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
