"""Application configuration via environment variables / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """All backend settings.  Loaded from ``.env`` + environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── general ──────────────────────────────────────────────────────────
    app_name: str = "Fast Biometric Face Search"
    api_v1_prefix: str = "/v1"
    log_level: str = "INFO"
    seed: int = Field(42, alias="SEED")
    testing: bool = Field(False, alias="TESTING")

    # ── database ─────────────────────────────────────────────────────────
    database_url: str = Field(..., alias="DATABASE_URL")

    # ── embedding extractor (plugin) ─────────────────────────────────────
    embedding_backend: Literal["dummy", "insightface", "torch", "onnx"] = Field(
        "dummy", alias="EMBEDDING_BACKEND"
    )
    embedding_dim: int = Field(512, alias="EMBEDDING_DIM")
    default_pipeline: Literal["pretrained", "custom"] = Field(
        "pretrained", alias="DEFAULT_PIPELINE"
    )
    enable_pretrained_pipeline: bool = Field(True, alias="ENABLE_PRETRAINED_PIPELINE")
    enable_custom_pipeline: bool = Field(True, alias="ENABLE_CUSTOM_PIPELINE")
    pretrained_backend: Literal["dummy", "insightface", "torch", "onnx"] = Field(
        "onnx", alias="PRETRAINED_BACKEND"
    )
    custom_backend: Literal["dummy", "insightface", "torch", "onnx"] = Field(
        "torch", alias="CUSTOM_BACKEND"
    )

    # insightface-specific
    model_name: str = Field("buffalo_l", alias="MODEL_NAME")

    # torch-specific
    torch_model_path: str = Field("", alias="TORCH_MODEL_PATH")
    torch_model_arch: str = Field("ir50", alias="TORCH_MODEL_ARCH")
    torch_input_size: int = Field(112, alias="TORCH_INPUT_SIZE")
    torch_device: str = Field("cpu", alias="TORCH_DEVICE")
    torch_use_fp16: bool = Field(False, alias="TORCH_USE_FP16")
    torch_norm_embeddings: bool = Field(True, alias="TORCH_NORM_EMBEDDINGS")

    # onnx-specific
    onnx_model_path: str = Field("", alias="ONNX_MODEL_PATH")  # legacy single-model
    onnx_detector_path: str = Field("", alias="ONNX_DETECTOR_PATH")
    onnx_embedder_path: str = Field("", alias="ONNX_EMBEDDER_PATH")

    # ── face detection / quality ─────────────────────────────────────────
    strict_single_face: bool = Field(True, alias="STRICT_SINGLE_FACE")
    match_threshold: float = Field(0.4, alias="MATCH_THRESHOLD")
    detection_backend: Literal["insightface", "opencv", "yolo", "none"] = Field(
        "none", alias="DETECTION_BACKEND"
    )
    custom_detection_backend: Literal["insightface", "opencv", "yolo", "none"] = Field(
        "opencv", alias="CUSTOM_DETECTION_BACKEND"
    )
    yolo_model_path: str = Field("", alias="YOLO_MODEL_PATH")
    allow_center_crop: bool = Field(False, alias="ALLOW_CENTER_CROP")
    custom_allow_center_crop: bool = Field(True, alias="CUSTOM_ALLOW_CENTER_CROP")
    min_det_score: float = Field(0.5, alias="MIN_DET_SCORE")

    # ── vector index ─────────────────────────────────────────────────────
    index_type: Literal["flat", "hnsw", "ivfpq"] = Field("hnsw", alias="INDEX_TYPE")
    index_path: str = Field(
        str(BASE_DIR / "backend" / "data" / "index" / "current.faiss"),
        alias="INDEX_PATH",
    )
    pretrained_index_path: str = Field(
        str(BASE_DIR / "backend" / "data" / "index" / "pretrained.faiss"),
        alias="PRETRAINED_INDEX_PATH",
    )
    custom_index_path: str = Field(
        str(BASE_DIR / "backend" / "data" / "index" / "custom.faiss"),
        alias="CUSTOM_INDEX_PATH",
    )
    hnsw_m: int = Field(32, alias="HNSW_M")
    hnsw_ef_construction: int = Field(200, alias="HNSW_EF_CONSTRUCTION")
    hnsw_ef_search: int = Field(64, alias="HNSW_EF_SEARCH")
    ivfpq_nlist: int = Field(100, alias="IVFPQ_NLIST")
    ivfpq_m: int = Field(16, alias="IVFPQ_M")
    ivfpq_nbits: int = Field(8, alias="IVFPQ_NBITS")

    # ── security / CORS ──────────────────────────────────────────────────
    api_key: str = Field("", alias="API_KEY")
    cors_origins: str = Field("*", alias="CORS_ORIGINS")

    # ── runtime ──────────────────────────────────────────────────────────
    auto_save_index: bool = Field(True, alias="AUTO_SAVE_INDEX")


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenience alias used by modules that import at module level.
settings = get_settings()
