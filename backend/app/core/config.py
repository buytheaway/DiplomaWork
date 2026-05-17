"""Application configuration via environment variables / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
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
    pretrained_match_threshold: float | None = Field(None, alias="PRETRAINED_MATCH_THRESHOLD")
    custom_match_threshold: float | None = Field(None, alias="CUSTOM_MATCH_THRESHOLD")
    custom_live_match_threshold: float | None = Field(None, alias="CUSTOM_LIVE_MATCH_THRESHOLD")
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
    custom_min_det_score: float | None = Field(
        None, ge=0.0, le=1.0, alias="CUSTOM_MIN_DET_SCORE"
    )
    min_face_size_px: int = Field(0, ge=0, alias="MIN_FACE_SIZE_PX")
    min_face_area_ratio: float = Field(0.0, ge=0.0, le=1.0, alias="MIN_FACE_AREA_RATIO")
    min_face_blur_variance: float = Field(0.0, ge=0.0, alias="MIN_FACE_BLUR_VARIANCE")
    face_crop_margin: float = Field(0.0, ge=0.0, le=1.0, alias="FACE_CROP_MARGIN")
    custom_face_crop_margin: float | None = Field(
        0.20, ge=0.0, le=1.0, alias="CUSTOM_FACE_CROP_MARGIN"
    )
    yolo_imgsz: int = Field(640, ge=160, alias="YOLO_IMGSZ")
    custom_yolo_imgsz: int | None = Field(None, ge=160, alias="CUSTOM_YOLO_IMGSZ")

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
    ivfpq_nprobe: int = Field(16, alias="IVFPQ_NPROBE")
    search_candidate_k: int = Field(200, alias="SEARCH_CANDIDATE_K")
    search_dynamic_enabled: bool = Field(True, alias="SEARCH_DYNAMIC_ENABLED")
    search_candidate_k_fast: int = Field(100, alias="SEARCH_CANDIDATE_K_FAST")
    search_candidate_k_safe: int = Field(500, alias="SEARCH_CANDIDATE_K_SAFE")
    ivfpq_nprobe_fast: int = Field(32, alias="IVFPQ_NPROBE_FAST")
    ivfpq_nprobe_safe: int = Field(128, alias="IVFPQ_NPROBE_SAFE")
    search_fallback_margin: float = Field(0.05, alias="SEARCH_FALLBACK_MARGIN")
    index_snapshot_retention: int = Field(3, alias="INDEX_SNAPSHOT_RETENTION")

    # ── security / CORS ──────────────────────────────────────────────────
    api_key: str = Field("", alias="API_KEY")
    admin_api_key: str = Field("", alias="ADMIN_API_KEY")
    cors_origins: str = Field(
        "http://127.0.0.1:8000,http://localhost:8000",
        alias="CORS_ORIGINS",
    )
    max_upload_bytes: int = Field(10 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")
    allowed_image_content_types: str = Field(
        "image/jpeg,image/png,image/webp,image/bmp",
        alias="ALLOWED_IMAGE_CONTENT_TYPES",
    )
    data_encryption_key: str = Field("", alias="DATA_ENCRYPTION_KEY")
    snapshot_encryption_key: str = Field("", alias="SNAPSHOT_ENCRYPTION_KEY")
    audit_retention_days: int = Field(30, alias="AUDIT_RETENTION_DAYS")
    enable_search_audit: bool = Field(True, alias="ENABLE_SEARCH_AUDIT")
    rate_limit_enabled: bool = Field(False, alias="RATE_LIMIT_ENABLED")
    rate_limit_search_per_min: int = Field(60, alias="RATE_LIMIT_SEARCH_PER_MIN")
    rate_limit_enroll_per_min: int = Field(20, alias="RATE_LIMIT_ENROLL_PER_MIN")
    rate_limit_admin_per_min: int = Field(10, alias="RATE_LIMIT_ADMIN_PER_MIN")

    # ── runtime ──────────────────────────────────────────────────────────
    auto_save_index: bool = Field(True, alias="AUTO_SAVE_INDEX")

    @model_validator(mode="after")
    def validate_runtime_configuration(self) -> Settings:
        if self.default_pipeline == "pretrained" and not self.enable_pretrained_pipeline:
            raise ValueError("DEFAULT_PIPELINE=pretrained requires ENABLE_PRETRAINED_PIPELINE=true")
        if self.default_pipeline == "custom" and not self.enable_custom_pipeline:
            raise ValueError("DEFAULT_PIPELINE=custom requires ENABLE_CUSTOM_PIPELINE=true")

        if self.testing:
            return self

        configured = [
            (
                "pretrained",
                self.enable_pretrained_pipeline,
                self.pretrained_backend,
                self.detection_backend,
            ),
            (
                "custom",
                self.enable_custom_pipeline,
                self.custom_backend,
                self.custom_detection_backend,
            ),
        ]
        for pipeline, enabled, backend, detection_backend in configured:
            if not enabled:
                continue
            if backend == "onnx":
                missing = [
                    name
                    for name, value in (
                        ("ONNX_DETECTOR_PATH", self.onnx_detector_path),
                        ("ONNX_EMBEDDER_PATH", self.onnx_embedder_path),
                    )
                    if not value
                ]
                if missing:
                    raise ValueError(f"{pipeline} ONNX backend requires {', '.join(missing)}")
            if backend == "torch":
                if not self.torch_model_path:
                    raise ValueError(f"{pipeline} torch backend requires TORCH_MODEL_PATH")
                if detection_backend == "yolo" and not self.yolo_model_path:
                    raise ValueError(f"{pipeline} YOLO detection requires YOLO_MODEL_PATH")

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenience alias used by modules that import at module level.
settings = get_settings()
