from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "DATABASE_URL": "sqlite+pysqlite:///:memory:",
        "TESTING": True,
    }
    values.update(overrides)
    return Settings(**values)


def test_settings_rejects_disabled_default_pipeline():
    with pytest.raises(ValidationError, match="DEFAULT_PIPELINE=custom"):
        _settings(DEFAULT_PIPELINE="custom", ENABLE_CUSTOM_PIPELINE=False)


def test_settings_rejects_missing_onnx_model_paths_when_not_testing():
    with pytest.raises(ValidationError, match="ONNX_DETECTOR_PATH"):
        _settings(
            TESTING=False,
            DEFAULT_PIPELINE="pretrained",
            ENABLE_PRETRAINED_PIPELINE=True,
            ENABLE_CUSTOM_PIPELINE=False,
            PRETRAINED_BACKEND="onnx",
            ONNX_DETECTOR_PATH="",
            ONNX_EMBEDDER_PATH="",
        )


def test_settings_rejects_missing_torch_model_path_when_not_testing():
    with pytest.raises(ValidationError, match="TORCH_MODEL_PATH"):
        _settings(
            TESTING=False,
            DEFAULT_PIPELINE="custom",
            ENABLE_PRETRAINED_PIPELINE=False,
            ENABLE_CUSTOM_PIPELINE=True,
            CUSTOM_BACKEND="torch",
            TORCH_MODEL_PATH="",
        )


def test_settings_rejects_missing_yolo_path_for_torch_yolo_detection():
    with pytest.raises(ValidationError, match="YOLO_MODEL_PATH"):
        _settings(
            TESTING=False,
            DEFAULT_PIPELINE="custom",
            ENABLE_PRETRAINED_PIPELINE=False,
            ENABLE_CUSTOM_PIPELINE=True,
            CUSTOM_BACKEND="torch",
            CUSTOM_DETECTION_BACKEND="yolo",
            TORCH_MODEL_PATH="weights.pth",
            YOLO_MODEL_PATH="",
        )


def test_settings_rejects_wildcard_cors_origins_when_not_testing():
    with pytest.raises(ValidationError, match="CORS_ORIGINS"):
        _settings(
            TESTING=False,
            ENABLE_PRETRAINED_PIPELINE=False,
            ENABLE_CUSTOM_PIPELINE=False,
            CORS_ORIGINS="*",
        )
