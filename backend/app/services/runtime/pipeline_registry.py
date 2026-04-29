from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.services.embeddings.interface import (
    DummyEmbeddingExtractor,
    EmbeddingExtractor,
    create_extractor,
)
from app.services.index.index_manager import IndexManager

PipelineKey = Literal["pretrained", "custom"]
EnrollSelection = Literal["pretrained", "custom", "both"]
SearchSelection = Literal["pretrained", "custom"]


@dataclass(frozen=True)
class PipelineRuntime:
    key: PipelineKey
    backend: str
    extractor: EmbeddingExtractor
    index_manager: IndexManager


class PipelineRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipelines: dict[PipelineKey, PipelineRuntime] = {}
        self._errors: dict[PipelineKey, str] = {}
        self.default_pipeline: PipelineKey = settings.default_pipeline
        self._logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        candidates: list[PipelineKey] = []
        if self._settings.enable_pretrained_pipeline:
            candidates.append("pretrained")
        if self._settings.enable_custom_pipeline:
            candidates.append("custom")

        for key in candidates:
            try:
                runtime = self._build_runtime(key)
                self._pipelines[key] = runtime
                self._logger.info(
                    "Pipeline ready: key=%s backend=%s model=%s index_path=%s",
                    key,
                    runtime.backend,
                    runtime.extractor.model_name,
                    runtime.index_manager.index_path,
                )
            except Exception as exc:  # pragma: no cover - exercised in real runtime
                self._errors[key] = str(exc)
                self._logger.warning("Pipeline %s unavailable: %s", key, exc)

        if not self._pipelines:
            raise RuntimeError(
                "No embedding pipelines were initialized successfully. "
                f"Errors: {self._errors}"
            )

        if self.default_pipeline not in self._pipelines:
            self.default_pipeline = next(iter(self._pipelines))

    def load_latest_snapshots(self, db: Session) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for key, runtime in self._pipelines.items():
            try:
                results[key] = runtime.index_manager.load_latest_snapshot(db)
            except Exception as exc:  # pragma: no cover - exercised in real runtime
                self._errors[key] = str(exc)
                self._logger.warning("Failed to load snapshot for %s: %s", key, exc)
                results[key] = False
        return results

    def get(self, key: PipelineKey | None = None) -> PipelineRuntime:
        selected = key or self.default_pipeline
        runtime = self._pipelines.get(selected)
        if runtime is None:
            available = ", ".join(self.available_pipelines()) or "none"
            raise KeyError(f"Pipeline '{selected}' is not available. Available: {available}")
        return runtime

    def resolve_enroll(self, selection: EnrollSelection | None) -> list[PipelineRuntime]:
        selected = selection or self.default_pipeline
        if selected == "both":
            return [self.get("pretrained"), self.get("custom")]
        return [self.get(selected)]

    def resolve_search(self, selection: SearchSelection | None) -> PipelineRuntime:
        return self.get(selection or self.default_pipeline)

    def available_pipelines(self) -> list[str]:
        return list(self._pipelines.keys())

    def unavailable_pipelines(self) -> dict[str, str]:
        return dict(self._errors)

    def status(self) -> dict[str, object]:
        models = {
            key: runtime.extractor.model_name
            for key, runtime in self._pipelines.items()
        }
        backends = {
            key: runtime.backend
            for key, runtime in self._pipelines.items()
        }
        return {
            "default_pipeline": self.default_pipeline,
            "available_pipelines": self.available_pipelines(),
            "unavailable_pipelines": self.unavailable_pipelines(),
            "models": models,
            "backends": backends,
        }

    @property
    def default_runtime(self) -> PipelineRuntime:
        return self.get(self.default_pipeline)

    def _build_runtime(self, key: PipelineKey) -> PipelineRuntime:
        pipeline_settings = self._settings_for(key)
        backend = pipeline_settings.embedding_backend

        if pipeline_settings.testing:
            extractor = DummyEmbeddingExtractor(dim=pipeline_settings.embedding_dim)
            if key != "pretrained":
                extractor.model_name = f"dummy_{key}"
        else:
            extractor = create_extractor(pipeline_settings)

        index_manager = IndexManager(
            pipeline_settings,
            model_name=extractor.model_name,
            pipeline=key,
            index_path_override=pipeline_settings.index_path,
        )
        return PipelineRuntime(
            key=key,
            backend=backend,
            extractor=extractor,
            index_manager=index_manager,
        )

    def _settings_for(self, key: PipelineKey) -> Settings:
        if key == "pretrained":
            backend = self._settings.pretrained_backend
            index_path = self._settings.pretrained_index_path
            detection_backend = self._settings.detection_backend
            allow_center_crop = self._settings.allow_center_crop
        else:
            backend = self._settings.custom_backend
            index_path = self._settings.custom_index_path
            detection_backend = self._settings.custom_detection_backend
            allow_center_crop = self._settings.custom_allow_center_crop

        return self._settings.model_copy(
            update={
                "embedding_backend": backend,
                "index_path": index_path,
                "detection_backend": detection_backend,
                "allow_center_crop": allow_center_crop,
            }
        )
