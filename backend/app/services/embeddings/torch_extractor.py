from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.core.config import BASE_DIR, Settings
from app.services.embeddings.interface import (
    EmbeddingExtractor,
    FaceEmbedding,
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.services.embeddings.torch_model import ModelConfig, build_model, forward_with_normalization
from app.services.face.align import AlignmentError, align_with_landmarks
from app.services.face.detector import InsightFaceDetector, OpenCVHaarDetector, decode_image
from app.services.face.quality import validate_face_quality


class TorchEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self, settings: Settings) -> None:
        import torch

        self.model_name = f"torch_{settings.torch_model_arch}"
        self.dim = 512
        self.strict_single_face = settings.strict_single_face
        self.min_det_score = settings.min_det_score
        self.min_face_size_px = settings.min_face_size_px
        self.min_face_area_ratio = settings.min_face_area_ratio
        self.min_face_blur_variance = settings.min_face_blur_variance
        self.face_crop_margin = settings.face_crop_margin
        self.allow_center_crop = settings.allow_center_crop
        self.input_size = settings.torch_input_size
        self.use_fp16 = settings.torch_use_fp16
        self.norm_embeddings = settings.torch_norm_embeddings
        self.device = torch.device(settings.torch_device)

        model_config = ModelConfig(
            arch=settings.torch_model_arch,
            embedding_dim=self.dim,
            norm_embeddings=settings.torch_norm_embeddings,
        )
        self.model = build_model(model_config).to(self.device).eval()

        if not settings.torch_model_path:
            raise ValueError("TORCH_MODEL_PATH is required for torch embedding backend")

        raw_path = Path(settings.torch_model_path)
        candidates = [raw_path, BASE_DIR / raw_path, BASE_DIR / "backend" / raw_path]
        weights_path = next((candidate for candidate in candidates if candidate.exists()), raw_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Torch model not found: {weights_path}")

        state = torch.load(str(weights_path), map_location=self.device, weights_only=False)
        state_dict = state.get("state_dict", state)
        self.model.load_state_dict(state_dict, strict=False)

        if settings.detection_backend == "yolo":
            from app.services.face.yolo_detector import YoloFaceDetector

            yolo_path = Path(settings.yolo_model_path)
            if not yolo_path.is_absolute():
                yolo_path = BASE_DIR / yolo_path
            self.detector = YoloFaceDetector(
                str(yolo_path),
                conf_threshold=settings.min_det_score,
                imgsz=settings.yolo_imgsz,
                device=str(self.device),
            )
        elif settings.detection_backend == "opencv":
            self.detector = OpenCVHaarDetector()
        else:
            self.detector = InsightFaceDetector(model_name=settings.model_name)
        logging.getLogger(__name__).info("Torch embedding model loaded: %s", weights_path)

    def _extract_face_embedding(self, image: np.ndarray, face) -> FaceEmbedding:
        import torch

        validate_face_quality(
            face,
            self.min_det_score,
            image=image,
            min_face_size_px=self.min_face_size_px,
            min_face_area_ratio=self.min_face_area_ratio,
            min_blur_variance=self.min_face_blur_variance,
        )

        try:
            align_bbox = self._expanded_bbox(image, face.bbox)
            aligned = align_with_landmarks(
                image, face.kps, self.allow_center_crop, bbox=align_bbox
            )
        except AlignmentError as exc:
            raise NoFaceDetectedError(str(exc)) from exc

        import cv2

        if aligned.shape[0] != self.input_size or aligned.shape[1] != self.input_size:
            aligned = cv2.resize(aligned, (self.input_size, self.input_size))

        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    embeddings = forward_with_normalization(
                        self.model, tensor, normalize=self.norm_embeddings
                    )
            else:
                embeddings = forward_with_normalization(
                    self.model, tensor, normalize=self.norm_embeddings
                )

        vector = embeddings.squeeze(0).float().cpu().numpy().astype(np.float32)
        if vector.shape[0] != self.dim:
            raise NoFaceDetectedError("Unexpected embedding dimension")

        bbox = tuple(float(value) for value in face.bbox.tolist()) if face.bbox is not None else None
        return FaceEmbedding(
            embedding=vector,
            detection_score=float(face.det_score),
            bbox=bbox,
        )

    def _expanded_bbox(self, image: np.ndarray, bbox: np.ndarray | None) -> np.ndarray | None:
        if bbox is None or len(bbox) < 4 or self.face_crop_margin <= 0.0:
            return bbox

        height, width = image.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        margin_x = box_w * self.face_crop_margin
        margin_y = box_h * self.face_crop_margin
        return np.array(
            [
                max(0.0, x1 - margin_x),
                max(0.0, y1 - margin_y),
                min(float(width), x2 + margin_x),
                min(float(height), y2 + margin_y),
            ],
            dtype=np.float32,
        )

    def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbedding]:
        if not image_bytes:
            raise InvalidImageError("Empty image bytes")

        image = decode_image(image_bytes)
        faces = self.detector.detect(image)
        if not faces:
            raise NoFaceDetectedError("No face detected")

        ordered_faces = sorted(faces, key=lambda detected: detected.det_score, reverse=True)
        embeddings: list[FaceEmbedding] = []
        for idx, face in enumerate(ordered_faces):
            try:
                embeddings.append(self._extract_face_embedding(image, face))
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Skipping torch face %s due to extraction error: %s",
                    idx,
                    exc,
                )

        if not embeddings:
            raise NoFaceDetectedError("No valid face crops could be embedded")

        return embeddings

    def extract_largest_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        if not image_bytes:
            raise InvalidImageError("Empty image bytes")

        image = decode_image(image_bytes)
        faces = self.detector.detect(image)
        if not faces:
            raise NoFaceDetectedError("No face detected")

        def area(face) -> float:
            if face.bbox is None:
                return 0.0
            x1, y1, x2, y2 = [float(value) for value in face.bbox[:4]]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        return self._extract_face_embedding(image, max(faces, key=area))

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        # Early check: reject multi-face images before expensive embedding step.
        if self.strict_single_face:
            if not image_bytes:
                raise InvalidImageError("Empty image bytes")
            image = decode_image(image_bytes)
            faces = self.detector.detect(image)
            if not faces:
                raise NoFaceDetectedError("No face detected")
            if len(faces) != 1:
                raise MultipleFacesDetectedError("Multiple faces detected")
            best = max(faces, key=lambda f: f.det_score)
            return self._extract_face_embedding(image, best).embedding

        embeddings = self.extract_embeddings(image_bytes)
        return embeddings[0].embedding
