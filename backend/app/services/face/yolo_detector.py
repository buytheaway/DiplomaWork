"""YOLOv8-based face detector using a custom-trained ultralytics model.

Loads a ``.pt`` checkpoint via ``ultralytics.YOLO`` and returns
``DetectedFace`` objects compatible with the existing detector interface.

Configure via ``.env``::

    CUSTOM_DETECTION_BACKEND=yolo
    YOLO_MODEL_PATH=../deploy/model_bundle/best.pt
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.services.face.detector import DetectedFace

logger = logging.getLogger(__name__)


class YoloFaceDetector:
    """Detect faces using an Ultralytics YOLO model."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for the YOLO detection backend. "
                "Install with: pip install ultralytics"
            ) from exc

        resolved = Path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(f"YOLO model not found: {resolved}")

        self._model = YOLO(str(resolved))
        self._conf = conf_threshold
        logger.info("YOLO face detector loaded: %s (conf=%.2f)", resolved.name, conf_threshold)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Run YOLO inference and return detected faces.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image (OpenCV format).

        Returns
        -------
        list[DetectedFace]
            One entry per detected face.  Landmarks (``kps``) are always
            ``None`` because YOLO does not produce facial keypoints.
        """
        results = self._model(image, conf=self._conf, verbose=False)

        faces: list[DetectedFace] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)
                conf = float(box.conf[0].cpu().numpy())
                faces.append(
                    DetectedFace(
                        bbox=xyxy,
                        kps=None,
                        det_score=conf,
                        embedding=None,
                        normed_embedding=None,
                    )
                )

        # Sort by confidence (highest first)
        faces.sort(key=lambda f: f.det_score, reverse=True)
        return faces
