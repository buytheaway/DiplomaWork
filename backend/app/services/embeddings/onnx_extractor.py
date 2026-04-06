"""ONNX embedding extractor — SCRFD face detector + ArcFace embedding model.

The extractor uses two separate ONNX models:

* **SCRFD** (``ONNX_DETECTOR_PATH``) — produces bounding boxes + 5-point
  landmarks for every detected face.
* **ArcFace** (``ONNX_EMBEDDER_PATH``) — takes aligned 112x112 face crops and
  produces 512-d (or configurable) L2-normalised embeddings.

Required extras::

    pip install onnxruntime opencv-python-headless

Configure via ``.env``::

    EMBEDDING_BACKEND=onnx
    ONNX_DETECTOR_PATH=models/scrfd_10g_bnkps.onnx
    ONNX_EMBEDDER_PATH=models/w600k_r50.onnx
    EMBEDDING_DIM=512
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.core.config import BASE_DIR, Settings
from app.services.embeddings.interface import (
    EmbeddingExtractor,
    FaceEmbedding,
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)

if TYPE_CHECKING:
    import onnxruntime as ort  # pragma: no cover

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference landmarks for ArcFace alignment (112x112)
# Source: insightface/utils/face_align.py — ``arcface_dst``
# ---------------------------------------------------------------------------
_ARCFACE_REF = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


# -- helpers ---------------------------------------------------------------


def _ensure_cv2():
    """Lazy-import ``cv2``; raise a clear message if absent."""
    try:
        import cv2  # noqa: WPS433

        return cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for the ONNX backend. "
            "Install with: pip install opencv-python-headless"
        ) from exc


def _ensure_ort():
    """Lazy-import ``onnxruntime``; raise a clear message if absent."""
    try:
        import onnxruntime as ort  # noqa: WPS433

        return ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for the ONNX embedding backend. "
            "Install with: pip install onnxruntime"
        ) from exc


def _preferred_ort_providers(ort_module) -> list[str]:
    """Pick only providers that exist in the current runtime.

    This avoids noisy warnings when the CPU-only wheel is installed on Windows.
    """
    available = set(ort_module.get_available_providers())
    ordered = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [provider for provider in ordered if provider in available]
    if providers:
        return providers
    return ["CPUExecutionProvider"]


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate 2-D similarity transform (Umeyama algorithm).

    Returns a 2x3 affine matrix mapping *src* -> *dst*.
    """
    num = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_dm = src - src_mean
    dst_dm = dst - dst_mean

    A = dst_dm.T @ src_dm / num
    d = np.ones(2, dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[1] = -1

    U, S, Vt = np.linalg.svd(A)

    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.eye(2, 3, dtype=np.float64)

    T = np.eye(3, dtype=np.float64)
    if rank == 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:2, :2] = U @ Vt
        else:
            d[1] = -1
            T[:2, :2] = U @ np.diag(d) @ Vt
    else:
        T[:2, :2] = U @ np.diag(d) @ Vt

    scale = 1.0 / src_dm.var(axis=0).sum() * (S @ d)
    T[:2, 2] = dst_mean - scale * (T[:2, :2] @ src_mean)
    T[:2, :2] *= scale
    return T[:2, :]


def _align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """Warp a face region to a canonical 112x112 crop using 5-point landmarks."""
    cv2 = _ensure_cv2()
    src_pts = landmarks.astype(np.float64)
    dst_pts = _ARCFACE_REF.astype(np.float64)
    M = _umeyama(src_pts, dst_pts)
    aligned = cv2.warpAffine(
        image, M, (output_size, output_size), borderValue=0.0,
    )
    return aligned


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy non-maximum suppression — returns indices to keep."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


# -- SCRFD detector --------------------------------------------------------


class _SCRFDDetector:
    """Lightweight SCRFD post-processor for the ONNX model-zoo export format.

    Supports models **with keypoints** (9 outputs) and **without** (6 outputs).
    Typical model: ``scrfd_10g_bnkps.onnx`` (640x640 input).
    """

    _FPN_STRIDES = (8, 16, 32)
    _NUM_ANCHORS = 2

    def __init__(
        self, session: ort.InferenceSession, input_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._input_size = input_size  # (w, h)
        self._input_mean = 127.5
        self._input_std = 128.0

        n_out = len(session.get_outputs())
        n_strides = len(self._FPN_STRIDES)
        self._has_kps = (n_out // n_strides) >= 3

    # -- public API --------------------------------------------------------

    def __call__(
        self,
        image: np.ndarray,
        score_thresh: float = 0.5,
        iou_thresh: float = 0.4,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Detect faces in *image* (BGR uint8).

        Returns ``(boxes_Nx4, keypoints_Nx5x2 | None, scores_N)``.
        """
        blob, scale, _pad = self._preprocess(image)
        outputs = self._session.run(None, {self._input_name: blob})
        return self._postprocess(outputs, scale, score_thresh, iou_thresh)

    # -- internals ---------------------------------------------------------

    def _preprocess(self, image: np.ndarray):
        cv2 = _ensure_cv2()
        iw, ih = self._input_size
        h, w = image.shape[:2]
        image_ratio = float(h) / float(w)
        model_ratio = float(ih) / float(iw)
        if image_ratio > model_ratio:
            nh = ih
            nw = int(nh / image_ratio)
        else:
            nw = iw
            nh = int(nw * image_ratio)

        det_scale = float(nh) / float(h)
        resized = cv2.resize(image, (nw, nh))
        det_img = np.zeros((ih, iw, 3), dtype=np.uint8)
        det_img[:nh, :nw, :] = resized
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / self._input_std,
            self._input_size,
            (self._input_mean, self._input_mean, self._input_mean),
            swapRB=True,
        )
        return blob.astype(np.float32), det_scale, (nw, nh)

    def _postprocess(self, outputs, scale, score_thresh, iou_thresh):
        fmc = len(self._FPN_STRIDES)
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_kps: list[np.ndarray] = []

        for idx, stride in enumerate(self._FPN_STRIDES):
            scores = outputs[idx].reshape(-1)
            bbox_deltas = outputs[idx + fmc].reshape(-1, 4) * stride

            fh = self._input_size[1] // stride
            fw = self._input_size[0] // stride
            anchors = self._make_anchors(fw, fh, stride)
            mask = scores >= score_thresh

            if not mask.any():
                continue

            scores_filt = scores[mask]
            bbox_filt = bbox_deltas[mask]
            anchors_filt = anchors[mask]

            boxes = self._distance2bbox(anchors_filt, bbox_filt)
            all_scores.append(scores_filt)
            all_boxes.append(boxes)

            if self._has_kps:
                kps_deltas = outputs[idx + fmc * 2].reshape(-1, 10) * stride
                kps_filt = kps_deltas[mask]
                kps = self._distance2kps(anchors_filt, kps_filt)
                all_kps.append(kps)

        if not all_scores:
            return (
                np.empty((0, 4), dtype=np.float32),
                None,
                np.empty((0,), dtype=np.float32),
            )

        boxes = np.vstack(all_boxes)
        scores_arr = np.concatenate(all_scores)
        kps_arr = np.vstack(all_kps) if all_kps else None

        order = scores_arr.argsort()[::-1]
        boxes = boxes[order]
        scores_arr = scores_arr[order]
        if kps_arr is not None:
            kps_arr = kps_arr[order]

        keep = _nms(boxes, scores_arr, iou_thresh)
        boxes = boxes[keep]
        scores_arr = scores_arr[keep]
        if kps_arr is not None:
            kps_arr = kps_arr[keep]

        # un-scale to original image coords
        boxes = boxes / scale
        if kps_arr is not None:
            kps_arr = kps_arr / scale
            kps_arr = kps_arr.reshape(-1, 5, 2)

        return boxes, kps_arr, scores_arr

    @staticmethod
    def _make_anchors(fw: int, fh: int, stride: int) -> np.ndarray:
        centres = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
        centres = (centres * stride).reshape(-1, 2)
        centres = np.stack([centres, centres], axis=1).reshape(-1, 2)
        return centres

    @staticmethod
    def _distance2bbox(points: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        x1 = points[:, 0] - deltas[:, 0]
        y1 = points[:, 1] - deltas[:, 1]
        x2 = points[:, 0] + deltas[:, 2]
        y2 = points[:, 1] + deltas[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(points: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        out = deltas.copy()
        for i in range(0, deltas.shape[1], 2):
            out[:, i] = points[:, 0] + deltas[:, i]
            out[:, i + 1] = points[:, 1] + deltas[:, i + 1]
        return out


# -- ArcFace embedder ------------------------------------------------------


class _ArcFaceEmbedder:
    """Run an ArcFace ONNX model on aligned 112x112 face crops."""

    def __init__(self, session: ort.InferenceSession) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        inp_shape = session.get_inputs()[0].shape
        self._input_size = int(inp_shape[-1]) if inp_shape[-1] is not None else 112

    def __call__(self, aligned_bgr: np.ndarray) -> np.ndarray:
        cv2 = _ensure_cv2()
        img = aligned_bgr
        if img.shape[:2] != (self._input_size, self._input_size):
            img = cv2.resize(img, (self._input_size, self._input_size))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5
        blob = img.transpose(2, 0, 1)[np.newaxis, ...]

        out = self._session.run(None, {self._input_name: blob.astype(np.float32)})
        vec = out[0].ravel().astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# -- public extractor ------------------------------------------------------


class OnnxEmbeddingExtractor(EmbeddingExtractor):
    """Production ONNX extractor: SCRFD detection -> align -> ArcFace embed.

    Requires two ONNX model files configured via env vars:

    * ``ONNX_DETECTOR_PATH`` — SCRFD model (e.g. *scrfd_10g_bnkps.onnx*)
    * ``ONNX_EMBEDDER_PATH`` — ArcFace model (e.g. *w600k_r50.onnx*)
    """

    def __init__(self, settings: Settings) -> None:
        ort = _ensure_ort()

        det_path = Path(settings.onnx_detector_path)
        emb_path = Path(settings.onnx_embedder_path)
        if not det_path.is_absolute():
            det_path = BASE_DIR / det_path
        if not emb_path.is_absolute():
            emb_path = BASE_DIR / emb_path

        if not det_path.exists():
            raise FileNotFoundError(
                f"ONNX detector model not found: {det_path}. "
                "Set ONNX_DETECTOR_PATH to a valid SCRFD .onnx file."
            )
        if not emb_path.exists():
            raise FileNotFoundError(
                f"ONNX embedder model not found: {emb_path}. "
                "Set ONNX_EMBEDDER_PATH to a valid ArcFace .onnx file."
            )

        providers = _preferred_ort_providers(ort)
        det_session = ort.InferenceSession(str(det_path), providers=providers)
        emb_session = ort.InferenceSession(str(emb_path), providers=providers)

        self._detector = _SCRFDDetector(det_session)
        self._embedder = _ArcFaceEmbedder(emb_session)

        self.model_name = f"onnx_{emb_path.stem}"
        self.dim = settings.embedding_dim
        self.strict_single_face = settings.strict_single_face
        self.min_det_score = settings.min_det_score

        logger.info(
            "ONNX backend ready: detector=%s embedder=%s dim=%d",
            det_path.name,
            emb_path.name,
            self.dim,
        )

    def _decode_and_detect(
        self, image_bytes: bytes
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        if not image_bytes:
            raise InvalidImageError("Empty image bytes")

        cv2 = _ensure_cv2()

        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("Cannot decode image bytes")

        boxes, kps, scores = self._detector(
            image, score_thresh=self.min_det_score,
        )

        if len(scores) == 0:
            raise NoFaceDetectedError("No face detected in image")

        return image, boxes, kps, scores

    def _extract_face_embedding(
        self,
        image: np.ndarray,
        box: np.ndarray,
        landmarks: np.ndarray | None,
        score: float,
    ) -> FaceEmbedding:
        cv2 = _ensure_cv2()

        if landmarks is not None:
            aligned = _align_face(image, landmarks)
        else:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2 = min(x2, image.shape[1])
            y2 = min(y2, image.shape[0])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                raise InvalidImageError("Empty face crop")
            aligned = cv2.resize(crop, (112, 112))

        vector = self._embedder(aligned)

        if vector.shape[0] != self.dim:
            raise NoFaceDetectedError(
                f"Model returned dim={vector.shape[0]}, expected {self.dim}"
            )

        return FaceEmbedding(
            embedding=vector,
            detection_score=float(score),
            bbox=tuple(float(value) for value in box.tolist()),
        )

    def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbedding]:
        image, boxes, kps, scores = self._decode_and_detect(image_bytes)

        embeddings: list[FaceEmbedding] = []
        for idx, score in enumerate(scores):
            try:
                embeddings.append(
                    self._extract_face_embedding(
                        image=image,
                        box=boxes[idx],
                        landmarks=None if kps is None else kps[idx],
                        score=float(score),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping ONNX face %s due to extraction error: %s", idx, exc)

        if not embeddings:
            raise NoFaceDetectedError("No valid face crops could be embedded")

        return embeddings

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        # Early check: reject multi-face images before expensive embedding step.
        if self.strict_single_face:
            image, boxes, kps, scores = self._decode_and_detect(image_bytes)
            if len(scores) != 1:
                raise MultipleFacesDetectedError(
                    f"Expected 1 face, detected {len(scores)}"
                )
            return self._extract_face_embedding(
                image=image,
                box=boxes[0],
                landmarks=None if kps is None else kps[0],
                score=float(scores[0]),
            ).embedding

        embeddings = self.extract_embeddings(image_bytes)
        return embeddings[0].embedding
