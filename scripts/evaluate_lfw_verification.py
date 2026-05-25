"""Evaluate real biometric verification metrics on the LFW pairs protocol.

The script is intentionally standalone: it reuses the project's embedding
extractors, but it does not call the backend API, database, desktop UI, enroll
flow, or FAISS index. It measures verification quality from labeled image pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.embeddings.interface import (  # noqa: E402
    EmbeddingExtractor,
    FaceEmbedding,
    FaceProcessingError,
    NoFaceDetectedError,
)
from training.verification_metrics import (  # noqa: E402
    best_accuracy_threshold,
    equal_error_rate,
    far_frr_by_threshold,
    tar_at_far,
)

PipelineName = Literal["custom", "pretrained"]
CustomPreprocess = Literal["runtime", "center_crop", "letterbox", "insightface_align"]
ColorOrder = Literal["rgb", "bgr"]
Normalization = Literal["minus1_1", "imagenet", "zero1", "none"]
DetectorMode = Literal["custom", "pretrained", "none"]
TtaMode = Literal["none", "hflip"]

PIPELINE_DISPLAY_NAMES: dict[PipelineName, str] = {
    "custom": "Proposed custom Torch IR-50 pipeline",
    "pretrained": "Pretrained ONNX/InsightFace baseline",
}

TARGET_FARS = [0.1, 0.01, 0.001]


@dataclass(frozen=True)
class LfwPair:
    pair_id: int
    line_number: int
    label_same: int
    name1: str
    index1: int
    name2: str
    index2: int
    path1: Path
    path2: Path


def lfw_image_path(lfw_root: Path, name: str, index: int) -> Path:
    return lfw_root / name / f"{name}_{index:04d}.jpg"


def parse_lfw_pairs(
    lfw_root: Path,
    pairs_file: Path,
    max_pairs: int | None = None,
) -> list[LfwPair]:
    """Parse standard LFW pairs.txt lines.

    Supported lines:
    * same person: ``name idx1 idx2``
    * different persons: ``name1 idx1 name2 idx2``
    """
    pairs: list[LfwPair] = []
    with pairs_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) in {1, 2} and all(part.isdigit() for part in parts):
                continue

            if len(parts) == 3:
                name1 = parts[0]
                index1 = int(parts[1])
                name2 = parts[0]
                index2 = int(parts[2])
                label_same = 1
            elif len(parts) == 4:
                name1 = parts[0]
                index1 = int(parts[1])
                name2 = parts[2]
                index2 = int(parts[3])
                label_same = 0
            else:
                raise ValueError(f"Unsupported LFW pair format at line {line_number}: {line!r}")

            pairs.append(
                LfwPair(
                    pair_id=len(pairs) + 1,
                    line_number=line_number,
                    label_same=label_same,
                    name1=name1,
                    index1=index1,
                    name2=name2,
                    index2=index2,
                    path1=lfw_image_path(lfw_root, name1, index1),
                    path2=lfw_image_path(lfw_root, name2, index2),
                )
            )
    if max_pairs is not None and len(pairs) > max_pairs:
        positives = [pair for pair in pairs if pair.label_same == 1]
        negatives = [pair for pair in pairs if pair.label_same == 0]
        if positives and negatives:
            positive_count = max_pairs // 2
            negative_count = max_pairs - positive_count
            selected = positives[:positive_count] + negatives[:negative_count]
            return sorted(selected, key=lambda pair: pair.pair_id)
        return pairs[:max_pairs]
    return pairs


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    embedding = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if norm > 0.0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class DirectTorchLfwExtractor(EmbeddingExtractor):
    """Custom Torch model evaluator with explicit LFW preprocessing ablations."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        preprocess: CustomPreprocess,
        color_order: ColorOrder,
        normalization: Normalization,
        detector: DetectorMode,
        device: str,
        input_size: int = 112,
        arch: str = "ir50",
        onnx_detector_path: str = "",
    ) -> None:
        import torch

        from app.services.embeddings.torch_model import (
            ModelConfig,
            build_model,
            forward_with_normalization,
        )

        self.model_name = f"torch_{arch}_{preprocess}_{color_order}_{normalization}"
        self.dim = 512
        self.preprocess = preprocess
        self.color_order = color_order
        self.normalization = normalization
        self.detector = detector
        self.input_size = input_size
        self.device = torch.device(device)
        self._forward_with_normalization = forward_with_normalization

        model_config = ModelConfig(arch=arch, embedding_dim=self.dim, norm_embeddings=True)
        self.model = build_model(model_config).to(self.device).eval()
        state = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state.get("state_dict", state), strict=False)

        self._pretrained_detector = None
        if preprocess == "insightface_align":
            if detector != "pretrained":
                raise ValueError("insightface_align requires --detector pretrained")
            if not onnx_detector_path:
                raise ValueError("insightface_align requires ONNX_DETECTOR_PATH")
            self._pretrained_detector = self._build_pretrained_detector(onnx_detector_path)

    def _build_pretrained_detector(self, detector_path: str):
        from app.services.embeddings.onnx_extractor import _ensure_ort, _SCRFDDetector

        ort = _ensure_ort()
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(detector_path), providers=providers)
        return _SCRFDDetector(session)

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        import cv2

        height, width = image.shape[:2]
        side = min(height, width)
        x1 = max((width - side) // 2, 0)
        y1 = max((height - side) // 2, 0)
        crop = image[y1 : y1 + side, x1 : x1 + side]
        return cv2.resize(crop, (self.input_size, self.input_size))

    def _letterbox(self, image: np.ndarray) -> np.ndarray:
        import cv2

        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_w, resized_h))
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        x1 = (self.input_size - resized_w) // 2
        y1 = (self.input_size - resized_h) // 2
        canvas[y1 : y1 + resized_h, x1 : x1 + resized_w] = resized
        return canvas

    def _align_with_pretrained_detector(self, image: np.ndarray) -> np.ndarray:
        import cv2

        from app.services.embeddings.onnx_extractor import _align_face

        if self._pretrained_detector is None:
            raise NoFaceDetectedError("Pretrained detector was not configured")
        boxes, keypoints, scores = self._pretrained_detector(image)
        if boxes.size == 0 or keypoints is None:
            raise NoFaceDetectedError("No face detected")
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        index = int(np.argmax(areas))
        aligned = _align_face(image, keypoints[index], self.input_size)
        if aligned.shape[0] != self.input_size or aligned.shape[1] != self.input_size:
            aligned = cv2.resize(aligned, (self.input_size, self.input_size))
        return aligned

    def _prepare_image(self, image_bytes: bytes) -> np.ndarray:
        from app.services.face.detector import decode_image

        image = decode_image(image_bytes)
        if self.preprocess == "center_crop":
            return self._center_crop(image)
        if self.preprocess == "letterbox":
            return self._letterbox(image)
        if self.preprocess == "insightface_align":
            return self._align_with_pretrained_detector(image)
        raise ValueError(f"Unsupported direct preprocess mode: {self.preprocess}")

    def _to_tensor(self, image: np.ndarray):
        import cv2
        import torch

        if self.color_order == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        array = image.astype(np.float32)

        if self.normalization == "minus1_1":
            array = (array / 255.0 - 0.5) / 0.5
        elif self.normalization == "zero1":
            array = array / 255.0
        elif self.normalization == "imagenet":
            array = array / 255.0
            if self.color_order == "rgb":
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            else:
                mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
                std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
            array = (array - mean) / std
        elif self.normalization == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalization: {self.normalization}")

        return torch.from_numpy(array).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

    def extract_largest_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        import torch

        if not image_bytes:
            raise NoFaceDetectedError("Empty image bytes")
        aligned = self._prepare_image(image_bytes)
        tensor = self._to_tensor(aligned)
        with torch.inference_mode():
            embedding = self._forward_with_normalization(self.model, tensor, normalize=True)
        vector = embedding.squeeze(0).float().cpu().numpy().astype(np.float32)
        if vector.shape[0] != self.dim:
            raise NoFaceDetectedError("Unexpected embedding dimension")
        return FaceEmbedding(embedding=vector, detection_score=None, bbox=None)

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        return self.extract_largest_embedding(image_bytes).embedding


def build_extractor(pipeline: PipelineName, device: str = "auto") -> tuple[EmbeddingExtractor, object]:
    """Build the configured project extractor for the requested pipeline."""
    os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    os.environ["DEFAULT_PIPELINE"] = pipeline
    os.environ["ENABLE_PRETRAINED_PIPELINE"] = "true" if pipeline == "pretrained" else "false"
    os.environ["ENABLE_CUSTOM_PIPELINE"] = "true" if pipeline == "custom" else "false"

    from app.core.config import get_settings
    from app.services.embeddings.interface import create_extractor

    get_settings.cache_clear()
    base_settings = get_settings()
    if pipeline == "custom":
        updates: dict[str, Any] = {
            "embedding_backend": base_settings.custom_backend,
            "detection_backend": base_settings.custom_detection_backend,
            "allow_center_crop": base_settings.custom_allow_center_crop,
            "torch_device": _resolve_device(device),
        }
        if base_settings.custom_min_det_score is not None:
            updates["min_det_score"] = base_settings.custom_min_det_score
        if base_settings.custom_face_crop_margin is not None:
            updates["face_crop_margin"] = base_settings.custom_face_crop_margin
        if base_settings.custom_yolo_imgsz is not None:
            updates["yolo_imgsz"] = base_settings.custom_yolo_imgsz
    else:
        updates = {
            "embedding_backend": base_settings.pretrained_backend,
            "detection_backend": base_settings.detection_backend,
            "allow_center_crop": base_settings.allow_center_crop,
        }
        if base_settings.pretrained_backend == "torch":
            updates["torch_device"] = _resolve_device(device)

    settings = base_settings.model_copy(update=updates)
    extractor = create_extractor(settings)

    model = getattr(extractor, "model", None)
    if model is not None and getattr(model, "training", False):
        model.eval()

    return extractor, settings


def build_custom_ablation_extractor(
    args: argparse.Namespace,
) -> tuple[EmbeddingExtractor, object]:
    """Build the custom evaluator requested by CLI ablation flags."""
    os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    os.environ["DEFAULT_PIPELINE"] = "custom"
    os.environ["ENABLE_PRETRAINED_PIPELINE"] = "false"
    os.environ["ENABLE_CUSTOM_PIPELINE"] = "true"
    checkpoint_value = args.checkpoint or os.environ.get("TORCH_MODEL_PATH", "")
    if not checkpoint_value:
        raise SystemExit("--checkpoint or TORCH_MODEL_PATH is required for custom ablations")

    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Custom checkpoint not found: {checkpoint}")

    os.environ["TORCH_MODEL_PATH"] = str(checkpoint)
    if args.custom_preprocess == "runtime":
        return build_extractor("custom", args.device)

    from app.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    device = _resolve_device(args.device)
    extractor = DirectTorchLfwExtractor(
        checkpoint=checkpoint,
        preprocess=args.custom_preprocess,
        color_order=args.color_order,
        normalization=args.normalization,
        detector=args.detector,
        device=device,
        input_size=settings.torch_input_size,
        arch=settings.torch_model_arch,
        onnx_detector_path=settings.onnx_detector_path,
    )
    return extractor, settings


def cache_key_from_args(args: argparse.Namespace) -> str:
    if args.pipeline == "pretrained":
        return f"pretrained_{args.tta}"
    checkpoint_name = Path(args.checkpoint).stem if args.checkpoint else "env_checkpoint"
    return (
        f"{args.custom_preprocess}_{args.detector}_"
        f"{args.color_order}_{args.normalization}_{checkpoint_name}_{args.tta}"
    )


def _cache_file_for_image(cache_dir: Path, lfw_root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(lfw_root).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]
    person_dir = image_path.parent.name
    return cache_dir / person_dir / f"{image_path.stem}-{digest}.npy"


def _hflip_image_bytes(image_path: Path) -> bytes:
    from PIL import Image, ImageOps

    image = Image.open(image_path).convert("RGB")
    flipped = ImageOps.mirror(image)
    buffer = io.BytesIO()
    flipped.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def load_or_extract_embedding(
    extractor: EmbeddingExtractor,
    image_path: Path,
    *,
    lfw_root: Path,
    pipeline_cache_dir: Path,
    tta: TtaMode,
) -> np.ndarray:
    cache_file = _cache_file_for_image(pipeline_cache_dir, lfw_root, image_path)
    if cache_file.exists():
        return normalize_embedding(np.load(cache_file))

    face = extractor.extract_largest_embedding(image_path.read_bytes())
    embedding = normalize_embedding(face.embedding)
    if tta == "hflip":
        flipped_face = extractor.extract_largest_embedding(_hflip_image_bytes(image_path))
        flipped_embedding = normalize_embedding(flipped_face.embedding)
        embedding = normalize_embedding(embedding + flipped_embedding)
    elif tta != "none":
        raise ValueError(f"Unsupported TTA mode: {tta}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, embedding)
    return embedding


def evaluate_pairs(
    extractor: EmbeddingExtractor,
    pairs: list[LfwPair],
    *,
    lfw_root: Path,
    pipeline_cache_dir: Path,
    tta: TtaMode,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for pair in pairs:
        row: dict[str, Any] = {
            "pair_id": pair.pair_id,
            "label_same": pair.label_same,
            "name1": pair.name1,
            "name2": pair.name2,
            "path1": str(pair.path1),
            "path2": str(pair.path2),
            "score": "",
            "status": "ok",
            "error": "",
        }

        missing = [str(path) for path in (pair.path1, pair.path2) if not path.exists()]
        if missing:
            row["status"] = "missing_image"
            row["error"] = "; ".join(missing)
            rows.append(row)
            continue

        try:
            embedding1 = load_or_extract_embedding(
                extractor,
                pair.path1,
                lfw_root=lfw_root,
                pipeline_cache_dir=pipeline_cache_dir,
                tta=tta,
            )
            embedding2 = load_or_extract_embedding(
                extractor,
                pair.path2,
                lfw_root=lfw_root,
                pipeline_cache_dir=pipeline_cache_dir,
                tta=tta,
            )
            row["score"] = float(np.dot(embedding1, embedding2))
        except NoFaceDetectedError as exc:
            row["status"] = "no_face"
            row["error"] = str(exc)
        except FaceProcessingError as exc:
            row["status"] = "face_processing_error"
            row["error"] = str(exc)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)

    return rows


def _valid_scores_and_labels(rows: list[dict[str, Any]]) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    for row in rows:
        if row["status"] != "ok":
            continue
        scores.append(float(row["score"]))
        labels.append(int(row["label_same"]))
    return scores, labels


def _rounded(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return round(float(value), 6)


def compute_verification_metrics(
    rows: list[dict[str, Any]],
    *,
    pipeline: PipelineName,
    extractor: EmbeddingExtractor | None = None,
    settings: object | None = None,
) -> dict[str, Any]:
    scores, labels = _valid_scores_and_labels(rows)
    if not scores:
        raise ValueError("No valid pairs; cannot compute biometric verification metrics")
    if 1 not in labels or 0 not in labels:
        raise ValueError("Both positive and negative valid pairs are required")

    eer = equal_error_rate(scores, labels)
    best = best_accuracy_threshold(scores, labels)
    tar_values = tar_at_far(scores, labels, TARGET_FARS)

    positive_scores = [score for score, label in zip(scores, labels, strict=False) if label == 1]
    negative_scores = [score for score, label in zip(scores, labels, strict=False) if label == 0]

    tar_by_far = {
        f"{item['target_far']:.3g}": {
            "tar": _rounded(item["tar"]),
            "threshold": _rounded(item["threshold"]),
            "far": _rounded(item["far"]),
            "frr": _rounded(item["frr"]),
        }
        for item in tar_values
    }

    return {
        "pipeline": pipeline,
        "pipeline_display_name": PIPELINE_DISPLAY_NAMES[pipeline],
        "extractor_model": getattr(extractor, "model_name", None),
        "embedding_dim": getattr(extractor, "dim", None),
        "settings_summary": settings_summary(settings) if settings is not None else {},
        "valid_pairs": len(scores),
        "positive_pairs": len(positive_scores),
        "negative_pairs": len(negative_scores),
        "skipped_pairs": len(rows) - len(scores),
        "mean_positive_score": _rounded(float(np.mean(positive_scores))),
        "mean_negative_score": _rounded(float(np.mean(negative_scores))),
        "eer": _rounded(eer["eer"]),
        "eer_threshold": _rounded(eer["threshold"]),
        "eer_far": _rounded(eer["far"]),
        "eer_frr": _rounded(eer["frr"]),
        "best_accuracy": _rounded(best["accuracy"]),
        "best_accuracy_threshold": _rounded(best["threshold"]),
        "far_at_best_threshold": _rounded(best["far"]),
        "frr_at_best_threshold": _rounded(best["frr"]),
        "tar_at_far": tar_by_far,
        "definitions": {
            "positive": "same person pair",
            "negative": "different person pair",
            "decision_rule": "score >= threshold means match",
            "far": "false accepts / total negative pairs",
            "frr": "false rejects / total positive pairs",
            "eer": "threshold where FAR and FRR are closest",
        },
    }


def settings_summary(settings: object) -> dict[str, Any]:
    return {
        "embedding_backend": getattr(settings, "embedding_backend", None),
        "detection_backend": getattr(settings, "detection_backend", None),
        "embedding_dim": getattr(settings, "embedding_dim", None),
        "strict_single_face": getattr(settings, "strict_single_face", None),
        "min_det_score": getattr(settings, "min_det_score", None),
        "torch_model_arch": getattr(settings, "torch_model_arch", None),
        "torch_input_size": getattr(settings, "torch_input_size", None),
        "torch_device": getattr(settings, "torch_device", None),
        "torch_norm_embeddings": getattr(settings, "torch_norm_embeddings", None),
        "onnx_detector_configured": bool(getattr(settings, "onnx_detector_path", "")),
        "onnx_embedder_configured": bool(getattr(settings, "onnx_embedder_path", "")),
        "torch_model_configured": bool(getattr(settings, "torch_model_path", "")),
    }


def threshold_curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores, labels = _valid_scores_and_labels(rows)
    curve = far_frr_by_threshold(scores, labels)
    output_rows: list[dict[str, Any]] = []
    for index, threshold in enumerate(curve["thresholds"]):
        output_rows.append(
            {
                "threshold": float(threshold),
                "far": float(curve["far"][index]),
                "frr": float(curve["frr"][index]),
                "tar": float(curve["tar"][index]),
                "accuracy": float(curve["accuracy"][index]),
                "tp": int(curve["tp"][index]),
                "fp": int(curve["fp"][index]),
                "tn": int(curve["tn"][index]),
                "fn": int(curve["fn"][index]),
            }
        )
    return output_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_markdown(path: Path, metrics: dict[str, Any]) -> None:
    tar_001 = metrics["tar_at_far"]["0.01"]["tar"]
    lines = [
        "# LFW Biometric Verification Metrics",
        "",
        "This report uses the LFW pairs verification protocol.",
        "A pair is predicted as the same identity when `score >= threshold`.",
        "",
        "| Pipeline | Valid pairs | Positive pairs | Negative pairs | EER | "
        "EER threshold | Best accuracy | Best threshold | FAR | FRR | TAR@FAR=0.01 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {metrics['pipeline_display_name']} | {metrics['valid_pairs']} | "
            f"{metrics['positive_pairs']} | {metrics['negative_pairs']} | "
            f"{metrics['eer']:.6f} | {metrics['eer_threshold']:.6f} | "
            f"{metrics['best_accuracy']:.6f} | "
            f"{metrics['best_accuracy_threshold']:.6f} | "
            f"{metrics['far_at_best_threshold']:.6f} | "
            f"{metrics['frr_at_best_threshold']:.6f} | "
            f"{tar_001 if tar_001 is not None else 'n/a'} |"
        ),
        "",
        "## Additional Statistics",
        "",
        f"- Mean positive score: `{metrics['mean_positive_score']}`",
        f"- Mean negative score: `{metrics['mean_negative_score']}`",
        f"- Skipped pairs: `{metrics['skipped_pairs']}`",
        "",
        "## Notes",
        "",
        "- FAR = false accepts / total negative pairs.",
        "- FRR = false rejects / total positive pairs.",
        "- EER is estimated at the threshold where FAR and FRR are closest.",
        "- These are real biometric verification metrics for the evaluated pairs.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    lfw_root = Path(args.lfw_root)
    pairs_file = Path(args.pairs_file)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) / args.pipeline / cache_key_from_args(args)

    if not lfw_root.exists():
        raise SystemExit(f"LFW root not found: {lfw_root}")
    if not pairs_file.exists():
        raise SystemExit(f"pairs.txt not found: {pairs_file}")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise SystemExit("--max-pairs must be positive")

    pairs = parse_lfw_pairs(lfw_root, pairs_file, max_pairs=args.max_pairs)
    if not pairs:
        raise SystemExit("No LFW pairs loaded")

    if args.pipeline == "custom":
        extractor, settings = build_custom_ablation_extractor(args)
    else:
        extractor, settings = build_extractor(args.pipeline, args.device)
    rows = evaluate_pairs(
        extractor,
        pairs,
        lfw_root=lfw_root,
        pipeline_cache_dir=cache_dir,
        tta=args.tta,
    )
    metrics = compute_verification_metrics(
        rows,
        pipeline=args.pipeline,
        extractor=extractor,
        settings=settings,
    )
    metrics.update(
        {
            "lfw_root": str(lfw_root),
            "pairs_file": str(pairs_file),
            "pairs_loaded": len(pairs),
            "cache_dir": str(cache_dir),
            "custom_preprocess": args.custom_preprocess if args.pipeline == "custom" else None,
            "color_order": args.color_order if args.pipeline == "custom" else None,
            "normalization": args.normalization if args.pipeline == "custom" else None,
            "detector": args.detector if args.pipeline == "custom" else None,
            "checkpoint": args.checkpoint if args.pipeline == "custom" else None,
            "tta": args.tta,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "pairs_scores.csv",
        rows,
        ["pair_id", "label_same", "name1", "name2", "path1", "path2", "score", "status", "error"],
    )
    write_csv(
        output_dir / "threshold_curve.csv",
        threshold_curve_rows(rows),
        ["threshold", "far", "frr", "tar", "accuracy", "tp", "fp", "tn", "fn"],
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_metrics_markdown(output_dir / "metrics.md", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LFW FAR/FRR/EER with the project embedding pipelines"
    )
    parser.add_argument("--lfw-root", required=True)
    parser.add_argument("--pairs-file", required=True)
    parser.add_argument("--pipeline", choices=["custom", "pretrained"], default="custom")
    parser.add_argument("--output-dir", default="reports/biometric_eval/lfw_custom")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--cache-dir", default="reports/biometric_eval/cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--custom-preprocess",
        choices=["runtime", "center_crop", "letterbox", "insightface_align"],
        default="runtime",
        help="Custom-only preprocessing ablation. Default preserves runtime behavior.",
    )
    parser.add_argument(
        "--color-order",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Custom direct preprocessing channel order.",
    )
    parser.add_argument(
        "--normalization",
        choices=["minus1_1", "imagenet", "zero1", "none"],
        default="minus1_1",
        help="Custom direct preprocessing normalization.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Custom Torch checkpoint path. Overrides TORCH_MODEL_PATH for evaluation only.",
    )
    parser.add_argument(
        "--detector",
        choices=["custom", "pretrained", "none"],
        default="custom",
        help="Custom-only detector/alignment source for ablations.",
    )
    parser.add_argument(
        "--tta",
        choices=["none", "hflip"],
        default="none",
        help="Evaluation-only test-time augmentation. hflip averages original and mirrored embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    metrics = run_evaluation(parse_args())
    print(
        "Saved LFW verification metrics for "
        f"{metrics['pipeline_display_name']} "
        f"({metrics['valid_pairs']} valid pairs, EER={metrics['eer']})"
    )


if __name__ == "__main__":
    main()
