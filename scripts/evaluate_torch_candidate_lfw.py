"""Evaluate an external Torch face-embedding checkpoint on LFW.

The script is isolated from runtime state: it does not change backend/desktop
configuration, does not replace the current custom checkpoint, and does not
rebuild FAISS. It supports the project's local IR model and official
InsightFace-style IResNet checkpoints used by ArcFace Torch model zoos.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.embeddings.interface import (  # noqa: E402
    EmbeddingExtractor,
    FaceEmbedding,
    FaceProcessingError,
    NoFaceDetectedError,
)
from app.services.face.align import align_with_landmarks  # noqa: E402
from app.services.face.detector import InsightFaceDetector, OpenCVHaarDetector  # noqa: E402
from scripts.evaluate_lfw_verification import (  # noqa: E402
    compute_verification_metrics,
    evaluate_pairs,
    normalize_embedding,
    parse_lfw_pairs,
    threshold_curve_rows,
    write_csv,
    write_metrics_markdown,
)
from training.models.ir_resnet import build_model as build_project_ir_model  # noqa: E402

CandidateArch = Literal[
    "project_ir18",
    "project_ir34",
    "project_ir50",
    "project_ir100",
    "insightface_iresnet50",
    "insightface_iresnet100",
]

ARCH_ALIASES: dict[str, CandidateArch] = {
    "ir18": "project_ir18",
    "ir34": "project_ir34",
    "ir50": "project_ir50",
    "irse50": "project_ir50",
    "ir100": "project_ir100",
    "iresnet50": "insightface_iresnet50",
    "iresnet100": "insightface_iresnet100",
}
AUTO_ARCHES: list[CandidateArch] = [
    "insightface_iresnet100",
    "insightface_iresnet50",
    "project_ir100",
    "project_ir50",
    "project_ir34",
    "project_ir18",
]
STATE_KEYS = ["state_dict", "model_state_dict", "backbone_state_dict", "backbone", "model", "net"]
PREFIXES = ["module.", "model.", "backbone.", "net.", "encoder.", "_model."]


@dataclass(frozen=True)
class CandidateInspection:
    checkpoint: str
    size_bytes: int
    architecture_requested: str
    architecture_selected: str | None
    loads: bool
    matched_tensors: int
    total_model_tensors: int
    embedding_dim: int | None
    input_size: str
    normalization: str
    color_order: str
    notes: str


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-5),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class InsightFaceIResNet(nn.Module):
    def __init__(self, layers: list[int], embedding_dim: int = 512) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=0.0, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [IBasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.features(x)


class CandidateTorchExtractor(EmbeddingExtractor):
    def __init__(
        self,
        *,
        model: nn.Module,
        model_name: str,
        device: str,
        preprocess: str,
        color_order: str,
        normalization: str,
    ) -> None:
        self.model = model.to(device).eval()
        self.model_name = model_name
        self.dim = 512
        self.device = torch.device(device)
        self.preprocess = preprocess
        self.color_order = color_order
        self.normalization = normalization
        self.last_preprocess_path = ""
        self.last_preprocess_error = ""

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        import cv2

        height, width = image.shape[:2]
        side = min(height, width)
        x1 = max((width - side) // 2, 0)
        y1 = max((height - side) // 2, 0)
        crop = image[y1 : y1 + side, x1 : x1 + side]
        return cv2.resize(crop, (112, 112))

    def _align_with_pretrained_detector(self, image: np.ndarray) -> np.ndarray:
        import cv2

        from app.services.embeddings.onnx_extractor import _align_face, _ensure_ort, _SCRFDDetector

        detector_path = ROOT / "models" / "det_10g.onnx"
        if not detector_path.exists():
            raise NoFaceDetectedError(f"Pretrained detector not found: {detector_path}")
        ort = _ensure_ort()
        detector = _SCRFDDetector(
            ort.InferenceSession(str(detector_path), providers=["CPUExecutionProvider"])
        )
        boxes, keypoints, _scores = detector(image)
        if boxes.size == 0 or keypoints is None:
            raise NoFaceDetectedError("No face detected")
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        aligned = _align_face(image, keypoints[int(np.argmax(areas))], 112)
        if aligned.shape[:2] != (112, 112):
            aligned = cv2.resize(aligned, (112, 112))
        return aligned

    def _runtime_preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2

        try:
            from app.core.config import Settings
            from app.services.face.yolo_detector import YoloFaceDetector

            settings = Settings()
            backend = settings.custom_detection_backend
            allow_center_crop = settings.custom_allow_center_crop
            crop_margin = (
                settings.custom_face_crop_margin
                if settings.custom_face_crop_margin is not None
                else settings.face_crop_margin
            )
            if backend == "yolo":
                detector = YoloFaceDetector(
                    settings.yolo_model_path,
                    conf_threshold=settings.min_det_score,
                    imgsz=settings.custom_yolo_imgsz or settings.yolo_imgsz,
                    device=str(self.device),
                )
            elif backend == "insightface":
                detector = InsightFaceDetector(model_name=settings.model_name)
            else:
                detector = OpenCVHaarDetector()
        except Exception:
            detector = OpenCVHaarDetector()
            allow_center_crop = True
            crop_margin = 0.0

        faces = detector.detect(image)
        if not faces:
            raise NoFaceDetectedError("No face detected")

        def area(face) -> float:
            if face.bbox is None:
                return 0.0
            x1, y1, x2, y2 = [float(value) for value in face.bbox[:4]]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        face = max(faces, key=area)
        bbox = face.bbox
        if bbox is not None and len(bbox) >= 4 and crop_margin > 0.0:
            height, width = image.shape[:2]
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            bbox = np.array(
                [
                    max(0.0, x1 - box_w * crop_margin),
                    max(0.0, y1 - box_h * crop_margin),
                    min(float(width), x2 + box_w * crop_margin),
                    min(float(height), y2 + box_h * crop_margin),
                ],
                dtype=np.float32,
            )

        aligned = align_with_landmarks(image, face.kps, allow_center_crop, bbox=bbox)
        if aligned.shape[:2] != (112, 112):
            aligned = cv2.resize(aligned, (112, 112))
        return aligned

    def _prepare(self, image_bytes: bytes) -> torch.Tensor:
        import cv2

        from app.services.face.detector import decode_image

        image = decode_image(image_bytes)
        if self.preprocess == "pretrained_align":
            image = self._align_with_pretrained_detector(image)
            self.last_preprocess_path = "pretrained_align"
            self.last_preprocess_error = ""
        elif self.preprocess == "runtime":
            image = self._runtime_preprocess(image)
            self.last_preprocess_path = "runtime"
            self.last_preprocess_error = ""
        elif self.preprocess == "runtime_fallback_center_crop":
            try:
                image = self._runtime_preprocess(image)
                self.last_preprocess_path = "runtime"
                self.last_preprocess_error = ""
            except Exception as exc:
                self.last_preprocess_error = str(exc)
                image = self._center_crop(image)
                self.last_preprocess_path = "fallback_center_crop"
        else:
            image = self._center_crop(image)
            self.last_preprocess_path = "center_crop"
            self.last_preprocess_error = ""

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
        elif self.normalization != "none":
            raise ValueError(f"Unsupported normalization: {self.normalization}")
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

    def extract_largest_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        with torch.inference_mode():
            embedding = self.model(self._prepare(image_bytes))
            embedding = F.normalize(embedding.float(), dim=1)
        vector = embedding.squeeze(0).cpu().numpy().astype(np.float32)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise NoFaceDetectedError("Invalid candidate embedding")
        return FaceEmbedding(embedding=vector, detection_score=None, bbox=None)

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        return self.extract_largest_embedding(image_bytes).embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Torch checkpoint on LFW")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--architecture",
        choices=["auto", "ir18", "ir34", "ir50", "irse50", "ir100", "iresnet50", "iresnet100"],
        default="auto",
    )
    parser.add_argument("--lfw-root", default="handoff_lfw_eval/lfw")
    parser.add_argument("--pairs-file", default="handoff_lfw_eval/lfw/pairs.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", choices=["none", "hflip"], default="none")
    parser.add_argument(
        "--normalization",
        choices=["minus1_1", "imagenet", "zero1", "none"],
        default="minus1_1",
    )
    parser.add_argument("--color-order", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument(
        "--preprocess",
        choices=["center_crop", "pretrained_align", "runtime", "runtime_fallback_center_crop"],
        default="center_crop",
    )
    parser.add_argument("--cache-dir", default="reports/biometric_eval/cache_candidates")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _unwrap_state(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in STATE_KEYS:
            value = raw.get(key)
            if isinstance(value, dict):
                return _unwrap_state(value)
        tensor_values = [value for value in raw.values() if torch.is_tensor(value)]
        if tensor_values:
            return {key: value for key, value in raw.items() if torch.is_tensor(value)}
    raise ValueError("Could not find a tensor state_dict in checkpoint")


def _strip_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def normalized_state(raw_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        clean_key = _strip_prefix(key)
        if clean_key.startswith("output_layer."):
            clean_key = clean_key.replace("output_layer.0.", "bn2.")
            clean_key = clean_key.replace("output_layer.3.", "fc.")
            clean_key = clean_key.replace("output_layer.4.", "features.")
        output[clean_key] = value
    return output


def build_candidate_model(arch: CandidateArch) -> nn.Module:
    if arch == "insightface_iresnet50":
        return InsightFaceIResNet([3, 4, 14, 3])
    if arch == "insightface_iresnet100":
        return InsightFaceIResNet([3, 13, 30, 3])
    project_arch = arch.replace("project_", "")
    return build_project_ir_model(project_arch, 512)


def _match_count(state: dict[str, torch.Tensor], arch: CandidateArch) -> tuple[int, int]:
    model_state = build_candidate_model(arch).state_dict()
    matched = sum(
        1
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    )
    return matched, len(model_state)


def select_architecture(state: dict[str, torch.Tensor], requested: str) -> tuple[CandidateArch | None, int, int]:
    if requested != "auto":
        arch = ARCH_ALIASES[requested]
        matched, total = _match_count(state, arch)
        return arch, matched, total

    best_arch: CandidateArch | None = None
    best_matched = -1
    best_total = 0
    for arch in AUTO_ARCHES:
        matched, total = _match_count(state, arch)
        if matched > best_matched:
            best_arch = arch
            best_matched = matched
            best_total = total
    return best_arch, best_matched, best_total


def load_model(state: dict[str, torch.Tensor], arch: CandidateArch) -> nn.Module:
    model = build_candidate_model(arch)
    model_state = model.state_dict()
    compatible = OrderedDict(
        (key, value.float())
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    )
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if len(compatible) < max(10, len(model_state) // 3):
        raise RuntimeError(
            f"Too few compatible tensors for {arch}: {len(compatible)}/{len(model_state)}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model


def cache_dir_for_candidate(base: str, checkpoint: Path, arch: str, args: argparse.Namespace) -> Path:
    payload = (
        f"{checkpoint.resolve()}|{checkpoint.stat().st_size}|{checkpoint.stat().st_mtime_ns}|"
        f"{arch}|{args.preprocess}|{args.color_order}|{args.normalization}|{args.tta}"
    )
    return Path(base) / f"{checkpoint.stem}_{hashlib.sha1(payload.encode()).hexdigest()[:16]}"


def write_inspection(output_dir: Path, inspection: CandidateInspection) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_inspection.json").write_text(
        json.dumps(asdict(inspection), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_candidate_markdown(
    output_dir: Path,
    inspection: CandidateInspection,
    metrics: dict[str, Any] | None,
) -> None:
    lines = [
        "# Torch Candidate LFW Evaluation",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Checkpoint | `{inspection.checkpoint}` |",
        f"| Size bytes | {inspection.size_bytes} |",
        f"| Requested architecture | {inspection.architecture_requested} |",
        f"| Selected architecture | {inspection.architecture_selected or 'n/a'} |",
        f"| Loads | {inspection.loads} |",
        f"| Matched tensors | {inspection.matched_tensors}/{inspection.total_model_tensors} |",
        f"| Embedding dim | {inspection.embedding_dim or 'n/a'} |",
        f"| Input | {inspection.input_size} |",
        f"| Normalization | {inspection.normalization} |",
        f"| Color order | {inspection.color_order} |",
        f"| Notes | {inspection.notes} |",
        "",
    ]
    if metrics is not None:
        lines.extend(
            [
                "## Metrics",
                "",
                "| EER | Best accuracy | TAR@FAR=0.01 | TAR@FAR=0.001 | Threshold | Valid pairs | Skipped |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| {metrics['eer']:.6f} | {metrics['best_accuracy']:.6f} | "
                    f"{metrics['tar_at_far']['0.01']['tar']:.6f} | "
                    f"{metrics['tar_at_far']['0.001']['tar']:.6f} | "
                    f"{metrics['best_accuracy_threshold']:.6f} | "
                    f"{metrics['valid_pairs']} | {metrics['skipped_pairs']} |"
                ),
                "",
            ]
        )
    (output_dir / "candidate_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_cache_file(cache_dir: Path, lfw_root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(lfw_root).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]
    return cache_dir / image_path.parent.name / f"{image_path.stem}-{digest}.npz"


def _candidate_hflip_bytes(image_path: Path) -> bytes:
    from PIL import Image, ImageOps

    image = Image.open(image_path).convert("RGB")
    flipped = ImageOps.mirror(image)
    buffer = io.BytesIO()
    flipped.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _extract_with_preprocess_meta(
    extractor: CandidateTorchExtractor,
    image_bytes: bytes,
) -> tuple[np.ndarray, str, str]:
    face = extractor.extract_largest_embedding(image_bytes)
    return (
        normalize_embedding(face.embedding),
        extractor.last_preprocess_path,
        extractor.last_preprocess_error,
    )


def _load_or_extract_candidate_embedding(
    extractor: CandidateTorchExtractor,
    image_path: Path,
    *,
    lfw_root: Path,
    pipeline_cache_dir: Path,
    tta: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_file = _candidate_cache_file(pipeline_cache_dir, lfw_root, image_path)
    meta_file = cache_file.with_suffix(".json")
    if cache_file.exists() and meta_file.exists():
        embedding = normalize_embedding(np.load(cache_file)["embedding"])
        return embedding, json.loads(meta_file.read_text(encoding="utf-8"))

    embedding, path1, error1 = _extract_with_preprocess_meta(extractor, image_path.read_bytes())
    preprocess_paths = [path1]
    runtime_errors = [error1] if error1 else []
    if tta == "hflip":
        flipped_embedding, path2, error2 = _extract_with_preprocess_meta(
            extractor,
            _candidate_hflip_bytes(image_path),
        )
        embedding = normalize_embedding(embedding + flipped_embedding)
        preprocess_paths.append(path2)
        if error2:
            runtime_errors.append(error2)
    elif tta != "none":
        raise ValueError(f"Unsupported TTA mode: {tta}")

    meta = {
        "preprocess_paths": preprocess_paths,
        "runtime_success_images": sum(1 for item in preprocess_paths if item == "runtime"),
        "fallback_center_crop_images": sum(
            1 for item in preprocess_paths if item == "fallback_center_crop"
        ),
        "runtime_errors": runtime_errors,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, embedding=embedding)
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return embedding, meta


def evaluate_candidate_pairs_with_stats(
    extractor: CandidateTorchExtractor,
    pairs,
    *,
    lfw_root: Path,
    pipeline_cache_dir: Path,
    tta: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "runtime_success_images": 0,
        "fallback_center_crop_images": 0,
        "fallback_pairs_count": 0,
        "final_no_face": 0,
        "final_extractor_error": 0,
    }

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
            "fallback_used": False,
            "path1_preprocess": "",
            "path2_preprocess": "",
        }

        missing = [str(path) for path in (pair.path1, pair.path2) if not path.exists()]
        if missing:
            row["status"] = "missing_image"
            row["error"] = "; ".join(missing)
            rows.append(row)
            continue

        try:
            embedding1, meta1 = _load_or_extract_candidate_embedding(
                extractor,
                pair.path1,
                lfw_root=lfw_root,
                pipeline_cache_dir=pipeline_cache_dir,
                tta=tta,
            )
            embedding2, meta2 = _load_or_extract_candidate_embedding(
                extractor,
                pair.path2,
                lfw_root=lfw_root,
                pipeline_cache_dir=pipeline_cache_dir,
                tta=tta,
            )
            row["score"] = float(np.dot(embedding1, embedding2))
            row["path1_preprocess"] = "+".join(meta1["preprocess_paths"])
            row["path2_preprocess"] = "+".join(meta2["preprocess_paths"])
            fallback_used = (
                meta1["fallback_center_crop_images"] > 0
                or meta2["fallback_center_crop_images"] > 0
            )
            row["fallback_used"] = fallback_used
            if fallback_used:
                stats["fallback_pairs_count"] += 1
            stats["runtime_success_images"] += int(meta1["runtime_success_images"]) + int(
                meta2["runtime_success_images"]
            )
            stats["fallback_center_crop_images"] += int(
                meta1["fallback_center_crop_images"]
            ) + int(meta2["fallback_center_crop_images"])
        except NoFaceDetectedError as exc:
            row["status"] = "no_face"
            row["error"] = str(exc)
            stats["final_no_face"] += 1
        except FaceProcessingError as exc:
            row["status"] = "face_processing_error"
            row["error"] = str(exc)
            stats["final_extractor_error"] += 1
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            stats["final_extractor_error"] += 1
        rows.append(row)

    return rows, stats


def run_candidate_evaluation(args: argparse.Namespace) -> tuple[CandidateInspection, dict[str, Any] | None]:
    checkpoint = _resolve_path(args.checkpoint)
    output_dir = _resolve_path(args.output_dir)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = normalized_state(_unwrap_state(raw))
    selected_arch, matched, total = select_architecture(state, args.architecture)
    loads = bool(selected_arch and matched > 0)
    inspection = CandidateInspection(
        checkpoint=str(checkpoint),
        size_bytes=checkpoint.stat().st_size,
        architecture_requested=args.architecture,
        architecture_selected=selected_arch,
        loads=loads,
        matched_tensors=matched,
        total_model_tensors=total,
        embedding_dim=512 if loads else None,
        input_size="112x112",
        normalization=args.normalization,
        color_order=args.color_order,
        notes="ok" if loads else "No compatible local candidate architecture matched",
    )
    write_inspection(output_dir, inspection)
    if not loads or selected_arch is None:
        write_candidate_markdown(output_dir, inspection, None)
        return inspection, None

    model = load_model(state, selected_arch)
    extractor = CandidateTorchExtractor(
        model=model,
        model_name=str(selected_arch),
        device=_resolve_device(args.device),
        preprocess=args.preprocess,
        color_order=args.color_order,
        normalization=args.normalization,
    )
    if args.inspect_only:
        write_candidate_markdown(output_dir, inspection, None)
        return inspection, None

    lfw_root = _resolve_path(args.lfw_root)
    pairs_file = _resolve_path(args.pairs_file)
    pairs = parse_lfw_pairs(lfw_root, pairs_file, max_pairs=args.max_pairs)
    cache_dir = cache_dir_for_candidate(args.cache_dir, checkpoint, str(selected_arch), args)
    fallback_stats: dict[str, Any] = {}
    if args.preprocess == "runtime_fallback_center_crop":
        rows, fallback_stats = evaluate_candidate_pairs_with_stats(
            extractor,
            pairs,
            lfw_root=lfw_root,
            pipeline_cache_dir=cache_dir,
            tta=args.tta,
        )
    else:
        rows = evaluate_pairs(
            extractor,
            pairs,
            lfw_root=lfw_root,
            pipeline_cache_dir=cache_dir,
            tta=args.tta,
        )
    metrics = compute_verification_metrics(rows, pipeline="custom", extractor=extractor, settings=None)
    metrics.update(
        {
            "candidate_checkpoint_source": str(checkpoint),
            "candidate_architecture": selected_arch,
            "candidate_matched_tensors": matched,
            "candidate_preprocess": args.preprocess,
            "candidate_color_order": args.color_order,
            "candidate_normalization": args.normalization,
            "tta": args.tta,
            "lfw_root": str(lfw_root),
            "pairs_file": str(pairs_file),
            "pairs_loaded": len(pairs),
            **fallback_stats,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    score_fields = ["pair_id", "label_same", "name1", "name2", "path1", "path2", "score", "status", "error"]
    if args.preprocess == "runtime_fallback_center_crop":
        score_fields.extend(["fallback_used", "path1_preprocess", "path2_preprocess"])
    write_csv(
        output_dir / "pairs_scores.csv",
        rows,
        score_fields,
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
    write_candidate_markdown(output_dir, inspection, metrics)
    return inspection, metrics


def main() -> None:
    inspection, metrics = run_candidate_evaluation(parse_args())
    if metrics is None:
        print(
            f"Candidate {Path(inspection.checkpoint).name}: "
            f"arch={inspection.architecture_selected}, inspection only"
        )
        return
    print(
        f"Candidate {Path(inspection.checkpoint).name}: "
        f"arch={inspection.architecture_selected}, valid={metrics['valid_pairs']}, "
        f"EER={metrics['eer']}, accuracy={metrics['best_accuracy']}"
    )


if __name__ == "__main__":
    main()
