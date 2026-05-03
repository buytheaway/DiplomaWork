"""Evaluate verification pairs with a configured backend embedding extractor.

This standalone script does not use the backend API, database, runtime registry,
or FAISS index. It only reuses the embedding extractor factory so FAR/FRR/EER
can be measured for the same extractor path used by the MVP.
"""

from __future__ import annotations

import argparse
import json
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

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings.interface import EmbeddingExtractor, create_extractor  # noqa: E402
from training.verification_metrics import (  # noqa: E402
    best_accuracy_threshold,
    equal_error_rate,
    roc_curve_points,
    tar_at_far,
)

BackendName = Literal["onnx", "insightface", "torch", "dummy"]


@dataclass(frozen=True)
class VerificationPair:
    name1: str
    index1: int
    name2: str
    index2: int
    label: int
    line_number: int


def parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return [float(value) for value in values]


def parse_lfw_pairs(pairs_path: Path, max_pairs: int | None = None) -> list[VerificationPair]:
    """Parse LFW-style pairs.txt.

    Supported pair lines:
    * same identity: ``name index1 index2``
    * different identity: ``name1 index1 name2 index2``

    Header lines such as ``10`` or ``10 300`` are skipped.
    """
    pairs: list[VerificationPair] = []
    with pairs_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) in {1, 2} and all(part.isdigit() for part in parts):
                continue

            if len(parts) == 3:
                pair = VerificationPair(
                    name1=parts[0],
                    index1=int(parts[1]),
                    name2=parts[0],
                    index2=int(parts[2]),
                    label=1,
                    line_number=line_number,
                )
            elif len(parts) == 4:
                pair = VerificationPair(
                    name1=parts[0],
                    index1=int(parts[1]),
                    name2=parts[2],
                    index2=int(parts[3]),
                    label=0,
                    line_number=line_number,
                )
            else:
                raise ValueError(f"Unsupported pairs.txt format at line {line_number}: {line!r}")

            pairs.append(pair)
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
    return pairs


def lfw_image_path(images_dir: Path, name: str, index: int) -> Path:
    return images_dir / name / f"{name}_{index:04d}.jpg"


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    embedding = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(embedding))
    if norm > 0.0:
        embedding = embedding / norm
    return embedding


def cached_embedding(
    extractor: EmbeddingExtractor,
    image_path: Path,
    cache: dict[Path, np.ndarray],
) -> np.ndarray:
    cached = cache.get(image_path)
    if cached is not None:
        return cached

    embedding = normalize_embedding(extractor.extract_embedding(image_path.read_bytes()))
    cache[image_path] = embedding
    return embedding


def evaluate_pair_scores(
    extractor: EmbeddingExtractor,
    images_dir: Path,
    pairs: list[VerificationPair],
    *,
    fail_on_missing: bool = False,
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[int] = []
    failed_pairs = 0
    missing_images = 0
    extraction_errors = 0
    cache: dict[Path, np.ndarray] = {}

    for pair in pairs:
        path1 = lfw_image_path(images_dir, pair.name1, pair.index1)
        path2 = lfw_image_path(images_dir, pair.name2, pair.index2)
        if not path1.exists() or not path2.exists():
            missing_images += 1
            failed_pairs += 1
            if fail_on_missing:
                missing = path1 if not path1.exists() else path2
                raise FileNotFoundError(f"Missing image for pair line {pair.line_number}: {missing}")
            continue

        try:
            embedding1 = cached_embedding(extractor, path1, cache)
            embedding2 = cached_embedding(extractor, path2, cache)
            scores.append(float(np.dot(embedding1, embedding2)))
            labels.append(pair.label)
        except Exception:
            extraction_errors += 1
            failed_pairs += 1

    return {
        "scores": scores,
        "labels": labels,
        "pairs_evaluated": len(scores),
        "failed_pairs": failed_pairs,
        "missing_images": missing_images,
        "extraction_errors": extraction_errors,
        "unique_images_embedded": len(cache),
    }


def compute_metric_summary(
    scores: list[float],
    labels: list[int],
    *,
    target_fars: list[float],
    thresholds: list[float] | None,
) -> dict[str, Any]:
    if not scores:
        raise ValueError("No evaluated pairs; cannot compute verification metrics")
    if 1 not in labels or 0 not in labels:
        raise ValueError("Verification metrics require both positive and negative pairs")

    eer = equal_error_rate(scores, labels, thresholds)
    best = best_accuracy_threshold(scores, labels, thresholds)
    tar_values = tar_at_far(scores, labels, target_fars, thresholds)

    return {
        "eer": round(eer["eer"], 6),
        "eer_threshold": round(eer["threshold"], 6),
        "best_accuracy_threshold": {
            "threshold": round(best["threshold"], 6),
            "accuracy": round(best["accuracy"], 6),
            "far": round(best["far"], 6),
            "frr": round(best["frr"], 6),
        },
        "tar_at_far": [
            {
                "target_far": item["target_far"],
                "tar": round(item["tar"], 6),
                "threshold": round(item["threshold"], 6),
                "far": round(item["far"], 6),
                "frr": round(item["frr"], 6),
            }
            for item in tar_values
        ],
        "selected_thresholds": {
            "eer": round(eer["threshold"], 6),
            "best_accuracy": round(best["threshold"], 6),
            "tar_at_far": {
                f"{item['target_far']:.6g}": round(item["threshold"], 6)
                for item in tar_values
            },
        },
        "far_frr_curve_points": roc_curve_points(scores, labels, thresholds, max_points=401),
    }


def settings_summary(settings: object, backend: BackendName) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "embedding_backend": backend,
        "embedding_dim": getattr(settings, "embedding_dim", None),
        "strict_single_face": getattr(settings, "strict_single_face", None),
        "min_det_score": getattr(settings, "min_det_score", None),
    }
    if backend == "onnx":
        summary.update(
            {
                "onnx_detector_model": Path(getattr(settings, "onnx_detector_path", "")).name,
                "onnx_embedder_model": Path(getattr(settings, "onnx_embedder_path", "")).name,
            }
        )
    elif backend == "insightface":
        summary["model_name"] = getattr(settings, "model_name", None)
    elif backend == "torch":
        summary.update(
            {
                "torch_model_arch": getattr(settings, "torch_model_arch", None),
                "torch_input_size": getattr(settings, "torch_input_size", None),
                "torch_device": getattr(settings, "torch_device", None),
                "torch_norm_embeddings": getattr(settings, "torch_norm_embeddings", None),
                "torch_model_path_present": bool(getattr(settings, "torch_model_path", "")),
                "detection_backend": getattr(settings, "detection_backend", None),
                "allow_center_crop": getattr(settings, "allow_center_crop", None),
            }
        )
    return summary


def create_backend_extractor(backend: BackendName, device: str | None) -> tuple[EmbeddingExtractor, object]:
    settings = get_settings().model_copy(
        update={
            "embedding_backend": backend,
            "testing": False,
            **({"torch_device": device} if device else {}),
        }
    )
    return create_extractor(settings), settings


def build_output_payload(
    *,
    backend: BackendName,
    extractor: EmbeddingExtractor,
    settings: object,
    images_dir: Path,
    pairs_file: Path,
    pairs_loaded: int,
    score_data: dict[str, Any],
    metric_data: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    labels = score_data["labels"]
    partial = score_data["failed_pairs"] > 0
    if partial:
        warnings.append("Evaluation is partial because at least one pair failed.")
    if backend == "dummy":
        warnings.append("Dummy backend is for plumbing tests only and has no biometric meaning.")

    return {
        "backend": backend,
        "extractor_model": extractor.model_name,
        "extractor_settings": settings_summary(settings, backend),
        "images_dir": str(images_dir),
        "pairs_file": str(pairs_file),
        "pairs_loaded": pairs_loaded,
        "pairs_evaluated": score_data["pairs_evaluated"],
        "failed_pairs": score_data["failed_pairs"],
        "missing_images": score_data["missing_images"],
        "extraction_errors": score_data["extraction_errors"],
        "unique_images_embedded": score_data["unique_images_embedded"],
        "positive_pair_count": int(sum(1 for label in labels if label == 1)),
        "negative_pair_count": int(sum(1 for label in labels if label == 0)),
        "warnings": warnings,
        **metric_data,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FAR/FRR/EER on LFW-style pairs with a backend extractor"
    )
    parser.add_argument("--backend", choices=["onnx", "insightface", "torch", "dummy"], required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--target-far", default="0.001,0.01,0.1")
    parser.add_argument("--thresholds", default=None, help="Optional comma-separated thresholds")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    pairs_file = Path(args.pairs)
    output_path = Path(args.output)

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not pairs_file.exists():
        raise SystemExit(f"Pairs file not found: {pairs_file}")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise SystemExit("--max-pairs must be positive")

    target_fars = parse_float_list(args.target_far)
    thresholds = parse_float_list(args.thresholds)
    if target_fars is None:
        raise SystemExit("--target-far must not be empty")

    pairs = parse_lfw_pairs(pairs_file, max_pairs=args.max_pairs)
    if not pairs:
        raise SystemExit("No pairs loaded")

    extractor, settings = create_backend_extractor(args.backend, args.device)
    score_data = evaluate_pair_scores(
        extractor,
        images_dir,
        pairs,
        fail_on_missing=args.fail_on_missing,
    )
    metric_data = compute_metric_summary(
        score_data["scores"],
        score_data["labels"],
        target_fars=target_fars,
        thresholds=thresholds,
    )
    payload = build_output_payload(
        backend=args.backend,
        extractor=extractor,
        settings=settings,
        images_dir=images_dir,
        pairs_file=pairs_file,
        pairs_loaded=len(pairs),
        score_data=score_data,
        metric_data=metric_data,
        warnings=[],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
