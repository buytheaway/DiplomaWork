"""LFW pairwise verification evaluation with fold-based threshold selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from training.models.ir_resnet import build_model
from training.utils import load_config, set_seed
from training.verification_metrics import (
    best_accuracy_threshold,
    equal_error_rate,
    roc_curve_points,
    tar_at_far,
)

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve as sklearn_roc_curve
except ImportError:  # pragma: no cover - depends on optional local environment
    roc_auc_score = None
    sklearn_roc_curve = None


def _load_pairs(pairs_path: Path) -> tuple[int, list[tuple[str, int, str, int]]]:
    num_folds = 10
    pairs: list[tuple[str, int, str, int]] = []

    with open(pairs_path, encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            parts = line.strip().split()
            if not parts:
                continue

            if line_index == 0 and len(parts) == 1 and parts[0].isdigit():
                num_folds = int(parts[0])
                continue

            if len(parts) == 3:
                name = parts[0]
                pairs.append((name, int(parts[1]), name, int(parts[2])))
            elif len(parts) == 4:
                pairs.append((parts[0], int(parts[1]), parts[2], int(parts[3])))

    return num_folds, pairs


def _image_path(lfw_dir: Path, name: str, idx: int) -> Path:
    return lfw_dir / name / f"{name}_{idx:04d}.jpg"


def _extract_embedding(
    model: torch.nn.Module,
    img_path: Path,
    transform: transforms.Compose,
    device: torch.device,
    cache: dict[Path, np.ndarray],
) -> np.ndarray:
    cached = cache.get(img_path)
    if cached is not None:
        return cached

    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    vector = embedding.cpu().numpy().flatten()
    cache[img_path] = vector
    return vector


def evaluate_lfw(
    model: torch.nn.Module,
    lfw_dir: Path,
    pairs_path: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> dict:
    num_folds, pairs = _load_pairs(pairs_path)
    if not pairs:
        raise ValueError(f"No pairs loaded from {pairs_path}")

    embedding_cache: dict[Path, np.ndarray] = {}
    similarities: list[float] = []
    labels: list[int] = []
    fold_ids: list[int] = []
    skipped_pairs = 0

    fold_size = max(1, len(pairs) // max(num_folds, 1))
    for pair_index, (name1, idx1, name2, idx2) in enumerate(
        tqdm(pairs, desc="LFW evaluation")
    ):
        path1 = _image_path(lfw_dir, name1, idx1)
        path2 = _image_path(lfw_dir, name2, idx2)
        if not path1.exists() or not path2.exists():
            skipped_pairs += 1
            continue

        emb1 = _extract_embedding(model, path1, transform, device, embedding_cache)
        emb2 = _extract_embedding(model, path2, transform, device, embedding_cache)

        similarities.append(float(np.dot(emb1, emb2)))
        labels.append(1 if name1 == name2 else 0)
        fold_ids.append(min(pair_index // fold_size, max(num_folds - 1, 0)))

    if not labels:
        raise ValueError("No valid LFW pairs were evaluated")

    scores = np.asarray(similarities, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int32)
    fold_ids_np = np.asarray(fold_ids, dtype=np.int32)
    thresholds = np.linspace(-1.0, 1.0, num=4001, dtype=np.float32)

    fold_metrics: list[dict] = []
    accuracies: list[float] = []
    best_thresholds: list[float] = []
    target_fars = [1e-1, 1e-2, 1e-3, 1e-4]
    tar_at_far_values = {
        "TAR@FAR=0.1": [],
        "TAR@FAR=0.01": [],
        "TAR@FAR=0.001": [],
        "TAR@FAR=0.0001": [],
    }

    for fold_id in range(max(fold_ids_np.max() + 1, 1)):
        test_mask = fold_ids_np == fold_id
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue

        train_scores = scores[train_mask]
        train_targets = targets[train_mask]
        test_scores = scores[test_mask]
        test_targets = targets[test_mask]

        best_threshold_result = best_accuracy_threshold(train_scores, train_targets, thresholds)
        best_threshold = best_threshold_result["threshold"]
        test_predictions = (test_scores >= best_threshold).astype(np.int32)
        test_accuracy = float((test_predictions == test_targets).mean())

        fold_result = {
            "fold": int(fold_id + 1),
            "accuracy": round(test_accuracy, 4),
            "threshold": round(best_threshold, 4),
        }

        for far in target_fars:
            far_selection = tar_at_far(train_scores, train_targets, [far], thresholds)[0]
            far_threshold = far_selection["threshold"]
            positive_test_scores = test_scores[test_targets == 1]
            tar = (
                float((positive_test_scores >= far_threshold).mean())
                if len(positive_test_scores) > 0 and np.isfinite(far_threshold)
                else 0.0
            )
            key = f"TAR@FAR={far}"
            tar_at_far_values[key].append(tar)
            fold_result[key] = {
                "tar": round(tar, 4),
                "threshold": round(far_threshold, 4),
                "train_far": round(far_selection["far"], 6),
                "train_frr": round(far_selection["frr"], 6),
            }

        fold_metrics.append(fold_result)
        accuracies.append(test_accuracy)
        best_thresholds.append(best_threshold)

    try:
        if roc_auc_score is None or sklearn_roc_curve is None:
            raise RuntimeError("scikit-learn metrics are unavailable")
        auc = float(roc_auc_score(targets, scores))
        fpr, tpr, _ = sklearn_roc_curve(targets, scores)
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    except Exception:
        auc = -1.0
        roc_data = {}

    eer_result = equal_error_rate(scores, targets, thresholds)
    global_best_threshold = best_accuracy_threshold(scores, targets, thresholds)
    global_tar_at_far = tar_at_far(scores, targets, target_fars, thresholds)
    far_frr_curve_points = roc_curve_points(scores, targets, thresholds, max_points=401)

    tar_summary = {}
    for key, values in tar_at_far_values.items():
        tar_summary[key] = {
            "tar": round(float(np.mean(values)) if values else 0.0, 4),
            "tar_std": round(float(np.std(values)) if values else 0.0, 4),
        }

    return {
        "num_pairs": int(len(targets)),
        "skipped_pairs": int(skipped_pairs),
        "num_positive": int(targets.sum()),
        "num_negative": int(len(targets) - targets.sum()),
        "accuracy": round(float(np.mean(accuracies)) if accuracies else 0.0, 4),
        "accuracy_std": round(float(np.std(accuracies)) if accuracies else 0.0, 4),
        "best_threshold": round(float(np.mean(best_thresholds)) if best_thresholds else 0.0, 4),
        "eer": round(eer_result["eer"], 6),
        "eer_threshold": round(eer_result["threshold"], 6),
        "auc": round(auc, 4),
        "selected_thresholds": {
            "mean_fold_best_accuracy": round(
                float(np.mean(best_thresholds)) if best_thresholds else 0.0,
                6,
            ),
            "global_best_accuracy": {
                "threshold": round(global_best_threshold["threshold"], 6),
                "accuracy": round(global_best_threshold["accuracy"], 6),
            },
            "eer": round(eer_result["threshold"], 6),
            "tar_at_far": {
                f"{item['target_far']:.4g}": round(item["threshold"], 6)
                for item in global_tar_at_far
            },
        },
        "tar_at_far": tar_summary,
        "tar_at_far_global": [
            {
                "target_far": item["target_far"],
                "tar": round(item["tar"], 6),
                "threshold": round(item["threshold"], 6),
                "far": round(item["far"], 6),
                "frr": round(item["frr"], 6),
            }
            for item in global_tar_at_far
        ],
        "far_frr_curve_points": far_frr_curve_points,
        "folds": fold_metrics,
        "roc_data": roc_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LFW pairwise verification benchmark")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint")
    parser.add_argument("--lfw-dir", default=None, help="Path to aligned LFW images")
    parser.add_argument("--pairs", default=None, help="Path to pairs.txt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    device = torch.device(args.device or config["device"])
    input_size = int(config["data"]["input_size"])
    lfw_dir = Path(args.lfw_dir or config["data"]["lfw_dir"])
    pairs_path = Path(args.pairs or config["data"]["pairs_file"])

    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()

    results = evaluate_lfw(model, lfw_dir, pairs_path, transform, device)

    print("\n" + "=" * 50)
    print("LFW Verification Results")
    print("=" * 50)
    print(f"  Pairs evaluated:  {results['num_pairs']}")
    print(f"  Skipped pairs:    {results['skipped_pairs']}")
    print(f"  Positive pairs:   {results['num_positive']}")
    print(f"  Negative pairs:   {results['num_negative']}")
    print(f"  Accuracy:         {results['accuracy']:.4f} +/- {results['accuracy_std']:.4f}")
    print(f"  Mean threshold:   {results['best_threshold']:.4f}")
    print(f"  EER:              {results['eer']:.4f}")
    print(f"  EER threshold:    {results['eer_threshold']:.4f}")
    print(f"  AUC:              {results['auc']:.4f}")
    print()
    for key, val in results["tar_at_far"].items():
        print(f"  {key}:  TAR={val['tar']:.4f} +/- {val['tar_std']:.4f}")
    print("=" * 50)

    output_path = args.output or "training/outputs/lfw_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_results = {k: v for k, v in results.items() if k != "roc_data"}
    save_results["roc_data_points"] = len(results.get("roc_data", {}).get("fpr", []))
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(save_results, handle, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
