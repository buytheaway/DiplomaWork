"""LFW pairwise verification evaluation.

Loads a trained IR-50 (or IR-18) checkpoint and evaluates it on the
standard Labeled Faces in the Wild (LFW) pairs protocol.

The script expects:
  - A directory of aligned 112×112 LFW images in the structure:
      lfw_aligned/
        Abel_Pacheco/
          Abel_Pacheco_0001.jpg
          Abel_Pacheco_0004.jpg
        ...
  - A pairs.txt file in the standard LFW format:
      10 (number of folds — informational, not used)
      Abel_Pacheco	1	4       ← positive pair
      Abdel_Madi_Shabneh	1	Dean_Barker	1  ← negative pair

Usage::

    python training/eval_lfw.py \\
        --weights training/outputs/checkpoint_epoch_028.pth \\
        --lfw-dir data/lfw_aligned \\
        --pairs data/pairs.txt \\
        --device cuda

Outputs: accuracy, AUC, TAR@FAR=1e-3, best threshold, and saves ROC
data to a JSON file.
"""

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


def _load_pairs(pairs_path: Path) -> list[tuple[str, int, str, int]]:
    """Parse LFW pairs.txt into a list of (name1, idx1, name2, idx2)."""
    pairs: list[tuple[str, int, str, int]] = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                # Positive pair: name idx1 idx2
                name = parts[0]
                pairs.append((name, int(parts[1]), name, int(parts[2])))
            elif len(parts) == 4:
                # Negative pair: name1 idx1 name2 idx2
                pairs.append((parts[0], int(parts[1]), parts[2], int(parts[3])))
    return pairs


def _image_path(lfw_dir: Path, name: str, idx: int) -> Path:
    """Construct the image path for a given LFW identity and index."""
    return lfw_dir / name / f"{name}_{idx:04d}.jpg"


def _extract_embedding(
    model: torch.nn.Module,
    img_path: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> np.ndarray:
    """Load an image and extract a normalized embedding."""
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().flatten()


def evaluate_lfw(
    model: torch.nn.Module,
    lfw_dir: Path,
    pairs_path: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> dict:
    """Run the full LFW verification benchmark."""
    pairs = _load_pairs(pairs_path)
    if not pairs:
        raise ValueError(f"No pairs loaded from {pairs_path}")

    similarities: list[float] = []
    labels: list[int] = []  # 1 = same person, 0 = different person

    for name1, idx1, name2, idx2 in tqdm(pairs, desc="LFW evaluation"):
        path1 = _image_path(lfw_dir, name1, idx1)
        path2 = _image_path(lfw_dir, name2, idx2)

        if not path1.exists() or not path2.exists():
            continue

        emb1 = _extract_embedding(model, path1, transform, device)
        emb2 = _extract_embedding(model, path2, transform, device)

        # Cosine similarity (embeddings are already L2-normalized)
        sim = float(np.dot(emb1, emb2))
        similarities.append(sim)
        labels.append(1 if name1 == name2 else 0)

    similarities = np.array(similarities)
    labels = np.array(labels)

    # Find optimal threshold
    thresholds = np.arange(-1.0, 1.0, 0.001)
    best_acc = 0.0
    best_threshold = 0.0

    for t in thresholds:
        preds = (similarities >= t).astype(int)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(t)

    # Compute TAR@FAR for various FAR levels
    positive_sims = similarities[labels == 1]
    negative_sims = similarities[labels == 0]

    tar_at_far = {}
    for target_far in [1e-1, 1e-2, 1e-3, 1e-4]:
        # Find threshold where FAR = target_far
        sorted_neg = np.sort(negative_sims)[::-1]
        idx = int(len(sorted_neg) * target_far)
        if idx >= len(sorted_neg):
            idx = len(sorted_neg) - 1
        threshold_at_far = float(sorted_neg[max(idx, 0)])
        tar = float((positive_sims >= threshold_at_far).mean())
        tar_at_far[f"TAR@FAR={target_far}"] = {
            "tar": round(tar, 4),
            "threshold": round(threshold_at_far, 4),
        }

    # Compute AUC (trapezoidal)
    from sklearn.metrics import roc_auc_score, roc_curve

    try:
        auc = float(roc_auc_score(labels, similarities))
        fpr, tpr, roc_thresholds = roc_curve(labels, similarities)
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
    except Exception:
        auc = -1.0
        roc_data = {}

    return {
        "num_pairs": len(labels),
        "num_positive": int(labels.sum()),
        "num_negative": int(len(labels) - labels.sum()),
        "accuracy": round(best_acc, 4),
        "best_threshold": round(best_threshold, 4),
        "auc": round(auc, 4),
        "tar_at_far": tar_at_far,
        "roc_data": roc_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LFW pairwise verification benchmark")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint")
    parser.add_argument("--lfw-dir", required=True, help="Path to aligned LFW images")
    parser.add_argument("--pairs", required=True, help="Path to pairs.txt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    device = torch.device(args.device or config["device"])
    input_size = int(config["data"]["input_size"])

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()

    results = evaluate_lfw(model, Path(args.lfw_dir), Path(args.pairs), transform, device)

    print("\n" + "=" * 50)
    print("LFW Verification Results")
    print("=" * 50)
    print(f"  Pairs evaluated:  {results['num_pairs']}")
    print(f"  Positive pairs:   {results['num_positive']}")
    print(f"  Negative pairs:   {results['num_negative']}")
    print(f"  Accuracy:         {results['accuracy']:.4f}")
    print(f"  Best threshold:   {results['best_threshold']:.4f}")
    print(f"  AUC:              {results['auc']:.4f}")
    print()
    for key, val in results["tar_at_far"].items():
        print(f"  {key}:  TAR={val['tar']:.4f}  (threshold={val['threshold']:.4f})")
    print("=" * 50)

    output_path = args.output or "training/outputs/lfw_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Don't save huge ROC arrays to printed output
    save_results = {k: v for k, v in results.items() if k != "roc_data"}
    save_results["roc_data_points"] = len(results.get("roc_data", {}).get("fpr", []))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
