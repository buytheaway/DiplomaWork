from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from training.datasets.folder_dataset import FolderFaceDataset
from training.models.ir_resnet import build_model
from training.utils import load_config, set_seed


def build_transforms(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate face embedding model")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    device = torch.device(args.device or config["device"])
    input_size = int(config["data"]["input_size"])
    val_dir = Path(args.val_dir or config["data"]["val_dir"])

    val_ds = FolderFaceDataset(val_dir, transform=build_transforms(input_size))
    val_loader = DataLoader(
        val_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in tqdm(val_loader, desc="Eval"):
            images = images.to(device, non_blocking=True)
            feats = model(images)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            embeddings.append(feats.cpu().numpy())
            labels.append(batch_labels.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    # simple top-1 nearest class centroid accuracy
    num_classes = int(labels.max()) + 1
    centroids = np.zeros((num_classes, embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros((num_classes,), dtype=np.int32)
    for emb, label in zip(embeddings, labels):
        centroids[label] += emb
        counts[label] += 1
    centroids /= np.maximum(counts[:, None], 1)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    sims = embeddings @ centroids.T
    preds = np.argmax(sims, axis=1)
    acc = float((preds == labels).mean())
    print(f"Top-1 centroid accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
