"""Run conservative CelebA fine-tuning experiments for the custom IR-50 model.

This script is deliberately training-only. It does not change backend runtime,
desktop UI, enroll/search behavior, or FAISS indexes. LFW is used only after
each epoch for verification evaluation and checkpoint selection.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from training.datasets.folder_dataset import FolderFaceDataset
from training.losses.arcface import ArcFace
from training.models.ir_resnet import build_model
from training.utils import set_seed
from training.verification_metrics import best_accuracy_threshold, equal_error_rate, tar_at_far

TARGET_FARS = [0.1, 0.01, 0.001]


@dataclass(frozen=True)
class Experiment:
    name: str
    output_dir: Path
    freeze_epochs: int
    epochs: int
    freeze_head_lr: float
    unfreeze_backbone_lr: float
    unfreeze_head_lr: float
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CelebA custom IR-50 fine-tuning")
    parser.add_argument("--train-root", default="datasets/celeba_faces/train")
    parser.add_argument("--val-root", default="datasets/celeba_faces/val")
    parser.add_argument("--lfw-root", default="handoff_lfw_eval/lfw")
    parser.add_argument("--pairs-file", default="handoff_lfw_eval/lfw/pairs.txt")
    parser.add_argument("--init-checkpoint", default="handoff_lfw_eval/artifacts/best_lfw.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=6000)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_backbone(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = not frozen


def build_optimizer(
    model: nn.Module,
    head: nn.Module,
    *,
    freeze_backbone: bool,
    backbone_lr: float,
    head_lr: float,
) -> SGD:
    set_backbone_frozen(model, freeze_backbone)
    params: list[dict] = [{"params": head.parameters(), "lr": head_lr}]
    if not freeze_backbone:
        params.insert(0, {"params": model.parameters(), "lr": backbone_lr})
    return SGD(params, momentum=0.9, weight_decay=0.0005)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    head: nn.Module,
    optimizer: SGD,
    scaler: GradScaler,
    experiment: Experiment,
    lfw_metrics: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "metadata": {
                "arch": "ir50",
                "embedding_dim": 512,
                "loss": "arcface",
                "experiment": asdict(experiment),
                "lfw_metrics": lfw_metrics,
            },
        },
        path,
    )


def parse_lfw_pairs(lfw_root: Path, pairs_file: Path, max_pairs: int | None) -> list[tuple[int, Path, Path]]:
    pairs: list[tuple[int, Path, Path]] = []
    with pairs_file.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) in {1, 2} and all(part.isdigit() for part in parts):
                continue
            if len(parts) == 3:
                name1 = parts[0]
                name2 = parts[0]
                idx1 = int(parts[1])
                idx2 = int(parts[2])
                same = 1
            elif len(parts) == 4:
                name1 = parts[0]
                idx1 = int(parts[1])
                name2 = parts[2]
                idx2 = int(parts[3])
                same = 0
            else:
                continue
            path1 = lfw_root / name1 / f"{name1}_{idx1:04d}.jpg"
            path2 = lfw_root / name2 / f"{name2}_{idx2:04d}.jpg"
            pairs.append((same, path1, path2))
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
    return pairs


def extract_lfw_embedding(
    model: nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
    cache: dict[Path, np.ndarray],
) -> np.ndarray:
    cached = cache.get(image_path)
    if cached is not None:
        return cached
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        embedding = model(tensor)
        embedding = torch.nn.functional.normalize(embedding.float(), p=2, dim=1)
    vector = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
    cache[image_path] = vector
    return vector


def evaluate_lfw_checkpoint(
    model: nn.Module,
    pairs: list[tuple[int, Path, Path]],
    device: torch.device,
) -> dict:
    model.eval()
    transform = eval_transform()
    cache: dict[Path, np.ndarray] = {}
    scores: list[float] = []
    labels: list[int] = []
    skipped = 0

    for same, path1, path2 in tqdm(pairs, desc="LFW", leave=False):
        if not path1.exists() or not path2.exists():
            skipped += 1
            continue
        embedding1 = extract_lfw_embedding(model, path1, transform, device, cache)
        embedding2 = extract_lfw_embedding(model, path2, transform, device, cache)
        scores.append(float(np.dot(embedding1, embedding2)))
        labels.append(same)

    if 1 not in labels or 0 not in labels:
        raise RuntimeError("LFW evaluation needs both positive and negative pairs")

    eer = equal_error_rate(scores, labels)
    best = best_accuracy_threshold(scores, labels)
    tar_values = tar_at_far(scores, labels, TARGET_FARS)
    positive_scores = [score for score, label in zip(scores, labels, strict=False) if label == 1]
    negative_scores = [score for score, label in zip(scores, labels, strict=False) if label == 0]

    return {
        "valid_pairs": len(scores),
        "skipped_pairs": skipped,
        "positive_pairs": len(positive_scores),
        "negative_pairs": len(negative_scores),
        "eer": round(float(eer["eer"]), 6),
        "eer_threshold": round(float(eer["threshold"]), 6),
        "eer_far": round(float(eer["far"]), 6),
        "eer_frr": round(float(eer["frr"]), 6),
        "best_accuracy": round(float(best["accuracy"]), 6),
        "best_accuracy_threshold": round(float(best["threshold"]), 6),
        "far_at_best_threshold": round(float(best["far"]), 6),
        "frr_at_best_threshold": round(float(best["frr"]), 6),
        "tar_at_far": {
            f"{item['target_far']:.3g}": {
                "tar": round(float(item["tar"]), 6),
                "threshold": round(float(item["threshold"]), 6),
                "far": round(float(item["far"]), 6),
                "frr": round(float(item["frr"]), 6),
            }
            for item in tar_values
        },
        "mean_positive_score": round(float(np.mean(positive_scores)), 6),
        "mean_negative_score": round(float(np.mean(negative_scores)), 6),
    }


def summarize_dataset(root: Path) -> dict:
    counts = []
    samples = []
    for identity in sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name):
        images = sorted(
            [
                path
                for path in identity.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        if images:
            counts.append(len(images))
            if len(samples) < 5:
                samples.append(str(images[0]))
    return {
        "root": str(root),
        "identities": len(counts),
        "images": sum(counts),
        "min_images_per_identity": min(counts) if counts else 0,
        "avg_images_per_identity": round(sum(counts) / len(counts), 2) if counts else 0.0,
        "max_images_per_identity": max(counts) if counts else 0,
        "sample_paths": samples,
    }


def write_run_summary(output_dir: Path, experiment: Experiment, history: list[dict]) -> None:
    rows = [
        "| Epoch | EER | Best accuracy | TAR@FAR=0.01 | Threshold | Train loss | Notes |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in history:
        metrics = item["lfw"]
        rows.append(
            f"| {item['epoch']} | {metrics['eer']:.6f} | {metrics['best_accuracy']:.6f} | "
            f"{metrics['tar_at_far']['0.01']['tar']:.6f} | "
            f"{metrics['best_accuracy_threshold']:.6f} | {item['train_loss']:.6f} | "
            f"{item['phase']} |"
        )

    best_eer = min(history, key=lambda item: item["lfw"]["eer"]) if history else None
    best_tar = max(history, key=lambda item: item["lfw"]["tar_at_far"]["0.01"]["tar"]) if history else None
    best_acc = max(history, key=lambda item: item["lfw"]["best_accuracy"]) if history else None

    lines = [
        f"# {experiment.name} CelebA Fine-Tuning Summary",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Freeze epochs: `{experiment.freeze_epochs}`",
        f"- Total configured epochs: `{experiment.epochs}`",
        f"- Batch size: `{experiment.batch_size}`",
        f"- Freeze head LR: `{experiment.freeze_head_lr}`",
        f"- Unfreeze backbone LR: `{experiment.unfreeze_backbone_lr}`",
        f"- Unfreeze head LR: `{experiment.unfreeze_head_lr}`",
        "",
        "## Epoch Metrics",
        "",
        *rows,
        "",
    ]
    if best_eer is not None and best_tar is not None and best_acc is not None:
        lines.extend(
            [
                "## Best Checkpoints",
                "",
                f"- Best by EER: epoch {best_eer['epoch']} -> `best_by_eer.pth`",
                f"- Best by TAR@FAR=0.01: epoch {best_tar['epoch']} -> `best_by_tar_far001.pth`",
                f"- Best by accuracy: epoch {best_acc['epoch']} -> `best_by_accuracy.pth`",
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def update_best_checkpoints(output_dir: Path, history: list[dict]) -> None:
    if not history:
        return
    best_eer = min(history, key=lambda item: item["lfw"]["eer"])
    best_tar = max(history, key=lambda item: item["lfw"]["tar_at_far"]["0.01"]["tar"])
    best_acc = max(history, key=lambda item: item["lfw"]["best_accuracy"])
    shutil.copy2(output_dir / f"checkpoint_epoch_{best_eer['epoch']:03d}.pth", output_dir / "best_by_eer.pth")
    shutil.copy2(
        output_dir / f"checkpoint_epoch_{best_tar['epoch']:03d}.pth",
        output_dir / "best_by_tar_far001.pth",
    )
    shutil.copy2(
        output_dir / f"checkpoint_epoch_{best_acc['epoch']:03d}.pth",
        output_dir / "best_by_accuracy.pth",
    )


def run_experiment(
    experiment: Experiment,
    *,
    args: argparse.Namespace,
    train_loader: DataLoader,
    num_classes: int,
    lfw_pairs: list[tuple[int, Path, Path]],
    device: torch.device,
) -> list[dict]:
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model("ir50", 512).to(device)
    head = ArcFace(embedding_dim=512, num_classes=num_classes, margin=0.5, scale=64.0).to(device)
    load_backbone(model, Path(args.init_checkpoint), device)

    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer: SGD | None = None
    history: list[dict] = []
    best_eer = float("inf")
    no_improvement_epochs = 0

    for epoch in range(1, experiment.epochs + 1):
        phase = "frozen_backbone" if epoch <= experiment.freeze_epochs else "unfrozen_backbone"
        should_freeze = epoch <= experiment.freeze_epochs
        if epoch == 1 or epoch == experiment.freeze_epochs + 1:
            optimizer = build_optimizer(
                model,
                head,
                freeze_backbone=should_freeze,
                backbone_lr=experiment.unfreeze_backbone_lr,
                head_lr=experiment.freeze_head_lr
                if should_freeze
                else experiment.unfreeze_head_lr,
            )
        assert optimizer is not None

        model.train()
        head.train()
        running_loss = 0.0
        total = 0
        epoch_start = time.time()

        loop = tqdm(train_loader, desc=f"{experiment.name} epoch {epoch}/{experiment.epochs}")
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                embeddings = model(images)
                logits = head(embeddings, labels)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()),
                max_norm=5.0,
            )
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / max(total, 1)
        checkpoint_path = experiment.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            head=head,
            optimizer=optimizer,
            scaler=scaler,
            experiment=experiment,
        )

        lfw_metrics = evaluate_lfw_checkpoint(model, lfw_pairs, device)
        save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            head=head,
            optimizer=optimizer,
            scaler=scaler,
            experiment=experiment,
            lfw_metrics=lfw_metrics,
        )

        epoch_result = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": round(train_loss, 6),
            "epoch_time_s": round(time.time() - epoch_start, 1),
            "checkpoint": str(checkpoint_path),
            "lfw": lfw_metrics,
        }
        history.append(epoch_result)
        (experiment.output_dir / f"metrics_epoch_{epoch:03d}.json").write_text(
            json.dumps(epoch_result, indent=2),
            encoding="utf-8",
        )
        update_best_checkpoints(experiment.output_dir, history)
        write_run_summary(experiment.output_dir, experiment, history)

        if lfw_metrics["eer"] < best_eer:
            best_eer = lfw_metrics["eer"]
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        print(
            f"{experiment.name} epoch {epoch}: loss={train_loss:.4f}, "
            f"EER={lfw_metrics['eer']:.6f}, "
            f"acc={lfw_metrics['best_accuracy']:.6f}, "
            f"TAR@FAR=0.01={lfw_metrics['tar_at_far']['0.01']['tar']:.6f}",
            flush=True,
        )
        if no_improvement_epochs >= args.early_stop_patience:
            print(
                f"{experiment.name}: early stop after {no_improvement_epochs} epochs "
                "without EER improvement.",
                flush=True,
            )
            break

    return history


def write_final_comparison(root: Path, histories: dict[str, list[dict]]) -> None:
    baseline_rows = [
        {
            "run": "old custom default",
            "epoch": "-",
            "eer": 0.166949,
            "best_accuracy": 0.834350,
            "tar": 0.496449,
            "threshold": "-",
            "notes": "previous runtime custom result",
        },
        {
            "run": "pretrained baseline",
            "epoch": "-",
            "eer": 0.027852,
            "best_accuracy": 0.984556,
            "tar": 0.971141,
            "threshold": "-",
            "notes": "external ONNX/InsightFace reference",
        },
    ]
    rows = [
        "| Run | Epoch | EER | Best accuracy | TAR@FAR=0.01 | Threshold | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in baseline_rows:
        rows.append(
            f"| {item['run']} | {item['epoch']} | {item['eer']:.6f} | "
            f"{item['best_accuracy']:.6f} | {item['tar']:.6f} | {item['threshold']} | "
            f"{item['notes']} |"
        )
    for run_name, history in histories.items():
        if not history:
            continue
        best = min(history, key=lambda item: item["lfw"]["eer"])
        metrics = best["lfw"]
        rows.append(
            f"| {run_name} | {best['epoch']} | {metrics['eer']:.6f} | "
            f"{metrics['best_accuracy']:.6f} | "
            f"{metrics['tar_at_far']['0.01']['tar']:.6f} | "
            f"{metrics['best_accuracy_threshold']:.6f} | best by EER |"
        )

    (root / "custom_finetune_comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_root = Path(args.train_root)
    val_root = Path(args.val_root)
    lfw_root = Path(args.lfw_root)
    pairs_file = Path(args.pairs_file)
    init_checkpoint = Path(args.init_checkpoint)
    for path in [train_root, val_root, lfw_root, pairs_file, init_checkpoint]:
        if not path.exists():
            raise FileNotFoundError(path)

    run_root = Path("training/runs")
    dataset_summary = {
        "train": summarize_dataset(train_root),
        "val": summarize_dataset(val_root),
    }
    (run_root / "celeba_finetune_dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2),
        encoding="utf-8",
    )

    train_ds = FolderFaceDataset(train_root, transform=train_transform())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    lfw_pairs = parse_lfw_pairs(lfw_root, pairs_file, args.max_pairs)
    if not lfw_pairs:
        raise RuntimeError("No LFW pairs loaded")

    experiments = {
        "A": Experiment(
            name="Run A safe fine-tune",
            output_dir=Path("training/runs/celeba_safe_finetune"),
            freeze_epochs=2,
            epochs=args.epochs,
            freeze_head_lr=0.001,
            unfreeze_backbone_lr=0.00001,
            unfreeze_head_lr=0.0005,
            batch_size=args.batch_size,
        ),
        "B": Experiment(
            name="Run B medium fine-tune",
            output_dir=Path("training/runs/celeba_medium_finetune"),
            freeze_epochs=1,
            epochs=args.epochs,
            freeze_head_lr=0.001,
            unfreeze_backbone_lr=0.00003,
            unfreeze_head_lr=0.001,
            batch_size=args.batch_size,
        ),
    }

    histories: dict[str, list[dict]] = {}
    for run_id in args.runs:
        histories[experiments[run_id].name] = run_experiment(
            experiments[run_id],
            args=args,
            train_loader=train_loader,
            num_classes=len(train_ds.class_to_idx),
            lfw_pairs=lfw_pairs,
            device=device,
        )
    write_final_comparison(run_root, histories)


if __name__ == "__main__":
    main()
