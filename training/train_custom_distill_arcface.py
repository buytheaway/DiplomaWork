"""Train custom IR-50 with ArcFace plus teacher embedding distillation.

This training script is standalone: it does not change backend runtime,
desktop UI, enroll/search behavior, runtime checkpoints, or FAISS indexes.
LFW is used only for post-epoch evaluation and checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.datasets.distill_face_dataset import DistillFaceDataset
from training.datasets.folder_dataset import FolderFaceDataset
from training.losses.arcface import ArcFace
from training.losses.distillation import CosineEmbeddingDistillationLoss
from training.models.ir_resnet import build_model
from training.utils import set_seed
from training.verification_metrics import best_accuracy_threshold, equal_error_rate, tar_at_far

OLD_CUSTOM = {
    "eer": 0.166949,
    "best_accuracy": 0.834350,
    "tar_far_0_01": 0.496449,
}
PRETRAINED_BASELINE = {
    "eer": 0.027852,
    "best_accuracy": 0.984556,
    "tar_far_0_01": 0.971141,
}
TARGET_FARS = [0.1, 0.01, 0.001]


@dataclass(frozen=True)
class DistillConfig:
    name: str
    output_dir: Path
    lambda_distill: float
    lr_backbone: float
    lr_head: float
    freeze_backbone_epochs: int
    epochs: int
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom IR-50 with teacher distillation")
    parser.add_argument("--train-root", default="datasets/celeba_faces/train")
    parser.add_argument("--val-root", default="datasets/celeba_faces/val")
    parser.add_argument("--teacher-cache", default="reports/teacher_cache/celeba_train_teacher_embeddings")
    parser.add_argument("--lfw-root", default="handoff_lfw_eval/lfw")
    parser.add_argument("--pairs-file", default="handoff_lfw_eval/lfw/pairs.txt")
    parser.add_argument("--init-checkpoint", default="handoff_lfw_eval/artifacts/best_lfw.pth")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume model/head/optimizer/scaler from a previous training checkpoint.",
    )
    parser.add_argument("--output-dir", default="training/runs/custom_distill_arcface")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--configs", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--max-pairs", type=int, default=6000)
    parser.add_argument("--tta", nargs="+", choices=["none", "hflip"], default=["none", "hflip"])
    parser.add_argument("--selection-tta", choices=["none", "hflip"], default="none")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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


def load_history(output_dir: Path) -> list[dict]:
    history: list[dict] = []
    for path in sorted(output_dir.glob("metrics_epoch_*.json")):
        try:
            history.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return history


def best_tracking_from_history(history: list[dict]) -> tuple[float, float]:
    primary = [item for item in history if item.get("selection_metrics")]
    if not primary:
        return float("inf"), -float("inf")
    best_eer = min(float(item["selection_metrics"]["eer"]) for item in primary)
    best_tar = max(
        float(item["selection_metrics"]["tar_at_far"]["0.01"]["tar"])
        for item in primary
    )
    return best_eer, best_tar


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = not frozen


def build_optimizer(
    model: nn.Module,
    head: nn.Module,
    *,
    freeze_backbone: bool,
    lr_backbone: float,
    lr_head: float,
) -> SGD:
    set_backbone_frozen(model, freeze_backbone)
    params: list[dict] = [{"params": head.parameters(), "lr": lr_head}]
    if not freeze_backbone:
        params.insert(0, {"params": model.parameters(), "lr": lr_backbone})
    return SGD(params, momentum=0.9, weight_decay=0.0005)


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
                index1 = int(parts[1])
                index2 = int(parts[2])
                same = 1
            elif len(parts) == 4:
                name1 = parts[0]
                index1 = int(parts[1])
                name2 = parts[2]
                index2 = int(parts[3])
                same = 0
            else:
                continue
            path1 = lfw_root / name1 / f"{name1}_{index1:04d}.jpg"
            path2 = lfw_root / name2 / f"{name2}_{index2:04d}.jpg"
            pairs.append((same, path1, path2))
    if max_pairs is not None and len(pairs) > max_pairs:
        positives = [pair for pair in pairs if pair[0] == 1]
        negatives = [pair for pair in pairs if pair[0] == 0]
        if positives and negatives:
            positive_count = max_pairs // 2
            negative_count = max_pairs - positive_count
            return positives[:positive_count] + negatives[:negative_count]
        return pairs[:max_pairs]
    return pairs


def lfw_embedding(
    model: nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
    cache: dict[tuple[Path, str], np.ndarray],
    tta: str,
) -> np.ndarray:
    cache_key = (image_path, tta)
    if cache_key in cache:
        return cache[cache_key]
    image = Image.open(image_path).convert("RGB")
    images = [image]
    if tta == "hflip":
        images.append(ImageOps.mirror(image))
    elif tta != "none":
        raise ValueError(f"Unsupported TTA: {tta}")

    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for item in images:
            tensor = transform(item).unsqueeze(0).to(device, non_blocking=True)
            cudnn_context = (
                torch.backends.cudnn.flags(enabled=False)
                if device.type == "cuda"
                else nullcontext()
            )
            with cudnn_context:
                embedding = model(tensor)
            embedding = torch.nn.functional.normalize(embedding.float(), p=2, dim=1)
            vectors.append(embedding.squeeze(0).cpu().numpy().astype(np.float32))
    vector = np.sum(vectors, axis=0)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    cache[cache_key] = vector.astype(np.float32)
    return cache[cache_key]


def evaluate_lfw(
    model: nn.Module,
    pairs: list[tuple[int, Path, Path]],
    device: torch.device,
    tta: str,
) -> dict:
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    transform = eval_transform()
    cache: dict[tuple[Path, str], np.ndarray] = {}
    scores: list[float] = []
    labels: list[int] = []
    skipped = 0
    for same, path1, path2 in tqdm(pairs, desc=f"LFW {tta}", leave=False):
        if not path1.exists() or not path2.exists():
            skipped += 1
            continue
        embedding1 = lfw_embedding(model, path1, transform, device, cache, tta)
        embedding2 = lfw_embedding(model, path2, transform, device, cache, tta)
        scores.append(float(np.dot(embedding1, embedding2)))
        labels.append(same)

    if 1 not in labels or 0 not in labels:
        raise RuntimeError("LFW evaluation needs positive and negative pairs")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    eer = equal_error_rate(scores, labels)
    best = best_accuracy_threshold(scores, labels)
    tar_values = tar_at_far(scores, labels, TARGET_FARS)
    positive_scores = [score for score, label in zip(scores, labels, strict=False) if label == 1]
    negative_scores = [score for score, label in zip(scores, labels, strict=False) if label == 0]

    return {
        "tta": tta,
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


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    head: nn.Module,
    optimizer: SGD,
    scaler: GradScaler,
    config: DistillConfig,
    metrics: dict | None,
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
                "loss": "arcface_plus_teacher_distillation",
                "config": asdict(config),
                "lfw_metrics": metrics,
            },
        },
        path,
    )


def write_summary(output_dir: Path, config: DistillConfig, history: list[dict]) -> None:
    rows = [
        "| Config | Epoch | EER | Accuracy | TAR@FAR=0.01 | TAR@FAR=0.001 | Threshold | TTA | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in history:
        for metrics in item["evaluations"]:
            rows.append(
                f"| {config.name} | {item['epoch']} | {metrics['eer']:.6f} | "
                f"{metrics['best_accuracy']:.6f} | "
                f"{metrics['tar_at_far']['0.01']['tar']:.6f} | "
                f"{metrics['tar_at_far']['0.001']['tar']:.6f} | "
                f"{metrics['best_accuracy_threshold']:.6f} | {metrics['tta']} | "
                f"{item['phase']} |"
            )

    primary = [item for item in history if item.get("selection_metrics")]
    best_eer = min(primary, key=lambda item: item["selection_metrics"]["eer"]) if primary else None
    best_tar = (
        max(primary, key=lambda item: item["selection_metrics"]["tar_at_far"]["0.01"]["tar"])
        if primary
        else None
    )
    best_acc = (
        max(primary, key=lambda item: item["selection_metrics"]["best_accuracy"]) if primary else None
    )

    lines = [
        f"# {config.name} Distillation Summary",
        "",
        f"- Output dir: `{output_dir}`",
        f"- lambda_distill: `{config.lambda_distill}`",
        f"- lr_backbone: `{config.lr_backbone}`",
        f"- lr_head: `{config.lr_head}`",
        f"- freeze_backbone_epochs: `{config.freeze_backbone_epochs}`",
        f"- batch_size: `{config.batch_size}`",
        "",
        *rows,
        "",
    ]
    if best_eer and best_tar and best_acc:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def update_best_checkpoints(output_dir: Path, history: list[dict]) -> None:
    primary = [item for item in history if item.get("selection_metrics")]
    if not primary:
        return
    best_eer = min(primary, key=lambda item: item["selection_metrics"]["eer"])
    best_tar = max(primary, key=lambda item: item["selection_metrics"]["tar_at_far"]["0.01"]["tar"])
    best_acc = max(primary, key=lambda item: item["selection_metrics"]["best_accuracy"])
    shutil.copy2(output_dir / f"checkpoint_epoch_{best_eer['epoch']:03d}.pth", output_dir / "best_by_eer.pth")
    shutil.copy2(
        output_dir / f"checkpoint_epoch_{best_tar['epoch']:03d}.pth",
        output_dir / "best_by_tar_far001.pth",
    )
    shutil.copy2(
        output_dir / f"checkpoint_epoch_{best_acc['epoch']:03d}.pth",
        output_dir / "best_by_accuracy.pth",
    )


def find_selection_metrics(evaluations: list[dict], selection_tta: str) -> dict:
    for item in evaluations:
        if item["tta"] == selection_tta:
            return item
    raise ValueError(f"Selection TTA {selection_tta!r} was not evaluated")


def run_config(
    config: DistillConfig,
    *,
    args: argparse.Namespace,
    train_loader: DataLoader,
    num_classes: int,
    lfw_pairs: list[tuple[int, Path, Path]],
    device: torch.device,
) -> list[dict]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model("ir50", 512).to(device)
    head = ArcFace(embedding_dim=512, num_classes=num_classes, margin=0.5, scale=64.0).to(device)
    load_backbone(model, Path(args.init_checkpoint), device)

    ce_loss = nn.CrossEntropyLoss()
    distill_loss = CosineEmbeddingDistillationLoss()
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    optimizer: SGD | None = None
    history: list[dict] = []
    best_eer = float("inf")
    best_tar = -float("inf")
    no_eer_improvement_epochs = 0
    no_tar_improvement_epochs = 0
    start_epoch = 1

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        if "head_state_dict" not in checkpoint:
            raise RuntimeError(f"Checkpoint has no ArcFace head state: {resume_path}")
        head.load_state_dict(checkpoint["head_state_dict"], strict=False)
        previous_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = previous_epoch + 1
        freeze_backbone = start_epoch <= config.freeze_backbone_epochs
        optimizer = build_optimizer(
            model,
            head,
            freeze_backbone=freeze_backbone,
            lr_backbone=config.lr_backbone,
            lr_head=config.lr_head,
        )
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        history = load_history(config.output_dir)
        best_eer, best_tar = best_tracking_from_history(history)
        print(
            f"{config.name}: resumed from {resume_path} at epoch {start_epoch}/{config.epochs}",
            flush=True,
        )

    if start_epoch > config.epochs:
        print(
            f"{config.name}: resume epoch {start_epoch} is beyond --epochs {config.epochs}; nothing to do.",
            flush=True,
        )
        return history

    for epoch in range(start_epoch, config.epochs + 1):
        freeze_backbone = epoch <= config.freeze_backbone_epochs
        phase = "frozen_backbone" if freeze_backbone else "unfrozen_backbone"
        if epoch == 1 or epoch == config.freeze_backbone_epochs + 1:
            optimizer = build_optimizer(
                model,
                head,
                freeze_backbone=freeze_backbone,
                lr_backbone=config.lr_backbone,
                lr_head=config.lr_head,
            )
        assert optimizer is not None

        model.train()
        head.train()
        epoch_start = time.time()
        total_items = 0
        loss_sum = 0.0
        arc_sum = 0.0
        distill_sum = 0.0

        loop = tqdm(train_loader, desc=f"{config.name} epoch {epoch}/{config.epochs}")
        for images, labels, teacher_embeddings, _paths in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            teacher_embeddings = teacher_embeddings.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=device.type == "cuda"):
                student_embeddings = model(images)
                logits = head(student_embeddings, labels)
                arc_loss = ce_loss(logits, labels)
                kd_loss = distill_loss(student_embeddings, teacher_embeddings)
                loss = arc_loss + (config.lambda_distill * kd_loss)

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

            batch_size = labels.size(0)
            total_items += batch_size
            loss_sum += loss.item() * batch_size
            arc_sum += arc_loss.item() * batch_size
            distill_sum += kd_loss.item() * batch_size
            loop.set_postfix(loss=loss.item(), arc=arc_loss.item(), distill=kd_loss.item())

        train_loss = loss_sum / max(total_items, 1)
        train_arc_loss = arc_sum / max(total_items, 1)
        train_distill_loss = distill_sum / max(total_items, 1)
        checkpoint_path = config.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            head=head,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            metrics=None,
        )

        evaluations = [evaluate_lfw(model, lfw_pairs, device, tta) for tta in args.tta]
        selection_metrics = find_selection_metrics(evaluations, args.selection_tta)
        save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            head=head,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            metrics={"evaluations": evaluations, "selection_tta": args.selection_tta},
        )

        result = {
            "config": config.name,
            "epoch": epoch,
            "phase": phase,
            "train_loss": round(train_loss, 6),
            "train_arcface_loss": round(train_arc_loss, 6),
            "train_distillation_loss": round(train_distill_loss, 6),
            "epoch_time_s": round(time.time() - epoch_start, 1),
            "checkpoint": str(checkpoint_path),
            "selection_tta": args.selection_tta,
            "selection_metrics": selection_metrics,
            "evaluations": evaluations,
        }
        history.append(result)
        (config.output_dir / f"metrics_epoch_{epoch:03d}.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        update_best_checkpoints(config.output_dir, history)
        write_summary(config.output_dir, config, history)

        if selection_metrics["eer"] < best_eer:
            best_eer = selection_metrics["eer"]
            no_eer_improvement_epochs = 0
        else:
            no_eer_improvement_epochs += 1
        current_tar = selection_metrics["tar_at_far"]["0.01"]["tar"]
        if current_tar > best_tar:
            best_tar = current_tar
            no_tar_improvement_epochs = 0
        else:
            no_tar_improvement_epochs += 1

        print(
            f"{config.name} epoch {epoch}: loss={train_loss:.4f}, arc={train_arc_loss:.4f}, "
            f"distill={train_distill_loss:.4f}, EER={selection_metrics['eer']:.6f}, "
            f"acc={selection_metrics['best_accuracy']:.6f}, "
            f"TAR@FAR=0.01={current_tar:.6f}",
            flush=True,
        )

        if epoch >= 3 and selection_metrics["eer"] > OLD_CUSTOM["eer"] + 0.02:
            print(f"{config.name}: stop, EER is worse than old custom by more than 0.02.", flush=True)
            break
        if no_eer_improvement_epochs >= args.early_stop_patience:
            print(f"{config.name}: early stop after {no_eer_improvement_epochs} EER stalls.", flush=True)
            break
        if no_tar_improvement_epochs >= args.early_stop_patience:
            print(
                f"{config.name}: early stop after {no_tar_improvement_epochs} TAR@FAR=0.01 stalls.",
                flush=True,
            )
            break

    return history


def summarize_dataset(root: Path) -> dict:
    dataset = FolderFaceDataset(root, allow_empty=False)
    counts: dict[int, int] = {}
    for sample in dataset.samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    values = list(counts.values())
    return {
        "root": str(root),
        "identities": len(values),
        "images": len(dataset.samples),
        "min_images_per_identity": min(values) if values else 0,
        "avg_images_per_identity": round(sum(values) / len(values), 2) if values else 0.0,
        "max_images_per_identity": max(values) if values else 0,
        "sample_paths": [str(sample.path) for sample in dataset.samples[:5]],
    }


def write_final_comparison(output_root: Path, histories: dict[str, list[dict]]) -> None:
    rows = [
        "| Config | Epoch | EER | Accuracy | TAR@FAR=0.01 | TAR@FAR=0.001 | Threshold | TTA | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
        (
            f"| old custom default | - | {OLD_CUSTOM['eer']:.6f} | "
            f"{OLD_CUSTOM['best_accuracy']:.6f} | {OLD_CUSTOM['tar_far_0_01']:.6f} | - | - | none | "
            "previous custom runtime result |"
        ),
        (
            f"| pretrained baseline | - | {PRETRAINED_BASELINE['eer']:.6f} | "
            f"{PRETRAINED_BASELINE['best_accuracy']:.6f} | "
            f"{PRETRAINED_BASELINE['tar_far_0_01']:.6f} | - | - | none | external baseline |"
        ),
    ]
    for config_name, history in histories.items():
        for tta in ["none", "hflip"]:
            candidates = [
                (item, metrics)
                for item in history
                for metrics in item["evaluations"]
                if metrics["tta"] == tta
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda pair: pair[1]["eer"])
            item, metrics = best
            rows.append(
                f"| {config_name} | {item['epoch']} | {metrics['eer']:.6f} | "
                f"{metrics['best_accuracy']:.6f} | "
                f"{metrics['tar_at_far']['0.01']['tar']:.6f} | "
                f"{metrics['tar_at_far']['0.001']['tar']:.6f} | "
                f"{metrics['best_accuracy_threshold']:.6f} | {tta} | best by EER |"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "distill_comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_configs(args: argparse.Namespace) -> dict[str, DistillConfig]:
    output_root = Path(args.output_dir)
    return {
        "A": DistillConfig(
            name="A_lambda0.25_lr1e-5",
            output_dir=output_root / "A_lambda0.25_lr1e-5",
            lambda_distill=0.25,
            lr_backbone=1e-5,
            lr_head=5e-4,
            freeze_backbone_epochs=1,
            epochs=args.epochs,
            batch_size=args.batch_size,
        ),
        "B": DistillConfig(
            name="B_lambda0.5_lr1e-5",
            output_dir=output_root / "B_lambda0.5_lr1e-5",
            lambda_distill=0.5,
            lr_backbone=1e-5,
            lr_head=5e-4,
            freeze_backbone_epochs=1,
            epochs=args.epochs,
            batch_size=args.batch_size,
        ),
        "C": DistillConfig(
            name="C_lambda1.0_lr5e-6",
            output_dir=output_root / "C_lambda1.0_lr5e-6",
            lambda_distill=1.0,
            lr_backbone=5e-6,
            lr_head=5e-4,
            freeze_backbone_epochs=1,
            epochs=args.epochs,
            batch_size=args.batch_size,
        ),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_root = Path(args.train_root)
    val_root = Path(args.val_root)
    teacher_cache = Path(args.teacher_cache)
    lfw_root = Path(args.lfw_root)
    pairs_file = Path(args.pairs_file)
    init_checkpoint = Path(args.init_checkpoint)
    for path in [train_root, val_root, teacher_cache, lfw_root, pairs_file, init_checkpoint]:
        if not path.exists():
            raise FileNotFoundError(path)

    train_dataset = DistillFaceDataset(train_root, teacher_cache, transform=train_transform())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    lfw_pairs = parse_lfw_pairs(lfw_root, pairs_file, args.max_pairs)
    if not lfw_pairs:
        raise RuntimeError("No LFW pairs loaded")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_summary = {
        "train": summarize_dataset(train_root),
        "val": summarize_dataset(val_root),
        "teacher_cache": str(teacher_cache),
        "teacher_cached_samples_used": len(train_dataset),
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2),
        encoding="utf-8",
    )

    configs = build_configs(args)
    histories: dict[str, list[dict]] = {}
    for config_id in args.configs:
        config = configs[config_id]
        histories[config.name] = run_config(
            config,
            args=args,
            train_loader=train_loader,
            num_classes=len(train_dataset.class_to_idx),
            lfw_pairs=lfw_pairs,
            device=device,
        )
    write_final_comparison(output_root, histories)


if __name__ == "__main__":
    main()
