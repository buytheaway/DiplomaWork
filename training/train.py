"""Train a face embedding model with ArcFace or AdaFace loss.

Usage::

    cd <project-root>
    python training/train.py --config training/config.yaml
    python training/train.py --config training/config.yaml --loss adaface
    python training/train.py --epochs 28 --batch-size 128 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from training.datasets.folder_dataset import FolderFaceDataset
from training.losses.adaface import AdaFace
from training.losses.arcface import ArcFace
from training.models.ir_resnet import build_model
from training.utils import ensure_dir, load_config, save_checkpoint, save_metrics, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def build_transforms(input_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_loss_head(
    loss_type: str,
    embedding_dim: int,
    num_classes: int,
    margin: float,
    scale: float,
) -> nn.Module:
    if loss_type == "adaface":
        logger.info("Using AdaFace loss (margin=%.2f, scale=%.1f)", margin, scale)
        return AdaFace(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )

    logger.info("Using ArcFace loss (margin=%.2f, scale=%.1f)", margin, scale)
    return ArcFace(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        margin=margin,
        scale=scale,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face embedding model (CUDA only)")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--loss",
        default=None,
        choices=["arcface", "adaface"],
        help="Override loss type from config",
    )
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def make_checkpoint_metadata(config: dict, loss_type: str) -> dict:
    return {
        "loss": loss_type,
        "arch": config["model"]["arch"],
        "embedding_dim": config["model"]["embedding_dim"],
    }


def maybe_run_lfw_validation(
    model: nn.Module,
    config: dict,
    input_size: int,
    device: torch.device,
    output_dir: Path,
    metrics: dict,
) -> None:
    lfw_dir_value = config["data"].get("lfw_dir")
    pairs_file_value = config["data"].get("pairs_file")
    if not lfw_dir_value or not pairs_file_value:
        logger.info("Skipping LFW verification: lfw_dir/pairs_file not configured")
        return

    lfw_dir = Path(lfw_dir_value)
    pairs_file = Path(pairs_file_value)
    if not lfw_dir.exists() or not pairs_file.exists():
        logger.warning(
            "Skipping LFW verification: missing paths lfw_dir=%s pairs=%s",
            lfw_dir,
            pairs_file,
        )
        return

    from training.eval_lfw import evaluate_lfw

    logger.info("Running LFW verification benchmark...")
    model.eval()
    results = evaluate_lfw(
        model=model,
        lfw_dir=lfw_dir,
        pairs_path=pairs_file,
        transform=build_transforms(input_size, False),
        device=device,
    )

    metrics["lfw"] = {
        "accuracy": results["accuracy"],
        "accuracy_std": results["accuracy_std"],
        "best_threshold": results["best_threshold"],
        "auc": results["auc"],
        "num_pairs": results["num_pairs"],
        "skipped_pairs": results["skipped_pairs"],
        "tar_at_far": results["tar_at_far"],
    }

    lfw_output_path = output_dir / "lfw_results.json"
    serializable = {k: v for k, v in results.items() if k != "roc_data"}
    serializable["roc_data_points"] = len(results.get("roc_data", {}).get("fpr", []))
    with lfw_output_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    logger.info(
        "LFW done: accuracy=%.4f +/- %.4f, threshold=%.4f, auc=%.4f",
        results["accuracy"],
        results["accuracy_std"],
        results["best_threshold"],
        results["auc"],
    )
    logger.info("LFW results saved to: %s", lfw_output_path)


def build_val_loader(
    train_ds: FolderFaceDataset,
    val_dir: Path,
    input_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[FolderFaceDataset, DataLoader | None]:
    val_ds = FolderFaceDataset(
        val_dir,
        transform=build_transforms(input_size, False),
        class_to_idx=train_ds.class_to_idx,
        ignore_unknown=True,
        allow_empty=True,
    )
    if len(val_ds) == 0:
        return val_ds, None

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return val_ds, val_loader


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size

    device_name = args.device or config["device"]
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Training supports CUDA only. Start it with --device cuda on a machine with a working NVIDIA GPU."
        )

    input_size = int(config["data"]["input_size"])
    train_dir = Path(args.train_dir or config["data"]["train_dir"])
    val_dir = Path(args.val_dir or config["data"]["val_dir"])
    loss_type = args.loss or config.get("loss", {}).get("type", "arcface")
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])

    logger.info("Configuration:")
    logger.info("  Device:      %s", device)
    logger.info("  Input size:  %d", input_size)
    logger.info("  Train dir:   %s", train_dir)
    logger.info("  Val dir:     %s", val_dir)
    logger.info("  Loss:        %s", loss_type)
    logger.info("  Epochs:      %d", config["train"]["epochs"])
    logger.info("  Batch size:  %d", batch_size)

    train_ds = FolderFaceDataset(train_dir, transform=build_transforms(input_size, True))
    val_ds, val_loader = build_val_loader(
        train_ds=train_ds,
        val_dir=val_dir,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    num_classes = len(train_ds.class_to_idx)
    logger.info("  Classes:     %d", num_classes)
    logger.info("  Train imgs:  %d", len(train_ds))
    logger.info("  Val imgs:    %d", len(val_ds))
    if val_loader is None:
        logger.warning(
            "  Val split:   skipped closed-set validation because val_dir shares no identities with train_dir"
        )
    else:
        logger.info("  Val classes: %d", len(list(val_ds.labels())))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
    head = build_loss_head(
        loss_type=loss_type,
        embedding_dim=config["model"]["embedding_dim"],
        num_classes=num_classes,
        margin=config["loss"]["margin"],
        scale=config["loss"]["scale"],
    ).to(device)

    optimizer = SGD(
        list(model.parameters()) + list(head.parameters()),
        lr=config["optimizer"]["lr"],
        momentum=config["optimizer"]["momentum"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    milestones = config["scheduler"].get("milestones", [8, 14, 20, 25])
    gamma = config["scheduler"].get("gamma", 0.1)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    max_grad_norm = config.get("train", {}).get("max_grad_norm", 5.0)
    amp_enabled = bool(config.get("amp", False))
    scaler = GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss()

    logger.info("  Scheduler:   MultiStepLR(milestones=%s, gamma=%.2f)", milestones, gamma)
    logger.info("  Grad clip:   %.1f", max_grad_norm)
    logger.info("  AMP:         %s", amp_enabled)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

        head_state = ckpt.get("head_state_dict")
        if head_state is not None:
            head.load_state_dict(head_state, strict=False)
        else:
            logger.warning(
                "Checkpoint %s has no head_state_dict; classifier head will restart from scratch",
                args.resume,
            )

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if amp_enabled and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

        logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch - 1)

    output_dir = ensure_dir(config["train"]["output_dir"])
    metrics = {
        "config": make_checkpoint_metadata(config, loss_type),
        "epochs": [],
    }

    logger.info("\nStarting training...\n")
    total_start = time.time()

    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        model.train()
        head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['train']['epochs']}")
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled):
                embeddings = model(images)
                logits = head(embeddings, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()),
                max_norm=max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct / total if total else 0.0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total if total else 0.0

        val_acc = None
        if val_loader is not None:
            model.eval()
            head.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    embeddings = model(images)
                    logits = head.inference_logits(embeddings)
                    _, preds = torch.max(logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total if val_total else None

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        val_acc_text = f"{val_acc:.4f}" if val_acc is not None else "n/a"

        logger.info(
            "Epoch %d/%d - loss=%.4f, train_acc=%.4f, val_acc=%s, lr=%.6f, time=%.1fs",
            epoch,
            config["train"]["epochs"],
            train_loss,
            train_acc,
            val_acc_text,
            current_lr,
            epoch_time,
        )

        metrics["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": current_lr,
                "epoch_time_s": round(epoch_time, 1),
            }
        )

        if epoch % config["train"]["save_every"] == 0:
            save_checkpoint(
                output_dir,
                epoch,
                model.state_dict(),
                optimizer.state_dict(),
                head_state=head.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict() if amp_enabled else None,
                metadata=make_checkpoint_metadata(config, loss_type),
            )

    total_time = time.time() - total_start
    metrics["total_time_min"] = round(total_time / 60, 1)

    final_ckpt = save_checkpoint(
        output_dir,
        config["train"]["epochs"],
        model.state_dict(),
        optimizer.state_dict(),
        head_state=head.state_dict(),
        scheduler_state=scheduler.state_dict(),
        scaler_state=scaler.state_dict() if amp_enabled else None,
        metadata=make_checkpoint_metadata(config, loss_type),
    )
    maybe_run_lfw_validation(model, config, input_size, device, output_dir, metrics)
    save_metrics(output_dir, metrics)

    logger.info("\nTraining complete! Total time: %.1f min", total_time / 60)
    logger.info("Final checkpoint: %s", final_ckpt)
    logger.info("Metrics saved to: %s/metrics.json", output_dir)
    logger.info("\nNext steps:")
    logger.info("  1. Export to ONNX:")
    logger.info("     python scripts/export_onnx.py --weights %s --validate", final_ckpt)


if __name__ == "__main__":
    main()
