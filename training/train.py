from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from training.datasets.folder_dataset import FolderFaceDataset
from training.losses.arcface import ArcFace
from training.models.ir_resnet import build_model
from training.utils import ensure_dir, load_config, save_checkpoint, save_metrics, set_seed


def build_transforms(input_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face embedding model (ArcFace)")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["seed"])
    set_seed(seed)

    device = args.device or config["device"]
    device = torch.device(device)

    input_size = int(config["data"]["input_size"])
    train_dir = Path(args.train_dir or config["data"]["train_dir"])
    val_dir = Path(args.val_dir or config["data"]["val_dir"])

    train_ds = FolderFaceDataset(train_dir, transform=build_transforms(input_size, True))
    val_ds = FolderFaceDataset(val_dir, transform=build_transforms(input_size, False))

    num_classes = len(list(train_ds.labels()))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size or config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size or config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    model = build_model(config["model"]["arch"], config["model"]["embedding_dim"]).to(device)
    head = ArcFace(
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
    scheduler = StepLR(
        optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"]
    )

    scaler = GradScaler(enabled=bool(config.get("amp", False)))
    criterion = nn.CrossEntropyLoss()

    output_dir = ensure_dir(config["train"]["output_dir"])
    metrics = {"epochs": []}

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        head.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['train']['epochs']}")
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(config.get("amp", False))):
                embeddings = model(images)
                logits = head(embeddings, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total if total else 0.0

        model.eval()
        head.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                embeddings = model(images)
                logits = head(embeddings, labels)
                _, preds = torch.max(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total else 0.0

        metrics["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        if epoch % config["train"]["save_every"] == 0:
            save_checkpoint(output_dir, epoch, model.state_dict(), optimizer.state_dict())

    save_metrics(output_dir, metrics)


if __name__ == "__main__":
    main()
