from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.datasets.folder_dataset import IMAGE_EXTS


@dataclass(frozen=True)
class DistillSample:
    path: Path
    label: int
    label_name: str
    teacher_embedding_path: Path


class DistillFaceDataset(Dataset):
    """Identity-folder face dataset paired with cached teacher embeddings."""

    def __init__(
        self,
        root: str | Path,
        teacher_cache: str | Path,
        transform=None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.teacher_cache = Path(teacher_cache)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        metadata_path = self.teacher_cache / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Teacher cache metadata not found: {metadata_path}")

        class_dirs = sorted([path for path in self.root.iterdir() if path.is_dir()])
        self.class_to_idx = (
            {path.name: index for index, path in enumerate(class_dirs)}
            if class_to_idx is None
            else dict(class_to_idx)
        )

        self.samples: list[DistillSample] = []
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = self.root / image_path
                if image_path.suffix.lower() not in IMAGE_EXTS or not image_path.exists():
                    continue

                label_name = row.get("identity") or image_path.parent.name
                if label_name not in self.class_to_idx:
                    continue

                embedding_path = Path(row["embedding_path"])
                if not embedding_path.is_absolute():
                    embedding_path = self.teacher_cache / embedding_path
                if not embedding_path.exists():
                    continue

                self.samples.append(
                    DistillSample(
                        path=image_path,
                        label=self.class_to_idx[label_name],
                        label_name=label_name,
                        teacher_embedding_path=embedding_path,
                    )
                )

        if not self.samples:
            raise ValueError(f"No cached teacher embeddings found for {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        teacher = np.load(sample.teacher_embedding_path).astype(np.float32)
        teacher_tensor = torch.from_numpy(teacher)
        return image, sample.label, teacher_tensor, str(sample.path)

    def labels(self) -> list[str]:
        seen = {}
        for sample in self.samples:
            seen.setdefault(sample.label, sample.label_name)
        return [seen[index] for index in sorted(seen.keys())]
