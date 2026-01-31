from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Sample:
    path: Path
    label: int
    label_name: str


class FolderFaceDataset(Dataset[Sample]):
    def __init__(self, root: str | Path, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[Sample] = []

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        class_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        class_to_idx = {p.name: idx for idx, p in enumerate(class_dirs)}

        for class_dir in class_dirs:
            for img_path in sorted(class_dir.rglob("*")):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                label = class_to_idx[class_dir.name]
                self.samples.append(
                    Sample(path=img_path, label=label, label_name=class_dir.name)
                )

        if not self.samples:
            raise ValueError(f"No images found under: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample.label

    def labels(self) -> Iterable[str]:
        seen = {}
        for sample in self.samples:
            seen.setdefault(sample.label, sample.label_name)
        return [seen[idx] for idx in sorted(seen.keys())]
