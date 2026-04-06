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
    def __init__(
        self,
        root: str | Path,
        transform=None,
        class_to_idx: dict[str, int] | None = None,
        ignore_unknown: bool = False,
        allow_empty: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[Sample] = []

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        class_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if class_to_idx is None:
            self.class_to_idx = {p.name: idx for idx, p in enumerate(class_dirs)}
        else:
            self.class_to_idx = dict(class_to_idx)

        for class_dir in class_dirs:
            if class_dir.name not in self.class_to_idx:
                if ignore_unknown:
                    continue
                raise ValueError(
                    f"Unknown class '{class_dir.name}' under {self.root}; "
                    "pass ignore_unknown=True to skip it"
                )
            for img_path in sorted(class_dir.rglob("*")):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                label = self.class_to_idx[class_dir.name]
                self.samples.append(
                    Sample(path=img_path, label=label, label_name=class_dir.name)
                )

        if not self.samples and not allow_empty:
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
