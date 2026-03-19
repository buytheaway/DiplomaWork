from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT = Path("datasets/celeba_faces")


@dataclass(frozen=True)
class CelebARecord:
    filename: str
    identity: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CelebA into datasets/<name>/{train,val,test}/<identity>/*.jpg"
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="CelebA root containing img_align_celeba/ and annotation text files",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT),
        help="Prepared dataset output root",
    )
    parser.add_argument(
        "--identity-file",
        default="",
        help="Optional explicit path to identity_CelebA.txt",
    )
    parser.add_argument(
        "--partition-file",
        default="",
        help="Optional explicit path to list_eval_partition.txt",
    )
    parser.add_argument(
        "--min-images-per-identity",
        type=int,
        default=10,
        help="Drop identities with fewer total images than this threshold",
    )
    parser.add_argument(
        "--max-identities",
        type=int,
        default=0,
        help="Optional cap for identities to keep, 0 keeps all",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="hardlink",
        help="Use hard links when possible to avoid duplicating CelebA files",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output root before preparing the dataset",
    )
    return parser.parse_args()


def resolve_existing_path(base_root: Path, explicit: str, candidates: list[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        return path

    for candidate in candidates:
        path = base_root / candidate
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not locate any of: {', '.join(candidates)} under {base_root}"
    )


def read_identity_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            filename, identity = parts
            mapping[filename] = identity
    if not mapping:
        raise ValueError(f"No identities parsed from {path}")
    return mapping


def read_partition_map(path: Path) -> dict[str, str]:
    split_lookup = {"0": "train", "1": "val", "2": "test"}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            filename, split_code = parts
            mapping[filename] = split_lookup.get(split_code, "train")
    if not mapping:
        raise ValueError(f"No partitions parsed from {path}")
    return mapping


def safe_link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def build_records(
    image_dir: Path,
    identity_map: dict[str, str],
    partition_map: dict[str, str],
    min_images_per_identity: int,
    max_identities: int,
) -> list[CelebARecord]:
    existing_files = sorted(path.name for path in image_dir.glob("*.jpg"))
    total_counts = Counter(identity_map[name] for name in existing_files if name in identity_map)

    allowed_identities = sorted(
        identity
        for identity, count in total_counts.items()
        if count >= min_images_per_identity
    )
    if max_identities > 0:
        allowed_identities = allowed_identities[:max_identities]
    allowed_set = set(allowed_identities)

    records: list[CelebARecord] = []
    for filename in existing_files:
        identity = identity_map.get(filename)
        if identity is None or identity not in allowed_set:
            continue
        split = partition_map.get(filename, "train")
        records.append(CelebARecord(filename=filename, identity=identity, split=split))

    if not records:
        raise ValueError("No CelebA records remained after filtering")
    return records


def prepare_dataset(args: argparse.Namespace) -> dict[str, object]:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()

    image_dir = resolve_existing_path(
        source_root,
        "",
        ["img_align_celeba", "Img/img_align_celeba"],
    )
    identity_file = resolve_existing_path(
        source_root,
        args.identity_file,
        ["identity_CelebA.txt", "Anno/identity_CelebA.txt"],
    )
    partition_file = resolve_existing_path(
        source_root,
        args.partition_file,
        ["list_eval_partition.txt", "Eval/list_eval_partition.txt"],
    )

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    for split in ("train", "val", "test"):
        (output_root / split).mkdir(parents=True, exist_ok=True)

    identity_map = read_identity_map(identity_file)
    partition_map = read_partition_map(partition_file)
    records = build_records(
        image_dir=image_dir,
        identity_map=identity_map,
        partition_map=partition_map,
        min_images_per_identity=args.min_images_per_identity,
        max_identities=args.max_identities,
    )

    stats_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        source = image_dir / record.filename
        if not source.exists():
            continue
        destination_dir = output_root / record.split / record.identity
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / record.filename
        safe_link_or_copy(source, destination, args.mode)
        stats_by_split[record.split][record.identity] += 1

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "mode": args.mode,
        "min_images_per_identity": args.min_images_per_identity,
        "max_identities": args.max_identities,
        "images": {
            split: int(sum(counter.values()))
            for split, counter in sorted(stats_by_split.items())
        },
        "identities": {
            split: len(counter)
            for split, counter in sorted(stats_by_split.items())
        },
    }
    with (output_root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
