from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from scripts.evaluate_verification_pairs import (
    VerificationPair,
    compute_metric_summary,
    evaluate_pair_scores,
    lfw_image_path,
    parse_lfw_pairs,
)


class FakeExtractor:
    model_name = "fake"
    dim = 2

    def __init__(self, vectors: dict[bytes, np.ndarray]) -> None:
        self._vectors = vectors

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        return self._vectors[image_bytes]


def test_parse_lfw_pairs_supports_same_and_different_pairs(tmp_path):
    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text(
        "\n".join(
            [
                "10 300",
                "Alice 1 2",
                "Alice 1 Bob 1",
                "Carol 3 4",
            ]
        ),
        encoding="utf-8",
    )

    pairs = parse_lfw_pairs(pairs_file)

    assert pairs == [
        VerificationPair("Alice", 1, "Alice", 2, 1, 2),
        VerificationPair("Alice", 1, "Bob", 1, 0, 3),
        VerificationPair("Carol", 3, "Carol", 4, 1, 4),
    ]


def test_parse_lfw_pairs_respects_max_pairs(tmp_path):
    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("10\nAlice 1 2\nAlice 1 Bob 1\n", encoding="utf-8")

    pairs = parse_lfw_pairs(pairs_file, max_pairs=1)

    assert len(pairs) == 1
    assert pairs[0].label == 1


def test_lfw_image_path_uses_standard_filename():
    path = lfw_image_path(Path("lfw"), "George_W_Bush", 12)

    assert path == Path("lfw") / "George_W_Bush" / "George_W_Bush_0012.jpg"


def test_metric_integration_uses_mocked_embeddings(tmp_path):
    images_dir = tmp_path / "lfw"
    (images_dir / "Alice").mkdir(parents=True)
    (images_dir / "Bob").mkdir(parents=True)
    (images_dir / "Alice" / "Alice_0001.jpg").write_bytes(b"alice-1")
    (images_dir / "Alice" / "Alice_0002.jpg").write_bytes(b"alice-2")
    (images_dir / "Bob" / "Bob_0001.jpg").write_bytes(b"bob-1")

    extractor = FakeExtractor(
        {
            b"alice-1": np.array([2.0, 0.0], dtype=np.float32),
            b"alice-2": np.array([4.0, 0.0], dtype=np.float32),
            b"bob-1": np.array([0.0, 3.0], dtype=np.float32),
        }
    )
    pairs = [
        VerificationPair("Alice", 1, "Alice", 2, 1, 1),
        VerificationPair("Alice", 1, "Bob", 1, 0, 2),
    ]

    score_data = evaluate_pair_scores(extractor, images_dir, pairs, fail_on_missing=True)
    metrics = compute_metric_summary(
        score_data["scores"],
        score_data["labels"],
        target_fars=[0.0, 0.1],
        thresholds=None,
    )

    assert score_data["pairs_evaluated"] == 2
    assert score_data["failed_pairs"] == 0
    assert score_data["scores"] == [1.0, 0.0]
    assert metrics["eer"] == 0.0
    assert metrics["best_accuracy_threshold"]["accuracy"] == 1.0
    assert metrics["tar_at_far"][0]["target_far"] == 0.0
