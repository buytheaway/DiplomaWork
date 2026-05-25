from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.evaluate_lfw_verification import (  # noqa: E402
    compute_verification_metrics,
    threshold_curve_rows,
)
from training.verification_metrics import far_frr_by_threshold, tar_at_far  # noqa: E402


def _row(pair_id: int, label_same: int, score: float) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "label_same": label_same,
        "name1": "a",
        "name2": "b",
        "path1": "a.jpg",
        "path2": "b.jpg",
        "score": score,
        "status": "ok",
        "error": "",
    }


def test_far_frr_formula_and_threshold_direction():
    scores = [0.95, 0.80, 0.40, 0.20]
    labels = [1, 1, 0, 0]
    curve = far_frr_by_threshold(scores, labels, thresholds=[0.90, 0.50, 0.30])

    np.testing.assert_allclose(curve["far"], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(curve["frr"], [0.5, 0.0, 0.0])

    strict = far_frr_by_threshold(scores, labels, thresholds=[0.30, 0.90])
    assert strict["far"][1] < strict["far"][0]
    assert strict["frr"][1] > strict["frr"][0]


def test_eer_and_best_accuracy_from_lfw_rows():
    rows = [
        _row(1, 1, 0.95),
        _row(2, 1, 0.85),
        _row(3, 0, 0.30),
        _row(4, 0, 0.10),
    ]

    metrics = compute_verification_metrics(rows, pipeline="custom")

    assert math.isclose(metrics["eer"], 0.0)
    assert math.isclose(metrics["best_accuracy"], 1.0)
    assert metrics["positive_pairs"] == 2
    assert metrics["negative_pairs"] == 2
    assert metrics["skipped_pairs"] == 0


def test_tar_at_far_uses_score_greater_equal_threshold_as_match():
    scores = [0.95, 0.85, 0.55, 0.90, 0.40, 0.20]
    labels = [1, 1, 1, 0, 0, 0]

    result = tar_at_far(scores, labels, target_fars=[0.0])[0]

    assert math.isclose(result["far"], 0.0)
    assert math.isclose(result["tar"], 1.0 / 3.0)
    assert math.isclose(result["threshold"], 0.95)


def test_threshold_curve_rows_include_confusion_counts():
    rows = [
        _row(1, 1, 0.90),
        _row(2, 0, 0.70),
        _row(3, 1, 0.60),
        _row(4, 0, 0.20),
    ]

    curve = threshold_curve_rows(rows)

    assert curve
    assert {"threshold", "far", "frr", "tar", "accuracy", "tp", "fp", "tn", "fn"} <= set(
        curve[0]
    )
