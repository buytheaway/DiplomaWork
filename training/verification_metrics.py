"""Pure NumPy helpers for biometric verification threshold calibration.

Conventions:
* Higher score means more similar.
* ``score >= threshold`` means predicted match.
* Labels use ``1`` for positive/same-identity pairs and ``0`` for negative pairs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _as_arrays(scores: Iterable[float], labels: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    scores_arr = np.asarray(list(scores), dtype=np.float64)
    labels_arr = np.asarray(list(labels), dtype=np.int32)
    if scores_arr.ndim != 1 or labels_arr.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if len(scores_arr) != len(labels_arr):
        raise ValueError("scores and labels must have the same length")
    if len(scores_arr) == 0:
        raise ValueError("scores and labels must not be empty")
    if not np.isfinite(scores_arr).all():
        raise ValueError("scores must be finite")
    if not np.isin(labels_arr, [0, 1]).all():
        raise ValueError("labels must contain only 0 or 1")
    return scores_arr, labels_arr


def default_thresholds(scores: Iterable[float]) -> np.ndarray:
    """Return deterministic threshold candidates covering all decision regions."""
    scores_arr = np.asarray(list(scores), dtype=np.float64)
    if scores_arr.ndim != 1 or len(scores_arr) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(scores_arr).all():
        raise ValueError("scores must be finite")

    unique_scores = np.unique(scores_arr)
    below_min = np.nextafter(unique_scores[0], -np.inf)
    above_max = np.nextafter(unique_scores[-1], np.inf)
    return np.concatenate(([below_min], unique_scores, [above_max]))


def _threshold_array(thresholds: Iterable[float] | None, scores: np.ndarray) -> np.ndarray:
    if thresholds is None:
        return default_thresholds(scores)
    thresholds_arr = np.asarray(list(thresholds), dtype=np.float64)
    if thresholds_arr.ndim != 1 or len(thresholds_arr) == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array")
    if not np.isfinite(thresholds_arr).all():
        raise ValueError("thresholds must be finite")
    return thresholds_arr


def confusion_counts_by_threshold(
    scores: Iterable[float],
    labels: Iterable[int],
    thresholds: Iterable[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute TP/FP/TN/FN arrays for each threshold."""
    scores_arr, labels_arr = _as_arrays(scores, labels)
    thresholds_arr = _threshold_array(thresholds, scores_arr)

    predictions = scores_arr[None, :] >= thresholds_arr[:, None]
    positives = labels_arr == 1
    negatives = labels_arr == 0

    tp = np.sum(predictions & positives[None, :], axis=1).astype(np.int64)
    fp = np.sum(predictions & negatives[None, :], axis=1).astype(np.int64)
    tn = np.sum(~predictions & negatives[None, :], axis=1).astype(np.int64)
    fn = np.sum(~predictions & positives[None, :], axis=1).astype(np.int64)

    return {
        "thresholds": thresholds_arr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def far_frr_by_threshold(
    scores: Iterable[float],
    labels: Iterable[int],
    thresholds: Iterable[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute FAR, FRR, TAR, and accuracy for each threshold."""
    counts = confusion_counts_by_threshold(scores, labels, thresholds)
    positive_count = counts["tp"] + counts["fn"]
    negative_count = counts["fp"] + counts["tn"]
    total_count = positive_count + negative_count

    with np.errstate(divide="ignore", invalid="ignore"):
        far = counts["fp"] / negative_count
        frr = counts["fn"] / positive_count
        tar = counts["tp"] / positive_count
        accuracy = (counts["tp"] + counts["tn"]) / total_count

    return {
        **counts,
        "far": far.astype(np.float64),
        "frr": frr.astype(np.float64),
        "tar": tar.astype(np.float64),
        "accuracy": accuracy.astype(np.float64),
    }


def best_accuracy_threshold(
    scores: Iterable[float],
    labels: Iterable[int],
    thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Return the threshold with maximum verification accuracy."""
    curve = far_frr_by_threshold(scores, labels, thresholds)
    accuracies = curve["accuracy"]
    best_index = int(np.nanargmax(accuracies))
    return {
        "threshold": float(curve["thresholds"][best_index]),
        "accuracy": float(accuracies[best_index]),
        "tp": int(curve["tp"][best_index]),
        "fp": int(curve["fp"][best_index]),
        "tn": int(curve["tn"][best_index]),
        "fn": int(curve["fn"][best_index]),
        "far": float(curve["far"][best_index]),
        "frr": float(curve["frr"][best_index]),
    }


def equal_error_rate(
    scores: Iterable[float],
    labels: Iterable[int],
    thresholds: Iterable[float] | None = None,
) -> dict[str, float]:
    """Estimate EER and threshold where FAR and FRR are closest.

    If adjacent curve points bracket ``FAR - FRR == 0``, linear interpolation is
    used between those points. Otherwise the closest sampled threshold is used.
    """
    scores_arr, labels_arr = _as_arrays(scores, labels)
    threshold_grid = _threshold_array(thresholds, scores_arr)
    threshold_grid = np.asarray(sorted(np.unique(threshold_grid)), dtype=np.float64)
    curve = far_frr_by_threshold(scores_arr, labels_arr, threshold_grid)
    far = curve["far"]
    frr = curve["frr"]
    threshold_values = curve["thresholds"]
    diff = far - frr

    finite = np.isfinite(diff)
    if not finite.any():
        raise ValueError("EER requires at least one positive and one negative pair")

    finite_indices = np.flatnonzero(finite)
    for left, right in zip(finite_indices[:-1], finite_indices[1:], strict=False):
        left_diff = diff[left]
        right_diff = diff[right]
        if left_diff == 0.0:
            eer_value = (far[left] + frr[left]) / 2.0
            return {
                "eer": float(eer_value),
                "threshold": float(threshold_values[left]),
                "far": float(far[left]),
                "frr": float(frr[left]),
            }
        if left_diff * right_diff < 0.0 and right_diff != left_diff:
            ratio = (0.0 - left_diff) / (right_diff - left_diff)
            threshold = threshold_values[left] + ratio * (
                threshold_values[right] - threshold_values[left]
            )
            interpolated_far = far[left] + ratio * (far[right] - far[left])
            interpolated_frr = frr[left] + ratio * (frr[right] - frr[left])
            eer_value = (interpolated_far + interpolated_frr) / 2.0
            return {
                "eer": float(eer_value),
                "threshold": float(threshold),
                "far": float(interpolated_far),
                "frr": float(interpolated_frr),
            }

    closest_index = int(finite_indices[np.nanargmin(np.abs(diff[finite]))])
    eer_value = (far[closest_index] + frr[closest_index]) / 2.0
    return {
        "eer": float(eer_value),
        "threshold": float(threshold_values[closest_index]),
        "far": float(far[closest_index]),
        "frr": float(frr[closest_index]),
    }


def tar_at_far(
    scores: Iterable[float],
    labels: Iterable[int],
    target_fars: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float]]:
    """Return maximum TAR values achieved under each target FAR."""
    curve = far_frr_by_threshold(scores, labels, thresholds)
    results: list[dict[str, float]] = []

    for target_far in target_fars:
        target = float(target_far)
        valid = np.flatnonzero(np.isfinite(curve["far"]) & (curve["far"] <= target))
        if len(valid) == 0:
            results.append(
                {
                    "target_far": target,
                    "tar": float("nan"),
                    "threshold": float("nan"),
                    "far": float("nan"),
                    "frr": float("nan"),
                }
            )
            continue

        best_valid_index = int(valid[np.nanargmax(curve["tar"][valid])])
        results.append(
            {
                "target_far": target,
                "tar": float(curve["tar"][best_valid_index]),
                "threshold": float(curve["thresholds"][best_valid_index]),
                "far": float(curve["far"][best_valid_index]),
                "frr": float(curve["frr"][best_valid_index]),
            }
        )

    return results


def roc_curve_points(
    scores: Iterable[float],
    labels: Iterable[int],
    thresholds: Iterable[float] | None = None,
    max_points: int | None = None,
) -> list[dict[str, float]]:
    """Return FAR/FRR/TAR curve points for JSON reporting."""
    scores_arr, labels_arr = _as_arrays(scores, labels)
    threshold_grid = _threshold_array(thresholds, scores_arr)
    threshold_grid = np.asarray(sorted(np.unique(threshold_grid)), dtype=np.float64)
    curve = far_frr_by_threshold(scores_arr, labels_arr, threshold_grid)
    indices = np.arange(len(curve["thresholds"]))
    if max_points is not None and len(indices) > max_points:
        indices = np.unique(np.linspace(0, len(indices) - 1, num=max_points, dtype=np.int64))

    points: list[dict[str, float]] = []
    for index in indices:
        points.append(
            {
                "threshold": float(curve["thresholds"][index]),
                "far": float(curve["far"][index]),
                "frr": float(curve["frr"][index]),
                "tar": float(curve["tar"][index]),
            }
        )
    return points
