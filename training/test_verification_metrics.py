from __future__ import annotations

import math

import numpy as np
import pytest

from training.verification_metrics import (
    best_accuracy_threshold,
    confusion_counts_by_threshold,
    equal_error_rate,
    far_frr_by_threshold,
    tar_at_far,
)


def test_confusion_counts_and_rates_by_threshold():
    scores = np.array([0.95, 0.80, 0.40, 0.20])
    labels = np.array([1, 1, 0, 0])
    thresholds = np.array([0.90, 0.50, 0.30])

    counts = confusion_counts_by_threshold(scores, labels, thresholds)

    assert counts["tp"].tolist() == [1, 2, 2]
    assert counts["fp"].tolist() == [0, 0, 1]
    assert counts["tn"].tolist() == [2, 2, 1]
    assert counts["fn"].tolist() == [1, 0, 0]

    rates = far_frr_by_threshold(scores, labels, thresholds)
    np.testing.assert_allclose(rates["far"], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(rates["frr"], [0.5, 0.0, 0.0])


def test_threshold_direction_higher_threshold_is_stricter():
    scores = np.array([0.90, 0.70, 0.60, 0.30])
    labels = np.array([1, 0, 1, 0])

    rates = far_frr_by_threshold(scores, labels, thresholds=[0.40, 0.80])

    low_threshold_far = rates["far"][0]
    high_threshold_far = rates["far"][1]
    low_threshold_frr = rates["frr"][0]
    high_threshold_frr = rates["frr"][1]

    assert high_threshold_far < low_threshold_far
    assert high_threshold_frr > low_threshold_frr


def test_equal_error_rate_exact_crossing():
    scores = np.array([0.90, 0.80, 0.40, 0.30, 0.70, 0.60, 0.20, 0.10])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    result = equal_error_rate(scores, labels, thresholds=[0.25, 0.35, 0.50, 0.65, 0.75])

    assert math.isclose(result["eer"], 0.5)
    assert math.isclose(result["threshold"], 0.5)
    assert math.isclose(result["far"], 0.5)
    assert math.isclose(result["frr"], 0.5)


def test_tar_at_far_selects_threshold_under_target_far():
    scores = np.array([0.95, 0.85, 0.55, 0.90, 0.40, 0.20])
    labels = np.array([1, 1, 1, 0, 0, 0])

    result = tar_at_far(scores, labels, target_fars=[0.0, 1.0 / 3.0])

    assert math.isclose(result[0]["far"], 0.0)
    assert math.isclose(result[0]["tar"], 1.0 / 3.0)
    assert math.isclose(result[0]["threshold"], 0.95)

    assert math.isclose(result[1]["far"], 1.0 / 3.0)
    assert math.isclose(result[1]["tar"], 1.0)
    assert math.isclose(result[1]["threshold"], 0.55)


def test_best_accuracy_threshold():
    scores = np.array([0.90, 0.80, 0.40, 0.30])
    labels = np.array([1, 1, 0, 0])

    result = best_accuracy_threshold(scores, labels)

    assert math.isclose(result["accuracy"], 1.0)
    assert math.isclose(result["far"], 0.0)
    assert math.isclose(result["frr"], 0.0)
    assert 0.40 < result["threshold"] <= 0.80


def test_no_positive_pairs_rates_and_eer_error():
    scores = np.array([0.90, 0.40, 0.20])
    labels = np.array([0, 0, 0])

    rates = far_frr_by_threshold(scores, labels, thresholds=[0.50])

    assert math.isclose(rates["far"][0], 1.0 / 3.0)
    assert math.isnan(rates["frr"][0])
    assert math.isnan(rates["tar"][0])
    with pytest.raises(ValueError, match="EER requires"):
        equal_error_rate(scores, labels)


def test_no_negative_pairs_rates_and_eer_error():
    scores = np.array([0.90, 0.70, 0.20])
    labels = np.array([1, 1, 1])

    rates = far_frr_by_threshold(scores, labels, thresholds=[0.50])

    assert math.isnan(rates["far"][0])
    assert math.isclose(rates["frr"][0], 1.0 / 3.0)
    assert math.isclose(rates["tar"][0], 2.0 / 3.0)
    with pytest.raises(ValueError, match="EER requires"):
        equal_error_rate(scores, labels)


def test_all_scores_equal_has_expected_eer():
    scores = np.array([0.50, 0.50, 0.50, 0.50])
    labels = np.array([1, 1, 0, 0])

    result = equal_error_rate(scores, labels)

    assert math.isclose(result["eer"], 0.5)
    assert math.isclose(result["far"], 0.5)
    assert math.isclose(result["frr"], 0.5)


def test_perfect_separation_has_zero_eer():
    scores = np.array([0.90, 0.80, 0.20, 0.10])
    labels = np.array([1, 1, 0, 0])

    result = equal_error_rate(scores, labels)
    best = best_accuracy_threshold(scores, labels)

    assert math.isclose(result["eer"], 0.0)
    assert math.isclose(result["far"], 0.0)
    assert math.isclose(result["frr"], 0.0)
    assert math.isclose(best["accuracy"], 1.0)


def test_fully_wrong_separation_has_high_eer():
    scores = np.array([0.20, 0.10, 0.90, 0.80])
    labels = np.array([1, 1, 0, 0])

    result = equal_error_rate(scores, labels)
    best = best_accuracy_threshold(scores, labels)

    assert math.isclose(result["eer"], 1.0)
    assert math.isclose(best["accuracy"], 0.5)


def test_target_far_not_exactly_present_uses_best_valid_threshold():
    scores = np.array([0.95, 0.85, 0.55, 0.90, 0.40, 0.20])
    labels = np.array([1, 1, 1, 0, 0, 0])

    result = tar_at_far(scores, labels, target_fars=[0.20])[0]

    assert math.isclose(result["far"], 0.0)
    assert math.isclose(result["tar"], 1.0 / 3.0)
    assert math.isclose(result["threshold"], 0.95)


def test_eer_interpolates_between_adjacent_curve_points():
    scores = np.array([0.90, 0.80, 0.10, 0.85, 0.20])
    labels = np.array([1, 1, 1, 0, 0])

    result = equal_error_rate(scores, labels, thresholds=[0.80, 0.85])

    assert math.isclose(result["eer"], 0.5)
    assert 0.80 < result["threshold"] < 0.85
