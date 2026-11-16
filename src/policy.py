"""Turning model probabilities into a triage tier.

Plain argmax treats every mistake the same. In triage, sending a High-risk
patient to the waiting room is much worse than seeing a Medium-risk patient
early, so the High tier gets its own, lower threshold.
"""
import numpy as np

LOW, MEDIUM, HIGH = 0, 1, 2


def apply_policy(proba: np.ndarray, high_threshold: float) -> np.ndarray:
    proba = np.atleast_2d(proba)
    tiers = proba[:, :2].argmax(axis=1)
    tiers[proba[:, HIGH] >= high_threshold] = HIGH
    return tiers


def choose_high_threshold(proba: np.ndarray, y_true, target_recall: float = 0.85) -> float:
    """Largest threshold that still catches `target_recall` of true High cases.

    Using the largest such value keeps over-triage as low as possible.
    """
    y_true = np.asarray(y_true)
    for threshold in np.arange(0.50, 0.04, -0.01):
        caught = apply_policy(proba, threshold)[y_true == HIGH] == HIGH
        if caught.mean() >= target_recall:
            return round(float(threshold), 2)
    return 0.05
