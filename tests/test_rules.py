import numpy as np
import pytest

from src.policy import apply_policy, choose_high_threshold
from triagecare.models import Tier
from triagecare.services.triage import check_vitals, choose_specialty, tier_from_flags

NORMAL = dict(heart_rate=78, resp_rate=16, systolic_bp=122, spo2=98, temp_c=36.9,
              blood_glucose=105, gcs_score=15, pain_score=1)


def vitals(**changes):
    return {**NORMAL, **changes}


def test_normal_vitals_raise_no_flags():
    assert check_vitals(NORMAL) == []
    assert tier_from_flags([]) == Tier.LOW


@pytest.mark.parametrize(
    "changes, vital",
    [
        ({"spo2": 85}, "spo2"),
        ({"systolic_bp": 82}, "systolic_bp"),
        ({"gcs_score": 7}, "gcs_score"),
        ({"resp_rate": 34}, "resp_rate"),
        ({"heart_rate": 150}, "heart_rate"),
        ({"temp_c": 40.5}, "temp_c"),
        ({"blood_glucose": 45}, "blood_glucose"),
    ],
)
def test_red_flags_force_high(changes, vital):
    flags = check_vitals(vitals(**changes))
    assert any(f.level == "red" and f.vital == vital for f in flags)
    assert tier_from_flags(flags) == Tier.HIGH


@pytest.mark.parametrize(
    "changes",
    [{"spo2": 91}, {"heart_rate": 125}, {"temp_c": 38.8}, {"pain_score": 9}, {"gcs_score": 14}],
)
def test_amber_flags_force_at_least_medium(changes):
    flags = check_vitals(vitals(**changes))
    assert flags and all(f.level == "amber" for f in flags)
    assert tier_from_flags(flags) == Tier.MEDIUM


def test_missing_optional_vitals_are_fine():
    assert check_vitals(vitals(systolic_bp=None, blood_glucose=None)) == []


def test_red_flags_listed_first():
    flags = check_vitals(vitals(temp_c=38.8, spo2=84))
    assert flags[0].level == "red"


def test_specialty_routing():
    resp = check_vitals(vitals(spo2=91))
    assert choose_specialty(Tier.MEDIUM, resp, "None") == "Pulmonology"
    cardiac = check_vitals(vitals(heart_rate=125))
    assert choose_specialty(Tier.MEDIUM, cardiac, "Cardiovascular") == "Cardiology"
    assert choose_specialty(Tier.HIGH, check_vitals(vitals(systolic_bp=80)), "None") == "Emergency Medicine"
    assert choose_specialty(Tier.LOW, [], "None") == "General Practice"


def test_policy_prefers_high_above_threshold():
    proba = np.array([[0.40, 0.35, 0.25], [0.60, 0.30, 0.10], [0.20, 0.50, 0.30]])
    assert apply_policy(proba, 0.25).tolist() == [2, 0, 2]
    assert apply_policy(proba, 0.50).tolist() == [0, 0, 1]


def test_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    noise = rng.random((500, 3))
    proba = noise + np.eye(3)[y] * 0.6
    proba /= proba.sum(axis=1, keepdims=True)
    t = choose_high_threshold(proba, y, target_recall=0.9)
    caught = (apply_policy(proba, t)[y == 2] == 2).mean()
    assert caught >= 0.9
