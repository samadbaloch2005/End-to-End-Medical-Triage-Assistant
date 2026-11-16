from triagecare.models import Tier
from triagecare.services.triage import get_model

BASE = dict(age=45, gender="Female", comorbidity="None", heart_rate=76, resp_rate=15,
            systolic_bp=120, spo2=98, temp_c=36.8, blood_glucose=100, gcs_score=15, pain_score=1)


def test_probabilities_are_valid():
    out = get_model().predict(BASE)
    probs = out["probabilities"]
    assert set(probs) == {"low", "medium", "high"}
    assert abs(sum(probs.values()) - 1) < 1e-6
    assert all(0 <= p <= 1 for p in probs.values())


def test_healthy_adult_is_low_risk():
    assert get_model().predict(BASE)["tier"] == Tier.LOW


def test_critical_patient_is_high_risk():
    critical = {**BASE, "age": 78, "heart_rate": 135, "resp_rate": 30, "systolic_bp": 80,
                "spo2": 85, "gcs_score": 9, "comorbidity": "Cardiovascular"}
    out = get_model().predict(critical)
    assert out["tier"] == Tier.HIGH
    assert out["probabilities"]["high"] > 0.5


def test_missing_values_are_imputed():
    out = get_model().predict({**BASE, "systolic_bp": None, "blood_glucose": None})
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-6


def test_explanation_points_at_the_abnormal_vital():
    hypoxic = {**BASE, "spo2": 86}
    drivers = get_model().predict(hypoxic)["drivers"]
    assert drivers[0]["feature"] == "spo2"
    assert drivers[0]["impact"] > 0


def test_model_card_has_metrics():
    card = get_model().card
    assert card["test_metrics"]["roc_auc_ovr"] > 0.8
    assert 0 < card["high_threshold"] < 0.5
    assert len(card["leaderboard"]) == 4
