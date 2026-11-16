"""Triage decision service.

The final tier for a patient comes from three layers, in this order:

1. The ML model gives calibrated probabilities for Low / Medium / High, and the
   High-risk threshold from training turns them into a tier.
2. Clinical safety rules check the raw vitals. A red flag forces High and an
   amber flag forces at least Medium. The model can raise a tier but rules can
   never be talked down by it.
3. A clinician can override the result afterwards, with a reason that is stored.

From the final tier and the abnormal vitals we then recommend a ward, a
specialty and an on-duty doctor.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.explain import explain
from src.policy import apply_policy
from triagecare.config import get_settings
from triagecare.models import (
    Role,
    Tier,
    TriageAssessment,
    User,
    Visit,
    VisitEvent,
    VisitStatus,
    Ward,
    WardLevel,
)
from triagecare.timeutil import utcnow

log = logging.getLogger(__name__)

FEATURE_LABELS = {
    "age": "Age",
    "gender": "Sex",
    "comorbidity": "Comorbidity",
    "heart_rate": "Heart rate",
    "resp_rate": "Respiratory rate",
    "systolic_bp": "Systolic BP",
    "spo2": "SpO₂",
    "temp_c": "Temperature",
    "blood_glucose": "Blood glucose",
    "gcs_score": "GCS",
    "pain_score": "Pain score",
}

FEATURE_UNITS = {
    "age": "y",
    "heart_rate": "bpm",
    "resp_rate": "/min",
    "systolic_bp": "mmHg",
    "spo2": "%",
    "temp_c": "°C",
    "blood_glucose": "mg/dL",
    "gcs_score": "/15",
    "pain_score": "/10",
}

# Maximum time to be seen by a doctor, per tier (minutes).
TARGET_MINUTES = {Tier.HIGH: 10, Tier.MEDIUM: 30, Tier.LOW: 120}

COMORBIDITIES = ["None", "Hypertension", "Diabetes", "Cardiovascular", "Asthma"]

SPECIALTIES = [
    "Emergency Medicine",
    "Cardiology",
    "Pulmonology",
    "Neurology",
    "Internal Medicine",
    "General Practice",
]


class ModelUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class TriageModel:
    """Loads the trained pipeline once and keeps it in memory."""

    def __init__(self, model_path, card_path):
        self.model_path = model_path
        self.card_path = card_path
        self._lock = threading.Lock()
        self._model = None
        self._card: dict | None = None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.model_path.exists():
                raise ModelUnavailable(
                    "No trained model found. Run `python -m triagecare.cli setup` "
                    "(or `python -m src.train`) first."
                )
            self._model = joblib.load(self.model_path)
            self._card = json.loads(self.card_path.read_text())
            log.info("Loaded triage model %s", self._card.get("version"))

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    @property
    def card(self) -> dict:
        self._load()
        return self._card

    @property
    def version(self) -> str:
        return f"{self.card['model_name']} {self.card['version']}"

    def predict(self, features: dict[str, Any]) -> dict:
        self._load()
        frame = pd.DataFrame([features], columns=self.card["features"])
        proba = self._model.predict_proba(frame)[0]
        tier_rank = int(apply_policy(proba, self.card["high_threshold"])[0])
        drivers = explain(self._model, features, self.card["reference_patient"])
        return {
            "probabilities": {"low": float(proba[0]), "medium": float(proba[1]), "high": float(proba[2])},
            "tier": Tier.from_rank(tier_rank),
            "drivers": drivers,
        }

    def reload(self) -> None:
        with self._lock:
            self._model = None
            self._card = None


_model: TriageModel | None = None


def get_model() -> TriageModel:
    global _model
    if _model is None:
        s = get_settings()
        _model = TriageModel(s.model_path, s.model_card_path)
    return _model


# --------------------------------------------------------------------------
# Safety rules
# --------------------------------------------------------------------------
@dataclass
class Flag:
    level: str  # "red" or "amber"
    vital: str
    message: str

    def as_dict(self) -> dict:
        return {"level": self.level, "vital": self.vital, "message": self.message}


def check_vitals(v: dict) -> list[Flag]:
    """Rule-based checks loosely based on NEWS2 and common ED red flags."""
    flags: list[Flag] = []

    def red(vital, msg):
        flags.append(Flag("red", vital, msg))

    def amber(vital, msg):
        flags.append(Flag("amber", vital, msg))

    spo2, hr, rr = v["spo2"], v["heart_rate"], v["resp_rate"]
    sbp, temp, gcs = v.get("systolic_bp"), v["temp_c"], v["gcs_score"]
    glucose, pain = v.get("blood_glucose"), v["pain_score"]

    if spo2 < 88:
        red("spo2", f"Severe hypoxia (SpO₂ {spo2:g}%)")
    elif spo2 < 92:
        amber("spo2", f"Low oxygen saturation (SpO₂ {spo2:g}%)")

    if sbp is not None:
        if sbp < 90:
            red("systolic_bp", f"Hypotension (SBP {sbp:g} mmHg)")
        elif sbp >= 200:
            red("systolic_bp", f"Severe hypertension (SBP {sbp:g} mmHg)")
        elif sbp < 100:
            amber("systolic_bp", f"Borderline low BP (SBP {sbp:g} mmHg)")
        elif sbp >= 180:
            amber("systolic_bp", f"High BP (SBP {sbp:g} mmHg)")

    if hr >= 140 or hr <= 40:
        red("heart_rate", f"Heart rate {hr:g} bpm")
    elif hr > 120 or hr < 50:
        amber("heart_rate", f"Heart rate {hr:g} bpm")

    if rr >= 30 or rr <= 8:
        red("resp_rate", f"Respiratory rate {rr:g}/min")
    elif rr > 24:
        amber("resp_rate", f"Raised respiratory rate ({rr:g}/min)")

    if gcs <= 8:
        red("gcs_score", f"Reduced consciousness (GCS {gcs})")
    elif gcs < 15:
        amber("gcs_score", f"Altered consciousness (GCS {gcs})")

    if temp >= 40 or temp <= 35:
        red("temp_c", f"Temperature {temp:g} °C")
    elif temp >= 38.5:
        amber("temp_c", f"Fever ({temp:g} °C)")

    if glucose is not None:
        if glucose < 60 or glucose > 400:
            red("blood_glucose", f"Blood glucose {glucose:g} mg/dL")
        elif glucose > 250:
            amber("blood_glucose", f"High blood glucose ({glucose:g} mg/dL)")

    if pain >= 8:
        amber("pain_score", f"Severe pain ({pain}/10)")

    return sorted(flags, key=lambda f: f.level != "red")


def tier_from_flags(flags: list[Flag]) -> Tier:
    if any(f.level == "red" for f in flags):
        return Tier.HIGH
    if flags:
        return Tier.MEDIUM
    return Tier.LOW


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def choose_specialty(tier: Tier, flags: list[Flag], comorbidity: str) -> str:
    vitals = {f.vital for f in flags}
    if "gcs_score" in vitals:
        return "Neurology" if tier != Tier.HIGH else "Emergency Medicine"
    if tier == Tier.HIGH:
        return "Emergency Medicine"
    if vitals & {"spo2", "resp_rate"}:
        return "Pulmonology"
    if vitals & {"heart_rate", "systolic_bp"} or comorbidity == "Cardiovascular" and vitals:
        return "Cardiology"
    if vitals & {"blood_glucose", "temp_c"}:
        return "Internal Medicine"
    if tier == Tier.MEDIUM:
        return "Emergency Medicine"
    return "General Practice"


def ward_occupancy(db: Session) -> dict[int, int]:
    rows = db.execute(
        select(Visit.ward_id, func.count())
        .where(Visit.ward_id.is_not(None), Visit.status != VisitStatus.COMPLETED)
        .group_by(Visit.ward_id)
    ).all()
    return {ward_id: count for ward_id, count in rows}


_WARD_PREFERENCE = {
    "critical": [WardLevel.RESUS, WardLevel.ACUTE],
    Tier.HIGH: [WardLevel.ACUTE, WardLevel.RESUS],
    Tier.MEDIUM: [WardLevel.URGENT, WardLevel.ACUTE],
    Tier.LOW: [WardLevel.GENERAL, WardLevel.URGENT],
}


def choose_ward(db: Session, tier: Tier, flags: list[Flag], exclude_visit: Visit | None = None) -> tuple[Ward | None, str]:
    """Pick a ward for the tier, falling back to the next suitable ward if full.

    Returns the ward and a short note explaining the choice.
    """
    red_vitals = {f.vital for f in flags if f.level == "red"}
    critical = tier == Tier.HIGH and red_vitals & {"gcs_score", "systolic_bp", "spo2", "resp_rate", "heart_rate"}
    preference = _WARD_PREFERENCE["critical" if critical else tier]

    wards = {w.level: w for w in db.scalars(select(Ward))}
    occupancy = ward_occupancy(db)
    if exclude_visit is not None and exclude_visit.ward_id:
        occupancy[exclude_visit.ward_id] = occupancy.get(exclude_visit.ward_id, 1) - 1

    for i, level in enumerate(preference):
        ward = wards.get(level)
        if ward and occupancy.get(ward.id, 0) < ward.capacity:
            note = "" if i == 0 else f"{wards[preference[0]].name} is full"
            return ward, note

    first = wards.get(preference[0])
    return first, "All suitable wards are full. Patient will wait for a bed."


def doctor_load(db: Session) -> dict[int, int]:
    rows = db.execute(
        select(Visit.doctor_id, func.count())
        .where(Visit.doctor_id.is_not(None), Visit.status != VisitStatus.COMPLETED)
        .group_by(Visit.doctor_id)
    ).all()
    return {doctor_id: count for doctor_id, count in rows}


def choose_doctor(db: Session, specialty: str) -> User | None:
    """Least-busy on-duty doctor in the specialty, else in Emergency Medicine, else anyone."""
    doctors = list(
        db.scalars(select(User).where(User.role == Role.DOCTOR, User.is_active, User.on_duty))
    )
    if not doctors:
        return None
    load = doctor_load(db)
    for wanted in (specialty, "Emergency Medicine", None):
        pool = [d for d in doctors if wanted is None or d.specialty == wanted]
        if pool:
            return min(pool, key=lambda d: (load.get(d.id, 0), d.id))
    return None


# --------------------------------------------------------------------------
# Running a triage
# --------------------------------------------------------------------------
@dataclass
class TriageResult:
    probabilities: dict
    model_tier: Tier
    final_tier: Tier
    flags: list[Flag]
    drivers: list[dict]
    specialty: str
    ward: Ward | None
    ward_note: str
    doctor: User | None
    model_version: str
    reasons: list[str] = field(default_factory=list)


def assess(db: Session, vitals: dict, age: int, gender: str, comorbidity: str, visit: Visit | None = None) -> TriageResult:
    """Score a patient without saving anything. Used by the triage form and the API."""
    model = get_model()
    features = {
        "age": age,
        "gender": gender if gender in ("Male", "Female") else "Female",
        "comorbidity": comorbidity if comorbidity in COMORBIDITIES else "None",
        "heart_rate": vitals["heart_rate"],
        "resp_rate": vitals["resp_rate"],
        "systolic_bp": vitals.get("systolic_bp"),
        "spo2": vitals["spo2"],
        "temp_c": vitals["temp_c"],
        "blood_glucose": vitals.get("blood_glucose"),
        "gcs_score": vitals["gcs_score"],
        "pain_score": vitals["pain_score"],
    }
    prediction = model.predict(features)
    flags = check_vitals(vitals)
    rule_tier = tier_from_flags(flags)
    model_tier = prediction["tier"]
    final_tier = Tier.from_rank(max(model_tier.rank, rule_tier.rank))

    reasons = [f"Model predicted {model_tier.value.title()} risk"]
    if rule_tier.rank > model_tier.rank:
        reasons.append(f"Raised to {final_tier.value.title()} by safety rules")

    specialty = choose_specialty(final_tier, flags, comorbidity)
    ward, ward_note = choose_ward(db, final_tier, flags, exclude_visit=visit)
    doctor = choose_doctor(db, specialty)

    return TriageResult(
        probabilities=prediction["probabilities"],
        model_tier=model_tier,
        final_tier=final_tier,
        flags=flags,
        drivers=prediction["drivers"],
        specialty=specialty,
        ward=ward,
        ward_note=ward_note,
        doctor=doctor,
        model_version=model.version,
        reasons=reasons,
    )


def record_triage(db: Session, visit: Visit, vitals: dict, nurse: User, notes: str | None = None) -> TriageAssessment:
    """Assess, save the assessment, and route the visit to a ward and doctor."""
    patient = visit.patient
    result = assess(db, vitals, patient.age, patient.gender, patient.comorbidity, visit=visit)

    assessment = TriageAssessment(
        visit_id=visit.id,
        nurse_id=nurse.id,
        heart_rate=vitals["heart_rate"],
        resp_rate=vitals["resp_rate"],
        systolic_bp=vitals.get("systolic_bp"),
        diastolic_bp=vitals.get("diastolic_bp"),
        spo2=vitals["spo2"],
        temp_c=vitals["temp_c"],
        blood_glucose=vitals.get("blood_glucose"),
        gcs_score=vitals["gcs_score"],
        pain_score=vitals["pain_score"],
        notes=notes,
        prob_low=result.probabilities["low"],
        prob_medium=result.probabilities["medium"],
        prob_high=result.probabilities["high"],
        model_tier=result.model_tier,
        model_version=result.model_version,
        drivers=result.drivers,
        red_flags=[f.as_dict() for f in result.flags],
        final_tier=result.final_tier,
        specialty=result.specialty,
        recommended_ward_id=result.ward.id if result.ward else None,
        recommended_doctor_id=result.doctor.id if result.doctor else None,
    )
    db.add(assessment)

    first_triage = visit.triaged_at is None
    visit.tier = result.final_tier
    visit.ward_id = assessment.recommended_ward_id
    if visit.status != VisitStatus.IN_TREATMENT:
        visit.doctor_id = assessment.recommended_doctor_id
    if visit.status == VisitStatus.WAITING:
        visit.status = VisitStatus.TRIAGED
    visit.triaged_at = visit.triaged_at or utcnow()

    where = result.ward.name if result.ward else "no ward"
    log_event(
        db,
        visit,
        nurse,
        "triage" if first_triage else "retriage",
        f"{'Triaged' if first_triage else 'Re-triaged'} as {result.final_tier.value.title()} risk, sent to {where}",
    )
    db.flush()
    return assessment


def override_tier(db: Session, visit: Visit, tier: Tier, reason: str, user: User) -> None:
    assessment = visit.latest_assessment
    if assessment is None:
        raise ValueError("Visit has not been triaged yet")
    previous = assessment.effective_tier
    assessment.override_tier = tier
    assessment.override_reason = reason
    assessment.overridden_by_id = user.id
    visit.tier = tier
    log_event(db, visit, user, "override", f"Changed priority {previous} → {tier.value}: {reason}")


def log_event(db: Session, visit: Visit, user: User | None, kind: str, message: str) -> None:
    db.add(VisitEvent(visit_id=visit.id, user_id=user.id if user else None, kind=kind, message=message))


def describe_driver(driver: dict) -> str:
    feature = driver["feature"]
    value = driver["value"]
    unit = FEATURE_UNITS.get(feature, "")
    if isinstance(value, float):
        value = f"{value:g}"
    return f"{value} {unit}".strip()
