"""Clinical read-outs built on top of a triage: which vitals are abnormal and what
that may point to, a suggested care plan for the doctor, and the alerts shown on
the dashboard.

Everything here is decision support for the clinician, based on common adult
emergency-department references (NEWS2, SIRS / qSOFA, Sepsis Six). It never
changes the triage tier; that comes from the model and the safety rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from triagecare.models import ArrivalMode, Tier, TriageAssessment, Visit, VisitStatus
from triagecare.timeutil import minutes_since, utcnow

# key: (label, unit, normal_low, normal_high, scale_min, scale_max)
VITAL_RANGES: dict[str, tuple[str, str, float, float, float, float]] = {
    "heart_rate": ("Heart rate", "bpm", 60, 100, 30, 180),
    "resp_rate": ("Respiratory rate", "/min", 12, 20, 4, 45),
    "spo2": ("SpO₂", "%", 95, 100, 75, 100),
    "systolic_bp": ("Systolic BP", "mmHg", 100, 140, 60, 220),
    "diastolic_bp": ("Diastolic BP", "mmHg", 60, 90, 30, 130),
    "temp_c": ("Temperature", "°C", 36.1, 37.8, 34, 42),
    "blood_glucose": ("Blood glucose", "mg/dL", 70, 180, 30, 450),
    "gcs_score": ("GCS", "/15", 15, 15, 3, 15),
    "pain_score": ("Pain", "/10", 0, 3, 0, 10),
}

# What an abnormal value may point to: (vital, direction) -> (name, likely causes)
MEANINGS: dict[tuple[str, str], tuple[str, str]] = {
    ("heart_rate", "high"): ("Tachycardia", "pain, fever, dehydration, blood loss, sepsis or an arrhythmia"),
    ("heart_rate", "low"): ("Bradycardia", "heart block, beta-blockers or other medicines, raised intracranial pressure"),
    ("resp_rate", "high"): ("Tachypnoea", "pneumonia, asthma or COPD flare, pulmonary embolism, or acidosis (sepsis, DKA)"),
    ("resp_rate", "low"): ("Slow breathing", "opioids or sedatives, exhaustion, reduced consciousness"),
    ("spo2", "low"): ("Hypoxia", "pneumonia, asthma or COPD, heart failure, pulmonary embolism"),
    ("systolic_bp", "low"): ("Hypotension", "shock from sepsis, bleeding, dehydration or a cardiac cause"),
    ("systolic_bp", "high"): ("Hypertension", "hypertensive urgency; look for stroke, chest pain or kidney injury"),
    ("diastolic_bp", "low"): ("Low diastolic BP", "vasodilation (sepsis), dehydration"),
    ("diastolic_bp", "high"): ("High diastolic BP", "poorly controlled hypertension"),
    ("temp_c", "high"): ("Fever", "infection; with a fast pulse or breathing, consider sepsis"),
    ("temp_c", "low"): ("Hypothermia", "exposure, sepsis in the elderly, hypothyroidism"),
    ("blood_glucose", "high"): ("Hyperglycaemia", "uncontrolled diabetes, DKA or HHS, stress response"),
    ("blood_glucose", "low"): ("Hypoglycaemia", "insulin or sulfonylurea excess, missed meals, sepsis, liver disease"),
    ("gcs_score", "low"): ("Reduced consciousness", "head injury, stroke, hypoglycaemia, intoxication, sepsis"),
    ("pain_score", "high"): ("Significant pain", "the cause needs prompt assessment and analgesia"),
}


def vitals_from_assessment(a: TriageAssessment) -> dict:
    return {k: getattr(a, k) for k in VITAL_RANGES}


# --------------------------------------------------------------------------
# Abnormal vitals
# --------------------------------------------------------------------------
@dataclass
class VitalReading:
    key: str
    label: str
    unit: str
    value: float | None
    low: float
    high: float
    status: str          # "normal", "high", "low" or "missing"
    severity: str        # "red", "amber", "watch" or "ok"
    meaning: str = ""
    causes: str = ""
    marker: float = 0    # position of the value on the scale, 0-100
    band: tuple[float, float] = (0, 100)  # normal range on the scale, 0-100

    @property
    def abnormal(self) -> bool:
        return self.status in ("high", "low")

    @property
    def display(self) -> str:
        if self.value is None:
            return "—"
        return f"{self.value:g}"


def _pos(value: float, lo: float, hi: float) -> float:
    return round(max(0.0, min(100.0, (value - lo) / (hi - lo) * 100)), 1)


def read_vitals(vitals: dict, flags: list[dict] | None = None) -> list[VitalReading]:
    """Every vital with its normal range, where it sits, and how worrying it is.

    `flags` are the safety-rule flags (dicts with `vital` and `level`); they set
    the severity. A value outside the normal range without a flag is "watch".
    """
    flag_level = {}
    for f in flags or []:
        f = f if isinstance(f, dict) else f.as_dict()
        if flag_level.get(f["vital"]) != "red":
            flag_level[f["vital"]] = f["level"]

    out = []
    for key, (label, unit, lo, hi, smin, smax) in VITAL_RANGES.items():
        value = vitals.get(key)
        if value is None:
            out.append(VitalReading(key, label, unit, None, lo, hi, "missing", "ok", band=(_pos(lo, smin, smax), _pos(hi, smin, smax))))
            continue
        value = float(value)
        status = "low" if value < lo else "high" if value > hi else "normal"
        severity = flag_level.get(key) or ("watch" if status != "normal" else "ok")
        meaning, causes = MEANINGS.get((key, status), ("", ""))
        out.append(
            VitalReading(
                key, label, unit, value, lo, hi, status, severity, meaning, causes,
                marker=_pos(value, smin, smax), band=(_pos(lo, smin, smax), _pos(hi, smin, smax)),
            )
        )
    return out


@dataclass
class Concern:
    name: str
    level: str   # "red" or "amber"
    because: list[str]
    note: str = ""


def possible_concerns(vitals: dict) -> list[Concern]:
    """Patterns across several vitals (SIRS, qSOFA, shock index and so on)."""
    hr, rr, spo2 = vitals.get("heart_rate"), vitals.get("resp_rate"), vitals.get("spo2")
    sbp, temp, gcs = vitals.get("systolic_bp"), vitals.get("temp_c"), vitals.get("gcs_score")
    glucose = vitals.get("blood_glucose")
    out: list[Concern] = []

    sirs = []
    if temp is not None and (temp > 38 or temp < 36):
        sirs.append(f"temperature {temp:g} °C")
    if hr is not None and hr > 90:
        sirs.append(f"heart rate {hr:g}")
    if rr is not None and rr > 20:
        sirs.append(f"respiratory rate {rr:g}")
    qsofa = []
    if rr is not None and rr >= 22:
        qsofa.append(f"RR {rr:g}")
    if sbp is not None and sbp <= 100:
        qsofa.append(f"SBP {sbp:g}")
    if gcs is not None and gcs < 15:
        qsofa.append(f"GCS {gcs}")

    if len(qsofa) >= 2:
        out.append(Concern("Possible sepsis, qSOFA ≥ 2", "red", qsofa,
                           "High risk of deterioration if infection is the cause."))
    elif len(sirs) >= 2 and temp is not None and (temp > 38 or temp < 36):
        out.append(Concern("Possible sepsis (SIRS criteria met)", "amber", sirs,
                           "Look for a source of infection."))

    if hr is not None and sbp:
        shock_index = hr / sbp
        if sbp < 90 or shock_index >= 1.0:
            out.append(Concern("Possible shock", "red", [f"heart rate {hr:g}", f"SBP {sbp:g}", f"shock index {shock_index:.1f}"],
                               "Shock index above 1 means the heart is beating faster than the pressure it keeps up."))

    if spo2 is not None and rr is not None and spo2 < 92 and rr > 24:
        out.append(Concern("Respiratory distress", "red", [f"SpO₂ {spo2:g}%", f"RR {rr:g}"]))
    elif spo2 is not None and spo2 < 92:
        out.append(Concern("Low oxygen", "amber" if spo2 >= 88 else "red", [f"SpO₂ {spo2:g}%"]))

    if glucose is not None and glucose > 250 and rr is not None and rr > 20:
        out.append(Concern("Possible diabetic ketoacidosis", "red", [f"glucose {glucose:g}", f"RR {rr:g}"],
                           "Fast breathing with high glucose can be the body clearing acid. Check ketones."))
    elif glucose is not None and glucose < 70:
        out.append(Concern("Hypoglycaemia", "red" if glucose < 54 else "amber", [f"glucose {glucose:g}"]))

    if gcs is not None and gcs < 15:
        out.append(Concern("Neurological concern", "red" if gcs <= 8 else "amber", [f"GCS {gcs}"]))

    if sbp is not None and sbp >= 180:
        out.append(Concern("Hypertensive urgency", "red" if sbp >= 200 else "amber", [f"SBP {sbp:g}"],
                           "Check for headache, chest pain, vision changes or neurological signs."))

    return sorted(out, key=lambda c: c.level != "red")


# --------------------------------------------------------------------------
# Suggested care plan
# --------------------------------------------------------------------------
@dataclass
class CareSection:
    title: str
    items: list[str] = field(default_factory=list)
    urgent: bool = False


def care_plan(
    tier: str,
    vitals: dict,
    flags: list[dict],
    complaint: str = "",
    comorbidity: str = "None",
    allergies: str | None = None,
) -> list[CareSection]:
    """A checklist the doctor can work through. Ordered: most urgent first."""
    tier = str(tier)
    flagged = {(f if isinstance(f, dict) else f.as_dict())["vital"] for f in flags}
    concerns = {c.name.split(" (")[0].split(",")[0] for c in possible_concerns(vitals)}
    text = (complaint or "").lower()
    sections: list[CareSection] = []

    hr, rr, spo2 = vitals.get("heart_rate"), vitals.get("resp_rate"), vitals.get("spo2")
    sbp, temp, gcs = vitals.get("systolic_bp"), vitals.get("temp_c"), vitals.get("gcs_score")
    glucose, pain = vitals.get("blood_glucose"), vitals.get("pain_score") or 0

    # Immediate actions
    now = CareSection("Immediate actions", urgent=tier == Tier.HIGH)
    if tier == Tier.HIGH:
        now.items += [
            "Senior doctor review now; move to resuscitation or acute bay",
            "ABCDE assessment, continuous cardiac and SpO₂ monitoring",
            "IV access with bloods: FBC, U&E, LFT, CRP, lactate, glucose",
        ]
    if gcs is not None and gcs <= 8:
        now.items.append("Protect the airway (GCS 8 or below); call anaesthetics")
    if "Possible shock" in concerns or (sbp is not None and sbp < 90):
        now.items += [
            "Two large-bore IV cannulas; fluid bolus per local protocol, then reassess BP and pulse",
            "Look for bleeding; catheter and hourly urine output",
        ]
    if "Possible sepsis" in concerns:
        now.items.append("Sepsis Six within 1 hour: oxygen, blood cultures, IV antibiotics, IV fluids, lactate, urine output")
    if spo2 is not None and spo2 < 94:
        now.items.append("Oxygen, titrate to SpO₂ 94–98% (88–92% if a known CO₂ retainer)")
    if glucose is not None and glucose < 70:
        if gcs is not None and gcs < 15:
            now.items.append("Hypoglycaemia with reduced consciousness: IV dextrose per protocol, recheck in 15 min")
        else:
            now.items.append("Hypoglycaemia: 15–20 g fast-acting oral glucose, recheck in 15 min")
    if now.items:
        sections.append(now)

    # Investigations
    tests = CareSection("Investigations")
    if "chest" in text or comorbidity == "Cardiovascular" or "heart_rate" in flagged:
        tests.items.append("12-lead ECG" + (" within 10 minutes of arrival" if "chest" in text else ""))
    if "chest" in text:
        tests.items.append("Troponin (repeat per local ACS pathway)")
    if (spo2 is not None and spo2 < 92) or "breath" in text or "cough" in text:
        tests.items.append("Chest X-ray")
    if spo2 is not None and spo2 < 92 or (rr is not None and rr > 24):
        tests.items.append("Arterial or venous blood gas")
    if glucose is not None and glucose > 250:
        tests.items.append("Blood ketones and venous gas (rule out DKA / HHS)")
    if temp is not None and temp >= 38:
        tests.items.append("Find the source: urine dip, chest X-ray, blood cultures before antibiotics")
    if gcs is not None and gcs < 15:
        tests.items.append("Capillary glucose now; CT head if no clear metabolic cause")
    if sbp is not None and sbp >= 180:
        tests.items.append("Repeat BP in both arms; urine dip, renal function, fundoscopy")
    if "abdom" in text or "stomach" in text:
        tests.items.append("Abdominal exam, urine dip, pregnancy test where relevant, consider ultrasound")
    if any(w in text for w in ("fall", "injur", "trauma", "accident", "fracture")):
        tests.items.append("Secondary survey and imaging of injured areas")
    if tests.items:
        sections.append(tests)

    # Treatment
    treat = CareSection("Treatment to consider")
    if "chest" in text:
        treat.items.append("If ACS suspected: aspirin unless contraindicated; follow the chest-pain pathway")
    if comorbidity == "Asthma" or "wheez" in text:
        if (spo2 is not None and spo2 < 95) or (rr is not None and rr > 20):
            treat.items.append("Nebulised bronchodilator; peak flow before and after")
    if temp is not None and temp >= 38 and "Possible sepsis" not in concerns:
        treat.items.append("Paracetamol for fever if no contraindication; oral fluids")
    if pain >= 4:
        treat.items.append(f"Analgesia for pain {pain}/10 by the pain ladder; reassess in 30 minutes")
    if glucose is not None and glucose > 250 and "Possible diabetic ketoacidosis" not in concerns:
        treat.items.append("Review diabetes medication; IV fluids if dehydrated")
    if "Possible diabetic ketoacidosis" in concerns:
        treat.items.append("If ketones raised: start the DKA protocol (fixed-rate insulin, fluids, potassium)")
    if hr is not None and hr < 50:
        treat.items.append("Bradycardia: review rate-limiting medicines; atropine per ALS if unstable")
    if treat.items:
        sections.append(treat)

    # Monitoring
    monitor = CareSection("Monitoring")
    monitor.items.append({
        Tier.HIGH: "Observations every 15 minutes (continuous if unstable)",
        Tier.MEDIUM: "Observations every 30–60 minutes",
        Tier.LOW: "Observations on arrival and before discharge",
    }.get(tier, "Observations per local policy"))
    if gcs is not None and gcs < 15:
        monitor.items.append("Neuro observations (GCS, pupils) every 30 minutes")
    if glucose is not None and (glucose < 70 or glucose > 250):
        monitor.items.append("Capillary glucose hourly")
    monitor.items.append("Re-triage if any vital worsens or the patient looks worse")
    sections.append(monitor)

    # Safety
    safety = CareSection("Safety")
    if allergies:
        safety.items.append(f"Allergy recorded: {allergies}. Check before prescribing")
    if comorbidity == "Diabetes":
        safety.items.append("Diabetic: check glucose before discharge and review medicines")
    if comorbidity == "Hypertension":
        safety.items.append("Hypertensive: note usual medicines and adherence")
    if tier == Tier.LOW and not flagged:
        safety.items.append("Likely suitable for OPD; give safety-net advice on discharge")
    if safety.items:
        sections.append(safety)

    return sections


# --------------------------------------------------------------------------
# Dashboard alerts
# --------------------------------------------------------------------------
@dataclass
class Alert:
    visit: Visit
    level: str          # "critical" or "warning"
    title: str
    details: list[str]
    waited: int


def critical_alerts(db: Session) -> list[Alert]:
    """Patients not yet with a doctor whose vitals (or history) need attention now."""
    now = utcnow()
    visits = list(
        db.scalars(
            select(Visit)
            .where(Visit.status.in_([VisitStatus.WAITING, VisitStatus.TRIAGED]))
            .options(selectinload(Visit.patient), selectinload(Visit.assessments), selectinload(Visit.ward))
        )
    )
    waiting_patient_ids = [v.patient_id for v in visits if v.status == VisitStatus.WAITING]
    high_before = set()
    if waiting_patient_ids:
        high_before = set(
            db.scalars(
                select(Visit.patient_id).where(
                    Visit.patient_id.in_(waiting_patient_ids),
                    Visit.status == VisitStatus.COMPLETED,
                    Visit.tier == Tier.HIGH,
                )
            )
        )

    alerts: list[Alert] = []
    for v in visits:
        waited = minutes_since(v.arrived_at, now)
        a = v.latest_assessment
        if a is not None:
            reds = [f["message"] for f in a.red_flags if f["level"] == "red"]
            if reds:
                alerts.append(Alert(v, "critical", "Critical vitals", reds, waited))
            elif a.effective_tier == Tier.HIGH:
                alerts.append(Alert(v, "warning", "High risk, waiting for a doctor",
                                    [f["message"] for f in a.red_flags] or ["Model predicts high risk"], waited))
        else:
            reasons = []
            if v.arrival_mode == ArrivalMode.AMBULANCE:
                reasons.append("Arrived by ambulance")
            if v.patient_id in high_before:
                reasons.append("High risk on a previous visit")
            if reasons:
                alerts.append(Alert(v, "warning", "Needs triage now", reasons, waited))

    return sorted(alerts, key=lambda x: (x.level != "critical", -x.waited))
