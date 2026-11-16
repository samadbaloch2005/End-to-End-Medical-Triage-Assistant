"""Database tables.

Patient flow in one line:
    Patient -> (Appointment) -> Visit -> TriageAssessment(s) -> Ward + Doctor -> discharge

All timestamps are stored as naive UTC. Conversion to local time happens only
when rendering (see `triagecare.timeutil`).
"""
from datetime import date, datetime
from enum import Enum

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from triagecare.db import Base
from triagecare.timeutil import utcnow

try:
    from enum import StrEnum
except ImportError:  # Python 3.10

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)

        def __format__(self, spec: str) -> str:
            return format(str(self.value), spec)


class Role(StrEnum):
    ADMIN = "admin"
    RECEPTIONIST = "receptionist"
    NURSE = "nurse"
    DOCTOR = "doctor"


class Tier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]

    @classmethod
    def from_rank(cls, rank: int) -> "Tier":
        return [cls.LOW, cls.MEDIUM, cls.HIGH][rank]


class WardLevel(StrEnum):
    RESUS = "resus"
    ACUTE = "acute"
    URGENT = "urgent"
    GENERAL = "general"


class AppointmentStatus(StrEnum):
    BOOKED = "booked"
    CHECKED_IN = "checked_in"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class VisitStatus(StrEnum):
    WAITING = "waiting"            # arrived, not yet triaged
    TRIAGED = "triaged"            # risk assigned, waiting for a doctor
    IN_TREATMENT = "in_treatment"  # a doctor has picked the patient up
    COMPLETED = "completed"        # left the department


class ArrivalMode(StrEnum):
    WALK_IN = "walk_in"
    APPOINTMENT = "appointment"
    AMBULANCE = "ambulance"


class Disposition(StrEnum):
    DISCHARGED = "discharged"
    ADMITTED = "admitted"
    REFERRED = "referred"
    LEFT = "left_without_being_seen"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(20), index=True)
    specialty: Mapped[str | None] = mapped_column(String(60))
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    on_duty: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    @property
    def display_name(self) -> str:
        return f"Dr. {self.full_name}" if self.role == Role.DOCTOR else self.full_name

    @property
    def initials(self) -> str:
        parts = self.full_name.split()
        return "".join(p[0] for p in parts[:2]).upper()


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True)
    mrn: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    first_name: Mapped[str] = mapped_column(String(60))
    last_name: Mapped[str] = mapped_column(String(60), index=True)
    date_of_birth: Mapped[date] = mapped_column(Date)
    gender: Mapped[str] = mapped_column(String(10))
    phone: Mapped[str | None] = mapped_column(String(30), index=True)
    cnic: Mapped[str | None] = mapped_column(String(20), index=True)
    address: Mapped[str | None] = mapped_column(String(255))
    comorbidity: Mapped[str] = mapped_column(String(40), default="None")
    allergies: Mapped[str | None] = mapped_column(String(255))
    emergency_contact: Mapped[str | None] = mapped_column(String(120))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    visits: Mapped[list["Visit"]] = relationship(
        back_populates="patient", order_by="desc(Visit.arrived_at)"
    )
    appointments: Mapped[list["Appointment"]] = relationship(
        back_populates="patient", order_by="desc(Appointment.scheduled_at)"
    )

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def age(self) -> int:
        today = date.today()
        dob = self.date_of_birth
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


class Ward(Base):
    __tablename__ = "wards"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(10), unique=True)
    name: Mapped[str] = mapped_column(String(80))
    level: Mapped[str] = mapped_column(String(20))
    capacity: Mapped[int] = mapped_column(Integer)
    description: Mapped[str | None] = mapped_column(String(255))


class Appointment(Base):
    __tablename__ = "appointments"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    scheduled_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    reason: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default=AppointmentStatus.BOOKED)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    patient: Mapped[Patient] = relationship(back_populates="appointments")
    doctor: Mapped[User | None] = relationship(foreign_keys=[doctor_id])
    visit: Mapped["Visit | None"] = relationship(back_populates="appointment")


class Visit(Base):
    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"), unique=True)
    arrival_mode: Mapped[str] = mapped_column(String(20), default=ArrivalMode.WALK_IN)
    chief_complaint: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default=VisitStatus.WAITING, index=True)

    # Current priority, copied from the latest assessment so the queue can sort cheaply.
    tier: Mapped[str | None] = mapped_column(String(10), index=True)
    ward_id: Mapped[int | None] = mapped_column(ForeignKey("wards.id"))
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))

    arrived_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    triaged_at: Mapped[datetime | None] = mapped_column(DateTime)
    seen_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    disposition: Mapped[str | None] = mapped_column(String(30))
    doctor_notes: Mapped[str | None] = mapped_column(Text)
    registered_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))

    patient: Mapped[Patient] = relationship(back_populates="visits")
    appointment: Mapped[Appointment | None] = relationship(back_populates="visit")
    ward: Mapped[Ward | None] = relationship()
    doctor: Mapped[User | None] = relationship(foreign_keys=[doctor_id])
    assessments: Mapped[list["TriageAssessment"]] = relationship(
        back_populates="visit", order_by="desc(TriageAssessment.created_at)"
    )
    events: Mapped[list["VisitEvent"]] = relationship(
        back_populates="visit", order_by="VisitEvent.created_at"
    )

    @property
    def latest_assessment(self) -> "TriageAssessment | None":
        return self.assessments[0] if self.assessments else None

    @property
    def is_active(self) -> bool:
        return self.status != VisitStatus.COMPLETED


class TriageAssessment(Base):
    __tablename__ = "triage_assessments"

    id: Mapped[int] = mapped_column(primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id"), index=True)
    nurse_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Vitals as entered
    heart_rate: Mapped[float] = mapped_column(Float)
    resp_rate: Mapped[float] = mapped_column(Float)
    systolic_bp: Mapped[float | None] = mapped_column(Float)
    diastolic_bp: Mapped[float | None] = mapped_column(Float)
    spo2: Mapped[float] = mapped_column(Float)
    temp_c: Mapped[float] = mapped_column(Float)
    blood_glucose: Mapped[float | None] = mapped_column(Float)
    gcs_score: Mapped[int] = mapped_column(Integer)
    pain_score: Mapped[int] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)

    # What the model said
    prob_low: Mapped[float] = mapped_column(Float)
    prob_medium: Mapped[float] = mapped_column(Float)
    prob_high: Mapped[float] = mapped_column(Float)
    model_tier: Mapped[str] = mapped_column(String(10))
    model_version: Mapped[str] = mapped_column(String(40))
    drivers: Mapped[list] = mapped_column(JSON, default=list)

    # Safety rules and the final decision
    red_flags: Mapped[list] = mapped_column(JSON, default=list)
    final_tier: Mapped[str] = mapped_column(String(10))
    specialty: Mapped[str | None] = mapped_column(String(60))
    recommended_ward_id: Mapped[int | None] = mapped_column(ForeignKey("wards.id"))
    recommended_doctor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))

    # Clinician override, if any
    override_tier: Mapped[str | None] = mapped_column(String(10))
    override_reason: Mapped[str | None] = mapped_column(String(255))
    overridden_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))

    visit: Mapped[Visit] = relationship(back_populates="assessments")
    nurse: Mapped[User | None] = relationship(foreign_keys=[nurse_id])
    recommended_ward: Mapped[Ward | None] = relationship(foreign_keys=[recommended_ward_id])
    recommended_doctor: Mapped[User | None] = relationship(foreign_keys=[recommended_doctor_id])
    overridden_by: Mapped[User | None] = relationship(foreign_keys=[overridden_by_id])

    @property
    def effective_tier(self) -> str:
        return self.override_tier or self.final_tier

    @property
    def probabilities(self) -> dict[str, float]:
        return {"low": self.prob_low, "medium": self.prob_medium, "high": self.prob_high}


class VisitEvent(Base):
    """Timeline entry for a visit: who did what, and when."""

    __tablename__ = "visit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    kind: Mapped[str] = mapped_column(String(30))
    message: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    visit: Mapped[Visit] = relationship(back_populates="events")
    user: Mapped[User | None] = relationship()
