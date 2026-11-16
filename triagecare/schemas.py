"""Input validation for forms and the JSON API."""
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from triagecare.models import ArrivalMode, Role, Tier
from triagecare.services.triage import COMORBIDITIES, SPECIALTIES


def _blank_to_none(v):
    if isinstance(v, str) and not v.strip():
        return None
    return v.strip() if isinstance(v, str) else v


class _Form(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class PatientIn(_Form):
    first_name: str = Field(min_length=1, max_length=60)
    last_name: str = Field(min_length=1, max_length=60)
    date_of_birth: date
    gender: str = Field(pattern="^(Male|Female|Other)$")
    phone: str | None = Field(default=None, max_length=30)
    cnic: str | None = Field(default=None, pattern=r"^\d{5}-?\d{7}-?\d$")
    address: str | None = Field(default=None, max_length=255)
    comorbidity: str = "None"
    allergies: str | None = Field(default=None, max_length=255)
    emergency_contact: str | None = Field(default=None, max_length=120)
    notes: str | None = None

    _blanks = field_validator(
        "phone", "cnic", "address", "allergies", "emergency_contact", "notes", mode="before"
    )(_blank_to_none)

    @field_validator("date_of_birth")
    @classmethod
    def not_in_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError("Date of birth can't be in the future")
        if v.year < 1900:
            raise ValueError("Please check the year")
        return v

    @field_validator("comorbidity")
    @classmethod
    def known_comorbidity(cls, v: str) -> str:
        if v not in COMORBIDITIES:
            raise ValueError("Unknown comorbidity")
        return v


class VitalsIn(_Form):
    heart_rate: float = Field(ge=20, le=250)
    resp_rate: float = Field(ge=4, le=60)
    systolic_bp: float | None = Field(default=None, ge=40, le=280)
    diastolic_bp: float | None = Field(default=None, ge=20, le=180)
    spo2: float = Field(ge=50, le=100)
    temp_c: float = Field(ge=30, le=44)
    blood_glucose: float | None = Field(default=None, ge=15, le=800)
    gcs_score: int = Field(ge=3, le=15)
    pain_score: int = Field(ge=0, le=10)
    notes: str | None = Field(default=None, max_length=1000)

    _blanks = field_validator("systolic_bp", "diastolic_bp", "blood_glucose", "notes", mode="before")(
        _blank_to_none
    )


class PredictIn(VitalsIn):
    """Stateless prediction request for the API."""

    age: int = Field(ge=0, le=120)
    gender: str = Field(pattern="^(Male|Female|Other)$")
    comorbidity: str = "None"


class VisitIn(_Form):
    patient_id: int
    chief_complaint: str = Field(min_length=2, max_length=255)
    arrival_mode: ArrivalMode = ArrivalMode.WALK_IN


class AppointmentIn(_Form):
    patient_id: int
    scheduled_at: datetime
    reason: str = Field(min_length=2, max_length=255)
    doctor_id: int | None = None

    _blanks = field_validator("doctor_id", mode="before")(_blank_to_none)


class OverrideIn(_Form):
    tier: Tier
    reason: str = Field(min_length=3, max_length=255)


class CompleteIn(_Form):
    disposition: str = Field(pattern="^(discharged|admitted|referred|left_without_being_seen)$")
    doctor_notes: str | None = Field(default=None, max_length=4000)


class UserIn(_Form):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-z0-9_.]+$")
    full_name: str = Field(min_length=2, max_length=120)
    role: Role
    specialty: str | None = None
    password: str = Field(min_length=8, max_length=128)

    _blanks = field_validator("specialty", mode="before")(_blank_to_none)

    @field_validator("specialty")
    @classmethod
    def known_specialty(cls, v):
        if v is not None and v not in SPECIALTIES:
            raise ValueError("Unknown specialty")
        return v


def parse_form(model: type[BaseModel], data) -> tuple[BaseModel | None, dict[str, str]]:
    """Validate form data. Returns (model, {}) or (None, {field: message})."""
    values = {k: v for k, v in dict(data).items() if k != "csrf_token"}
    try:
        return model.model_validate(values), {}
    except ValidationError as exc:
        errors: dict[str, str] = {}
        for err in exc.errors():
            field = str(err["loc"][0]) if err["loc"] else "__all__"
            msg = err["msg"].removeprefix("Value error, ")
            if err["type"] in ("missing", "string_too_short") and not values.get(field):
                msg = "Required"
            elif err["type"] == "string_pattern_mismatch":
                msg = "Invalid format"
            errors.setdefault(field, msg)
        return None, errors
