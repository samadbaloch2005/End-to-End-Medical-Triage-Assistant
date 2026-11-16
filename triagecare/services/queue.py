"""The department queue and the numbers on the dashboard."""
from collections import Counter
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from triagecare.models import (
    Appointment,
    AppointmentStatus,
    ArrivalMode,
    Tier,
    Visit,
    VisitStatus,
    Ward,
)
from triagecare.services.triage import TARGET_MINUTES, ward_occupancy
from triagecare.timeutil import local_day_bounds_utc, local_today, minutes_since, to_local, utcnow

TIER_ORDER = {Tier.HIGH: 0, Tier.MEDIUM: 1, Tier.LOW: 2, None: 3}


@dataclass
class QueueEntry:
    visit: Visit
    position: int
    waited: int        # minutes since arrival
    target: int | None
    overdue: bool

    @property
    def minutes_left(self) -> int | None:
        if self.target is None:
            return None
        return self.target - self.waited


def _active_visits(db: Session) -> list[Visit]:
    return list(
        db.scalars(
            select(Visit)
            .where(Visit.status != VisitStatus.COMPLETED)
            .options(
                selectinload(Visit.patient),
                selectinload(Visit.ward),
                selectinload(Visit.doctor),
                selectinload(Visit.assessments),
            )
        )
    )


def build_queue(db: Session, doctor_id: int | None = None) -> dict[str, list[QueueEntry]]:
    """Active visits in three lanes, each in the order patients should be seen."""
    now = utcnow()
    visits = _active_visits(db)
    if doctor_id is not None:
        visits = [v for v in visits if v.doctor_id == doctor_id]

    awaiting = sorted(
        (v for v in visits if v.status == VisitStatus.WAITING),
        key=lambda v: (v.arrival_mode != ArrivalMode.AMBULANCE, v.arrived_at),
    )
    triaged = sorted(
        (v for v in visits if v.status == VisitStatus.TRIAGED),
        key=lambda v: (TIER_ORDER.get(v.tier, 3), v.arrived_at),
    )
    treating = sorted(
        (v for v in visits if v.status == VisitStatus.IN_TREATMENT),
        key=lambda v: (TIER_ORDER.get(v.tier, 3), v.seen_at or v.arrived_at),
    )

    def entries(items, with_target):
        out = []
        for i, v in enumerate(items, start=1):
            waited = minutes_since(v.arrived_at, now)
            target = TARGET_MINUTES.get(Tier(v.tier)) if with_target and v.tier else None
            out.append(QueueEntry(v, i, waited, target, target is not None and waited > target))
        return out

    return {
        "awaiting": entries(awaiting, False),
        "triaged": entries(triaged, True),
        "treating": entries(treating, False),
    }


def ward_board(db: Session) -> list[dict]:
    occupancy = ward_occupancy(db)
    order = {"resus": 0, "acute": 1, "urgent": 2, "general": 3}
    wards = sorted(db.scalars(select(Ward)), key=lambda w: order.get(w.level, 9))
    board = []
    for w in wards:
        used = occupancy.get(w.id, 0)
        board.append(
            {
                "ward": w,
                "occupied": used,
                "free": max(0, w.capacity - used),
                "percent": min(100, round(100 * used / w.capacity)) if w.capacity else 0,
            }
        )
    return board


def todays_appointments(db: Session, day: date | None = None) -> list[Appointment]:
    start, end = local_day_bounds_utc(day or local_today())
    return list(
        db.scalars(
            select(Appointment)
            .where(Appointment.scheduled_at >= start, Appointment.scheduled_at < end)
            .options(selectinload(Appointment.patient), selectinload(Appointment.doctor))
            .order_by(Appointment.scheduled_at)
        )
    )


def dashboard_stats(db: Session) -> dict:
    start, end = local_day_bounds_utc(local_today())
    today = list(
        db.scalars(
            select(Visit).where(Visit.arrived_at >= start, Visit.arrived_at < end)
        )
    )
    active = _active_visits(db)

    by_tier = Counter(v.tier for v in today if v.tier)
    waits = [minutes_since(v.arrived_at, v.seen_at) for v in today if v.seen_at]

    hourly = [0] * 24
    for v in today:
        hourly[to_local(v.arrived_at).hour] += 1

    queue = build_queue(db)
    overdue = sum(1 for e in queue["triaged"] if e.overdue)

    appointments = todays_appointments(db)
    return {
        "arrivals_today": len(today),
        "in_department": len(active),
        "awaiting_triage": len(queue["awaiting"]),
        "waiting_for_doctor": len(queue["triaged"]),
        "overdue": overdue,
        "avg_wait": round(sum(waits) / len(waits)) if waits else None,
        "by_tier": {t.value: by_tier.get(t.value, 0) for t in (Tier.HIGH, Tier.MEDIUM, Tier.LOW)},
        "hourly": hourly,
        "appointments_today": len(appointments),
        "appointments_pending": sum(
            1 for a in appointments if a.status == AppointmentStatus.BOOKED
        ),
        "completed_today": sum(1 for v in today if v.status == VisitStatus.COMPLETED),
    }
