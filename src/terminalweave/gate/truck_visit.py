from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, StateTransitionError

class VisitState(str, Enum):
    EXPECTED = 'expected'
    ARRIVED = 'arrived'
    INSIDE = 'inside'
    RELEASED = 'released'
    EXITED = 'exited'
    REJECTED = 'rejected'

@dataclass
class TruckVisit:
    visit_id: str
    truck_id: str
    appointment_id: str | None
    driver_id: str | None = None
    state: VisitState = VisitState.EXPECTED
    timestamps: dict[str, datetime] = field(default_factory=dict)
    exceptions: list[str] = field(default_factory=list)

class TruckVisitFlow:
    sequence = [VisitState.EXPECTED, VisitState.ARRIVED, VisitState.INSIDE, VisitState.RELEASED, VisitState.EXITED]

    def transition(self, visit: TruckVisit, target: VisitState, at: datetime) -> TruckVisit:
        at = ensure_utc(at)
        if target == VisitState.REJECTED:
            if visit.state in {VisitState.RELEASED, VisitState.EXITED}:
                raise StateTransitionError('released visit cannot be rejected')
            visit.state = target
            visit.timestamps[target.value] = at
            return visit
        if visit.state == VisitState.REJECTED:
            raise StateTransitionError('rejected visit cannot continue')
        current = self.sequence.index(visit.state)
        expected = self.sequence[current + 1] if current + 1 < len(self.sequence) else None
        if target != expected:
            raise StateTransitionError(f'expected transition to {expected}')
        if visit.timestamps and at < max(visit.timestamps.values()):
            raise ValidationError('visit time cannot move backwards')
        visit.state = target
        visit.timestamps[target.value] = at
        return visit

    def turn_time_minutes(self, visit: TruckVisit) -> float | None:
        arrived = visit.timestamps.get('arrived')
        exited = visit.timestamps.get('exited')
        return (exited - arrived).total_seconds() / 60 if arrived and exited else None

    def add_exception(self, visit: TruckVisit, code: str) -> None:
        if not code.strip():
            raise ValidationError('exception code required')
        if code not in visit.exceptions:
            visit.exceptions.append(code)
