from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import StateTransitionError
from terminalweave.gate.appointments import Appointment, AppointmentState
from terminalweave.gate.exceptions import GateException, GateExceptionDesk
from terminalweave.gate.lane_control import LaneController, LaneMode
from terminalweave.gate.ocr import ContainerOcrResolver, OcrCandidate, OcrDecision
from terminalweave.gate.security import AccessController, Credential
from terminalweave.gate.token import GateToken, GateTokenIssuer
from terminalweave.gate.truck_visit import TruckVisit, TruckVisitFlow, VisitState
from terminalweave.gate.weighbridge import WeighbridgeBook, WeightTicket


class PassageDirection(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


class PassageStatus(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class PassageEvidence:
    """Evidence captured at the moment a physical passage was commanded."""

    captured_at: datetime
    visit_state: VisitState
    direction: PassageDirection
    lane_id: str
    credential_id: str
    credential_holder: str
    token_id: str
    container_id: str
    weight_ticket_id: str
    ocr_confidence: float
    gross_kg: Decimal
    tare_kg: Decimal
    net_kg: Decimal
    declared_kg: Decimal
    delta_kg: Decimal
    tolerance_kg: Decimal
    permission: str
    appointment_id: str | None = None


@dataclass(frozen=True)
class PassageDecision:
    status: PassageStatus
    direction: PassageDirection
    visit_id: str
    truck_id: str
    at: datetime
    reasons: tuple[str, ...] = ()
    lane_id: str | None = None
    evidence: PassageEvidence | None = None
    idempotent: bool = False

    @property
    def allowed(self) -> bool:
        return self.status is PassageStatus.ALLOWED

    @property
    def denied(self) -> bool:
        return self.status is PassageStatus.DENIED

    @property
    def duplicate(self) -> bool:
        return self.status is PassageStatus.DUPLICATE


@dataclass
class PassageRecord:
    visit_id: str
    entry: PassageDecision | None = None
    exit: PassageDecision | None = None

    @property
    def evidence(self) -> PassageEvidence | None:
        return self.entry.evidence if self.entry is not None else None


class GatePassageCoordinator:
    """Make the gate's evidence checks and lane actuation one fail-closed decision."""

    ENTRY_PERMISSION = "gate.entry"
    EXIT_PERMISSION = "gate.exit"

    def __init__(
        self,
        *,
        access: AccessController,
        tokens: GateTokenIssuer,
        weighbridge: WeighbridgeBook,
        lanes: LaneController,
        exceptions: GateExceptionDesk | None = None,
        visits: TruckVisitFlow | None = None,
        ocr: ContainerOcrResolver | None = None,
    ):
        self.access = access
        self.tokens = tokens
        self.weighbridge = weighbridge
        self.lanes = lanes
        self.exceptions = exceptions or GateExceptionDesk()
        self.visits = visits or TruckVisitFlow()
        self.ocr = ocr or ContainerOcrResolver()
        self.records: dict[str, PassageRecord] = {}
        self.decisions: list[PassageDecision] = []
        self._exception_sequence = 0

    def request_entry(
        self,
        *,
        visit: TruckVisit,
        credential: Credential | str,
        token: GateToken,
        ocr: OcrDecision | list[OcrCandidate],
        ticket: WeightTicket | str,
        declared_kg: Decimal | str | int,
        lane_id: str,
        at: datetime,
        tolerance_kg: Decimal | str | int = Decimal(0),
        appointment: Appointment | None = None,
        expected_container_id: str | None = None,
        permission: str | None = None,
    ) -> PassageDecision:
        return self._request(
            direction=PassageDirection.ENTRY,
            visit=visit,
            credential=credential,
            token=token,
            ocr=ocr,
            ticket=ticket,
            declared_kg=declared_kg,
            lane_id=lane_id,
            at=at,
            tolerance_kg=tolerance_kg,
            appointment=appointment,
            expected_container_id=expected_container_id,
            permission=permission or self.ENTRY_PERMISSION,
        )

    def request_exit(
        self,
        *,
        visit: TruckVisit,
        credential: Credential | str,
        token: GateToken,
        ocr: OcrDecision | list[OcrCandidate],
        ticket: WeightTicket | str,
        declared_kg: Decimal | str | int,
        lane_id: str,
        at: datetime,
        tolerance_kg: Decimal | str | int = Decimal(0),
        expected_container_id: str | None = None,
        permission: str | None = None,
    ) -> PassageDecision:
        return self._request(
            direction=PassageDirection.EXIT,
            visit=visit,
            credential=credential,
            token=token,
            ocr=ocr,
            ticket=ticket,
            declared_kg=declared_kg,
            lane_id=lane_id,
            at=at,
            tolerance_kg=tolerance_kg,
            appointment=None,
            expected_container_id=expected_container_id,
            permission=permission or self.EXIT_PERMISSION,
        )

    def enter_yard(self, visit: TruckVisit, at: datetime) -> TruckVisit:
        record = self.records.get(visit.visit_id)
        if record is None or record.entry is None or record.entry.lane_id is None:
            raise StateTransitionError("visit has no active gate passage")
        self.lanes.leave(record.entry.lane_id)
        return self.visits.transition(visit, VisitState.INSIDE, at)

    def release_visit(self, visit: TruckVisit, at: datetime) -> TruckVisit:
        return self.visits.transition(visit, VisitState.RELEASED, at)

    def passage_for(self, visit_id: str) -> PassageRecord:
        try:
            return self.records[visit_id]
        except KeyError:
            raise StateTransitionError("no passage recorded for visit") from None

    def evidence_for(self, visit_id: str) -> PassageEvidence:
        record = self.passage_for(visit_id)
        if record.evidence is None:
            raise StateTransitionError("no passage evidence for visit")
        return record.evidence

    def _request(
        self,
        *,
        direction: PassageDirection,
        visit: TruckVisit,
        credential: Credential | str,
        token: GateToken,
        ocr: OcrDecision | list[OcrCandidate],
        ticket: WeightTicket | str,
        declared_kg: Decimal | str | int,
        lane_id: str,
        at: datetime,
        tolerance_kg: Decimal | str | int,
        appointment: Appointment | None,
        expected_container_id: str | None,
        permission: str,
    ) -> PassageDecision:
        at = ensure_utc(at)
        record = self.records.get(visit.visit_id)
        duplicate = self._duplicate_decision(direction, visit, record, at)
        if duplicate is not None:
            self.decisions.append(duplicate)
            return duplicate

        reasons: list[str] = []
        expected_state, target_state = self._states_for(direction)
        if record is None and direction is PassageDirection.EXIT:
            reasons.append("missing_entry_evidence")
        if visit.state is not expected_state:
            reasons.append("unexpected_visit_state")
        if visit.timestamps and at < max(visit.timestamps.values()):
            reasons.append("visit_time_moving_backwards")

        credential_id, credential_record = self._credential(credential)
        if credential_record is None:
            reasons.append("missing_credential")
        else:
            if credential_record.holder != visit.truck_id:
                reasons.append("identity_mismatch")
            if not self.access.authorize(credential_id, permission, at):
                reasons.append("access_denied")

        token_ok = isinstance(token, GateToken) and self.tokens.verify(token, at)
        if not isinstance(token, GateToken):
            reasons.append("missing_token")
        elif not token_ok:
            reasons.append("invalid_token")
        if not isinstance(token, GateToken) or token.visit_id != visit.visit_id:
            reasons.append("token_visit_mismatch")

        ocr_decision = self._ocr_decision(ocr, reasons)
        weight_ticket = self._weight_ticket(ticket, reasons)
        declared = self._decimal(declared_kg, "declared_weight", reasons)
        tolerance = self._decimal(tolerance_kg, "weight_tolerance", reasons, allow_zero=True)

        container_id: str | None = None
        if ocr_decision is not None:
            if ocr_decision.manual_review or not ocr_decision.value:
                reasons.append("incomplete_container_recognition")
            else:
                container_id = ocr_decision.value

        if weight_ticket is not None:
            if weight_ticket.truck_id != visit.truck_id:
                reasons.append("identity_mismatch")
            if not weight_ticket.container_id:
                reasons.append("missing_container_on_weight_ticket")
            elif container_id and weight_ticket.container_id != container_id:
                reasons.append("container_mismatch")
            elif weight_ticket.container_id:
                container_id = weight_ticket.container_id

        if expected_container_id is not None:
            expected = str(expected_container_id).upper()
            if not container_id or container_id != expected:
                reasons.append("container_mismatch")

        delta: Decimal | None = None
        if weight_ticket is not None and declared is not None:
            delta = weight_ticket.net_kg - declared
            if tolerance is None or abs(delta) > tolerance:
                reasons.append("weight_mismatch")

        if direction is PassageDirection.ENTRY:
            self._check_appointment(visit, appointment, reasons)

        lane_reason = self._lane_reason(lane_id, direction)
        if lane_reason:
            reasons.append(lane_reason)

        if reasons:
            decision = PassageDecision(
                PassageStatus.DENIED,
                direction,
                visit.visit_id,
                visit.truck_id,
                at,
                tuple(reasons),
                lane_id=lane_id,
            )
            self._open_exception(visit, direction, at, reasons)
            self.decisions.append(decision)
            return decision

        # All evidence has been validated before either the lane or visit state changes.
        assert credential_record is not None
        assert isinstance(token, GateToken)
        assert ocr_decision is not None and ocr_decision.value is not None
        assert weight_ticket is not None
        assert declared is not None
        assert tolerance is not None
        assert delta is not None
        assert container_id is not None

        evidence = PassageEvidence(
            captured_at=at,
            visit_state=visit.state,
            direction=direction,
            lane_id=lane_id,
            credential_id=credential_id,
            credential_holder=credential_record.holder,
            token_id=token.token_id,
            container_id=container_id,
            weight_ticket_id=weight_ticket.ticket_id,
            ocr_confidence=ocr_decision.confidence,
            gross_kg=weight_ticket.gross_kg,
            tare_kg=weight_ticket.tare_kg,
            net_kg=weight_ticket.net_kg,
            declared_kg=declared,
            delta_kg=delta,
            tolerance_kg=tolerance,
            permission=permission,
            appointment_id=visit.appointment_id,
        )

        self.lanes.enter(lane_id, direction.value)
        self.visits.transition(visit, target_state, at)
        if direction is PassageDirection.EXIT:
            self.lanes.leave(lane_id)

        decision = PassageDecision(
            PassageStatus.ALLOWED,
            direction,
            visit.visit_id,
            visit.truck_id,
            at,
            ("accepted",),
            lane_id=lane_id,
            evidence=evidence,
        )
        record = self.records.setdefault(visit.visit_id, PassageRecord(visit.visit_id))
        if direction is PassageDirection.ENTRY:
            record.entry = decision
        else:
            record.exit = decision
        self.decisions.append(decision)
        return decision

    def _duplicate_decision(
        self,
        direction: PassageDirection,
        visit: TruckVisit,
        record: PassageRecord | None,
        at: datetime,
    ) -> PassageDecision | None:
        if direction is PassageDirection.ENTRY:
            if record is not None and record.entry is not None:
                return PassageDecision(
                    PassageStatus.DUPLICATE,
                    direction,
                    visit.visit_id,
                    visit.truck_id,
                    at,
                    ("duplicate_arrival",),
                    lane_id=record.entry.lane_id,
                    evidence=record.entry.evidence,
                    idempotent=True,
                )
        elif record is not None and record.exit is not None:
            return PassageDecision(
                PassageStatus.DUPLICATE,
                direction,
                visit.visit_id,
                visit.truck_id,
                at,
                ("duplicate_exit",),
                lane_id=record.exit.lane_id,
                evidence=record.exit.evidence,
                idempotent=True,
            )
        return None

    @staticmethod
    def _states_for(direction: PassageDirection) -> tuple[VisitState, VisitState]:
        if direction is PassageDirection.ENTRY:
            return VisitState.EXPECTED, VisitState.ARRIVED
        return VisitState.RELEASED, VisitState.EXITED

    def _credential(self, value: Credential | str) -> tuple[str, Credential | None]:
        credential_id = value.credential_id if isinstance(value, Credential) else str(value)
        return credential_id, self.access.credentials.get(credential_id)

    def _ocr_decision(
        self,
        value: OcrDecision | list[OcrCandidate] | None,
        reasons: list[str],
    ) -> OcrDecision | None:
        if isinstance(value, OcrDecision):
            return value
        if isinstance(value, (list, tuple)):
            candidates = list(value)
            if not candidates:
                reasons.append("missing_container_recognition")
                return None
            if not all(isinstance(item, OcrCandidate) for item in candidates):
                reasons.append("invalid_container_recognition")
                return None
            return self.ocr.resolve(candidates)
        reasons.append("missing_container_recognition")
        return None

    def _weight_ticket(
        self,
        value: WeightTicket | str | None,
        reasons: list[str],
    ) -> WeightTicket | None:
        if isinstance(value, WeightTicket):
            return value
        if value is None:
            reasons.append("missing_weight_ticket")
            return None
        ticket = self.weighbridge.tickets.get(str(value))
        if ticket is None:
            reasons.append("missing_weight_ticket")
        return ticket

    @staticmethod
    def _decimal(
        value: Decimal | str | int | None,
        code: str,
        reasons: list[str],
        *,
        allow_zero: bool = False,
    ) -> Decimal | None:
        if value is None:
            reasons.append(code)
            return None
        try:
            result = Decimal(str(value))
        except (ArithmeticError, ValueError, TypeError):
            reasons.append(code)
            return None
        if not result.is_finite() or result < Decimal(0) or (not allow_zero and result <= 0):
            reasons.append(code)
            return None
        return result

    def _check_appointment(
        self,
        visit: TruckVisit,
        appointment: Appointment | None,
        reasons: list[str],
    ) -> None:
        if appointment is None:
            if visit.appointment_id is not None:
                reasons.append("missing_appointment")
            return
        if visit.appointment_id != appointment.appointment_id:
            reasons.append("appointment_mismatch")
        if appointment.truck_id != visit.truck_id:
            reasons.append("identity_mismatch")
        if appointment.state is not AppointmentState.CHECKED_IN:
            reasons.append("appointment_not_checked_in")

    def _lane_reason(self, lane_id: str, direction: PassageDirection) -> str | None:
        state = self.lanes.states.get(lane_id)
        if state is None:
            return "unknown_lane"
        if state.occupied:
            return "lane_occupied"
        if state.mode is LaneMode.CLOSED:
            return "lane_closed"
        if state.mode is not LaneMode.FLEX and state.mode is not LaneMode(direction.value):
            return "lane_direction_mismatch"
        return None

    def _open_exception(
        self,
        visit: TruckVisit,
        direction: PassageDirection,
        at: datetime,
        reasons: list[str],
    ) -> None:
        self._exception_sequence += 1
        code = reasons[0]
        exception = GateException(
            exception_id=f"{visit.visit_id}:{direction.value}:{self._exception_sequence}",
            truck_id=visit.truck_id,
            code=code,
            opened_at=at,
        )
        self.exceptions.open(exception)
        self.visits.add_exception(visit, code)


GatePassageService = GatePassageCoordinator
