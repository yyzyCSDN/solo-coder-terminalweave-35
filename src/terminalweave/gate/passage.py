from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, NotFoundError
from terminalweave.gate.appointments import GateBook
from terminalweave.gate.lane_control import LaneController
from terminalweave.gate.exceptions import GateExceptionDesk, GateException
from terminalweave.gate.ocr import ContainerOcrResolver, OcrCandidate, OcrDecision
from terminalweave.gate.security import AccessController
from terminalweave.gate.token import GateTokenIssuer, GateToken
from terminalweave.gate.truck_visit import TruckVisit, TruckVisitFlow, VisitState
from terminalweave.gate.weighbridge import WeighbridgeBook

ENTRY_PERMISSION = 'gate_entry'
EXIT_PERMISSION = 'gate_exit'


class PassageResult(str, Enum):
    ADMITTED = 'admitted'
    RELEASED = 'released'
    EXCEPTION = 'exception'
    REPLAY = 'replay'


@dataclass(frozen=True)
class PassageRequest:
    """Everything observed at the booth for one passage attempt.

    A missing reading is ``None``; the decision fails closed rather than
    treating absence as consent.
    """
    visit_id: str
    direction: str
    at: datetime
    credential_id: str | None
    token: GateToken | None
    ocr_candidates: tuple[OcrCandidate, ...] = ()
    expected_container_id: str | None = None
    weight_ticket_id: str | None = None
    declared_weight_kg: Decimal | None = None
    lane_id: str | None = None
    permission: str | None = None

    def __post_init__(self):
        if self.direction not in {'entry', 'exit'}:
            raise ValidationError('invalid passage direction')
        object.__setattr__(self, 'at', ensure_utc(self.at))
        if self.direction == 'entry' and not self.lane_id:
            raise ValidationError('entry passage requires a lane')


@dataclass(frozen=True)
class PassageDecision:
    visit_id: str
    direction: str
    result: PassageResult
    at: datetime
    lane_id: str | None
    container_id: str | None
    reasons: tuple[str, ...] = ()
    exception_id: str | None = None
    replayed: bool = False
    evidence: 'PassageEvidence | None' = None


@dataclass(frozen=True)
class PassageEvidence:
    """Snapshot of what the decision was based on, kept after release."""
    visit_id: str
    direction: str
    decided_at: datetime
    result: PassageResult
    visit_state: VisitState
    lane_id: str | None
    credential_id: str | None
    token_id: str | None
    container_id: str | None
    ocr: OcrDecision | None
    weight_ticket_id: str | None
    weight_check: dict[str, object] | None
    reasons: tuple[str, ...] = ()
    exception_id: str | None = None


class GatePassageDirector:
    """Unified gate decision over identification, documents, token, weight,
    visit state, access status and lane control.

    All four facts (identity, container number, weight, access status) must
    agree. When the evidence is insufficient the attempt is routed to the
    exception desk instead of being released.
    """

    def __init__(self, *, ocr: ContainerOcrResolver, access: AccessController,
                 tokens: GateTokenIssuer, weighbridge: WeighbridgeBook,
                 flow: TruckVisitFlow, lanes: LaneController,
                 exceptions: GateExceptionDesk, visits: list[TruckVisit] | None = None,
                 appointments: GateBook | None = None,
                 weight_tolerance_kg: Decimal = Decimal('200'),
                 ocr_minimum_confidence: float = 0.92):
        if weight_tolerance_kg < 0:
            raise ValidationError('weight tolerance cannot be negative')
        self.ocr = ocr
        self.access = access
        self.tokens = tokens
        self.weighbridge = weighbridge
        self.flow = flow
        self.lanes = lanes
        self.exceptions = exceptions
        self.appointments = appointments
        self.weight_tolerance_kg = weight_tolerance_kg
        self.ocr_minimum_confidence = ocr_minimum_confidence
        self.visits: dict[str, TruckVisit] = {}
        self._lane_of_visit: dict[str, str] = {}
        self._seq = 0
        self.history: list[PassageDecision] = []
        for visit in visits or []:
            self.register(visit)

    # -- registration / lookup -------------------------------------------------

    def register(self, visit: TruckVisit) -> TruckVisit:
        if visit.visit_id in self.visits:
            raise ValidationError('duplicate visit registration')
        self.visits[visit.visit_id] = visit
        return visit

    def _visit(self, visit_id: str) -> TruckVisit:
        visit = self.visits.get(visit_id)
        if visit is None:
            raise NotFoundError(f'unknown visit {visit_id}')
        return visit

    # -- entry / exit -----------------------------------------------------------

    def decide(self, req: PassageRequest) -> PassageDecision:
        visit = self._visit(req.visit_id)

        if req.direction == 'entry':
            if visit.state == VisitState.INSIDE:
                # Duplicate arrival must never occupy the lane a second time.
                return self._replay(req, visit, self._lane_of_visit.get(visit.visit_id))
            if visit.state in {VisitState.RELEASED, VisitState.EXITED}:
                return self._exception(req, visit, ('visit already passed entry',), None, None)
            return self._enter(req, visit)

        # exit
        if visit.state == VisitState.EXITED:
            lane_id = self._lane_of_visit.pop(visit.visit_id, None)
            return self._replay(req, visit, lane_id)
        if visit.state not in {VisitState.INSIDE, VisitState.RELEASED}:
            return self._exception(req, visit, (f'visit is not ready for exit (state={visit.state.value})',),
                                   None, None)
        return self._exit(req, visit)

    def _enter(self, req: PassageRequest, visit: TruckVisit) -> PassageDecision:
        container, ocr_decision, weight_check, reasons = self._evaluate(
            req, visit, ENTRY_PERMISSION)
        if visit.state not in {VisitState.EXPECTED, VisitState.ARRIVED}:
            reasons = (*reasons, f'visit is not ready for entry (state={visit.state.value})')
        if reasons:
            return self._exception(req, visit, reasons, ocr_decision, weight_check,
                                   container_id=container)

        # All evidence agrees: lane is acquired before the visit advances, so a
        # lane failure leaves the visit retryable rather than half-transitioned.
        self.lanes.enter(req.lane_id, 'entry')
        if visit.state == VisitState.EXPECTED:
            self.flow.transition(visit, VisitState.ARRIVED, req.at)
        self.flow.transition(visit, VisitState.INSIDE, req.at)
        if visit.appointment_id and self.appointments is not None:
            appointment = self.appointments._items.get(visit.appointment_id)
            if appointment is not None and appointment.state.value == 'booked':
                self.appointments.check_in(visit.appointment_id, req.at)
        self._lane_of_visit[visit.visit_id] = req.lane_id
        return self._record(req, visit, PassageResult.ADMITTED, req.lane_id, container,
                            ocr_decision, weight_check, ())

    def _exit(self, req: PassageRequest, visit: TruckVisit) -> PassageDecision:
        lane_id = req.lane_id or self._lane_of_visit.get(visit.visit_id)
        container, ocr_decision, weight_check, reasons = self._evaluate(
            req, visit, EXIT_PERMISSION)
        if reasons:
            return self._exception(req, visit, reasons, ocr_decision, weight_check,
                                   lane_id=lane_id, container_id=container)

        occupied = self._lane_of_visit.get(visit.visit_id)
        target = occupied or lane_id
        if target:
            self.lanes.leave(target)
        self._lane_of_visit.pop(visit.visit_id, None)
        if visit.state == VisitState.INSIDE:
            self.flow.transition(visit, VisitState.RELEASED, req.at)
        self.flow.transition(visit, VisitState.EXITED, req.at)
        if visit.appointment_id and self.appointments is not None:
            appointment = self.appointments._items.get(visit.appointment_id)
            if appointment is not None and appointment.state.value == 'checked_in':
                self.appointments.complete(visit.appointment_id, req.at)
        # Evidence is retained on the visit even though the lane is released.
        return self._record(req, visit, PassageResult.RELEASED, occupied, container,
                            ocr_decision, weight_check, ())

    # -- shared evaluation ------------------------------------------------------

    def _evaluate(self, req: PassageRequest, visit: TruckVisit,
                  default_permission: str
                  ) -> tuple[str | None, OcrDecision | None, dict[str, object] | None, tuple[str, ...]]:
        reasons: list[str] = []
        permission = req.permission or default_permission

        # Identity: credential, holder, token and visit must all agree.
        if not req.credential_id:
            reasons.append('credential missing')
        elif not self.access.authorize(req.credential_id, permission, req.at):
            reasons.append('credential not authorized')
        elif visit.driver_id is not None:
            credential = self.access.credentials.get(req.credential_id)
            if credential is not None and credential.holder != visit.driver_id:
                reasons.append('credential holder does not match visit driver')

        if req.token is None:
            reasons.append('token missing')
        elif not self.tokens.verify(req.token, req.at):
            reasons.append('token invalid or expired')
        elif req.token.visit_id != visit.visit_id:
            reasons.append('token does not belong to this visit')

        # Container number: OCR must be certain and match the expected number.
        container = None
        ocr_decision: OcrDecision | None = None
        if not req.ocr_candidates:
            reasons.append('container reading missing')
        else:
            ocr_decision = self.ocr.resolve(list(req.ocr_candidates),
                                           minimum_confidence=self.ocr_minimum_confidence)
            if ocr_decision.manual_review or ocr_decision.value is None:
                reasons.append(f'container reading unreliable: {ocr_decision.reason}')
            else:
                container = ocr_decision.value
                if req.expected_container_id and container != req.expected_container_id.upper():
                    reasons.append('container number does not match visit')

        # Weight: ticket truck must agree, declared net must be within tolerance.
        weight_check = None
        if not req.weight_ticket_id:
            reasons.append('weight ticket missing')
        elif req.declared_weight_kg is None:
            reasons.append('declared weight missing')
        else:
            ticket = self.weighbridge.tickets.get(req.weight_ticket_id)
            if ticket is None:
                reasons.append('weight ticket unknown')
            else:
                if ticket.truck_id != visit.truck_id:
                    reasons.append('weight ticket belongs to another truck')
                if container is not None and ticket.container_id \
                        and ticket.container_id != container:
                    reasons.append('weight ticket container does not match reading')
                weight_check = self.weighbridge.verify_container(
                    req.weight_ticket_id, req.declared_weight_kg, self.weight_tolerance_kg)
                if not weight_check['within_tolerance']:
                    reasons.append('weight exceeds declared tolerance')

        # Access status: an appointment linked to the visit must agree with it.
        if visit.appointment_id:
            if self.appointments is None:
                reasons.append('appointment cannot be verified')
            else:
                appointment = self.appointments._items.get(visit.appointment_id)
                if appointment is None:
                    reasons.append('appointment unknown')
                elif appointment.truck_id != visit.truck_id:
                    reasons.append('appointment belongs to another truck')
                elif req.direction == 'entry' and appointment.state.value not in {'booked', 'checked_in'}:
                    reasons.append(f'appointment not usable (state={appointment.state.value})')
                elif req.direction == 'exit' and appointment.state.value != 'checked_in':
                    reasons.append(f'appointment not checked in (state={appointment.state.value})')

        return container, ocr_decision, weight_check, tuple(reasons)

    # -- outcomes ---------------------------------------------------------------

    def _exception(self, req: PassageRequest, visit: TruckVisit,
                   reasons: tuple[str, ...], ocr_decision: OcrDecision | None,
                   weight_check: dict[str, object] | None, *,
                   lane_id: str | None = None, container_id: str | None = None) -> PassageDecision:
        for code in reasons:
            self.flow.add_exception(visit, code)
        exception_id = None
        merged = '; '.join(reasons)
        open_codes = {x.code for x in self.exceptions.open_for_truck(visit.truck_id)}
        if merged not in open_codes:
            self._seq += 1
            exception_id = f'GX-{visit.visit_id}-{req.direction}-{self._seq}'
            self.exceptions.open(GateException(
                exception_id=exception_id,
                truck_id=visit.truck_id,
                code=merged,
                opened_at=req.at,
            ))
        return self._record(req, visit, PassageResult.EXCEPTION, lane_id, container_id,
                            ocr_decision, weight_check, reasons, exception_id=exception_id)

    def _replay(self, req: PassageRequest, visit: TruckVisit, lane_id: str | None) -> PassageDecision:
        decision = PassageDecision(
            visit_id=visit.visit_id, direction=req.direction, result=PassageResult.REPLAY,
            at=req.at, lane_id=lane_id, container_id=None, replayed=True,
            reasons=('duplicate passage attempt ignored',))
        self.history.append(decision)
        return decision

    def _record(self, req: PassageRequest, visit: TruckVisit, result: PassageResult,
                lane_id: str | None, container_id: str | None,
                ocr_decision: OcrDecision | None, weight_check: dict[str, object] | None,
                reasons: tuple[str, ...], *, exception_id: str | None = None) -> PassageDecision:
        evidence = PassageEvidence(
            visit_id=visit.visit_id, direction=req.direction, decided_at=req.at,
            result=result, visit_state=visit.state, lane_id=lane_id,
            credential_id=req.credential_id, token_id=req.token.token_id if req.token else None,
            container_id=container_id, ocr=ocr_decision,
            weight_ticket_id=req.weight_ticket_id, weight_check=weight_check,
            reasons=reasons, exception_id=exception_id,
        )
        decision = PassageDecision(
            visit_id=visit.visit_id, direction=req.direction, result=result, at=req.at,
            lane_id=lane_id, container_id=container_id, reasons=reasons,
            exception_id=exception_id, evidence=evidence,
        )
        self.history.append(decision)
        return decision

    # -- queries ----------------------------------------------------------------

    def evidence_for(self, visit_id: str) -> tuple[PassageEvidence, ...]:
        return tuple(d.evidence for d in self.history
                     if d.visit_id == visit_id and d.evidence is not None)

    def active_visits(self) -> tuple[str, ...]:
        return tuple(vid for vid in self._lane_of_visit
                     if self.visits[vid].state != VisitState.EXITED)
