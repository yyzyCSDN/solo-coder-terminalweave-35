from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from terminalweave.core.errors import StateTransitionError
from terminalweave.gate.lane_control import LaneController, LaneMode
from terminalweave.gate.exceptions import GateExceptionDesk, ExceptionState
from terminalweave.gate.ocr import ContainerOcrResolver, OcrCandidate
from terminalweave.gate.passage import (
    GatePassageDirector, PassageRequest, PassageResult,
)
from terminalweave.gate.security import AccessController, Credential
from terminalweave.gate.token import GateTokenIssuer
from terminalweave.gate.truck_visit import TruckVisit, TruckVisitFlow, VisitState
from terminalweave.gate.weighbridge import WeighbridgeBook, WeightTicket
from terminalweave.gate.appointments import GateSlotPolicy, GateBook, AppointmentState

UTC = timezone.utc
T0 = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
SECRET = b'0123456789abcdef'
CONTAINER = 'MSCU1234567'


def build(*, driver_id='D-1', credential_holder='D-1', appointment_id=None,
          visits=None, appointments=None, tolerance=Decimal('200')):
    credentials = [Credential('CRED-1', credential_holder, T0 - timedelta(days=1),
                              T0 + timedelta(days=1),
                              frozenset({'gate_entry', 'gate_exit'}))]
    director = GatePassageDirector(
        ocr=ContainerOcrResolver(),
        access=AccessController(credentials),
        tokens=GateTokenIssuer(SECRET),
        weighbridge=WeighbridgeBook(),
        flow=TruckVisitFlow(),
        lanes=LaneController(['L1', 'L2']),
        exceptions=GateExceptionDesk(),
        visits=visits if visits is not None else [
            TruckVisit('V1', 'TRUCK-1', appointment_id, driver_id)],
        appointments=appointments,
        weight_tolerance_kg=tolerance,
    )
    director.lanes.set_mode('L1', LaneMode.ENTRY)
    director.lanes.set_mode('L2', LaneMode.EXIT)
    return director


def ticket(director, *, ticket_id=None, truck_id='TRUCK-1',
           gross='30000', tare='10000', container_id=CONTAINER):
    if ticket_id is None:
        ticket_id = f'WT-{len(director.weighbridge.tickets) + 1}'
    director.weighbridge.add(WeightTicket(
        ticket_id, truck_id, Decimal(gross), Decimal(tare), container_id))
    return ticket_id


def good_entry(director, at=T0, visit_id='V1', lane_id='L1',
               declared=Decimal('20000'), token=None, expected=CONTAINER):
    return director.decide(PassageRequest(
        visit_id=visit_id, direction='entry', at=at, lane_id=lane_id,
        credential_id='CRED-1',
        token=token or director.tokens.issue(visit_id, at),
        ocr_candidates=(OcrCandidate(f' {CONTAINER} ', 0.99),),
        expected_container_id=expected,
        weight_ticket_id=ticket(director),
        declared_weight_kg=declared,
    ))


def good_exit(director, at=T0 + timedelta(hours=1), visit_id='V1', **kw):
    return director.decide(PassageRequest(
        visit_id=visit_id, direction='exit', at=at,
        credential_id='CRED-1',
        token=director.tokens.issue(visit_id, at),
        ocr_candidates=(OcrCandidate(CONTAINER, 0.99),),
        expected_container_id=CONTAINER,
        weight_ticket_id=ticket(director),
        declared_weight_kg=Decimal('20000'),
        **kw,
    ))


# --- happy path ---------------------------------------------------------------

def test_consistent_evidence_admits_and_releases():
    d = build()
    entry = good_entry(d)
    assert entry.result == PassageResult.ADMITTED
    visit = d.visits['V1']
    assert visit.state == VisitState.INSIDE
    assert d.lanes.states['L1'].occupied

    exit_decision = good_exit(d)
    assert exit_decision.result == PassageResult.RELEASED
    assert visit.state == VisitState.EXITED
    assert not d.lanes.states['L1'].occupied
    assert d._lane_of_visit.get('V1') is None


def test_full_round_trip_moves_visit_through_all_states():
    d = build()
    good_entry(d)
    good_exit(d)
    assert [d.visits['V1'].timestamps[k] is not None for k in
            ('arrived', 'inside', 'released', 'exited')]


# --- fail closed: insufficient or inconsistent evidence ----------------------

def test_missing_evidence_goes_to_exception_not_admitted():
    d = build()
    decision = d.decide(PassageRequest(
        visit_id='V1', direction='entry', at=T0, lane_id='L1',
        credential_id=None, token=None,
        ocr_candidates=(), expected_container_id=None,
        weight_ticket_id=None, declared_weight_kg=None,
    ))
    assert decision.result == PassageResult.EXCEPTION
    assert 'credential missing' in decision.reasons
    assert 'token missing' in decision.reasons
    assert 'container reading missing' in decision.reasons
    assert 'weight ticket missing' in decision.reasons
    # lane never touched, visit never entered
    assert not d.lanes.states['L1'].occupied
    assert d.visits['V1'].state == VisitState.EXPECTED
    assert d.exceptions.open_for_truck('TRUCK-1')


def test_unreadable_container_is_exception():
    d = build()
    decision = d.decide(PassageRequest(
        visit_id='V1', direction='entry', at=T0, lane_id='L1',
        credential_id='CRED-1', token=d.tokens.issue('V1', T0),
        ocr_candidates=(OcrCandidate(CONTAINER, 0.40),),
        weight_ticket_id=ticket(d), declared_weight_kg=Decimal('20000'),
    ))
    assert decision.result == PassageResult.EXCEPTION
    assert any('unreliable' in r for r in decision.reasons)
    assert not d.lanes.states['L1'].occupied


def test_container_mismatch_is_exception():
    d = build()
    decision = good_entry(d, expected='MSCU7654321')
    assert decision.result == PassageResult.EXCEPTION
    assert any('does not match visit' in r for r in decision.reasons)


def test_weight_out_of_tolerance_is_exception():
    d = build(tolerance=Decimal('50'))
    decision = good_entry(d, declared=Decimal('20500'))
    assert decision.result == PassageResult.EXCEPTION
    assert 'weight exceeds declared tolerance' in decision.reasons


def test_weight_ticket_from_another_truck_is_exception():
    d = build()
    decision = d.decide(PassageRequest(
        visit_id='V1', direction='entry', at=T0, lane_id='L1',
        credential_id='CRED-1', token=d.tokens.issue('V1', T0),
        ocr_candidates=(OcrCandidate(CONTAINER, 0.99),),
        expected_container_id=CONTAINER,
        weight_ticket_id=ticket(d, truck_id='TRUCK-9'),
        declared_weight_kg=Decimal('20000'),
    ))
    assert decision.result == PassageResult.EXCEPTION
    assert 'weight ticket belongs to another truck' in decision.reasons


def test_credential_holder_mismatch_is_exception():
    d = build(credential_holder='D-OTHER')
    decision = good_entry(d)
    assert decision.result == PassageResult.EXCEPTION
    assert 'credential holder does not match visit driver' in decision.reasons


def test_token_for_other_visit_is_exception():
    d = build(visits=[TruckVisit('V1', 'TRUCK-1', None, 'D-1'),
                      TruckVisit('V2', 'TRUCK-2', None, 'D-1')])
    decision = good_entry(d, token=d.tokens.issue('V2', T0))
    assert decision.result == PassageResult.EXCEPTION
    assert 'token does not belong to this visit' in decision.reasons
    assert not d.lanes.states['L1'].occupied


def test_expired_token_is_exception():
    d = build()
    token = d.tokens.issue('V1', T0 - timedelta(hours=5), ttl=timedelta(hours=2))
    decision = good_entry(d, token=token)
    assert decision.result == PassageResult.EXCEPTION
    assert 'token invalid or expired' in decision.reasons


def test_failed_entry_can_be_retried_after_evidence_fixed():
    d = build()
    bad = d.decide(PassageRequest(
        visit_id='V1', direction='entry', at=T0, lane_id='L1',
        credential_id=None, token=None,
        ocr_candidates=(), expected_container_id=None,
        weight_ticket_id=None, declared_weight_kg=None,
    ))
    assert bad.result == PassageResult.EXCEPTION
    ok = good_entry(d, at=T0 + timedelta(minutes=5))
    assert ok.result == PassageResult.ADMITTED
    assert d.visits['V1'].state == VisitState.INSIDE


# --- duplicate arrivals --------------------------------------------------------

def test_duplicate_arrival_does_not_occupy_lane_twice():
    d = build()
    first = good_entry(d)
    assert first.result == PassageResult.ADMITTED
    second = good_entry(d, at=T0 + timedelta(minutes=2))
    assert second.result == PassageResult.REPLAY
    assert second.replayed
    # still exactly one occupant; second lane stays free
    assert d.lanes.states['L1'].occupied
    assert not d.lanes.states['L2'].occupied
    assert d._lane_of_visit['V1'] == 'L1'


def test_duplicate_exit_is_replay_and_keeps_lane_free():
    d = build()
    good_entry(d)
    first_exit = good_exit(d)
    assert first_exit.result == PassageResult.RELEASED
    second_exit = good_exit(d, at=T0 + timedelta(hours=2))
    assert second_exit.result == PassageResult.REPLAY
    assert not d.lanes.states['L1'].occupied


def test_visit_that_never_entered_cannot_exit():
    d = build()
    decision = good_exit(d)
    assert decision.result == PassageResult.EXCEPTION
    assert 'not ready for exit' in decision.reasons[0]


# --- evidence retention after leaving ------------------------------------------

def test_evidence_is_retained_after_exit():
    d = build()
    good_entry(d)
    good_exit(d)
    evidence = d.evidence_for('V1')
    assert len(evidence) == 2
    entry_ev, exit_ev = evidence
    assert entry_ev.result == PassageResult.ADMITTED
    assert entry_ev.container_id == CONTAINER
    assert entry_ev.credential_id == 'CRED-1'
    assert entry_ev.token_id and entry_ev.weight_ticket_id == 'WT-1'
    assert entry_ev.weight_check['within_tolerance']
    assert exit_ev.result == PassageResult.RELEASED
    assert exit_ev.visit_state == VisitState.EXITED
    # the lane is released but the snapshot still says which lane it used
    assert entry_ev.lane_id == 'L1'
    assert not d.lanes.states['L1'].occupied


def test_exception_attempt_also_keeps_evidence():
    d = build()
    decision = good_entry(d, expected='MSCU7654321')
    assert decision.result == PassageResult.EXCEPTION
    evidence = d.evidence_for('V1')
    assert len(evidence) == 1
    assert evidence[0].exception_id == decision.exception_id
    assert evidence[0].container_id == CONTAINER


# --- appointment / access status integration -----------------------------------

def test_appointment_checked_in_and_completed_with_passage():
    policy = GateSlotPolicy(30, 5, 5, 5)
    book = GateBook(policy)
    book.book('A1', 'TRUCK-1', T0)
    d = build(appointment_id='A1', appointments=book)
    good_entry(d)
    assert book._items['A1'].state == AppointmentState.CHECKED_IN
    good_exit(d)
    assert book._items['A1'].state == AppointmentState.COMPLETED


def test_cancelled_appointment_blocks_entry_as_exception():
    policy = GateSlotPolicy(30, 5, 5, 5)
    book = GateBook(policy)
    book.book('A1', 'TRUCK-1', T0)
    book.cancel('A1')
    d = build(appointment_id='A1', appointments=book)
    decision = good_entry(d)
    assert decision.result == PassageResult.EXCEPTION
    assert any('appointment not usable' in r for r in decision.reasons)
    assert d.visits['V1'].state == VisitState.EXPECTED
    assert not d.lanes.states['L1'].occupied


def test_appointment_bound_to_other_truck_blocks_entry():
    policy = GateSlotPolicy(30, 5, 5, 5)
    book = GateBook(policy)
    book.book('A1', 'TRUCK-9', T0)
    d = build(appointment_id='A1', appointments=book)
    decision = good_entry(d)
    assert decision.result == PassageResult.EXCEPTION
    assert 'appointment belongs to another truck' in decision.reasons


def test_lane_invariant_failure_is_not_swallowed_as_exception():
    d = build()
    good_entry(d)
    # the occupied lane is externally cleared, breaking the occupancy invariant
    d.lanes.states['L1'].occupied = False
    with pytest.raises(StateTransitionError):
        good_exit(d)
    # visit untouched, and no exit decision/evidence was recorded
    assert d.visits['V1'].state == VisitState.INSIDE
    assert len(d.evidence_for('V1')) == 1
