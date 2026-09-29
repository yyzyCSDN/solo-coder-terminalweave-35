from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from terminalweave.core.serialization import to_primitive
from terminalweave.gate.appointments import GateBook, GateSlotPolicy
from terminalweave.gate.exceptions import ExceptionState
from terminalweave.gate.lane_control import LaneController, LaneMode
from terminalweave.gate.ocr import OcrCandidate
from terminalweave.gate.passage import (
    GatePassageCoordinator,
    PassageStatus,
    VisitState,
)
from terminalweave.gate.security import AccessController, Credential
from terminalweave.gate.token import GateTokenIssuer
from terminalweave.gate.truck_visit import TruckVisit
from terminalweave.gate.weighbridge import WeighbridgeBook, WeightTicket

UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)


def build_gate(*lane_modes: tuple[str, LaneMode]):
    credentials = [
        Credential(
            "CARD-1",
            "TRUCK-1",
            D - timedelta(hours=1),
            D + timedelta(hours=4),
            frozenset({"gate.entry", "gate.exit"}),
        )
    ]
    access = AccessController(credentials)
    tokens = GateTokenIssuer(b"0123456789abcdef")
    weighbridge = WeighbridgeBook()
    ticket = WeightTicket(
        "WT-1", "TRUCK-1", Decimal("18000"), Decimal("8000"), "ABCD1234567"
    )
    weighbridge.add(ticket)
    controller = LaneController([lane_id for lane_id, _ in lane_modes])
    for lane_id, mode in lane_modes:
        controller.set_mode(lane_id, mode)
    coordinator = GatePassageCoordinator(
        access=access,
        tokens=tokens,
        weighbridge=weighbridge,
        lanes=controller,
    )
    return coordinator, access, tokens, ticket, controller


def checked_in_appointment(visit: TruckVisit):
    book = GateBook(GateSlotPolicy(60, 10, 10, 10))
    appointment = book.book(visit.appointment_id, visit.truck_id, D)
    book.check_in(visit.appointment_id, D)
    return appointment


def visit():
    return TruckVisit("VISIT-1", "TRUCK-1", "APP-1")


def request_kwargs(visit, token, ticket, lane_id, **overrides):
    values = {
        "visit": visit,
        "credential": "CARD-1",
        "token": token,
        "ocr": [OcrCandidate("ABCD1234567", 0.99)],
        "ticket": ticket,
        "declared_kg": Decimal("10000"),
        "lane_id": lane_id,
        "at": D,
        "tolerance_kg": Decimal("20"),
    }
    values.update(overrides)
    return values


def test_consistent_entry_evidence_occupies_only_commanded_lane():
    coordinator, _, issuer, ticket, lanes = build_gate(
        ("L1", LaneMode.ENTRY), ("L2", LaneMode.ENTRY)
    )
    truck_visit = visit()
    token = issuer.issue(truck_visit.visit_id, D)
    appointment = checked_in_appointment(truck_visit)

    decision = coordinator.request_entry(
        **request_kwargs(truck_visit, token, ticket, "L1", appointment=appointment)
    )

    assert decision.status is PassageStatus.ALLOWED
    assert decision.lane_id == "L1"
    assert truck_visit.state is VisitState.ARRIVED
    assert lanes.states["L1"].occupied
    assert not lanes.states["L2"].occupied
    evidence = decision.evidence
    assert evidence.container_id == "ABCD1234567"
    assert evidence.net_kg == Decimal("10000")
    assert evidence.credential_holder == "TRUCK-1"
    assert evidence.appointment_id == "APP-1"


def test_incomplete_or_inconsistent_evidence_denies_and_opens_exception():
    coordinator, _, issuer, ticket, lanes = build_gate(("L1", LaneMode.ENTRY))
    truck_visit = visit()
    token = issuer.issue("OTHER-VISIT", D)

    decision = coordinator.request_entry(
        **request_kwargs(
            truck_visit,
            token,
            ticket,
            "L1",
            ocr=[OcrCandidate("ABCD1234567", 0.80)],
            declared_kg=Decimal("9000"),
            tolerance_kg=Decimal("20"),
        )
    )

    assert decision.status is PassageStatus.DENIED
    assert "incomplete_container_recognition" in decision.reasons
    assert "token_visit_mismatch" in decision.reasons
    assert "weight_mismatch" in decision.reasons
    assert "missing_appointment" in decision.reasons
    assert truck_visit.state is VisitState.EXPECTED
    assert truck_visit.exceptions
    assert not lanes.states["L1"].occupied
    exceptions = coordinator.exceptions.open_for_truck("TRUCK-1")
    assert len(exceptions) == 1
    assert exceptions[0].state is ExceptionState.OPEN
    assert decision.evidence is None


def test_duplicate_arrival_is_idempotent_and_does_not_take_second_lane():
    coordinator, _, issuer, ticket, lanes = build_gate(
        ("L1", LaneMode.ENTRY), ("L2", LaneMode.ENTRY)
    )
    truck_visit = visit()
    appointment = checked_in_appointment(truck_visit)
    token = issuer.issue(truck_visit.visit_id, D)
    kwargs = request_kwargs(
        truck_visit, token, ticket, "L1", appointment=appointment
    )

    first = coordinator.request_entry(**kwargs)
    duplicate = coordinator.request_entry(**kwargs)

    assert first.status is PassageStatus.ALLOWED
    assert duplicate.status is PassageStatus.DUPLICATE
    assert duplicate.idempotent
    assert duplicate.lane_id == "L1"
    assert duplicate.evidence == first.evidence
    assert lanes.states["L1"].occupied
    assert not lanes.states["L2"].occupied
    assert truck_visit.state is VisitState.ARRIVED


def test_exit_records_passage_evidence_and_retains_it_after_exit():
    coordinator, _, issuer, ticket, lanes = build_gate(
        ("ENTRY", LaneMode.ENTRY), ("EXIT", LaneMode.EXIT)
    )
    truck_visit = visit()
    token = issuer.issue(truck_visit.visit_id, D)
    appointment = checked_in_appointment(truck_visit)

    coordinator.request_entry(
        **request_kwargs(
            truck_visit, token, ticket, "ENTRY", appointment=appointment
        )
    )
    coordinator.enter_yard(truck_visit, D + timedelta(minutes=2))
    coordinator.release_visit(truck_visit, D + timedelta(minutes=20))
    exit_decision = coordinator.request_exit(
        **request_kwargs(
            truck_visit,
            token,
            ticket,
            "EXIT",
            at=D + timedelta(minutes=25),
        )
    )

    assert exit_decision.status is PassageStatus.ALLOWED
    assert truck_visit.state is VisitState.EXITED
    assert not lanes.states["EXIT"].occupied
    record = coordinator.passage_for(truck_visit.visit_id)
    assert record.entry.evidence.container_id == "ABCD1234567"
    assert record.exit.evidence.direction.value == "exit"
    assert coordinator.evidence_for(truck_visit.visit_id).weight_ticket_id == "WT-1"

    repeated = coordinator.request_exit(
        **request_kwargs(
            truck_visit,
            token,
            ticket,
            "EXIT",
            at=D + timedelta(minutes=26),
        )
    )
    assert repeated.status is PassageStatus.DUPLICATE
    assert not lanes.states["EXIT"].occupied
    assert coordinator.evidence_for(truck_visit.visit_id).container_id == "ABCD1234567"


def test_exit_without_entry_fails_closed():
    coordinator, _, issuer, ticket, lanes = build_gate(("EXIT", LaneMode.EXIT))
    truck_visit = visit()
    token = issuer.issue(truck_visit.visit_id, D)

    decision = coordinator.request_exit(
        **request_kwargs(truck_visit, token, ticket, "EXIT")
    )

    assert decision.status is PassageStatus.DENIED
    assert "missing_entry_evidence" in decision.reasons


def test_passage_evidence_is_serializable():
    coordinator, _, issuer, ticket, _ = build_gate(("L1", LaneMode.ENTRY))
    truck_visit = visit()
    token = issuer.issue(truck_visit.visit_id, D)
    appointment = checked_in_appointment(truck_visit)

    decision = coordinator.request_entry(
        **request_kwargs(truck_visit, token, ticket, "L1", appointment=appointment)
    )

    payload = to_primitive(decision)
    assert payload["status"] == "allowed"
    assert payload["evidence"]["net_kg"] == "10000"
    assert payload["evidence"]["captured_at"] == "2026-01-01T00:00:00Z"
