from src.event_resolution import resolve_event


def _clip(message_id, phase, category, reason="reason"):
    return {
        "channel_id": "source",
        "message_id": message_id,
        "phase": phase,
        "category": category,
        "reason": reason,
    }


def test_completed_guard_resolves_communication_but_preserves_incident_verdict():
    result = resolve_event(
        [
            _clip(1, "initial", "incident_candidate", "outside_no_colour"),
            _clip(2, "complete", "guard_candidate", "green_light"),
        ]
    )
    assert result.category == "incident_candidate"
    assert result.resolution_state == "likely_resolved"
    assert result.representative["message_id"] == 1
    assert result.notification_representative["message_id"] == 2


def test_blank_or_environment_completion_cannot_clear_startup_incident():
    result = resolve_event(
        [
            _clip(1, "initial", "incident_candidate"),
            _clip(2, "complete", "environment_candidate"),
        ]
    )
    assert result.category == "incident_candidate"
    assert result.resolution_state == "conflicting"


def test_startup_incident_without_completion_is_unconfirmed():
    result = resolve_event([_clip(1, "initial", "incident_candidate")])
    assert result.category == "incident_candidate"
    assert result.resolution_state == "unconfirmed"


def test_completed_incident_promotes_guard_startup_and_is_confirmed():
    result = resolve_event(
        [
            _clip(1, "initial", "guard_candidate"),
            _clip(2, "complete", "incident_candidate"),
        ]
    )
    assert result.category == "incident_candidate"
    assert result.resolution_state == "confirmed"
    assert result.representative["message_id"] == 2

