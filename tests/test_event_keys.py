from src.event_keys import event_key, event_phase, message_event_key


def test_event_key_groups_lifecycle_siblings():
    initial = "(Initial) Camera 10 @ 14-03-24 20:44:22"
    stopped = "(Stopped) Camera 10 @ 14-03-24 20:44:22"
    assert event_key("cam10", initial) == event_key("cam10", stopped)


def test_message_event_key_has_collision_free_fallback():
    assert message_event_key("cam01", "Motion detected", 42) == "cam01|message:42"


def test_event_phase_recognizes_camera_lifecycle():
    assert event_phase("(Initial*) motion") == "initial"
    assert event_phase("(Stopped*) motion") == "complete"
    assert event_phase("(Timeout) motion") == "complete"
    assert event_phase("motion") == "standalone"
