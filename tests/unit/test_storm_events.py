from src.storm_events import ClipSignal, find_corroborated_events, neighbor_cameras

ORDER = {"cam01": 0, "cam02": 1, "cam03": 2, "cam04": 3, "cam05": 4}


def _clip(camera_id, message_id, timestamp, is_candidate=True):
    return ClipSignal(
        camera_id=camera_id, message_id=message_id, timestamp=timestamp, is_candidate=is_candidate
    )


def test_neighbor_cameras_within_distance():
    assert set(neighbor_cameras("cam03", ORDER, distance=1)) == {"cam02", "cam04"}
    assert set(neighbor_cameras("cam03", ORDER, distance=2)) == {"cam01", "cam02", "cam04", "cam05"}


def test_neighbor_cameras_excludes_self():
    assert "cam03" not in neighbor_cameras("cam03", ORDER, distance=5)


def test_neighbor_cameras_empty_for_unconfigured_camera():
    assert neighbor_cameras("cam99", ORDER, distance=5) == []


def test_find_corroborated_events_links_adjacent_cameras_within_window():
    clips = [
        _clip("cam03", 1, "2024-01-01T10:00:00Z"),
        _clip("cam04", 2, "2024-01-01T10:05:00Z"),
    ]
    events = find_corroborated_events(clips, ORDER, window_minutes=15, neighbor_distance=1)
    assert len(events) == 1
    assert {(c.camera_id, c.message_id) for c in events[0]} == {("cam03", 1), ("cam04", 2)}


def test_find_corroborated_events_ignores_pairs_outside_the_window():
    clips = [
        _clip("cam03", 1, "2024-01-01T10:00:00Z"),
        _clip("cam04", 2, "2024-01-01T11:00:00Z"),
    ]
    events = find_corroborated_events(clips, ORDER, window_minutes=15, neighbor_distance=1)
    assert events == []


def test_find_corroborated_events_ignores_cameras_outside_the_distance():
    clips = [
        _clip("cam01", 1, "2024-01-01T10:00:00Z"),
        _clip("cam04", 2, "2024-01-01T10:05:00Z"),
    ]
    events = find_corroborated_events(clips, ORDER, window_minutes=15, neighbor_distance=1)
    assert events == []


def test_find_corroborated_events_ignores_non_candidate_clips():
    clips = [
        _clip("cam03", 1, "2024-01-01T10:00:00Z", is_candidate=False),
        _clip("cam04", 2, "2024-01-01T10:05:00Z"),
    ]
    events = find_corroborated_events(clips, ORDER, window_minutes=15, neighbor_distance=1)
    assert events == []


def test_find_corroborated_events_merges_a_transitive_chain_across_three_cameras():
    # cam01-cam03 aren't neighbours at distance=1, but cam01-cam02-cam03 chains
    # together into a single event via the shared cam02 link.
    clips = [
        _clip("cam01", 1, "2024-01-01T10:00:00Z"),
        _clip("cam02", 2, "2024-01-01T10:05:00Z"),
        _clip("cam03", 3, "2024-01-01T10:10:00Z"),
    ]
    events = find_corroborated_events(clips, ORDER, window_minutes=15, neighbor_distance=1)
    assert len(events) == 1
    assert len(events[0]) == 3


def test_find_corroborated_events_keeps_unrelated_events_separate():
    clips = [
        _clip("cam01", 1, "2024-01-01T10:00:00Z"),
        _clip("cam02", 2, "2024-01-01T10:05:00Z"),
        _clip("cam04", 3, "2024-06-01T10:00:00Z"),
        _clip("cam05", 4, "2024-06-01T10:05:00Z"),
    ]
    events = find_corroborated_events(clips, ORDER, window_minutes=15, neighbor_distance=1)
    assert len(events) == 2
    assert events[0][0].timestamp < events[1][0].timestamp


def test_find_corroborated_events_a_lone_candidate_with_no_neighbour_is_dropped():
    clips = [_clip("cam03", 1, "2024-01-01T10:00:00Z")]
    assert find_corroborated_events(clips, ORDER, window_minutes=15, neighbor_distance=1) == []


def test_find_corroborated_events_members_sorted_chronologically():
    clips = [
        _clip("cam04", 2, "2024-01-01T10:05:00Z"),
        _clip("cam03", 1, "2024-01-01T10:00:00Z"),
    ]
    events = find_corroborated_events(clips, ORDER, window_minutes=15, neighbor_distance=1)
    assert [c.message_id for c in events[0]] == [1, 2]
