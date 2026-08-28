from src.config import Camera, CamerasConfig, CameraZone
from src.message_parsing import parse_message

_ZONE = CameraZone(fence=None, far_side=None, depth_cutoff=1.0, ignore=())


def _cameras(*specs: tuple[str, tuple[str, ...]]) -> CamerasConfig:
    cameras = tuple(
        Camera(id=cam_id, aliases=aliases, order=i, zone=_ZONE, threshold_overrides={})
        for i, (cam_id, aliases) in enumerate(specs)
    )
    return CamerasConfig(cameras=cameras, unknown_camera_id="unknown")


def test_clip_with_exact_alias_match():
    cameras = _cameras(("cam_north", ("North Gate", "cam1")))
    parsed = parse_message(text="cam1", has_media=True, cameras=cameras)
    assert parsed.kind == "clip"
    assert parsed.camera_id == "cam_north"


def test_clip_with_alias_embedded_in_longer_caption():
    cameras = _cameras(("cam_north", ("North Gate",)))
    parsed = parse_message(
        text="Motion detected - North Gate - 03:12", has_media=True, cameras=cameras
    )
    assert parsed.kind == "clip"
    assert parsed.camera_id == "cam_north"


def test_clip_with_no_match_is_unknown():
    cameras = _cameras(("cam_north", ("North Gate",)))
    parsed = parse_message(text="something unrelated", has_media=True, cameras=cameras)
    assert parsed.kind == "clip"
    assert parsed.camera_id == "unknown"


def test_clip_with_no_caption_is_unknown():
    cameras = _cameras(("cam_north", ("North Gate",)))
    parsed = parse_message(text=None, has_media=True, cameras=cameras)
    assert parsed.kind == "clip"
    assert parsed.camera_id == "unknown"


def test_clip_alias_does_not_match_longer_numeric_suffix():
    cameras = _cameras(("cam01", ("MOTIONVIEWER 1",)), ("cam10", ("MOTIONVIEWER 10",)))
    parsed = parse_message(
        text="Cam Alert: NATURE RIDGE COMPLEX, MOTIONVIEWER 10 @ 20-11-23 18:05:42",
        has_media=True,
        cameras=cameras,
    )
    assert parsed.camera_id == "cam10"


def test_battery_dead_keyword_detected():
    cameras = _cameras(("cam_north", ("North Gate",)))
    parsed = parse_message(
        text="North Gate camera: battery low", has_media=False, cameras=cameras
    )
    assert parsed.kind == "system_event"
    assert parsed.event_type == "battery_dead"
    assert parsed.camera_id == "cam_north"


def test_power_out_keyword_detected():
    cameras = _cameras(("cam_north", ("North Gate",)))
    parsed = parse_message(text="North Gate camera offline", has_media=False, cameras=cameras)
    assert parsed.kind == "system_event"
    assert parsed.event_type == "power_out"


def test_back_online_keyword_detected():
    cameras = _cameras(("cam_north", ("North Gate",)))
    parsed = parse_message(text="North Gate camera back online", has_media=False, cameras=cameras)
    assert parsed.kind == "system_event"
    assert parsed.event_type == "back_online"


def test_unrecognized_text_is_ignored():
    cameras = _cameras(("cam_north", ("North Gate",)))
    parsed = parse_message(text="good evening team", has_media=False, cameras=cameras)
    assert parsed.kind == "ignored"
    assert parsed.camera_id is None
    assert parsed.event_type is None
