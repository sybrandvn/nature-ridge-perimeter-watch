import cv2
import numpy as np
import pytest

from scripts.render_debug import DEFAULTS, render_clip
from scripts.spike import detect_clip
from src.config import CameraZone

HEIGHT, WIDTH = 48, 64


@pytest.fixture
def zone() -> CameraZone:
    return CameraZone(
        fence=((0.5, 0.0), (0.5, 1.0)),
        outside="left",
        depth_cutoff=0.1,
        ignore=(((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)),),
    )


def _write_clip(path, *, frames: int = 12, flare_at: int | None = None) -> str:
    """A synthetic clip with a blob tracking left-to-right, optionally with a
    whole-frame brightness step standing in for an IR gain change."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (WIDTH, HEIGHT)
    )
    for i in range(frames):
        base = 30 if flare_at is None or i < flare_at else 90
        frame = np.full((HEIGHT, WIDTH, 3), base, dtype=np.uint8)
        x = 4 + i * 4
        cv2.rectangle(frame, (x, 20), (x + 8, 32), (240, 240, 240), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def test_render_clip_writes_a_readable_video_taller_than_the_source(tmp_path, zone):
    clip = _write_clip(tmp_path / "in.mp4")
    out = render_clip(clip, zone, out_path=str(tmp_path / "out.mp4"))

    assert out is not None
    cap = cv2.VideoCapture(out)
    try:
        assert cap.isOpened()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    # 2x upscale plus the HUD panel below the frame.
    assert width == WIDTH * 2
    assert height > HEIGHT * 2


def test_render_clip_frame_count_matches_the_detector(tmp_path, zone):
    clip = _write_clip(tmp_path / "in.mp4", frames=14)
    out = render_clip(clip, zone, out_path=str(tmp_path / "out.mp4"))
    detection = detect_clip(clip)

    cap = cv2.VideoCapture(out)
    try:
        rendered = 0
        while cap.read()[0]:
            rendered += 1
    finally:
        cap.release()
    assert detection is not None
    assert rendered == len(detection.frames)


def test_render_clip_returns_none_for_an_unreadable_clip(tmp_path, zone):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert render_clip(str(empty), zone, out_path=str(tmp_path / "out.mp4")) is None


def test_render_clip_honours_scale(tmp_path, zone):
    clip = _write_clip(tmp_path / "in.mp4")
    out = render_clip(clip, zone, out_path=str(tmp_path / "out.mp4"), scale=3)
    cap = cv2.VideoCapture(out)
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == WIDTH * 3
    finally:
        cap.release()


def test_render_clip_creates_missing_output_directories(tmp_path, zone):
    clip = _write_clip(tmp_path / "in.mp4")
    out_path = tmp_path / "nested" / "deeper" / "out.mp4"
    assert render_clip(clip, zone, out_path=str(out_path)) is not None
    assert out_path.exists()


def test_detector_defaults_match_the_spike():
    # The renderer is only trustworthy for tuning if its untouched defaults are
    # the values that actually scored the corpus.
    import inspect

    from scripts.spike import extract_clip_features

    params = inspect.signature(extract_clip_features).parameters
    assert DEFAULTS["threshold"] == params["threshold"].default
    assert DEFAULTS["min_area"] == params["min_blob_area_fraction"].default
    assert DEFAULTS["max_area"] == params["max_area_fraction"].default
    assert DEFAULTS["flare_tolerance"] == params["flare_tolerance"].default
    assert DEFAULTS["max_flare_fraction"] == params["max_flare_fraction"].default


def test_detect_clip_marks_flare_frames_left_by_the_cap(tmp_path, zone):
    # detect_clip drops everything through the *last* flare-flagged frame, so
    # a clip that settles cleanly never has is_flare=True left in what's kept
    # -- only a clip whose brightness never fully settles within
    # max_flare_fraction should still show flagged frames, as a warning that
    # the kept footage wasn't fully clean.
    values = [min(v, 255) for v in range(0, 320, 20)] + [255] * 4  # ramps for 16 frames of 20
    path = tmp_path / "long_flare.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (WIDTH, HEIGHT))
    for v in values:
        writer.write(np.full((HEIGHT, WIDTH, 3), v, dtype=np.uint8))
    writer.release()

    detection = detect_clip(str(path))

    assert detection is not None
    assert 0 < detection.warmup_dropped < len(values)
    assert any(f.is_flare for f in detection.frames)


def test_detect_clip_reports_no_flare_on_steady_illumination(tmp_path, zone):
    clip = _write_clip(tmp_path / "steady.mp4", frames=12)
    detection = detect_clip(clip)

    assert detection is not None
    assert not any(f.is_flare for f in detection.frames)


def test_detect_clip_computes_the_cutoff_per_clip_not_a_fixed_window(tmp_path, zone):
    # A clip that flares should drop those frames; a clip that never flares
    # should drop none -- neither is a hardcoded frame count.
    flaring = detect_clip(_write_clip(tmp_path / "flare.mp4", frames=12, flare_at=6))
    steady = detect_clip(_write_clip(tmp_path / "steady.mp4", frames=12))

    assert flaring is not None and steady is not None
    assert flaring.warmup_dropped > 0
    assert steady.warmup_dropped == 0
