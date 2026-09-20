from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import render_debug
from scripts.render_debug import (
    DEFAULTS,
    _draw_dashed_rect,
    _draw_light_mask,
    _draw_multi_tracks,
    render_clip,
)
from src import db
from src.config import CameraZone, load_thresholds_config
from src.motion import TrackedObject, detect_clip

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


def test_render_clip_scores_the_same_detection_it_draws(monkeypatch, tmp_path, zone):
    clip = _write_clip(tmp_path / "in.mp4", frames=14, flare_at=2)
    real_detect = render_debug.detect_clip
    real_features = render_debug.features_from_detection
    real_classify = render_debug.classify_detailed
    real_warmup = render_debug.warmup_motion_analysis
    real_draw_warmup = render_debug._draw_warmup_motion_objects
    seen = {}

    def capture_detect(*args, **kwargs):
        seen["ignore_polygons"] = kwargs.get("ignore_polygons")
        seen["detector_settings"] = kwargs
        seen["detection"] = real_detect(*args, **kwargs)
        return seen["detection"]

    def capture_features(detection, *args, **kwargs):
        seen["scored_detection"] = detection
        seen["scored_warmup"] = kwargs.get("warmup_motion")
        return real_features(detection, *args, **kwargs)

    def capture_warmup(*args, **kwargs):
        seen["warmup_analysis"] = real_warmup(*args, **kwargs)
        return seen["warmup_analysis"]

    def capture_draw_warmup(canvas, objects, **kwargs):
        seen.setdefault("drawn_warmup_objects", []).extend(objects)
        return real_draw_warmup(canvas, objects, **kwargs)

    def capture_classification(features):
        seen["classified_features"] = features
        return real_classify(features)

    monkeypatch.setattr(render_debug, "detect_clip", capture_detect)
    monkeypatch.setattr(render_debug, "features_from_detection", capture_features)
    monkeypatch.setattr(render_debug, "classify_detailed", capture_classification)
    monkeypatch.setattr(render_debug, "warmup_motion_analysis", capture_warmup)
    monkeypatch.setattr(render_debug, "_draw_warmup_motion_objects", capture_draw_warmup)

    configured_motion = load_thresholds_config("config/thresholds.yaml").motion_thresholds()
    out = render_clip(
        clip,
        zone,
        out_path=str(tmp_path / "out.mp4"),
        timestamp="2026-01-01T12:00:00Z",
        motion_thresholds=configured_motion,
    )

    assert out is not None
    assert seen["scored_detection"] is seen["detection"]
    assert seen["scored_warmup"] is seen["warmup_analysis"]
    assert seen["drawn_warmup_objects"] == list(seen["warmup_analysis"].objects)
    assert seen["ignore_polygons"] == zone.ignore
    assert seen["detection"] is not None
    assert seen["detector_settings"]["compensate_warmup"] is False
    assert seen["detector_settings"]["anchor_refine"] is True
    assert seen["classified_features"]["is_daylight"] == 1.0


def test_draw_light_mask_outlines_green_pixels_when_not_daylight_gated():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (0, 255, 0)  # pure green, BGR
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    _draw_light_mask(canvas, frame, daylight_gated=False)
    assert canvas.any()


def test_draw_light_mask_marks_but_does_not_return_a_daylight_gated_candidate():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (0, 255, 0)  # pure green, BGR -- would draw if not gated
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    moving = _draw_light_mask(canvas, frame, daylight_gated=True)
    assert canvas.any()
    assert not moving.any()


def test_widen_year_shows_the_century_on_a_caption_timestamp():
    assert render_debug._widen_year("25-11-24 05:55:34") == "25-11-2024 05:55:34"


def test_widen_year_leaves_an_already_four_digit_year_alone():
    assert render_debug._widen_year("25-11-2024 05:55:34") == "25-11-2024 05:55:34"


def test_draw_light_mask_splits_stationary_from_moving_by_ignore_mask():
    # Left half is a "known stationary light" (inside ignore_mask), right half
    # is the guard's moving flashlight (outside it) -- both green, but only
    # the right half should come back as the returned moving-light mask.
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (0, 255, 0)  # pure green, BGR
    ignore_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    ignore_mask[:, : WIDTH // 2] = True
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    moving = _draw_light_mask(canvas, frame, daylight_gated=False, ignore_mask=ignore_mask)

    assert canvas.any()  # both regions still drawn (in different colours)
    assert not moving[:, : WIDTH // 2].any()  # stationary half excluded
    assert moving[:, WIDTH // 2 :].any()  # moving half included


def test_draw_light_mask_returns_empty_moving_mask_when_daylight_gated():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (0, 255, 0)
    moving = _draw_light_mask(canvas=np.zeros_like(frame), frame=frame, daylight_gated=True)
    assert not moving.any()


def test_draw_light_mask_outlines_every_blob_and_reports_the_count(monkeypatch):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(frame, (3, 15), (6, 18), (0, 255, 0), thickness=-1)
    cv2.rectangle(frame, (40, 25), (43, 28), (0, 255, 0), thickness=-1)
    labels = []
    monkeypatch.setattr(render_debug, "_text", lambda _img, text, *_a, **_k: labels.append(text))

    canvas = np.zeros_like(frame)
    _draw_light_mask(canvas, frame, daylight_gated=False)

    assert labels == ["FLASHLIGHT PIXELS (2 blobs)"]
    assert tuple(int(c) for c in canvas[15, 3]) == render_debug.COLOR_LIGHT
    assert tuple(int(c) for c in canvas[25, 40]) == render_debug.COLOR_LIGHT


def test_draw_dashed_rect_draws_fewer_pixels_than_a_solid_rectangle():
    # The whole point of the dashed style is that an inferred (recovered /
    # reverse-filled) box reads as visually less certain than a genuine
    # detection's solid box -- confirm it actually leaves gaps rather than
    # drawing a continuous outline.
    dashed = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    _draw_dashed_rect(dashed, (5, 5), (55, 40), (255, 0, 255), thickness=2)
    solid = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(solid, (5, 5), (55, 40), (255, 0, 255), 2)
    assert dashed.any()
    assert np.count_nonzero(dashed) < np.count_nonzero(solid)


def test_draw_multi_tracks_marks_a_real_flashlight_object_distinctly():
    # Two persistent objects in the same frame: track 0 sits on a real,
    # highly-saturated green patch (the flashlight); track 1 sits on plain
    # grey (the actual subject). Only track 0's box/label should be drawn in
    # COLOR_LIGHT and carry "FLASHLIGHT" -- confirms the fix for the render
    # never marking ANY multi-track object as a flashlight before this.
    frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    cv2.rectangle(frame, (2, 2), (12, 12), (40, 255, 40), thickness=-1)  # real flashlight hue
    flashlight_track = TrackedObject(track_id=0, bbox=(2, 2, 12, 12))
    subject_track = TrackedObject(track_id=1, bbox=(40, 30, 60, 46))

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    _draw_multi_tracks(
        canvas,
        [flashlight_track, subject_track],
        scale=1,
        multi_track_history={},
        frame_bgr=frame,
    )

    # The flashlight track's box border is drawn in COLOR_LIGHT.
    assert tuple(int(c) for c in canvas[2, 7]) == render_debug.COLOR_LIGHT
    # The subject track's box border is its own stable per-id colour, not
    # COLOR_LIGHT -- confirms the check is per-object, not clip-wide.
    subject_color = render_debug._track_color(1)
    assert tuple(int(c) for c in canvas[30, 50]) == subject_color
    assert subject_color != render_debug.COLOR_LIGHT


def test_draw_multi_tracks_uses_the_scorers_two_percent_light_threshold():
    # Nine green pixels in a 20x20 track are 2.25%: enough for the
    # multi-object feature, but far below the single-track bbox-overlap rule's
    # 50%. The renderer must follow the former for multi-object labels.
    frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    cv2.rectangle(frame, (2, 2), (4, 4), (40, 255, 40), thickness=-1)
    track = TrackedObject(track_id=0, bbox=(2, 2, 22, 22))
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    _draw_multi_tracks(
        canvas, [track], scale=1, multi_track_history={}, frame_bgr=frame
    )

    assert tuple(int(c) for c in canvas[2, 10]) == render_debug.COLOR_LIGHT


def test_draw_multi_tracks_keeps_a_peak_scoring_flashlight_track_marked():
    # Production passes whole-clip peak scores. A faint later frame must keep
    # the same persistent flashlight identity that caused classification.
    frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    track = TrackedObject(track_id=4, bbox=(2, 2, 22, 22))
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    _draw_multi_tracks(
        canvas,
        [track],
        scale=1,
        multi_track_history={},
        frame_bgr=frame,
        flashlight_scores={4: 0.08},
    )

    assert tuple(int(c) for c in canvas[2, 10]) == render_debug.COLOR_LIGHT


def test_draw_multi_tracks_skips_the_flashlight_check_without_a_frame():
    # frame_bgr=None (the default, and every pre-existing caller) must behave
    # exactly as before this feature existed -- no crash, no flashlight check.
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    track = TrackedObject(track_id=0, bbox=(2, 2, 12, 12))

    _draw_multi_tracks(canvas, [track], scale=1, multi_track_history={})

    assert tuple(int(c) for c in canvas[2, 7]) == render_debug._track_color(0)


def test_draw_multi_tracks_accepts_fractional_dead_reckoned_bbox():
    canvas = np.zeros((80, 80, 3), dtype=np.uint8)
    track = TrackedObject(track_id=0, bbox=(2.5, 3.5, 12.5, 13.5))

    _draw_multi_tracks(canvas, [track], scale=2, multi_track_history={})

    assert tuple(int(c) for c in canvas[8, 14]) == render_debug._track_color(0)


def test_draw_multi_tracks_daylight_gated_suppresses_the_flashlight_check():
    # Same real flashlight-hue patch as the first test, but daylight_gated --
    # must not mark it, same "a flashlight in daylight footage isn't
    # necessarily the guard's" caution the single-track check already
    # applies via its own daylight_gated parameter.
    frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    cv2.rectangle(frame, (2, 2), (12, 12), (40, 255, 40), thickness=-1)
    track = TrackedObject(track_id=0, bbox=(2, 2, 12, 12))
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    _draw_multi_tracks(
        canvas, [track], scale=1, multi_track_history={}, frame_bgr=frame, daylight_gated=True
    )

    assert tuple(int(c) for c in canvas[2, 7]) == render_debug._track_color(0)


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
    result = render_clip(clip, zone, out_path=str(out_path))
    assert result is not None
    assert Path(result).exists()


def test_detector_defaults_match_the_spike():
    # The renderer is only trustworthy for tuning if its untouched defaults are
    # the values that actually scored the corpus.
    import inspect

    from src.scoring import extract_clip_features

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
    flaring = detect_clip(_write_clip(tmp_path / "flare.mp4", frames=12, flare_at=2))
    steady = detect_clip(_write_clip(tmp_path / "steady.mp4", frames=12))

    assert flaring is not None and steady is not None
    assert flaring.warmup_dropped > 0
    assert steady.warmup_dropped == 0


def test_render_clip_includes_dropped_warmup_frames_in_output(tmp_path, zone):
    # The pre-cutoff frames are excluded from settled-background scoring but
    # feed dedicated warmup features and must be visible for audit.
    clip = _write_clip(tmp_path / "flare.mp4", frames=14, flare_at=2)
    detection = detect_clip(clip)
    assert detection is not None
    assert detection.warmup_dropped > 0

    out = render_clip(clip, zone, out_path=str(tmp_path / "out.mp4"))
    cap = cv2.VideoCapture(out)
    try:
        rendered = 0
        while cap.read()[0]:
            rendered += 1
    finally:
        cap.release()
    assert rendered == detection.warmup_dropped + len(detection.frames)


def test_render_clip_draws_flashlight_from_each_raw_warmup_frame(
    monkeypatch, tmp_path, zone
):
    clip = _write_clip(tmp_path / "flare.mp4", frames=14, flare_at=2)
    real_detect = render_debug.detect_clip
    real_draw = render_debug._draw_light_mask
    seen = {"calls": []}

    def capture_detect(*args, **kwargs):
        seen["detection"] = real_detect(*args, **kwargs)
        return seen["detection"]

    def capture_draw(canvas, frame, **kwargs):
        seen["calls"].append((frame.copy(), kwargs["daylight_gated"]))
        return real_draw(canvas, frame, **kwargs)

    monkeypatch.setattr(render_debug, "detect_clip", capture_detect)
    monkeypatch.setattr(render_debug, "_draw_light_mask", capture_draw)

    render_clip(clip, zone, out_path=str(tmp_path / "out.mp4"))

    detection = seen["detection"]
    warmup_calls = seen["calls"][: detection.warmup_dropped]
    expected_gate, _ratios = render_debug.warmup_flashlight_diagnostics(
        detection,
        exclude_mask=render_debug.feature_exclude_mask(detection, zone),
        daylight_color_fraction=0.15,
        daylight_hint=None,
    )
    assert len(warmup_calls) == detection.warmup_dropped
    assert all(call[1] is expected_gate for call in warmup_calls)
    assert all(
        np.array_equal(call[0], raw)
        for call, raw in zip(warmup_calls, detection.dropped_frames, strict=True)
    )


def test_prefer_longest_per_event_keeps_only_the_longer_sibling(tmp_path):
    from scripts.render_debug import _prefer_longest_per_event

    short_clip = _write_clip(tmp_path / "short.mp4", frames=6)
    long_clip = _write_clip(tmp_path / "long.mp4", frames=20)
    caption = "Initial alert @ 26-08-30 01:02:03"
    sibling_caption = "Motion stopped @ 26-08-30 01:02:03"

    clips = [
        {
            "camera_id": "cam06",
            "message_id": 1,
            "caption": caption,
            "file_path": short_clip,
            "label": "incident",
        },
        {
            "camera_id": "cam06",
            "message_id": 2,
            "caption": sibling_caption,
            "file_path": long_clip,
            "label": "incident",
        },
    ]

    result = _prefer_longest_per_event(clips)

    assert len(result) == 1
    assert result[0]["message_id"] == 2


def test_prefer_longest_per_event_tie_prefers_clear_over_duplicate(tmp_path):
    from scripts.render_debug import _prefer_longest_per_event

    duplicate = _write_clip(tmp_path / "duplicate.mp4", frames=10)
    clear = _write_clip(tmp_path / "clear.mp4", frames=10)
    caption = "Initial alert @ 26-08-30 01:02:03"
    clips = [
        {
            "camera_id": "cam06",
            "message_id": 1,
            "caption": caption,
            "file_path": duplicate,
            "label": "guard",
            "startup_state": "duplicate",
        },
        {
            "camera_id": "cam06",
            "message_id": 2,
            "caption": caption,
            "file_path": clear,
            "label": "guard",
            "startup_state": None,
        },
    ]

    assert _prefer_longest_per_event(clips)[0]["message_id"] == 2


def test_resolve_explicit_duplicate_id_loads_and_prefers_clear_sibling(tmp_path):
    duplicate = _write_clip(tmp_path / "duplicate.mp4", frames=10)
    clear = _write_clip(tmp_path / "clear.mp4", frames=10)
    conn = db.connect(tmp_path / "test.db")
    for message_id, caption, path in (
        (1, "Initial alert @ 26-08-30 01:02:03", duplicate),
        (2, "Motion stopped @ 26-08-30 01:02:03", clear),
    ):
        db.upsert_clip(
            conn,
            channel_id="chan",
            message_id=message_id,
            camera_id="cam06",
            timestamp="2026-08-30T01:02:03Z",
            caption=caption,
            file_path=path,
            source="backfill",
        )
    db.upsert_label(
        conn,
        channel_id="chan",
        message_id=1,
        label="guard",
        startup_state="duplicate",
    )
    db.upsert_label(conn, channel_id="chan", message_id=2, label="guard")
    conn.commit()
    args = SimpleNamespace(
        clip=None,
        camera=None,
        message_id=[1],
        message_ids_file=None,
        label=[],
        limit=None,
    )

    result = render_debug._resolve_clips(args, conn)

    assert [row["message_id"] for row in result] == [2]


def test_resolve_exact_message_id_keeps_requested_startup_clip(tmp_path):
    duplicate = _write_clip(tmp_path / "duplicate.mp4", frames=10)
    clear = _write_clip(tmp_path / "clear.mp4", frames=10)
    conn = db.connect(tmp_path / "test.db")
    for message_id, caption, path in (
        (1, "Initial alert @ 26-08-30 01:02:03", duplicate),
        (2, "Motion stopped @ 26-08-30 01:02:03", clear),
    ):
        db.upsert_clip(
            conn,
            channel_id="chan",
            message_id=message_id,
            camera_id="cam06",
            timestamp="2026-08-30T01:02:03Z",
            caption=caption,
            file_path=path,
            source="backfill",
        )
    db.upsert_label(
        conn,
        channel_id="chan",
        message_id=1,
        label="guard",
        startup_state="duplicate",
    )
    db.upsert_label(conn, channel_id="chan", message_id=2, label="guard")
    conn.commit()
    args = SimpleNamespace(
        clip=None,
        camera=None,
        message_id=[],
        exact_message_id=[1],
        message_ids_file=None,
        label=[],
        limit=None,
    )

    result = render_debug._resolve_clips(args, conn)

    assert [row["message_id"] for row in result] == [1]
    assert result[0]["timestamp"] == "2026-08-30T01:02:03Z"


def test_prefer_longest_per_event_leaves_unrelated_clips_alone(tmp_path):
    from scripts.render_debug import _prefer_longest_per_event

    clip_a = _write_clip(tmp_path / "a.mp4", frames=6)
    clip_b = _write_clip(tmp_path / "b.mp4", frames=6)
    clips = [
        {"camera_id": "cam06", "message_id": 1, "caption": None, "file_path": clip_a, "label": ""},
        {"camera_id": "cam07", "message_id": 2, "caption": None, "file_path": clip_b, "label": ""},
    ]

    result = _prefer_longest_per_event(clips)

    assert result == clips
