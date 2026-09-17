from dataclasses import replace

import numpy as np

from src import db
from src.config import CameraZone
from src.motion import (
    EXTRACTOR_VERSION,
    ClipDetection,
    FrameDetection,
    GeometryObservations,
    TrackedObject,
    contour_centroid,
    extraction_fingerprint,
    get_cached_features,
    largest_contour,
    put_cached_features,
)

_ZONE = CameraZone(fence=((0.0, 0.5), (1.0, 0.5)), outside="left", depth_cutoff=0.1, ignore=())


def test_detector_dtos_live_in_motion_and_spike_reexports_them():
    # This is the first detector/feature split seam. Existing callers remain
    # source-compatible through scripts.spike while detector-owned data now
    # has one neutral home for a later serialisable raw-track payload.
    from scripts import spike

    assert spike.TrackedObject is TrackedObject
    assert spike.FrameDetection is FrameDetection
    assert spike.ClipDetection is ClipDetection
    assert spike.largest_contour is largest_contour
    assert spike.contour_centroid is contour_centroid

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    observed = FrameDetection(
        index=0,
        frame=frame,
        mask=np.zeros((2, 2), dtype=np.uint8),
        all_contours=[],
        blobs=[],
        largest=None,
        centroid=None,
        motion_pixel_fraction=0.0,
        median_grey=0.0,
        is_flare=False,
    )
    clip = ClipDetection(
        frames=[observed],
        background=frame,
        frame_width=2,
        frame_height=2,
        warmup_dropped=0,
        total_frames=1,
        dropped_frames=[],
        dropped_frame_boxes=[],
    )
    assert clip.frames[0].frame is frame


def test_motion_detect_clip_dispatches_to_the_private_compatibility_body(monkeypatch):
    from scripts import spike
    from src import motion

    sentinel = object()
    monkeypatch.setattr(spike, "_detect_clip", lambda *args, **kwargs: sentinel)

    assert motion.detect_clip("clip.mp4", threshold=18) is sentinel


def test_geometry_observations_round_trip_as_json_safe_payload():
    observed = GeometryObservations(
        frame_width=60,
        frame_height=40,
        best_contour_points=((0.1, 0.2), (0.3, 0.4)),
        genuine_contour_points=(((0.1, 0.2),),),
        centroid_track=((0.2, 0.3),),
        multi_tracks=((TrackedObject(track_id=4, bbox=(1, 2, 3, 4), merged_ids=(5,)),),),
    )

    payload = observed.to_payload()
    assert GeometryObservations.from_payload(payload) == observed


def test_extraction_fingerprint_covers_video_zone_reference_and_daylight(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-a")
    reference = np.zeros((2, 3), dtype=np.uint8)

    def fingerprint(**overrides):
        values = {
            "video_path": video,
            "motion_fingerprint": "motion-a",
            "zone": _ZONE,
            "reference_background": reference,
            "daylight_hint": False,
        }
        values.update(overrides)
        return extraction_fingerprint(**values)

    baseline = fingerprint()
    assert fingerprint() == baseline
    assert fingerprint(motion_fingerprint="motion-b") != baseline
    assert fingerprint(zone=replace(_ZONE, depth_cutoff=0.2)) != baseline
    assert fingerprint(reference_background=np.ones((2, 3), dtype=np.uint8)) != baseline
    assert fingerprint(daylight_hint=True) != baseline

    video.write_bytes(b"video-b")
    assert fingerprint() != baseline


def test_cache_round_trips_features_and_no_motion(tmp_path):
    conn = db.connect(tmp_path / "cache.db")
    put_cached_features(
        conn,
        channel_id="channel",
        message_id=1,
        fingerprint="fp-1",
        features={"jitter": 2.5},
    )
    assert get_cached_features(
        conn, channel_id="channel", message_id=1, fingerprint="fp-1"
    ) == (True, {"jitter": 2.5})

    put_cached_features(
        conn,
        channel_id="channel",
        message_id=2,
        fingerprint="fp-2",
        features=None,
    )
    assert get_cached_features(
        conn, channel_id="channel", message_id=2, fingerprint="fp-2"
    ) == (True, None)
    assert get_cached_features(
        conn, channel_id="channel", message_id=2, fingerprint="different"
    ) == (False, None)

    row = conn.execute(
        "SELECT extractor_version FROM blob_tracks WHERE message_id = 1"
    ).fetchone()
    assert row["extractor_version"] == EXTRACTOR_VERSION
    conn.close()
