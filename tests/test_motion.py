from dataclasses import replace

import numpy as np

from src import db
from src.config import CameraZone
from src.motion import (
    EXTRACTOR_VERSION,
    extraction_fingerprint,
    get_cached_features,
    put_cached_features,
)

_ZONE = CameraZone(fence=((0.0, 0.5), (1.0, 0.5)), outside="left", depth_cutoff=0.1, ignore=())


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
