from pathlib import Path

import pytest

from scripts import backtest
from src import db
from src.config import Camera, CamerasConfig, CameraZone

_ZONE = CameraZone(fence=((0.0, 0.5), (1.0, 0.5)), outside="left", depth_cutoff=0.0, ignore=())


def _features(**overrides) -> dict[str, float]:
    base = {
        "aspect_ratio": 1.0,
        "solidity": 0.9,
        "green_light_ratio": 0.0,
        "green_light_flicker": 0.0,
        "jitter": 1.0,
        "persistence": 0.5,
        "outside_pixel_fraction": 0.0,
        "median_fence_distance": 0.0,
        "color_fraction": 0.0,
        "blob_count": 1,
    }
    base.update(overrides)
    return base


def test_classify_no_motion_when_features_none():
    assert backtest.classify(None) == "no_motion"


def test_classify_guard_candidate_on_green_light_ratio():
    assert backtest.classify(_features(green_light_ratio=0.2)) == "guard_candidate"


def test_classify_guard_candidate_on_flicker():
    assert backtest.classify(_features(green_light_flicker=0.05)) == "guard_candidate"


def test_classify_environment_candidate_on_blob_count():
    assert backtest.classify(_features(blob_count=11)) == "environment_candidate"


def test_classify_guard_wins_over_environment_blob_count():
    features = _features(blob_count=11, green_light_ratio=0.2)
    assert backtest.classify(features) == "guard_candidate"


def test_classify_environment_candidate_on_implausible_height():
    features = _features(uncalibrated=0.0, implausible_height_fraction=0.6)
    assert backtest.classify(features) == "environment_candidate"


def test_classify_implausible_height_does_not_gate_at_or_below_threshold():
    features = _features(uncalibrated=0.0, implausible_height_fraction=0.5)
    assert backtest.classify(features) != "environment_candidate"


def test_classify_implausible_height_ignored_when_uncalibrated():
    # cam01b/cam15/cam16 have no usable picket trace -- must fall through to
    # the pixel-space rules exactly as before, never silently suppress.
    features = _features(uncalibrated=1.0, implausible_height_fraction=1.0)
    assert backtest.classify(features) != "environment_candidate"


def test_classify_implausible_height_ignored_when_key_absent():
    # Callers that never ran the metric gate (feature dict predates it) must
    # behave exactly as before -- missing key is not the same as 0.0.
    features = _features()
    assert backtest.classify(features) == "unclassified"


def test_classify_guard_wins_over_implausible_height():
    features = _features(uncalibrated=0.0, implausible_height_fraction=1.0, green_light_ratio=0.2)
    assert backtest.classify(features) == "guard_candidate"


def test_classify_environment_wins_over_animal_incident_shape():
    # high blob_count AND a fence-crossing geometry read -- environment_candidate
    # takes priority over the animal/incident geometry rule.
    features = _features(blob_count=11, outside_pixel_fraction=0.9, median_fence_distance=0.5)
    assert backtest.classify(features) == "environment_candidate"


def test_classify_animal_candidate_on_daylight_color():
    features = _features(outside_pixel_fraction=0.9, median_fence_distance=0.5, color_fraction=0.3)
    assert backtest.classify(features) == "animal_candidate"


def test_classify_incident_candidate_on_night_color():
    features = _features(outside_pixel_fraction=0.9, median_fence_distance=0.5, color_fraction=0.0)
    assert backtest.classify(features) == "incident_candidate"


def test_classify_guard_wins_over_animal_incident_geometry():
    # green light present AND a fence-crossing geometry read -- guard_candidate
    # takes priority, since guards routinely register as "outside" too (they walk
    # close to the fence and shine a flashlight across it).
    features = _features(
        outside_pixel_fraction=0.9, median_fence_distance=0.5, green_light_ratio=0.2
    )
    assert backtest.classify(features) == "guard_candidate"


def test_classify_requires_both_outside_fraction_and_fence_distance():
    # High outside_pixel_fraction alone (e.g. a guard hugging the fence, visible
    # through the mesh) must not fire without also being far from the fence line.
    features = _features(outside_pixel_fraction=0.9, median_fence_distance=0.05)
    assert backtest.classify(features) == "unclassified"


def test_classify_insect_candidate():
    assert backtest.classify(_features(jitter=60, solidity=0.5)) == "insect_candidate"


def test_classify_unclassified_when_nothing_fires():
    assert backtest.classify(_features()) == "unclassified"


def test_iter_clips_with_files_includes_unlabelled_and_labelled(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=1,
        camera_id="cam01",
        timestamp="2026-01-01T20:00:00Z",
        caption=None,
        file_path="clip1.mp4",
        source="backfill",
    )
    db.upsert_label(conn, channel_id="chan1", message_id=1, label="guard")
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=2,
        camera_id="cam01",
        timestamp="2026-01-01T20:01:00Z",
        caption=None,
        file_path="clip2.mp4",
        source="backfill",
    )
    db.upsert_clip(  # no file -- should be skipped
        conn,
        channel_id="chan1",
        message_id=3,
        camera_id="cam01",
        timestamp="2026-01-01T20:02:00Z",
        caption=None,
        file_path=None,
        source="backfill",
    )

    rows = list(backtest.iter_clips_with_files(conn))

    assert [r["message_id"] for r in rows] == [1, 2]
    assert rows[0]["label"] == "guard"
    assert rows[1]["label"] is None
    conn.close()


def test_run_backtest_assembles_rows_and_skips_unknown_camera(tmp_path: Path, capsys):
    conn = db.connect(tmp_path / "t.db")
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=1,
        camera_id="cam01",
        timestamp="2026-01-01T20:00:00Z",
        caption=None,
        file_path="clip1.mp4",
        source="backfill",
    )
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=2,
        camera_id="camXX",  # not in cameras config
        timestamp="2026-01-01T20:01:00Z",
        caption=None,
        file_path="clip2.mp4",
        source="backfill",
    )

    camera = Camera(id="cam01", aliases=(), order=0, zone=_ZONE, threshold_overrides={})
    cameras = CamerasConfig(cameras=(camera,), unknown_camera_id="unknown")

    def fake_extract(file_path, zone, *, reference_row=None):
        return _features(green_light_ratio=0.2)

    rows = list(backtest.run_backtest(conn, cameras, extract_fn=fake_extract))

    assert len(rows) == 1
    assert rows[0]["message_id"] == 1
    assert rows[0]["category"] == "guard_candidate"
    assert "camXX" in capsys.readouterr().err
    conn.close()


def test_write_csv_round_trip(tmp_path: Path):
    rows = [
        {
            "channel_id": "chan1",
            "message_id": 1,
            "camera_id": "cam01",
            "label": "guard",
            "category": "guard_candidate",
            "aspect_ratio": 1.0,
            "solidity": 0.9,
            "green_light_ratio": 0.2,
            "green_light_flicker": 0.0,
            "jitter": 1.0,
            "persistence": 0.5,
            "outside_pixel_fraction": 0.0,
        }
    ]
    out_path = tmp_path / "backtest.csv"
    backtest.write_csv(rows, str(out_path))

    import csv

    with out_path.open() as f:
        read_rows = list(csv.DictReader(f))
    assert read_rows[0]["message_id"] == "1"
    assert read_rows[0]["category"] == "guard_candidate"


def test_classify_raises_on_missing_feature_key():
    # Guard against a features dict missing an expected key -- should raise loudly
    # (KeyError) rather than silently miscategorising.
    with pytest.raises(KeyError):
        backtest.classify({"aspect_ratio": 1.0})
