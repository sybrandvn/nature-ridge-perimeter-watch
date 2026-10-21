import json
from pathlib import Path

from src import db
from src.clip_analysis import analyze_clip
from src.config import Camera, CameraZone, load_thresholds_config


def test_analyze_clip_threads_timestamp_and_classifies(tmp_path: Path):
    conn = db.connect(tmp_path / "test.db")
    camera = Camera(
        id="cam01",
        aliases=(),
        order=0,
        zone=CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=()),
        threshold_overrides={},
    )
    seen = {}

    def extract(path, zone, **kwargs):
        seen.update(kwargs)
        return {"green_light_ratio": 0.2, "green_light_flicker": 0.0}

    result = analyze_clip(
        conn=conn,
        channel_id="source",
        message_id=1,
        video_path="clip.mp4",
        timestamp="2026-01-01T20:00:00Z",
        camera=camera,
        thresholds=load_thresholds_config("config/thresholds.yaml"),
        extract_fn=extract,
        use_cache=False,
    )
    assert seen["daylight_hint"] is False
    assert result.features["is_daylight"] == 0.0
    assert result.classification.category == "guard_candidate"
    conn.close()


def test_analyze_clip_applies_cam15_rule_through_production_entry_point():
    from src.config import load_cameras_config

    path = Path(__file__).parents[1] / "fixtures/environment_hypothesis.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in rows if row["message_id"] == 24395)

    result = analyze_clip(
        conn=None, channel_id=row["channel_id"], message_id=row["message_id"],
        video_path="clip.mp4", timestamp=row["timestamp"],
        camera=load_cameras_config("config/cameras.yaml").by_id("cam15"),
        thresholds=load_thresholds_config("config/thresholds.yaml"),
        extract_fn=lambda *_args, **_kwargs: dict(row["features"]), use_cache=False,
    )

    assert result.classification.category == "environment_candidate"
    assert result.classification.reason == "cam15_mixed_environment"
