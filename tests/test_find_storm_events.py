from pathlib import Path

from scripts import find_storm_events as fse
from src import db
from src.config import Camera, CamerasConfig, CameraZone


def _features(**overrides) -> dict[str, float]:
    base = {
        "aspect_ratio": 1.0,
        "solidity": 0.9,
        "green_light_ratio": 0.0,
        "green_light_flicker": 0.0,
        "jitter": 1.0,
        "persistence": 0.5,
        "outside_pixel_fraction": 0.0,
        "blob_count": 1,
    }
    base.update(overrides)
    return base


def _camera(camera_id: str, order: int) -> Camera:
    return Camera(
        id=camera_id,
        aliases=(),
        order=order,
        zone=CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=()),
        threshold_overrides={},
    )


def _cameras() -> CamerasConfig:
    return CamerasConfig(
        cameras=(_camera("cam03", 2), _camera("cam04", 3)), unknown_camera_id="unknown"
    )


def _seed_clip(conn, *, camera_id, message_id, timestamp):
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=message_id,
        camera_id=camera_id,
        timestamp=timestamp,
        caption=None,
        file_path=f"{camera_id}_{message_id}.mp4",
        source="backfill",
    )


def test_collect_signals_uses_injected_extract_fn_and_classify(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed_clip(conn, camera_id="cam03", message_id=1, timestamp="2024-01-01T10:00:00Z")
    _seed_clip(conn, camera_id="cam04", message_id=2, timestamp="2024-01-01T10:05:00Z")
    db.upsert_label(conn, channel_id="chan1", message_id=1, label="environment")

    def fake_extract(file_path, zone, **_kwargs):
        # cam03's clip has scattered motion, cam04's doesn't.
        return _features(blob_count=20.0) if "cam03" in file_path else _features(blob_count=1.0)

    rows = fse.collect_signals(conn, _cameras(), extract_fn=fake_extract)

    by_camera = {r["camera_id"]: r for r in rows}
    assert by_camera["cam03"]["is_candidate"] is True
    assert by_camera["cam03"]["label"] == "environment"
    assert by_camera["cam04"]["is_candidate"] is False
    assert by_camera["cam04"]["label"] is None


def test_collect_signals_skips_clips_from_unconfigured_cameras(tmp_path: Path, capsys):
    conn = db.connect(tmp_path / "t.db")
    _seed_clip(conn, camera_id="cam99", message_id=1, timestamp="2024-01-01T10:00:00Z")

    rows = fse.collect_signals(conn, _cameras(), extract_fn=lambda *_a, **_k: None)

    assert rows == []
    assert "cam99" in capsys.readouterr().err


def test_events_from_rows_finds_a_corroborated_pair():
    rows = [
        {
            "camera_id": "cam03",
            "message_id": 1,
            "timestamp": "2024-01-01T10:00:00Z",
            "is_candidate": True,
        },
        {
            "camera_id": "cam04",
            "message_id": 2,
            "timestamp": "2024-01-01T10:05:00Z",
            "is_candidate": True,
        },
    ]
    events = fse.events_from_rows(rows, _cameras(), window_minutes=15, neighbor_distance=1)
    assert len(events) == 1
    assert len(events[0]) == 2


def test_events_from_rows_ignores_rows_without_a_timestamp():
    rows = [
        {"camera_id": "cam03", "message_id": 1, "timestamp": None, "is_candidate": True},
    ]
    assert fse.events_from_rows(rows, _cameras(), window_minutes=15, neighbor_distance=1) == []


def test_events_from_rows_excludes_twilight_clips_from_candidacy():
    # 2026-01-21T17:18:57Z is 24 minutes after the January sunset (18:55 local) --
    # inside a 60-minute twilight margin, so it should not corroborate anything
    # when exclude_twilight_minutes is set, even though is_candidate is True.
    rows = [
        {
            "camera_id": "cam03",
            "message_id": 1,
            "timestamp": "2026-01-21T17:18:57.000000Z",
            "is_candidate": True,
        },
        {
            "camera_id": "cam04",
            "message_id": 2,
            "timestamp": "2026-01-21T17:20:00.000000Z",
            "is_candidate": True,
        },
    ]
    without_filter = fse.events_from_rows(rows, _cameras(), window_minutes=15, neighbor_distance=1)
    assert len(without_filter) == 1

    with_filter = fse.events_from_rows(
        rows,
        _cameras(),
        window_minutes=15,
        neighbor_distance=1,
        exclude_twilight_minutes=60,
    )
    assert with_filter == []


def test_write_events_writes_csv_and_message_ids_sidecar(tmp_path: Path):
    events = fse.events_from_rows(
        [
            {
                "camera_id": "cam03",
                "message_id": 1,
                "timestamp": "2024-01-01T10:00:00Z",
                "is_candidate": True,
            },
            {
                "camera_id": "cam04",
                "message_id": 2,
                "timestamp": "2024-01-01T10:05:00Z",
                "is_candidate": True,
            },
        ],
        _cameras(),
        window_minutes=15,
        neighbor_distance=1,
    )
    out_path = str(tmp_path / "events.csv")
    label_by_key = {("cam03", 1): "environment", ("cam04", 2): None}

    to_review = fse.write_events(events, label_by_key, out_path)

    assert to_review == [("cam04", 2)]  # cam03/1 already confirmed, excluded
    csv_text = Path(out_path).read_text()
    assert "cam03,cam04" in csv_text
    ids_text = Path(out_path).with_suffix(".message_ids").read_text()
    assert "2  # cam04" in ids_text
    assert "1  # cam03" not in ids_text
