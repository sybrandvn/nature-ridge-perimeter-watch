from pathlib import Path
from typing import Any

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

    seen = {}

    def fake_extract(file_path, zone, *, reference_row=None, **kwargs):
        seen.update(kwargs)
        return _features(green_light_ratio=0.2)

    rows = list(backtest.run_backtest(conn, cameras, extract_fn=fake_extract))

    assert len(rows) == 1
    assert rows[0]["message_id"] == 1
    assert rows[0]["category"] == "guard_candidate"
    assert rows[0]["reason"] == "green_light"
    assert seen["daylight_hint"] is False  # 22:00 local, the sun table says night
    # is_daylight must survive into the reported row -- a stored backtest_results
    # row with this dropped could not be replayed through classify_detailed()
    # faithfully (it would silently default to "night" for every clip).
    assert rows[0]["is_daylight"] == 0.0
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
            "reason": "green_light",
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
    assert read_rows[0]["reason"] == "green_light"


def _row(camera_id: str, label: str | None, category: str) -> dict[str, Any]:
    return {"camera_id": camera_id, "label": label, "category": category}


def test_summarize_labelled_ignores_unlabelled_rows():
    rows = [
        _row("cam01", None, "guard_candidate"),
        _row("cam01", "guard", "guard_candidate"),
    ]
    summary = backtest.summarize_labelled(rows)
    assert summary["n_labelled"] == 1
    assert summary["confusion"] == {"guard": {"guard_candidate": 1}}


def test_summarize_labelled_confusion_matrix_counts_every_label_category_pair():
    rows = [
        _row("cam01", "guard", "guard_candidate"),
        _row("cam01", "guard", "guard_candidate"),
        _row("cam01", "guard", "environment_candidate"),
        _row("cam02", "animal", "animal_candidate"),
    ]
    summary = backtest.summarize_labelled(rows)
    assert summary["confusion"] == {
        "guard": {"guard_candidate": 2, "environment_candidate": 1},
        "animal": {"animal_candidate": 1},
    }


def test_summarize_labelled_alert_channel_precision_recall():
    rows = [
        _row("cam01", "incident", "incident_candidate"),  # tp
        _row("cam01", "animal", "unclassified"),  # fn
        _row("cam01", "guard", "incident_candidate"),  # fp (leak)
        _row("cam01", "guard", "guard_candidate"),  # tn
        _row("cam01", "environment", "environment_candidate"),  # tn
    ]
    summary = backtest.summarize_labelled(rows)
    alert = summary["alert_channel"]
    assert (alert["tp"], alert["fp"], alert["fn"], alert["tn"]) == (1, 1, 1, 2)
    assert alert["precision"] == pytest.approx(0.5)
    assert alert["recall"] == pytest.approx(0.5)
    assert alert["f1"] == pytest.approx(0.5)


def test_summarize_labelled_per_camera_leak_only_covers_guard_and_environment():
    rows = [
        _row("cam01", "guard", "guard_candidate"),
        _row("cam01", "guard", "incident_candidate"),  # leaked
        _row("cam02", "environment", "environment_candidate"),
        _row("cam01", "animal", "animal_candidate"),  # not guard/environment
    ]
    summary = backtest.summarize_labelled(rows)
    assert summary["per_camera_leak"]["guard"] == {
        "cam01": {"n": 2, "leaked": 1, "leak_rate": 0.5},
    }
    assert summary["per_camera_leak"]["environment"] == {
        "cam02": {"n": 1, "leaked": 0, "leak_rate": 0.0},
    }


def test_summarize_labelled_empty_rows_has_zeroed_metrics_not_a_crash():
    summary = backtest.summarize_labelled([])
    assert summary["n_labelled"] == 0
    assert summary["alert_channel"]["precision"] == 0.0
    assert summary["alert_channel"]["recall"] == 0.0


def test_print_summary_does_not_crash_on_a_real_summary(capsys):
    rows = [
        _row("cam01", "guard", "guard_candidate"),
        _row("cam01", "incident", "incident_candidate"),
    ]
    backtest.print_summary(backtest.summarize_labelled(rows))
    out = capsys.readouterr().out
    assert "confusion matrix" in out
    assert "alert channel" in out
