from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import rank_candidates as rc
from src import db
from src.config import CameraZone


def test_auc_perfect_separation():
    pos = np.array([0.9, 0.8, 0.7])
    neg = np.array([0.1, 0.2, 0.3])
    assert rc.auc(pos, neg) == pytest.approx(1.0)


def test_auc_ties_count_as_half():
    pos = np.array([0.5])
    neg = np.array([0.5])
    assert rc.auc(pos, neg) == pytest.approx(0.5)


def test_auc_no_positives_is_nan():
    assert np.isnan(rc.auc(np.array([]), np.array([0.1])))


def test_fit_logistic_separates_a_trivial_dataset():
    rng = np.random.default_rng(0)
    x_pos = rng.normal(loc=3.0, scale=0.3, size=(20, 2))
    x_neg = rng.normal(loc=-3.0, scale=0.3, size=(20, 2))
    x = np.vstack([x_pos, x_neg])
    y = np.array([1.0] * 20 + [0.0] * 20)

    weights = rc.fit_logistic(x, y, iterations=1000)
    scores = rc.predict_proba(x, weights)

    assert rc.auc(scores[y == 1], scores[y == 0]) > 0.95


def test_leave_one_out_auc_on_separable_data():
    rng = np.random.default_rng(1)
    x_pos = rng.normal(loc=3.0, scale=0.3, size=(15, 2))
    x_neg = rng.normal(loc=-3.0, scale=0.3, size=(15, 2))
    x = np.vstack([x_pos, x_neg])
    y = np.array([1.0] * 15 + [0.0] * 15)

    assert rc.leave_one_out_auc(x, y, iterations=500) > 0.9


def test_stratified_top_n_caps_per_camera_not_globally():
    rows = [
        {"camera_id": "cam01", "score": s} for s in (0.9, 0.8, 0.7, 0.6)
    ] + [{"camera_id": "cam02", "score": s} for s in (0.5, 0.4)]

    queue = rc.stratified_top_n(rows, top_n=2)

    assert sum(1 for r in queue if r["camera_id"] == "cam01") == 2
    assert sum(1 for r in queue if r["camera_id"] == "cam02") == 2
    # sorted overall by score descending
    assert [r["score"] for r in queue] == [0.9, 0.8, 0.5, 0.4]


def test_stratified_top_n_sorts_within_camera_first():
    rows = [{"camera_id": "cam01", "score": s} for s in (0.1, 0.9, 0.5)]
    queue = rc.stratified_top_n(rows, top_n=2)
    assert [r["score"] for r in queue] == [0.9, 0.5]


def _detected_row(camera_id, message_id, label, **feature_overrides):
    row = {
        "channel_id": "chan1",
        "message_id": message_id,
        "camera_id": camera_id,
        "timestamp": "2026-01-01T00:00:00Z",
        "label": label,
        "detected": True,
    }
    for feat in rc.NUMERIC_FEATURES:
        row[feat] = 0.0
    row.update(feature_overrides)
    return row


def test_rank_and_write_scores_and_writes_only_unlabelled(tmp_path: Path):
    rows = [
        _detected_row("cam01", 1, "animal", aspect_ratio=0.5, jitter=1.0),
        _detected_row("cam01", 2, "guard", aspect_ratio=1.2, green_light_ratio=0.3),
        _detected_row("cam01", 3, None, aspect_ratio=0.5, jitter=1.0),  # unlabelled, animal-like
        _detected_row("cam01", 4, None, aspect_ratio=1.2, green_light_ratio=0.3),  # guard-like
    ]
    out_path = tmp_path / "candidates.csv"

    queue = rc.rank_and_write(rows, top_per_camera=10, out_path=str(out_path))

    assert {r["message_id"] for r in queue} == {3, 4}
    assert out_path.exists()
    ids_path = out_path.with_suffix(".message_ids")
    assert ids_path.exists()
    assert set(ids_path.read_text().split()) == {"3", "4"}


def _write_video(path: Path, num_frames: int, fps: float = 5.0) -> str:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (16, 12))
    try:
        for i in range(num_frames):
            writer.write(np.full((12, 16, 3), i * 10 % 255, dtype=np.uint8))
    finally:
        writer.release()
    return str(path)


def test_rank_and_write_drops_sub_1s_clip_with_an_event_sibling(tmp_path: Path):
    short_path = _write_video(tmp_path / "short.mp4", num_frames=2)  # 0.4s @ 5fps
    long_path = _write_video(tmp_path / "long.mp4", num_frames=10)  # 2.0s @ 5fps
    caption = "Cam Alert: (Initial) NATURE RIDGE COMPLEX, MOTIONVIEWER 1 @ 01-01-26 00:00:00"
    rows = [
        _detected_row("cam01", 1, "animal", aspect_ratio=0.5, jitter=1.0),
        _detected_row("cam01", 2, "guard", aspect_ratio=1.2, green_light_ratio=0.3),
        _detected_row(
            "cam01", 3, None, aspect_ratio=0.5, jitter=1.0, caption=caption, file_path=short_path
        ),
        _detected_row(
            "cam01", 4, None, aspect_ratio=0.5, jitter=1.0, caption=caption, file_path=long_path
        ),
    ]

    queue = rc.rank_and_write(rows, top_per_camera=10, out_path=str(tmp_path / "candidates.csv"))

    assert {r["message_id"] for r in queue} == {4}  # the short sibling is dropped


def test_rank_and_write_keeps_sub_1s_clip_with_no_sibling(tmp_path: Path):
    short_path = _write_video(tmp_path / "short.mp4", num_frames=2)  # 0.4s @ 5fps
    rows = [
        _detected_row("cam01", 1, "animal", aspect_ratio=0.5, jitter=1.0),
        _detected_row("cam01", 2, "guard", aspect_ratio=1.2, green_light_ratio=0.3),
        _detected_row(
            "cam01",
            3,
            None,
            aspect_ratio=0.5,
            jitter=1.0,
            caption="Cam Alert: (Initial) NATURE RIDGE COMPLEX, MOTIONVIEWER 1 @ 01-01-26 00:00:00",
            file_path=short_path,
        ),
    ]

    queue = rc.rank_and_write(rows, top_per_camera=10, out_path=str(tmp_path / "candidates.csv"))

    assert {r["message_id"] for r in queue} == {3}  # no sibling -- kept despite being short


def test_rank_and_write_excludes_blinding_foreground(tmp_path: Path):
    rows = [
        _detected_row("cam01", 1, "animal", aspect_ratio=0.5, jitter=1.0),
        _detected_row("cam01", 2, "guard", aspect_ratio=1.2, green_light_ratio=0.3),
        _detected_row("cam01", 3, None, aspect_ratio=0.5, jitter=1.0, blob_white_fraction=0.9),
        _detected_row("cam01", 4, None, aspect_ratio=0.5, jitter=1.0),
    ]

    queue = rc.rank_and_write(rows, top_per_camera=10, out_path=str(tmp_path / "candidates.csv"))

    assert {r["message_id"] for r in queue} == {4}  # 3 is blinding-foreground, excluded


def test_write_maintenance_candidates_only_blinding_rows(tmp_path: Path):
    rows = [
        _detected_row("cam01", 1, "animal", aspect_ratio=0.5, jitter=1.0),
        _detected_row("cam01", 2, "guard", blob_white_fraction=0.9),
        _detected_row("cam01", 3, None, long_flare_frames=20),
        _detected_row("cam01", 4, None, aspect_ratio=0.5, jitter=1.0),
    ]

    queue = rc.write_maintenance_candidates(
        rows, top_per_camera=10, out_path=str(tmp_path / "maintenance.csv")
    )

    # both the labelled guard clip and the unlabelled one belong here -- this
    # is a "clean the camera" report, not an incident review queue
    assert {r["message_id"] for r in queue} == {2, 3}


def test_write_maintenance_candidates_excludes_no_maintenance_cameras(tmp_path: Path):
    rows = [
        _detected_row("cam04", 1, "guard", blob_white_fraction=0.9),
        _detected_row("cam01", 2, "guard", blob_white_fraction=0.9),
    ]

    queue = rc.write_maintenance_candidates(
        rows, top_per_camera=10, out_path=str(tmp_path / "maintenance.csv")
    )

    # cam04 confirmed 2026-09-07 to be flashlight-into-lens, not a real
    # obstruction -- excluded from the "clean the camera" report entirely
    assert {r["message_id"] for r in queue} == {2}


def test_write_maintenance_candidates_excludes_flashlight_into_lens(tmp_path: Path):
    rows = [
        # overexposed AND the whole frame settles red after the flash -- a
        # guard's flashlight, not something needing cleaning
        _detected_row("cam01", 1, "guard", blob_white_fraction=0.9, post_flash_red_shift=0.25),
        # overexposed with no post-flash colour change -- a real obstruction
        _detected_row("cam01", 2, "guard", blob_white_fraction=0.9, post_flash_red_shift=0.01),
    ]

    queue = rc.write_maintenance_candidates(
        rows, top_per_camera=10, out_path=str(tmp_path / "maintenance.csv")
    )

    assert {r["message_id"] for r in queue} == {2}


def _series(camera_id, n, flagged_indices, *, start_day=1):
    rows = []
    for i in range(n):
        day, hour = divmod(i, 4)
        overrides = {"blob_white_fraction": 0.9} if i in flagged_indices else {}
        row = _detected_row(camera_id, 1000 + i, None, **overrides)
        row["timestamp"] = f"2026-01-{start_day + day:02d}T{hour + 1:02d}:00:00Z"
        rows.append(row)
    return rows


def test_obstruction_windows_flags_a_sustained_elevation():
    # 12 flagged clips inside the first 3 days of a 40-clip, 10-day history:
    # concentrated enough to clear both the absolute floor and this camera's
    # own baseline, which is what a real obstruction looks like.
    rows = _series("cam01", 40, set(range(12)))
    windows = rc.obstruction_windows(rows, window_days=14, min_clips=12, min_rate=0.25)
    assert windows
    assert windows[0]["camera_id"] == "cam01"
    assert windows[0]["flagged"] == 12


def test_obstruction_windows_ignores_an_isolated_flag():
    rows = _series("cam01", 40, {5})
    assert rc.obstruction_windows(rows, min_clips=12, min_rate=0.25) == []


def test_obstruction_windows_needs_enough_clips():
    # 3 clips, all flagged -- 100% but meaningless
    rows = _series("cam01", 3, {0, 1, 2})
    assert rc.obstruction_windows(rows, min_clips=12) == []


def test_obstruction_windows_excludes_flashlight_into_lens():
    rows = _series("cam01", 40, set(range(12)))
    for row in rows:
        row["post_flash_red_shift"] = 0.25
    assert rc.obstruction_windows(rows, min_clips=12) == []


def test_obstruction_windows_skips_rows_without_a_usable_timestamp():
    rows = _series("cam01", 40, set(range(12)))
    for row in rows:
        row["timestamp"] = None
    assert rc.obstruction_windows(rows) == []


def test_obstruction_windows_honours_no_maintenance_cameras():
    rows = _series("cam04", 40, set(range(12)))
    assert rc.obstruction_windows(rows, min_clips=12) == []


def test_write_obstruction_windows_round_trip(tmp_path: Path):
    rows = _series("cam01", 40, set(range(12)))
    out_path = tmp_path / "maintenance.windows.csv"
    windows = rc.write_obstruction_windows(rows, out_path=str(out_path))
    assert out_path.exists()
    lines = out_path.read_text().splitlines()
    assert lines[0].startswith("camera_id,window_start,window_end,clips,flagged,rate")
    assert len(lines) == len(windows) + 1


def test_rank_and_write_raises_without_labelled_rows(tmp_path: Path):
    rows = [_detected_row("cam01", 1, None)]
    with pytest.raises(ValueError, match="no labelled"):
        rc.rank_and_write(rows, top_per_camera=10, out_path=str(tmp_path / "out.csv"))


def test_collect_features_uses_injected_extract_fn(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=1,
        camera_id="cam01",
        timestamp="2026-01-01T00:00:00Z",
        caption=None,
        file_path="clip1.mp4",
        source="backfill",
    )
    db.upsert_label(conn, channel_id="chan1", message_id=1, label="guard")

    class _Cameras:
        def by_id(self, camera_id):
            return _Camera()

    class _Camera:
        zone = CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=())

        def zone_at(self, timestamp):
            return self.zone

    seen = {}

    def fake_extract(file_path, zone, **kwargs):
        seen.update(kwargs)
        return {"aspect_ratio": 1.0}

    rows = rc.collect_features(conn, _Cameras(), extract_fn=fake_extract)

    assert len(rows) == 1
    assert rows[0]["label"] == "guard"
    assert rows[0]["detected"] is True
    assert rows[0]["aspect_ratio"] == 1.0
    assert "daylight_hint" in seen  # the exogenous sun-time signal reaches the extractor
