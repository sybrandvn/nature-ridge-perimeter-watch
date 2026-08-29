from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import spike
from src import db
from src.config import Camera, CamerasConfig, CameraZone

_ZONE = CameraZone(fence=((0.0, 0.5), (1.0, 0.5)), outside="left", depth_cutoff=0.0, ignore=())


def _blank_frame(size: int = 60, value: int = 0) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _frame_with_square(pos: int, size: int = 60) -> np.ndarray:
    frame = _blank_frame(size)
    cv2.rectangle(frame, (pos, pos), (pos + 8, pos + 8), (255, 255, 255), thickness=-1)
    return frame


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self._idx = 0

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame

    def release(self) -> None:
        pass


def test_largest_contour_returns_none_for_empty_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    assert spike.largest_contour(mask) is None


def test_largest_contour_finds_blob():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    contour = spike.largest_contour(mask)
    assert contour is not None
    assert cv2.contourArea(contour) > 0


def test_largest_contour_rejects_blob_over_max_area():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[0:35, 0:35] = 255  # whole-frame illumination change
    mask[37:39, 37:39] = 255  # small real subject
    unfiltered = spike.largest_contour(mask)
    filtered = spike.largest_contour(mask, max_area=100)
    assert cv2.contourArea(unfiltered) > 100
    assert filtered is not None
    assert cv2.contourArea(filtered) <= 100


def test_largest_contour_returns_none_when_all_blobs_over_max_area():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[0:35, 0:35] = 255
    assert spike.largest_contour(mask, max_area=100) is None


def test_contour_centroid_of_square():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4:10, 4:10] = 255
    contour = spike.largest_contour(mask)
    centroid = spike.contour_centroid(contour)
    assert centroid == pytest.approx((6.5, 6.5), abs=1.0)


def test_normalized_contour_points_scales_to_unit_range():
    contour = np.array([[[0, 0]], [[50, 0]], [[50, 100]], [[0, 100]]], dtype=np.int32)
    points = spike.normalized_contour_points(contour, frame_width=100, frame_height=100)
    assert points == [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]


def test_extract_clip_features_returns_none_without_motion(monkeypatch, tmp_path):
    frames = [_blank_frame() for _ in range(5)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is None


def test_extract_clip_features_computes_all_features_with_motion(monkeypatch, tmp_path):
    frames = [_frame_with_square(pos) for pos in (5, 10, 15, 20, 25)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    for key in (
        "outside_pixel_fraction",
        "aspect_ratio",
        "solidity",
        "saturation_ratio",
        "green_light_flicker",
        "row_normalised_area",
        "edge_density",
        "path_length",
        "jitter",
        "persistence",
    ):
        assert key in result
    assert result["persistence"] > 0


def test_extract_clip_features_ignores_ir_warmup_brightness_swing(monkeypatch, tmp_path):
    # Mirrors the real cam15 failure: the first frames swing globally as the IR
    # gain settles, which dwarfs the actual subject's motion.
    warmup = [_blank_frame(value=0 if i % 2 else 220) for i in range(10)]
    subject = [_frame_with_square(pos) for pos in (5, 12, 19, 26, 33, 40, 5, 12, 19, 26)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(warmup + subject))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    # The 8x8 square, not the 60x60 flare -- a whole-frame blob would be far from square.
    assert result["aspect_ratio"] == pytest.approx(1.0, abs=0.3)
    assert result["persistence"] > 0.5


def test_extract_clip_features_keeps_warmup_frames_on_short_clips(monkeypatch, tmp_path):
    # Startup clips are shorter than the warmup window; dropping it would leave nothing.
    frames = [_frame_with_square(pos) for pos in (5, 12, 19, 26, 33, 40)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    assert result["persistence"] > 0


def test_extract_clip_features_flags_swinging_flashlight(monkeypatch, tmp_path):
    # A beam that sweeps in and out of frame, not just present in one frame.
    def _frame_with_square_and_green(pos: int, green: bool) -> np.ndarray:
        frame = _frame_with_square(pos)
        if green:
            cv2.rectangle(frame, (0, 0), (30, 30), (0, 255, 0), thickness=-1)
        return frame

    positions = (5, 12, 19, 26, 33, 40, 5, 12, 19, 26)
    frames = [
        _frame_with_square_and_green(pos, green=i % 2 == 0) for i, pos in enumerate(positions)
    ]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    assert result["green_light_flicker"] > 0.1


def test_iter_labelled_clips_with_files_requires_both_file_and_label(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=1,
        camera_id="cam01",
        timestamp="2026-01-01T20:00:00Z",
        caption=None,
        file_path="data/history/cam01/1.mp4",
        source="backfill",
    )
    db.upsert_label(conn, channel_id="chan1", message_id=1, label="guard")

    db.upsert_clip(  # no file_path -- should be skipped
        conn,
        channel_id="chan1",
        message_id=2,
        camera_id="cam01",
        timestamp="2026-01-01T20:01:00Z",
        caption=None,
        file_path=None,
        source="backfill",
    )
    db.upsert_label(conn, channel_id="chan1", message_id=2, label="guard")

    db.upsert_clip(  # not labelled -- should be skipped
        conn,
        channel_id="chan1",
        message_id=3,
        camera_id="cam01",
        timestamp="2026-01-01T20:02:00Z",
        caption=None,
        file_path="data/history/cam01/3.mp4",
        source="backfill",
    )

    rows = list(spike.iter_labelled_clips_with_files(conn, camera_id="cam01"))

    assert [r["message_id"] for r in rows] == [1]
    conn.close()


def test_run_spike_raises_for_unknown_camera(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    cameras = CamerasConfig(cameras=(), unknown_camera_id="unknown")
    with pytest.raises(ValueError, match="Unknown camera_id"):
        spike.run_spike(conn, cameras, camera_id="missing")
    conn.close()


def test_run_spike_assembles_rows_and_skips_undetected(monkeypatch, tmp_path: Path):
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
    db.upsert_label(conn, channel_id="chan1", message_id=2, label="animal")

    camera = Camera(id="cam01", aliases=(), order=0, zone=_ZONE, threshold_overrides={})
    cameras = CamerasConfig(cameras=(camera,), unknown_camera_id="unknown")

    def fake_extract(file_path, zone, *, reference_row=None):
        if file_path == "clip1.mp4":
            return {"outside_pixel_fraction": 0.9}
        return None  # simulate no motion detected in clip2

    monkeypatch.setattr(spike, "extract_clip_features", fake_extract)

    rows = spike.run_spike(conn, cameras, camera_id="cam01")

    assert len(rows) == 1
    assert rows[0]["message_id"] == 1
    assert rows[0]["label"] == "guard"
    assert rows[0]["outside_pixel_fraction"] == 0.9
    conn.close()


def test_write_csv_round_trip(tmp_path: Path):
    rows = [
        {
            "channel_id": "chan1",
            "message_id": 1,
            "camera_id": "cam01",
            "label": "guard",
            "outside_pixel_fraction": 0.1,
            "aspect_ratio": 1.5,
            "solidity": 0.9,
            "saturation_ratio": 0.0,
            "row_normalised_area": 100.0,
            "edge_density": 0.2,
            "path_length": 10.0,
            "jitter": 0.5,
            "persistence": 0.8,
        }
    ]
    out_path = tmp_path / "features.csv"

    spike.write_csv(rows, str(out_path))

    content = out_path.read_text()
    assert "outside_pixel_fraction" in content
    assert "guard" in content
