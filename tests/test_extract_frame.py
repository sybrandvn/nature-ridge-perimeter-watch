from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.extract_frame import extract_frame

WIDTH, HEIGHT = 16, 12


def _write_synthetic_video(path: Path, num_frames: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (WIDTH, HEIGHT)
    )
    try:
        for i in range(num_frames):
            frame = np.full((HEIGHT, WIDTH, 3), i * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_extract_frame_defaults_to_middle_frame(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=10)

    frame = extract_frame(str(video_path))

    # lossy mp4 compression means exact pixel values aren't preserved, so allow slack
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert abs(int(frame[0, 0, 0]) - 100) < 20


def test_extract_frame_honours_explicit_frame_index(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=10)

    frame = extract_frame(str(video_path), frame_index=0)

    assert abs(int(frame[0, 0, 0]) - 0) < 20


def test_extract_frame_raises_for_out_of_range_index(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=3)

    with pytest.raises(ValueError, match="999"):
        extract_frame(str(video_path), frame_index=999)
