import cv2
import numpy as np
import pytest

from scripts.motion_heatmap import (
    _circular_hue_diff,
    _filter_small_blobs,
    _local_contrast,
    _normalise,
    _threshold_channel,
    colorize_heatmap,
    compute_motion_heatmap,
    overlay_heatmap,
    render_motion_heatmap,
)

WIDTH, HEIGHT = 32, 24


def _write_clip(path, *, warmup_frames: int = 4, motion_frames: int = 10) -> str:
    """A synthetic clip: a few frames of ramping global brightness (stands in
    for the IR gain warmup), then a settled background with a bright square
    sweeping left to right."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (WIDTH, HEIGHT))
    try:
        for i in range(warmup_frames):
            base = 30 + i * 15
            writer.write(np.full((HEIGHT, WIDTH, 3), base, dtype=np.uint8))
        for i in range(motion_frames):
            frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
            x = 2 + i * 2
            cv2.rectangle(frame, (x, 10), (x + 4, 16), (240, 240, 240), -1)
            writer.write(frame)
    finally:
        writer.release()
    return str(path)


def test_normalise_clips_to_unit_range():
    diff = np.array([-300.0, -50.0, 0.0, 50.0, 300.0], dtype=np.float32)
    out = _normalise(diff, scale=100.0)
    assert out[0] == 1.0
    assert out[-1] == 1.0
    assert out[2] == 0.0
    assert 0.0 < out[1] < 1.0


def test_circular_hue_diff_wraps_around_179():
    hue_a = np.array([[178]], dtype=np.uint8)
    hue_b = np.array([[2]], dtype=np.uint8)
    # raw difference is 176, but the wrapped (true) distance is only 4
    assert _circular_hue_diff(hue_a, hue_b)[0, 0] == pytest.approx(4 / 90)


def test_local_contrast_is_zero_on_a_flat_patch():
    flat = np.full((20, 20), 100, dtype=np.uint8)
    assert np.allclose(_local_contrast(flat, window=5), 0.0)


def test_local_contrast_is_positive_across_an_edge():
    half_and_half = np.zeros((20, 20), dtype=np.uint8)
    half_and_half[:, 10:] = 200
    variance = _local_contrast(half_and_half, window=5)
    assert variance[10, 9] > 0
    assert variance[10, 0] == pytest.approx(0.0, abs=1e-3)


def test_threshold_channel_zeroes_values_at_or_below_the_floor():
    diff = np.array([0.0, 0.05, 0.1, 0.2], dtype=np.float32)
    out = _threshold_channel(diff, 0.1)
    assert list(out) == [0.0, 0.0, 0.0, 0.2]


def test_filter_small_blobs_drops_scattered_single_pixel_noise():
    combined = np.zeros((30, 30), dtype=np.float32)
    # scattered single-pixel "noise" flecks, nowhere near each other
    for y, x in [(2, 2), (5, 20), (25, 4), (18, 27)]:
        combined[y, x] = 0.8
    filtered = _filter_small_blobs(combined, min_area_px=9.0, open_kernel=1)
    assert not filtered.any()


def test_filter_small_blobs_keeps_a_subject_sized_region():
    combined = np.zeros((30, 30), dtype=np.float32)
    combined[10:16, 10:16] = 0.8  # a solid 6x6=36px patch
    filtered = _filter_small_blobs(combined, min_area_px=9.0, open_kernel=1)
    assert filtered[12, 12] == pytest.approx(0.8)


def test_compute_motion_heatmap_drops_warmup_and_highlights_the_moving_square(tmp_path):
    clip = _write_clip(tmp_path / "clip.mp4", warmup_frames=4, motion_frames=10)

    result = compute_motion_heatmap(clip)

    assert result is not None
    mean_change, background_bgr, drop = result
    assert drop >= 1
    assert background_bgr.shape == (HEIGHT, WIDTH, 3)
    # the square sweeps through row band 10-16 -- that band should show
    # noticeably more accumulated change than a row untouched by it
    swept_band = mean_change[10:16, :].mean()
    untouched_band = mean_change[0:5, :].mean()
    assert swept_band > untouched_band


def test_compute_motion_heatmap_returns_none_for_unreadable_clip(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert compute_motion_heatmap(str(empty)) is None


def test_colorize_heatmap_produces_a_bgr_image_same_size():
    mean_change = np.random.default_rng(0).random((20, 30)).astype(np.float32)
    colored = colorize_heatmap(mean_change)
    assert colored.shape == (20, 30, 3)
    assert colored.dtype == np.uint8


def test_colorize_heatmap_handles_a_uniform_input_without_crashing():
    mean_change = np.zeros((10, 10), dtype=np.float32)
    colored = colorize_heatmap(mean_change)
    assert colored.shape == (10, 10, 3)


def test_overlay_heatmap_blends_toward_the_background_at_low_alpha():
    heatmap = np.full((10, 10, 3), 255, dtype=np.uint8)
    background = np.zeros((10, 10, 3), dtype=np.uint8)
    blended = overlay_heatmap(heatmap, background, alpha=0.0)
    assert np.allclose(blended, background)


def test_render_motion_heatmap_writes_heatmap_and_overlay_files(tmp_path):
    clip = _write_clip(tmp_path / "clip.mp4")
    out = tmp_path / "heatmap.png"
    overlay_out = tmp_path / "overlay.png"

    result = render_motion_heatmap(
        clip, out_path=str(out), overlay_path=str(overlay_out), scale=2
    )

    assert result == str(out)
    assert out.exists()
    assert overlay_out.exists()
    image = cv2.imread(str(out))
    assert image.shape == (HEIGHT * 2, WIDTH * 2, 3)


def test_render_motion_heatmap_returns_none_for_unreadable_clip(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert render_motion_heatmap(str(empty), out_path=str(tmp_path / "out.png")) is None
