import cv2
import numpy as np
import pytest

from src.features import (
    aspect_ratio,
    edge_density,
    green_light_ratio,
    jitter,
    path_length,
    persistence,
    row_normalised_area,
    saturation_ratio,
    solidity,
    time_of_day,
)


def _rect_contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)


def test_aspect_ratio_tall_rectangle():
    # cv2.boundingRect is pixel-inclusive: a contour spanning width w covers w+1
    # columns, so the true ratio is (h+1)/(w+1), not h/w.
    assert aspect_ratio(_rect_contour(0, 0, 10, 20)) == pytest.approx(21 / 11)


def test_aspect_ratio_wide_rectangle():
    assert aspect_ratio(_rect_contour(0, 0, 20, 10)) == pytest.approx(11 / 21)


def test_solidity_of_convex_rectangle_is_one():
    assert solidity(_rect_contour(0, 0, 10, 20)) == pytest.approx(1.0)


def test_solidity_of_notched_shape_is_less_than_one():
    notched = np.array(
        [[[0, 0]], [[10, 0]], [[10, 5]], [[5, 5]], [[5, 10]], [[0, 10]]], dtype=np.int32
    )
    assert solidity(notched) < 1.0


def test_row_normalised_area_scales_up_far_objects():
    near_contour = _rect_contour(0, 90, 10, 10)  # centroid row ~95, close to camera
    far_contour = _rect_contour(0, 10, 10, 10)  # centroid row ~15, far from camera
    reference_row = 95.0
    near_scaled = row_normalised_area(near_contour, reference_row)
    far_scaled = row_normalised_area(far_contour, reference_row)
    assert far_scaled > near_scaled  # same real area, farther one scaled up more


def test_saturation_ratio_detects_colour_content():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (0, 0, 255), thickness=-1)  # solid red block
    contour = _rect_contour(10, 10, 20, 20)
    assert saturation_ratio(frame, contour) > 0.9


def test_saturation_ratio_near_zero_for_greyscale_content():
    frame = np.full((50, 50, 3), 128, dtype=np.uint8)  # uniform grey, zero saturation
    contour = _rect_contour(10, 10, 20, 20)
    assert saturation_ratio(frame, contour) == pytest.approx(0.0, abs=1e-6)


def test_green_light_ratio_detects_green_flashlight():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (0, 255, 0), thickness=-1)  # solid green (BGR)
    contour = _rect_contour(10, 10, 20, 20)
    assert green_light_ratio(frame, contour) > 0.9


def test_green_light_ratio_ignores_non_green_colour():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (0, 0, 255), thickness=-1)  # solid red (BGR)
    contour = _rect_contour(10, 10, 20, 20)
    assert green_light_ratio(frame, contour) == pytest.approx(0.0)


def test_green_light_ratio_zero_for_greyscale_content():
    frame = np.full((50, 50, 3), 128, dtype=np.uint8)  # uniform grey
    contour = _rect_contour(10, 10, 20, 20)
    assert green_light_ratio(frame, contour) == pytest.approx(0.0)


def test_green_light_ratio_ignores_dim_green():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (0, 40, 0), thickness=-1)  # dim green, low value
    contour = _rect_contour(10, 10, 20, 20)
    assert green_light_ratio(frame, contour) == pytest.approx(0.0)


def test_edge_density_higher_for_textured_region():
    checker = np.zeros((20, 20, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    uniform = np.full((20, 20, 3), 128, dtype=np.uint8)
    contour = _rect_contour(0, 0, 20, 20)
    assert edge_density(checker, contour) > edge_density(uniform, contour)


def test_path_length_sums_step_distances():
    centroids = [(0.0, 0.0), (3.0, 4.0), (3.0, 0.0)]
    assert path_length(centroids) == pytest.approx(9.0)


def test_path_length_single_point_is_zero():
    assert path_length([(1.0, 1.0)]) == 0.0


def test_jitter_zero_for_perfectly_uniform_motion():
    centroids = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    assert jitter(centroids) == pytest.approx(0.0, abs=1e-9)


def test_jitter_higher_for_erratic_motion():
    smooth = [(float(i), 0.0) for i in range(10)]
    erratic = [(0.0, 0.0), (5.0, 3.0), (1.0, -2.0), (6.0, 4.0), (0.0, 0.0), (7.0, -3.0)]
    assert jitter(erratic) > jitter(smooth)


def test_persistence_fraction():
    assert persistence(15, 30) == pytest.approx(0.5)


def test_persistence_zero_total_frames_is_zero():
    assert persistence(0, 0) == 0.0


def test_time_of_day_deep_night_utc_is_night():
    # 22:00 UTC + 2h offset = 00:00 local, well inside 18:00-06:00.
    assert time_of_day("2026-01-01T22:00:00Z") == "night"


def test_time_of_day_midday_utc_is_day():
    # 10:00 UTC + 2h offset = 12:00 local, well outside 18:00-06:00.
    assert time_of_day("2026-01-01T10:00:00Z") == "day"


def test_time_of_day_respects_custom_window():
    assert (
        time_of_day("2026-01-01T10:00:00Z", window_start="08:00", window_end="20:00") == "night"
    )


def test_time_of_day_at_window_start_boundary_is_night():
    # 16:00 UTC + 2h offset = 18:00 local, exactly the window start (inclusive).
    assert time_of_day("2026-01-01T16:00:00Z") == "night"


def test_time_of_day_at_window_end_boundary_is_day():
    # 04:00 UTC + 2h offset = 06:00 local, exactly the window end (exclusive).
    assert time_of_day("2026-01-01T04:00:00Z") == "day"
