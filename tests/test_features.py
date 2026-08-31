import cv2
import numpy as np
import pytest

from src.features import (
    area_stability,
    aspect_ratio,
    color_saturation_fraction,
    edge_density,
    flare_frames,
    flare_settle_index,
    green_light_flicker,
    green_light_mask,
    green_light_ratio,
    heading_change,
    ignore_region_mask,
    jitter,
    longest_detection_run,
    normalised_speed,
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


def test_color_saturation_fraction_high_for_broad_daylight_colour():
    # Green foliage covering most of the frame -- genuine ambient colour, not
    # a small lit source.
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[:, :] = (0, 180, 0)  # BGR green, saturated
    assert color_saturation_fraction(frame) > 0.9


def test_color_saturation_fraction_low_for_ir_greyscale_with_small_light():
    # Near-monochrome IR frame with only a small saturated (flashlight) patch --
    # should not read as broad daylight colour.
    frame = np.full((50, 50, 3), 100, dtype=np.uint8)  # grey, zero saturation
    cv2.rectangle(frame, (5, 5), (9, 9), (0, 255, 0), thickness=-1)  # tiny green light
    assert color_saturation_fraction(frame) < 0.05


def test_color_saturation_fraction_zero_for_pure_greyscale():
    frame = np.full((50, 50, 3), 128, dtype=np.uint8)
    assert color_saturation_fraction(frame) == pytest.approx(0.0, abs=1e-6)


def test_green_light_mask_flags_only_the_green_pixels():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (0, 255, 0), thickness=-1)  # solid green (BGR)
    mask = green_light_mask(frame)
    assert mask[20, 20]
    assert not mask[5, 5]


def test_green_light_mask_ignores_dim_green():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (0, 40, 0), thickness=-1)  # dim green, low value
    assert not np.any(green_light_mask(frame))


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


def test_green_light_flicker_high_for_swinging_beam():
    # on/off/on/off, like a flashlight beam swinging in and out of frame
    assert green_light_flicker([0.0, 0.8, 0.0, 0.7, 0.0]) > 0.3


def test_ignore_region_mask_flags_only_the_configured_polygon():
    # Normalised top-left quarter of a 100x100 frame.
    polygon = [((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))]
    mask = ignore_region_mask(100, 100, polygon)
    assert mask[10, 10]  # inside the polygon
    assert not mask[90, 90]  # outside the polygon


def test_ignore_region_mask_empty_when_no_polygons():
    mask = ignore_region_mask(100, 100, [])
    assert not np.any(mask)


def test_ignore_region_mask_skips_degenerate_polygons():
    # A polygon with fewer than 3 points can't enclose an area -- should be
    # silently skipped rather than raising or filling the whole frame.
    mask = ignore_region_mask(100, 100, [((0.0, 0.0), (0.5, 0.5))])
    assert not np.any(mask)


def test_green_light_ratio_exclude_mask_ignores_a_known_light_region():
    # A stationary light (or lens edge/vignette artifact) sitting inside a
    # configured exclude region should not count towards the ratio, even
    # though it reads exactly like the guard's flashlight to the hue check.
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (0, 255, 0), thickness=-1)  # solid green (BGR)
    contour = _rect_contour(10, 10, 20, 20)
    exclude_mask = np.zeros((50, 50), dtype=bool)
    exclude_mask[10:30, 10:30] = True  # covers the whole green region

    assert green_light_ratio(frame, contour, exclude_mask=exclude_mask) == pytest.approx(0.0)


def test_green_light_ratio_exclude_mask_leaves_other_green_pixels_intact():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (0, 255, 0), thickness=-1)  # solid green (BGR)
    contour = _rect_contour(10, 10, 20, 20)
    exclude_mask = np.zeros((50, 50), dtype=bool)
    exclude_mask[0:5, 0:5] = True  # nowhere near the green region

    assert green_light_ratio(frame, contour, exclude_mask=exclude_mask) > 0.9


def test_green_light_flicker_zero_for_steady_signal():
    assert green_light_flicker([0.2, 0.2, 0.2, 0.2]) == pytest.approx(0.0)


def test_green_light_flicker_zero_for_fewer_than_two_values():
    assert green_light_flicker([]) == 0.0
    assert green_light_flicker([0.5]) == 0.0


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


def test_longest_detection_run_prefers_unbroken_detections():
    scattered = longest_detection_run([0, 2, 4, 6, 8, 10], 12)
    unbroken = longest_detection_run([0, 1, 2, 3, 4, 5], 12)
    assert scattered == pytest.approx(1 / 12)
    assert unbroken == pytest.approx(0.5)


def test_longest_detection_run_empty_is_zero():
    assert longest_detection_run([], 10) == 0.0
    assert longest_detection_run([0, 1], 0) == 0.0


def test_area_stability_zero_for_constant_area():
    assert area_stability([100.0, 100.0, 100.0]) == pytest.approx(0.0)


def test_area_stability_higher_for_pulsing_blob():
    steady = area_stability([100.0, 105.0, 98.0, 102.0])
    pulsing = area_stability([10.0, 400.0, 20.0, 350.0])
    assert pulsing > steady


def test_area_stability_degenerate_inputs_are_zero():
    assert area_stability([]) == 0.0
    assert area_stability([50.0]) == 0.0
    assert area_stability([0.0, 0.0]) == 0.0


def test_normalised_speed_is_scale_invariant():
    # Same motion-to-size ratio at two different apparent sizes.
    small = normalised_speed([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)], blob_width=4.0)
    large = normalised_speed([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)], blob_width=20.0)
    assert small == pytest.approx(large)
    assert small == pytest.approx(0.5)


def test_normalised_speed_degenerate_inputs_are_zero():
    assert normalised_speed([(0.0, 0.0)], blob_width=10.0) == 0.0
    assert normalised_speed([(0.0, 0.0), (5.0, 0.0)], blob_width=0.0) == 0.0


def test_heading_change_zero_for_straight_track():
    straight = [(float(i), 0.0) for i in range(6)]
    assert heading_change(straight) == pytest.approx(0.0, abs=1e-9)


def test_heading_change_near_pi_for_reversing_track():
    zigzag = [(0.0, 0.0), (10.0, 0.0), (0.0, 0.0), (10.0, 0.0), (0.0, 0.0)]
    assert heading_change(zigzag) == pytest.approx(np.pi, abs=1e-6)


def test_heading_change_ignores_sub_pixel_steps():
    # Steps below min_step carry no meaningful direction, so a jittering but
    # stationary blob must not read as a turning track.
    jittering = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)]
    assert heading_change(jittering) == 0.0


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


def test_flare_frames_marks_both_sides_of_a_gain_step():
    # Settled at 40, one 15-level step up to 55, settled again.
    assert flare_frames([40, 40, 55, 55, 55]) == [False, True, True, False, False]


def test_flare_frames_ignores_subject_sized_median_drift():
    # A subject crossing shifts the whole-frame median by a level or two at most.
    assert flare_frames([40, 41, 42, 41, 40]) == [False] * 5


def test_flare_frames_catches_a_mid_clip_illuminator_kick():
    flagged = flare_frames([40, 40, 40, 60, 60, 60])
    assert flagged == [False, False, True, True, False, False]


def test_flare_frames_degenerate_inputs():
    assert flare_frames([]) == []
    assert flare_frames([40.0]) == [False]


def test_flare_settle_index_is_after_the_opening_ramp():
    # 10 -> 25 -> 40 are gain steps; the trailing 40 -> 42 is below tolerance,
    # so the ramp is considered settled from frame 3 onwards.
    medians = [10, 25, 40, 42, 42, 42, 42, 42, 42, 42]
    assert flare_settle_index(medians) == 3


def test_flare_settle_index_is_zero_for_a_clip_that_never_flares():
    assert flare_settle_index([44] * 10) == 0


def test_flare_settle_index_never_discards_more_than_max_fraction():
    # A short clip that ramps most of the way through must still keep footage:
    # the cameras are motion-triggered, so the subject is already in frame.
    medians = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert flare_settle_index(medians, max_fraction=0.4) == 4
