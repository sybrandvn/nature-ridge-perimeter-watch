import cv2
import numpy as np
import pytest

from src.features import (
    apply_photometric_match,
    area_stability,
    aspect_ratio,
    blob_white_fraction,
    color_saturation_fraction,
    daylight_hint,
    depth_progression,
    detect_stationary_light_mask,
    edge_density,
    flare_frames,
    flare_settle_index,
    flashlight_bbox_overlap,
    green_light_flicker,
    green_light_mask,
    green_light_ratio,
    heading_change,
    ignore_region_mask,
    is_daylight,
    is_twilight,
    jitter,
    longest_detection_run,
    minutes_from_daylight_boundary,
    normalised_speed,
    optical_flow_direction_coherence,
    path_length,
    persistence,
    photometric_match,
    photometric_match_color,
    post_flash_red_shift,
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


def test_blob_white_fraction_detects_overexposed_content():
    frame = np.full((50, 50, 3), 50, dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (250, 250, 250), thickness=-1)  # overexposed
    contour = _rect_contour(10, 10, 20, 20)
    assert blob_white_fraction(frame, contour) > 0.9


def test_blob_white_fraction_zero_for_dim_content():
    frame = np.full((50, 50, 3), 50, dtype=np.uint8)
    contour = _rect_contour(10, 10, 20, 20)
    assert blob_white_fraction(frame, contour) == pytest.approx(0.0)


def _bgr(b: int, g: int, r: int) -> np.ndarray:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (b, g, r)
    return frame


def test_post_flash_red_shift_detects_red_settle_after_a_flash():
    # 3 neutral frames, a bright flash, then 3 visibly red-tinted frames
    frames = [_bgr(60, 60, 60)] * 3 + [_bgr(250, 250, 250)] + [_bgr(60, 60, 90)] * 3
    assert post_flash_red_shift(frames) > 0.4


def test_post_flash_red_shift_zero_when_tint_unchanged():
    # a flash with no colour change afterwards -- an obstruction, not a flashlight
    frames = [_bgr(60, 60, 60)] * 3 + [_bgr(250, 250, 250)] + [_bgr(60, 60, 60)] * 3
    assert post_flash_red_shift(frames) == pytest.approx(0.0, abs=1e-6)


def test_post_flash_red_shift_zero_for_too_few_frames():
    assert post_flash_red_shift([_bgr(60, 60, 60)] * 3) == 0.0


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


def test_flashlight_bbox_overlap_high_when_box_is_mostly_the_beam():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (0, 255, 0), thickness=-1)  # solid green (BGR)
    assert flashlight_bbox_overlap(frame, (10, 10, 20, 20)) > 0.9


def test_flashlight_bbox_overlap_zero_for_non_green_box():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (200, 200, 200), thickness=-1)  # bright white
    assert flashlight_bbox_overlap(frame, (10, 10, 20, 20)) == pytest.approx(0.0)


def test_flashlight_bbox_overlap_excludes_a_known_stationary_light():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (29, 29), (0, 255, 0), thickness=-1)  # solid green (BGR)
    exclude_mask = np.zeros((50, 50), dtype=bool)
    exclude_mask[10:30, 10:30] = True  # covers the whole green region
    assert flashlight_bbox_overlap(frame, (10, 10, 20, 20), exclude_mask=exclude_mask) == (
        pytest.approx(0.0)
    )


def test_green_light_flicker_zero_for_fewer_than_two_values():
    assert green_light_flicker([]) == 0.0
    assert green_light_flicker([0.5]) == 0.0


def _frame_with_green_patch(x: int, y: int, w: int, h: int) -> np.ndarray:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), thickness=-1)
    return frame


def test_detect_stationary_light_mask_flags_a_light_fixed_across_every_frame():
    frames = [_frame_with_green_patch(5, 5, 6, 6) for _ in range(10)]
    mask = detect_stationary_light_mask(frames)
    assert mask[5:11, 5:11].all()


def test_detect_stationary_light_mask_ignores_a_light_that_only_appears_briefly():
    # The guard's flashlight lighting up a different spot almost every frame --
    # never stable enough at any one pixel to count as a fixed light.
    frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(10)]
    for i, frame in enumerate(frames):
        cv2.rectangle(frame, (i, i), (i + 3, i + 3), (0, 255, 0), thickness=-1)
    mask = detect_stationary_light_mask(frames)
    assert not np.any(mask)


def test_detect_stationary_light_mask_respects_exclude_region():
    # A guard standing still with the flashlight held on one spot for the whole
    # clip reads exactly like a fixed light -- exclude_region (the tracked
    # subject's own box) must protect it from being auto-classified as one.
    frames = [_frame_with_green_patch(5, 5, 6, 6) for _ in range(10)]
    exclude_region = np.zeros((50, 50), dtype=bool)
    exclude_region[0:20, 0:20] = True
    mask = detect_stationary_light_mask(frames, exclude_region=exclude_region)
    assert not np.any(mask)


def test_detect_stationary_light_mask_empty_frames_returns_empty_mask():
    mask = detect_stationary_light_mask([])
    assert mask.shape == (0, 0)


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


def test_depth_progression_one_for_steady_one_way_travel():
    assert depth_progression([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)


def test_depth_progression_zero_for_no_net_movement():
    # out then back to the start -- total variation is nonzero but net change is 0
    assert depth_progression([5.0, 8.0, 5.0]) == pytest.approx(0.0)


def test_depth_progression_partial_for_mixed_travel():
    steady = depth_progression([1.0, 2.0, 3.0, 4.0])
    mixed = depth_progression([1.0, 3.0, 2.0, 4.0])
    assert 0.0 < mixed < steady


def test_depth_progression_degenerate_inputs_are_zero():
    assert depth_progression([]) == 0.0
    assert depth_progression([5.0]) == 0.0
    assert depth_progression([5.0, 5.0, 5.0]) == 0.0


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


def _textured_patch(size: int = 60, seed: int = 0) -> np.ndarray:
    """Blurred noise, not a flat patch or raw independent-pixel noise --
    goodFeaturesToTrack/calcOpticalFlowPyrLK need real LOCAL spatial
    correlation to find and follow corners reliably (matching detect_clip's
    own preprocessing, which Gaussian-blurs every frame before diffing --
    see `scripts.spike.detect_clip`)."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def _shifted(patch: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    """`patch` translated by (dx, dy), edge-replicated rather than zero-filled
    so the shift itself doesn't manufacture a false hard edge for
    goodFeaturesToTrack to key on."""
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        patch, matrix, (patch.shape[1], patch.shape[0]), borderMode=cv2.BORDER_REPLICATE
    )


def test_optical_flow_direction_coherence_high_for_uniform_translation():
    # The whole patch shifts 5px right -- every tracked corner should move the
    # same way, a textbook rigid-body translation.
    prev = _textured_patch()
    curr = _shifted(prev, dx=5)
    contour = _rect_contour(10, 10, 40, 40)

    result = optical_flow_direction_coherence(prev, curr, contour)

    assert result is not None
    assert result > 0.9


def test_optical_flow_direction_coherence_low_for_opposing_motion():
    # Top half of the contour shifts right, bottom half shifts left -- two
    # equal-sized clusters of opposite-direction vectors, the signature this
    # feature exists to catch (a branch swinging one way while another part
    # of it swings back).
    prev = _textured_patch()
    curr = prev.copy()
    curr[:30, :] = _shifted(prev, dx=6)[:30, :]
    curr[30:, :] = _shifted(prev, dx=-6)[30:, :]
    contour = _rect_contour(5, 5, 50, 50)

    result = optical_flow_direction_coherence(prev, curr, contour)

    assert result is not None
    assert result < 0.5


def test_optical_flow_direction_coherence_none_for_textureless_patch():
    # No texture at all -- goodFeaturesToTrack finds nothing to track, so
    # there is no direction to measure, which must not read as a confident 0.0.
    prev = np.full((60, 60), 128, dtype=np.uint8)
    curr = np.full((60, 60), 128, dtype=np.uint8)
    contour = _rect_contour(10, 10, 40, 40)

    assert optical_flow_direction_coherence(prev, curr, contour) is None


def test_optical_flow_direction_coherence_none_when_nothing_moves():
    # Real texture, but the two frames are identical -- every tracked point's
    # displacement is below min_displacement, so direction is undefined, not
    # a confident 0.0 (which would misread as "maximally incoherent").
    prev = _textured_patch()
    curr = prev.copy()
    contour = _rect_contour(10, 10, 40, 40)

    assert optical_flow_direction_coherence(prev, curr, contour) is None


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


def test_is_daylight_true_for_confirmed_january_dawn_clip():
    # cam03/18574: 03:57 UTC = 05:57 local on 2026-01-25, frame is full-colour daylight.
    assert is_daylight("2026-01-25T03:57:32Z") is True


def test_is_daylight_true_for_confirmed_december_dawn_clip():
    # cam05/9430: 03:46 UTC = 05:46 local on 2024-12-02, frame is full-colour daylight.
    assert is_daylight("2024-12-02T03:46:55Z") is True


def test_is_daylight_true_for_confirmed_december_dusk_clip():
    # cam05/9690: 16:05 UTC = 18:05 local on 2024-12-08, frame is full-colour daylight.
    assert is_daylight("2024-12-08T16:05:20Z") is True


def test_is_daylight_false_for_deep_night():
    # 22:00 UTC + 2h offset = 00:00 local, well outside any month's sunrise-sunset span.
    assert is_daylight("2026-01-01T22:00:00Z") is False


def test_is_daylight_false_for_pre_sunrise_winter_morning():
    # 03:30 UTC + 2h offset = 05:30 local in June, before the ~06:50 winter sunrise.
    assert is_daylight("2026-06-15T03:30:00Z") is False


def test_minutes_from_daylight_boundary_positive_after_sunset():
    # cam03/18512: 17:18:57 UTC = 19:18 local on 2026-01-21, January sunset 18:55.
    assert minutes_from_daylight_boundary("2026-01-21T17:18:57Z") == pytest.approx(23.0)


def test_minutes_from_daylight_boundary_negative_before_sunrise():
    # 03:30 UTC + 2h offset = 05:30 local in June, 80 min before the 06:50 sunrise.
    assert minutes_from_daylight_boundary("2026-06-15T03:30:00Z") == pytest.approx(-80.0)


def test_minutes_from_daylight_boundary_picks_the_nearer_boundary():
    # January (sunrise 05:30, sunset 18:55): 17:00 local is 115min before sunset,
    # nearer than its 690min-past-sunrise distance.
    assert minutes_from_daylight_boundary("2026-01-01T15:00:00Z") == pytest.approx(-115.0)


def test_is_twilight_true_just_after_sunset():
    # Same clip as the batch-2 false positive: 24 minutes after January sunset.
    assert is_twilight("2026-01-21T17:18:57Z", margin_minutes=60) is True


def test_is_twilight_false_well_after_dusk():
    assert is_twilight("2026-01-21T20:00:00Z", margin_minutes=60) is False


def test_is_twilight_false_at_deep_night():
    assert is_twilight("2026-01-01T22:00:00Z", margin_minutes=60) is False


def test_is_twilight_true_near_sunrise():
    # 05:00 local in June is 110 min before the 06:50 sunrise -- outside a 60min
    # margin, but inside a wider one.
    assert is_twilight("2026-06-15T03:00:00Z", margin_minutes=60) is False
    assert is_twilight("2026-06-15T03:00:00Z", margin_minutes=120) is True


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


def test_photometric_match_recovers_a_known_gain_and_offset():
    rng = np.random.default_rng(0)
    source = rng.uniform(0, 200, size=(20, 20)).astype(np.float64)
    reference = source * 1.4 + 20.0
    gain, offset = photometric_match(source, reference)
    assert gain == pytest.approx(1.4, abs=1e-6)
    assert offset == pytest.approx(20.0, abs=1e-6)


def test_apply_photometric_match_reconstructs_the_reference():
    rng = np.random.default_rng(1)
    source = rng.uniform(0, 200, size=(20, 20)).astype(np.uint8)
    reference = np.clip(source.astype(np.float64) * 0.8 + 10.0, 0, 255).astype(np.uint8)
    gain, offset = photometric_match(source, reference)
    corrected = apply_photometric_match(source, gain, offset)
    assert np.mean(np.abs(corrected.astype(np.float64) - reference.astype(np.float64))) < 1.0


def test_photometric_match_falls_back_to_offset_for_a_blank_frame():
    # Zero variance (e.g. a fully saturated/blank frame) makes a gain fit
    # undefined -- must fall back to a pure brightness shift, not NaN/inf.
    source = np.full((10, 10), 200.0)
    reference = np.full((10, 10), 150.0)
    gain, offset = photometric_match(source, reference)
    assert gain == pytest.approx(1.0)
    assert offset == pytest.approx(-50.0)


def test_photometric_match_color_corrects_each_channel_independently():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[:, :] = (40, 60, 80)  # BGR, a colour cast
    reference = np.zeros((10, 10, 3), dtype=np.uint8)
    reference[:, :] = (100, 100, 100)  # neutral grey
    corrected = photometric_match_color(frame, reference)
    assert np.mean(np.abs(corrected.astype(np.float64) - 100.0)) < 1.0


def test_daylight_hint_true_in_the_middle_of_the_day():
    assert daylight_hint("2026-01-15T10:00:00Z") is True


def test_daylight_hint_true_just_after_sunset_via_twilight():
    # 19:30 local in January, 35 minutes past the 18:55 sunset -- not daylight,
    # but still carrying real ambient colour, so the gate should stay on.
    assert is_daylight("2026-01-15T17:30:00Z") is False
    assert daylight_hint("2026-01-15T17:30:00Z") is True


def test_daylight_hint_false_in_the_middle_of_the_night():
    assert daylight_hint("2026-01-15T22:00:00Z") is False


def test_daylight_hint_none_without_a_timestamp():
    assert daylight_hint(None) is None
    assert daylight_hint("") is None


def test_daylight_hint_none_on_an_unparseable_timestamp():
    assert daylight_hint("not-a-timestamp") is None
