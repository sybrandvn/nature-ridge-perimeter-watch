import pytest

from src.config import CameraZone
from src.zones import (
    _picket_direction_at_y,
    classify_zone,
    effective_fence,
    entered_band_from_outside,
    estimated_height_m,
    estimated_speed_mps,
    estimated_width_m,
    fence_separation_at_y,
    fence_vanishing_point,
    in_fence_band,
    in_ignore_region,
    is_beyond_depth_cutoff,
    is_grounded_at_fence,
    median_fence_distance,
    outside_pixel_fraction,
    pixels_per_metre_at_y,
    side_name,
    signed_side,
    subject_base_y,
    track_crosses_fence,
)

# Vertical fence at x=0.5, spanning the frame top to bottom -- the typical shape
# for these cameras (looking down the fence line).
_VERTICAL_FENCE = ((0.5, 0.0), (0.5, 1.0))


def test_signed_side_requires_at_least_two_points():
    with pytest.raises(ValueError, match="fence needs"):
        signed_side((0.5, 0.5), [(0.0, 0.0)])


def test_side_name_right_of_vertical_fence():
    assert side_name((0.7, 0.5), _VERTICAL_FENCE) == "right"


def test_side_name_left_of_vertical_fence():
    assert side_name((0.3, 0.5), _VERTICAL_FENCE) == "left"


def test_side_name_on_line():
    assert side_name((0.5, 0.5), _VERTICAL_FENCE) == "on_line"


def test_side_name_ignores_point_order():
    # Plain screen position, not direction-of-travel -- reversing the polyline's
    # point order must not change which side a point falls on.
    reversed_fence = tuple(reversed(_VERTICAL_FENCE))
    assert side_name((0.7, 0.5), reversed_fence) == "right"
    assert side_name((0.3, 0.5), reversed_fence) == "left"


def test_side_name_picks_segment_by_row_for_bent_polyline():
    bent_fence = ((0.2, 0.0), (0.5, 0.5), (0.2, 1.0))
    # Row 0.25 sits in the first segment (fence x interpolates to ~0.35).
    assert side_name((0.5, 0.25), bent_fence) == "right"
    assert side_name((0.2, 0.25), bent_fence) == "left"
    # Row 0.75 sits in the second segment (fence x interpolates to ~0.35).
    assert side_name((0.5, 0.75), bent_fence) == "right"
    assert side_name((0.2, 0.75), bent_fence) == "left"


def test_side_name_extrapolates_beyond_fence_row_range():
    # y=-0.1 is above the fence's first point -- extrapolate along the nearest
    # (first) segment rather than erroring.
    fence = ((0.4, 0.1), (0.4, 0.9))
    assert side_name((0.6, -0.1), fence) == "right"
    assert side_name((0.2, 1.2), fence) == "left"



def test_is_beyond_depth_cutoff():
    zone = CameraZone(fence=None, outside=None, depth_cutoff=0.3, ignore=())
    assert is_beyond_depth_cutoff((0.5, 0.1), zone) is True
    assert is_beyond_depth_cutoff((0.5, 0.5), zone) is False


def test_in_ignore_region_detects_containment():
    polygon = ((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2))
    zone = CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=(polygon,))
    assert in_ignore_region((0.1, 0.1), zone) is True
    assert in_ignore_region((0.9, 0.9), zone) is False


def test_classify_zone_priorities_ignore_over_outside():
    polygon = ((0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0))
    zone = CameraZone(
        fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=(polygon,)
    )
    # (0.9, 0.5) is outside (right of the vertical fence) but inside the ignore polygon.
    assert classify_zone((0.9, 0.5), zone) == "ignored"


def test_classify_zone_priorities_depth_cutoff_over_outside():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="left", depth_cutoff=0.3, ignore=())
    # (0.3, 0.1) is outside (left of the fence) but also beyond depth cutoff.
    assert classify_zone((0.3, 0.1), zone) == "ambiguous"


def test_classify_zone_outside_and_inside():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="left", depth_cutoff=0.0, ignore=())
    assert classify_zone((0.3, 0.5), zone) == "outside"
    assert classify_zone((0.7, 0.5), zone) == "inside"


def test_classify_zone_ambiguous_without_fence_config():
    zone = CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=())
    assert classify_zone((0.5, 0.5), zone) == "ambiguous"


def test_outside_pixel_fraction_mixed_points():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="left", depth_cutoff=0.0, ignore=())
    points = [(0.3, 0.5), (0.35, 0.5), (0.7, 0.5)]  # 2 outside, 1 inside
    assert outside_pixel_fraction(points, zone) == pytest.approx(2 / 3)


def test_outside_pixel_fraction_excludes_ignored_and_ambiguous():
    polygon = ((0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0))
    zone = CameraZone(
        fence=_VERTICAL_FENCE, outside="left", depth_cutoff=0.3, ignore=(polygon,)
    )
    points = [
        (0.3, 0.1),  # beyond depth cutoff -> ambiguous, excluded
        (0.9, 0.5),  # inside ignore polygon -> ignored, excluded
        (0.55, 0.5),  # inside (right of the fence line), classifiable
    ]
    assert outside_pixel_fraction(points, zone) == 0.0


def test_outside_pixel_fraction_empty_relevant_set_is_zero():
    zone = CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=())
    assert outside_pixel_fraction([(0.5, 0.5)], zone) == 0.0


def test_track_crosses_fence_detects_side_change():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="left", depth_cutoff=0.0, ignore=())
    crossing = [(0.2, 0.4), (0.4, 0.5), (0.7, 0.6)]
    assert track_crosses_fence(crossing, zone) is True


def test_track_crosses_fence_false_when_track_stays_one_side():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="left", depth_cutoff=0.0, ignore=())
    assert track_crosses_fence([(0.2, 0.4), (0.3, 0.5), (0.1, 0.6)], zone) is False


def test_track_crosses_fence_needs_fence_and_two_points():
    with_fence = CameraZone(fence=_VERTICAL_FENCE, outside="left", depth_cutoff=0.0, ignore=())
    without_fence = CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=())
    assert track_crosses_fence([(0.2, 0.5)], with_fence) is False
    assert track_crosses_fence([(0.2, 0.5), (0.8, 0.5)], without_fence) is False


def test_median_fence_distance_even_and_odd_lengths():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="left", depth_cutoff=0.0, ignore=())
    assert median_fence_distance([(0.6, 0.5), (0.9, 0.5), (0.7, 0.5)], zone) == pytest.approx(0.2)
    assert median_fence_distance([(0.6, 0.5), (0.9, 0.5)], zone) == pytest.approx(0.25)


def test_median_fence_distance_without_fence_or_track_is_zero():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="left", depth_cutoff=0.0, ignore=())
    no_fence = CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=())
    assert median_fence_distance([], zone) == 0.0
    assert median_fence_distance([(0.9, 0.5)], no_fence) == 0.0


# --------------------------------------------------------------------------
# effective_fence + the classification functions preferring fence_bottom
# --------------------------------------------------------------------------

_BOTTOM_FENCE = ((0.7, 0.0), (0.7, 1.0))


def test_effective_fence_prefers_fence_bottom():
    zone = CameraZone(
        fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=(),
        fence_bottom=_BOTTOM_FENCE,
    )
    assert effective_fence(zone) == _BOTTOM_FENCE


def test_effective_fence_falls_back_to_fence():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=())
    assert effective_fence(zone) == _VERTICAL_FENCE


def test_classify_zone_uses_fence_bottom_when_present():
    # A point between the top (x=0.5) and bottom (x=0.7) lines reads "inside"
    # under fence_bottom even though it would read "outside" under fence alone.
    zone = CameraZone(
        fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=(),
        fence_bottom=_BOTTOM_FENCE,
    )
    assert classify_zone((0.6, 0.5), zone) == "inside"
    assert classify_zone((0.6, 0.5), CameraZone(
        fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=()
    )) == "outside"


def test_track_crosses_fence_uses_fence_bottom_when_present():
    zone = CameraZone(
        fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=(),
        fence_bottom=_BOTTOM_FENCE,
    )
    # Both points sit between the two lines -- crosses the top-only fence but
    # not the base line actually driving classification.
    assert track_crosses_fence([(0.55, 0.4), (0.6, 0.6)], zone) is False


def test_median_fence_distance_uses_fence_bottom_when_present():
    zone = CameraZone(
        fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=(),
        fence_bottom=_BOTTOM_FENCE,
    )
    assert median_fence_distance([(0.9, 0.5)], zone) == pytest.approx(0.2)


# --------------------------------------------------------------------------
# Fence-pair calibration (top + base line -> pixels-per-metre ruler)
# --------------------------------------------------------------------------


def _calibrated_zone(**overrides) -> CameraZone:
    defaults = dict(
        fence=((0.5, 0.0), (0.5, 1.0)),
        outside="right",
        depth_cutoff=0.0,
        ignore=(),
        fence_bottom=((0.6, 0.0), (0.6, 1.0)),
        fence_height_m=2.0,
    )
    defaults.update(overrides)
    return CameraZone(**defaults)


def test_is_grounded_at_fence_true_within_base_line_range():
    zone = _calibrated_zone()
    assert is_grounded_at_fence(0.5, zone) is True


def test_is_grounded_at_fence_false_above_base_line_range():
    zone = _calibrated_zone(fence_bottom=((0.6, 0.2), (0.6, 0.9)))
    assert is_grounded_at_fence(0.1, zone) is False


def test_is_grounded_at_fence_false_without_fence_bottom():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=())
    assert is_grounded_at_fence(0.5, zone) is False


def test_fence_separation_at_y_computes_pixel_gap():
    zone = _calibrated_zone()
    # 0.1 normalised x separation * 320px frame width = 32px.
    assert fence_separation_at_y(0.5, zone, frame_width=320, frame_height=240) == pytest.approx(
        32.0
    )


def test_fence_separation_at_y_none_without_fence_bottom():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=())
    assert fence_separation_at_y(0.5, zone, frame_width=320, frame_height=240) is None


def test_fence_separation_at_y_none_below_min_separation():
    zone = _calibrated_zone(fence_bottom=((0.501, 0.0), (0.501, 1.0)))
    assert fence_separation_at_y(0.5, zone, frame_width=320, frame_height=240) is None


def test_fence_separation_at_y_none_when_not_grounded():
    zone = _calibrated_zone(fence_bottom=((0.6, 0.2), (0.6, 0.9)))
    assert fence_separation_at_y(0.05, zone, frame_width=320, frame_height=240) is None


def test_pixels_per_metre_at_y():
    zone = _calibrated_zone()
    # 32px separation / 2.0m fence height = 16 px/m.
    assert pixels_per_metre_at_y(0.5, zone, frame_width=320, frame_height=240) == pytest.approx(
        16.0
    )


def test_pixels_per_metre_at_y_none_when_uncalibrated():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=())
    assert pixels_per_metre_at_y(0.5, zone, frame_width=320, frame_height=240) is None


def test_pixels_per_metre_at_y_none_below_min_pixels_per_metre():
    # 32px separation / 4.0m "fence height" = 8 px/m, below the MIN_PIXELS_PER_METRE
    # floor -- far enough away that ordinary pixel noise isn't trustworthy.
    zone = _calibrated_zone(fence_height_m=4.0)
    assert pixels_per_metre_at_y(0.5, zone, frame_width=320, frame_height=240) is None


def test_subject_base_y_uses_bbox_bottom_edge():
    # bbox (x, y, w, h) = (10, 20, 30, 40) in a 240px-tall frame -> bottom row
    # (20 + 40) / 240.
    assert subject_base_y((10, 20, 30, 40), frame_height=240) == pytest.approx(60 / 240)


def test_estimated_height_m_uses_pixels_per_metre_scale():
    zone = _calibrated_zone()
    # 32px scale @ 16 px/m -> 2.0m tall.
    assert estimated_height_m(
        32.0, 0.5, zone, frame_width=320, frame_height=240
    ) == pytest.approx(2.0)


def test_estimated_height_m_none_when_uncalibrated():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=())
    assert estimated_height_m(32.0, 0.5, zone, frame_width=320, frame_height=240) is None


def test_estimated_width_m_uses_pixels_per_metre_scale():
    zone = _calibrated_zone()
    assert estimated_width_m(8.0, 0.5, zone, frame_width=320, frame_height=240) == pytest.approx(
        0.5
    )


def test_estimated_speed_mps_uses_pixels_per_metre_scale_and_dt():
    zone = _calibrated_zone()
    # 32px / 16 px/m = 2.0m travelled in 2s -> 1.0 m/s.
    assert estimated_speed_mps(
        32.0, 2.0, 0.5, zone, frame_width=320, frame_height=240
    ) == pytest.approx(1.0)


def test_estimated_speed_mps_none_for_non_positive_dt():
    zone = _calibrated_zone()
    assert estimated_speed_mps(32.0, 0.0, 0.5, zone, frame_width=320, frame_height=240) is None


# --------------------------------------------------------------------------
# fence_picket angle-correction: projects along a traced picket's own tilt
# instead of assuming the fence/fence_bottom pair is vertical at the same row.
# --------------------------------------------------------------------------


def test_fence_separation_at_y_uses_picket_angle_when_configured():
    # Top rail and base are both vertical (x=0.5 / x=0.6), but the picket
    # itself is tilted -- its projection crosses the top rail at a DIFFERENT
    # row than the base point, not row 0.5.
    zone = _calibrated_zone(fence_picket=((0.6, 0.4), (0.65, 0.5)))
    # origin = (0.6, 0.5); direction = (0.6-0.65, 0.4-0.5) = (-0.05, -0.1);
    # crosses fence (x=0.5) at t=2 -> point (0.5, 0.3).
    # dx_px = -0.1*320 = -32, dy_px = -0.2*240 = -48 -> hypot = ~57.69px.
    assert fence_separation_at_y(0.5, zone, frame_width=320, frame_height=240) == pytest.approx(
        57.6879, rel=1e-4
    )


def test_fence_separation_at_y_falls_back_without_picket_crossing():
    # A picket direction that never crosses the top rail within its segment
    # range must fall back to the naive same-row method, not return None.
    zone = _calibrated_zone(fence_picket=((0.6, 0.5), (0.65, 0.5)))  # horizontal, never rises
    assert fence_separation_at_y(0.5, zone, frame_width=320, frame_height=240) == pytest.approx(
        32.0
    )


def test_fence_separation_at_y_ignores_picket_when_not_configured():
    zone = _calibrated_zone()
    assert zone.fence_picket is None
    assert fence_separation_at_y(0.5, zone, frame_width=320, frame_height=240) == pytest.approx(
        32.0
    )


# --------------------------------------------------------------------------
# Picket tilt is itself depth-adjusted: full strength at the picket's own
# row, shrinking to upright (vertical) at depth_cutoff -- poles genuinely
# lean less the farther they are from the camera.
# --------------------------------------------------------------------------


def test_picket_direction_full_tilt_at_its_own_row():
    zone = _calibrated_zone(fence_picket=((0.6, 0.4), (0.65, 0.5)), depth_cutoff=0.1)
    assert _picket_direction_at_y(0.5, zone) == pytest.approx((-0.05, -0.1))


def test_picket_direction_upright_at_depth_cutoff():
    zone = _calibrated_zone(fence_picket=((0.6, 0.4), (0.65, 0.5)), depth_cutoff=0.1)
    dx, dy = _picket_direction_at_y(0.1, zone)
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(-0.1)


def test_picket_direction_interpolates_halfway():
    zone = _calibrated_zone(fence_picket=((0.6, 0.4), (0.65, 0.5)), depth_cutoff=0.1)
    # halfway between the picket's own row (0.5) and depth_cutoff (0.1) is 0.3.
    dx, dy = _picket_direction_at_y(0.3, zone)
    assert dx == pytest.approx(-0.025)
    assert dy == pytest.approx(-0.1)


def test_picket_direction_clamps_beyond_depth_cutoff():
    zone = _calibrated_zone(fence_picket=((0.6, 0.4), (0.65, 0.5)), depth_cutoff=0.1)
    dx, dy = _picket_direction_at_y(0.0, zone)
    assert dx == pytest.approx(0.0)


def test_picket_direction_clamps_nearer_than_its_own_row():
    zone = _calibrated_zone(fence_picket=((0.6, 0.4), (0.65, 0.5)), depth_cutoff=0.1)
    dx, dy = _picket_direction_at_y(0.9, zone)
    assert dx == pytest.approx(-0.05)


def test_picket_direction_none_without_fence_picket():
    zone = _calibrated_zone()
    assert _picket_direction_at_y(0.5, zone) is None


def test_picket_direction_falls_back_to_full_tilt_on_degenerate_span():
    # depth_cutoff at/beyond the picket's own row leaves no room to
    # interpolate -- return the raw (unmodified) direction rather than divide
    # by a non-positive span.
    zone = _calibrated_zone(fence_picket=((0.6, 0.4), (0.65, 0.5)), depth_cutoff=0.5)
    assert _picket_direction_at_y(0.5, zone) == pytest.approx((-0.05, -0.1))


# --------------------------------------------------------------------------
# fence_vanishing_point: where the top-rail and base lines converge if
# extended -- a geometrically-derived cap, preferred over depth_cutoff.
# --------------------------------------------------------------------------

# fence and fence_bottom converge (extrapolated) at (0.5667, -0.2) -- above
# the frame, but still a well-defined intersection of the two lines.
_CONVERGING_FENCE = ((0.5, 0.2), (0.4, 0.8))
_CONVERGING_BOTTOM = ((0.6, 0.2), (0.65, 0.8))


def test_fence_vanishing_point_computes_convergence():
    zone = CameraZone(
        fence=_CONVERGING_FENCE, fence_bottom=_CONVERGING_BOTTOM, outside="right",
        depth_cutoff=0.0, ignore=(),
    )
    point = fence_vanishing_point(zone)
    assert point[0] == pytest.approx(0.5667, abs=1e-3)
    assert point[1] == pytest.approx(-0.2, abs=1e-3)


def test_fence_vanishing_point_none_when_parallel():
    zone = _calibrated_zone()  # fence and fence_bottom are both vertical
    assert fence_vanishing_point(zone) is None


def test_fence_vanishing_point_none_without_both_lines():
    zone = CameraZone(fence=None, fence_bottom=_CONVERGING_BOTTOM, outside="right",
                       depth_cutoff=0.0, ignore=())
    assert fence_vanishing_point(zone) is None


def test_picket_direction_prefers_vanishing_point_over_depth_cutoff():
    zone = CameraZone(
        fence=_CONVERGING_FENCE, fence_bottom=_CONVERGING_BOTTOM, outside="right",
        depth_cutoff=0.05, ignore=(), fence_picket=((0.6, 0.6), (0.65, 0.7)),
    )
    # Cap should be the vanishing point's y (-0.2), not depth_cutoff (0.05) --
    # at row 0.7 (the picket's own base row) tilt is still full strength.
    dx, dy = _picket_direction_at_y(0.7, zone)
    assert dx == pytest.approx(-0.05)


def test_picket_direction_falls_back_to_depth_cutoff_when_no_vanishing_point():
    # Parallel lines have no vanishing point -- must fall back to depth_cutoff,
    # matching the pre-vanishing-point behaviour exactly.
    zone = _calibrated_zone(fence_picket=((0.6, 0.4), (0.65, 0.5)), depth_cutoff=0.1)
    dx, dy = _picket_direction_at_y(0.1, zone)
    assert dx == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Fence "band" (the structure itself, between the two lines) -- discovery
# stage only, see docs/plan.md.
# --------------------------------------------------------------------------


def test_in_fence_band_true_between_the_two_lines():
    zone = _calibrated_zone()  # fence x=0.5, fence_bottom x=0.6, both vertical
    assert in_fence_band((0.55, 0.5), zone) is True


def test_in_fence_band_false_outside_both_lines():
    zone = _calibrated_zone()
    assert in_fence_band((0.9, 0.5), zone) is False
    assert in_fence_band((0.1, 0.5), zone) is False


def test_in_fence_band_false_without_fence_bottom():
    zone = CameraZone(fence=_VERTICAL_FENCE, outside="right", depth_cutoff=0.0, ignore=())
    assert in_fence_band((0.5, 0.5), zone) is False


def test_in_fence_band_false_when_not_grounded():
    zone = _calibrated_zone(fence_bottom=((0.6, 0.2), (0.6, 0.9)))
    assert in_fence_band((0.55, 0.05), zone) is False


def test_entered_band_from_outside_true_when_outside_precedes_band():
    zone = _calibrated_zone()
    track = [(0.9, 0.5), (0.55, 0.5)]  # outside, then on the fence band
    assert entered_band_from_outside(track, zone) is True


def test_entered_band_from_outside_false_when_band_precedes_outside():
    zone = _calibrated_zone()
    track = [(0.55, 0.5), (0.9, 0.5)]  # on the band first, then outside
    assert entered_band_from_outside(track, zone) is False


def test_entered_band_from_outside_false_without_any_outside_point():
    zone = _calibrated_zone()
    track = [(0.55, 0.5), (0.58, 0.5)]  # stays on the band the whole time
    assert entered_band_from_outside(track, zone) is False
