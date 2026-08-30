import pytest

from src.config import CameraZone
from src.zones import (
    classify_zone,
    in_ignore_region,
    is_beyond_depth_cutoff,
    median_fence_distance,
    outside_pixel_fraction,
    side_name,
    signed_side,
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
