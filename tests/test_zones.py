import pytest

from src.config import CameraZone
from src.zones import (
    classify_zone,
    far_side_pixel_fraction,
    in_ignore_region,
    is_beyond_depth_cutoff,
    side_name,
    signed_side,
)

# Horizontal fence, walked left-to-right along the bottom-middle of frame.
_HORIZONTAL_FENCE = ((0.0, 0.5), (1.0, 0.5))


def test_signed_side_requires_at_least_two_points():
    with pytest.raises(ValueError, match="fence needs"):
        signed_side((0.5, 0.5), [(0.0, 0.0)])


def test_side_name_right_below_horizontal_fence():
    # Facing east (left-to-right) with y-down image coords, "below" (larger y) is right.
    assert side_name((0.5, 0.9), _HORIZONTAL_FENCE) == "right"


def test_side_name_left_above_horizontal_fence():
    assert side_name((0.5, 0.1), _HORIZONTAL_FENCE) == "left"


def test_side_name_on_line():
    assert side_name((0.5, 0.5), _HORIZONTAL_FENCE) == "on_line"


def test_signed_side_uses_nearest_segment_for_bent_polyline():
    bent_fence = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0))
    # Point near the vertical second segment, clearly to its right (larger x).
    assert side_name((0.6, 0.5), bent_fence) == "left"
    assert side_name((0.4, 0.5), bent_fence) == "right"


def test_is_beyond_depth_cutoff():
    zone = CameraZone(fence=None, far_side=None, depth_cutoff=0.3, ignore=())
    assert is_beyond_depth_cutoff((0.5, 0.1), zone) is True
    assert is_beyond_depth_cutoff((0.5, 0.5), zone) is False


def test_in_ignore_region_detects_containment():
    polygon = ((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2))
    zone = CameraZone(fence=None, far_side=None, depth_cutoff=0.0, ignore=(polygon,))
    assert in_ignore_region((0.1, 0.1), zone) is True
    assert in_ignore_region((0.9, 0.9), zone) is False


def test_classify_zone_priorities_ignore_over_far_side():
    polygon = ((0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0))
    zone = CameraZone(
        fence=_HORIZONTAL_FENCE, far_side="right", depth_cutoff=0.0, ignore=(polygon,)
    )
    # (0.5, 0.9) is far_side (right of horizontal fence) but inside the ignore polygon.
    assert classify_zone((0.5, 0.9), zone) == "ignored"


def test_classify_zone_priorities_depth_cutoff_over_far_side():
    zone = CameraZone(fence=_HORIZONTAL_FENCE, far_side="left", depth_cutoff=0.3, ignore=())
    # (0.5, 0.1) is far_side (left/above the fence) but also beyond depth cutoff.
    assert classify_zone((0.5, 0.1), zone) == "ambiguous"


def test_classify_zone_far_and_near_side():
    zone = CameraZone(fence=_HORIZONTAL_FENCE, far_side="left", depth_cutoff=0.0, ignore=())
    assert classify_zone((0.5, 0.1), zone) == "far_side"
    assert classify_zone((0.5, 0.9), zone) == "near_side"


def test_classify_zone_ambiguous_without_fence_config():
    zone = CameraZone(fence=None, far_side=None, depth_cutoff=0.0, ignore=())
    assert classify_zone((0.5, 0.5), zone) == "ambiguous"


def test_far_side_pixel_fraction_mixed_points():
    zone = CameraZone(fence=_HORIZONTAL_FENCE, far_side="left", depth_cutoff=0.0, ignore=())
    points = [(0.5, 0.1), (0.5, 0.2), (0.5, 0.9)]  # 2 far, 1 near
    assert far_side_pixel_fraction(points, zone) == pytest.approx(2 / 3)


def test_far_side_pixel_fraction_excludes_ignored_and_ambiguous():
    polygon = ((0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0))
    zone = CameraZone(
        fence=_HORIZONTAL_FENCE, far_side="left", depth_cutoff=0.3, ignore=(polygon,)
    )
    points = [
        (0.5, 0.1),  # beyond depth cutoff -> ambiguous, excluded
        (0.5, 0.9),  # inside ignore polygon -> ignored, excluded
        (0.5, 0.55),  # near side (right, below the fence line), classifiable
    ]
    assert far_side_pixel_fraction(points, zone) == 0.0


def test_far_side_pixel_fraction_empty_relevant_set_is_zero():
    zone = CameraZone(fence=None, far_side=None, depth_cutoff=0.0, ignore=())
    assert far_side_pixel_fraction([(0.5, 0.5)], zone) == 0.0
