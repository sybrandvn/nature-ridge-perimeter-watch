"""Tests for src/ground_calibration.py.

The core tests build a SYNTHETIC camera with a known focal length, mounting
height and downward tilt, project a known fence and known pickets through it,
and check the calibration recovers the truth. Testing metrology against
hand-traced real geometry could only ever check self-consistency; projecting
known ground truth checks correctness.
"""

from __future__ import annotations

import math

import pytest

from src.config import CameraZone
from src.ground_calibration import (
    DEFAULT_MAX_RANGE_M,
    FLEET_FOCAL_PX,
    MAX_SUBJECT_HEIGHT_M,
    calibrate,
)

WIDTH, HEIGHT = 320, 240


class _SyntheticCamera:
    """Pinhole camera at height `cam_h`, tilted `tilt_deg` downward, looking
    along +y of a world whose ground plane is z=0."""

    def __init__(self, focal=200.0, cam_h=3.0, tilt_deg=25.0):
        self.focal, self.cam_h = focal, cam_h
        t = math.radians(tilt_deg)
        self.centre = (0.0, 0.0, cam_h)
        self.forward = (0.0, math.cos(t), -math.sin(t))
        self.down = (0.0, -math.sin(t), -math.cos(t))
        self.right = (1.0, 0.0, 0.0)

    def project(self, point):
        d = tuple(point[i] - self.centre[i] for i in range(3))
        dot = lambda a, b: sum(a[i] * b[i] for i in range(3))  # noqa: E731
        z = dot(d, self.forward)
        assert z > 0, "point is behind the camera"
        return (
            self.focal * dot(d, self.right) / z + WIDTH / 2,
            self.focal * dot(d, self.down) / z + HEIGHT / 2,
        )

    def normalised(self, point):
        px, py = self.project(point)
        return (px / WIDTH, py / HEIGHT)


def _synthetic_zone(cam=None, fence_h=2.0, fence_x=1.5, **kwargs) -> CameraZone:
    cam = cam or _SyntheticCamera()
    top = tuple(cam.normalised((fence_x, y, fence_h)) for y in (5.0, 25.0))
    bottom = tuple(cam.normalised((fence_x, y, 0.0)) for y in (5.0, 25.0))
    pickets = tuple(
        (cam.normalised((fence_x, y, fence_h)), cam.normalised((fence_x, y, 0.0)))
        for y in (6.0, 10.0, 16.0)
    )
    defaults = dict(
        fence=top,
        outside="right",
        depth_cutoff=0.0,
        ignore=(),
        fence_bottom=bottom,
        fence_height_m=fence_h,
        fence_pickets=pickets,
        metric_calibration=True,
    )
    defaults.update(kwargs)
    return CameraZone(**defaults)


def test_recovers_known_focal_length_and_camera_height():
    cam = _SyntheticCamera(focal=200.0, cam_h=3.0, tilt_deg=25.0)
    cal = calibrate(_synthetic_zone(cam), WIDTH, HEIGHT)
    assert cal is not None
    assert cal.focal_px == pytest.approx(200.0, rel=1e-6)
    assert cal.camera_height_m == pytest.approx(3.0, rel=1e-6)


@pytest.mark.parametrize(
    ("focal", "cam_h", "tilt"),
    [(150.0, 2.0, 15.0), (250.0, 4.5, 30.0), (187.0, 2.6, 40.0)],
)
def test_recovers_truth_across_mountings(focal, cam_h, tilt):
    cam = _SyntheticCamera(focal=focal, cam_h=cam_h, tilt_deg=tilt)
    cal = calibrate(_synthetic_zone(cam), WIDTH, HEIGHT)
    assert cal is not None
    assert cal.focal_px == pytest.approx(focal, rel=1e-5)
    assert cal.camera_height_m == pytest.approx(cam_h, rel=1e-5)


def test_distance_matches_true_ground_distance():
    cam = _SyntheticCamera()
    cal = calibrate(_synthetic_zone(cam), WIDTH, HEIGHT)
    for true_y in (6.0, 10.0, 18.0, 25.0):
        px = cam.project((0.4, true_y, 0.0))
        # Camera sits above the origin, so horizontal distance is hypot(x, y).
        expected = math.hypot(0.4, true_y)
        assert cal.distance_m(px) == pytest.approx(expected, rel=1e-6)


def test_height_matches_true_subject_height():
    cam = _SyntheticCamera()
    cal = calibrate(_synthetic_zone(cam), WIDTH, HEIGHT)
    for true_h in (1.2, 1.75, 2.4):
        base = cam.project((0.4, 14.0, 0.0))
        top = cam.project((0.4, 14.0, true_h))
        assert cal.height_m(base, top[1]) == pytest.approx(true_h, rel=1e-6)


def test_height_is_independent_of_distance():
    """The whole point of the model: the same person measures the same at any
    range, which the pixels-per-metre ruler could not manage."""
    cam = _SyntheticCamera()
    cal = calibrate(_synthetic_zone(cam), WIDTH, HEIGHT)
    measured = []
    for y in (6.0, 12.0, 20.0, 30.0):
        base = cam.project((0.0, y, 0.0))
        top = cam.project((0.0, y, 1.8))
        measured.append(cal.height_m(base, top[1]))
    assert all(m == pytest.approx(1.8, rel=1e-6) for m in measured)


# -- opting in / out -------------------------------------------------------


def test_returns_none_when_not_opted_in():
    assert calibrate(_synthetic_zone(metric_calibration=False), WIDTH, HEIGHT) is None


def test_returns_none_without_fence_bottom():
    assert calibrate(_synthetic_zone(fence_bottom=None), WIDTH, HEIGHT) is None


def test_returns_none_with_no_pickets():
    """A picket is always required: it is the only thing carrying camera roll,
    which varies enough across the real fleet to matter (~20% of scale)."""
    zone = _synthetic_zone()
    none = CameraZone(
        fence=zone.fence,
        outside=zone.outside,
        depth_cutoff=zone.depth_cutoff,
        ignore=zone.ignore,
        fence_bottom=zone.fence_bottom,
        fence_height_m=zone.fence_height_m,
        fence_pickets=(),
        metric_calibration=True,
    )
    assert calibrate(none, WIDTH, HEIGHT) is None


def test_single_picket_with_known_focal_recovers_truth():
    """One picket plus an already-known focal length is enough: the conjugacy
    that normally derives f instead pins the vertical vanishing point."""
    cam = _SyntheticCamera(focal=200.0, cam_h=3.0, tilt_deg=25.0)
    zone = _synthetic_zone(cam)
    single = CameraZone(
        fence=zone.fence,
        outside=zone.outside,
        depth_cutoff=zone.depth_cutoff,
        ignore=zone.ignore,
        fence_bottom=zone.fence_bottom,
        fence_height_m=zone.fence_height_m,
        fence_pickets=zone.fence_pickets[:1],
        metric_calibration=True,
        metric_focal_px=200.0,
    )
    cal = calibrate(single, WIDTH, HEIGHT)
    assert cal is not None
    assert cal.focal_px == pytest.approx(200.0)
    assert cal.camera_height_m == pytest.approx(3.0, rel=1e-5)
    px = cam.project((0.4, 14.0, 0.0))
    assert cal.distance_m(px) == pytest.approx(math.hypot(0.4, 14.0), rel=1e-5)


def test_single_picket_matches_multi_picket_result():
    cam = _SyntheticCamera(focal=200.0, cam_h=3.0, tilt_deg=25.0)
    zone = _synthetic_zone(cam)
    multi = calibrate(zone, WIDTH, HEIGHT)
    single = calibrate(
        CameraZone(
            fence=zone.fence,
            outside=zone.outside,
            depth_cutoff=zone.depth_cutoff,
            ignore=zone.ignore,
            fence_bottom=zone.fence_bottom,
            fence_height_m=zone.fence_height_m,
            fence_pickets=zone.fence_pickets[1:2],
            metric_calibration=True,
            metric_focal_px=200.0,
        ),
        WIDTH,
        HEIGHT,
    )
    assert single.camera_height_m == pytest.approx(multi.camera_height_m, rel=1e-5)


def test_single_picket_falls_back_to_fleet_focal():
    cam = _SyntheticCamera(focal=FLEET_FOCAL_PX, cam_h=2.6, tilt_deg=30.0)
    zone = _synthetic_zone(cam)
    cal = calibrate(
        CameraZone(
            fence=zone.fence,
            outside=zone.outside,
            depth_cutoff=zone.depth_cutoff,
            ignore=zone.ignore,
            fence_bottom=zone.fence_bottom,
            fence_height_m=zone.fence_height_m,
            fence_pickets=zone.fence_pickets[:1],
            metric_calibration=True,
        ),
        WIDTH,
        HEIGHT,
    )
    assert cal is not None
    assert cal.focal_px == pytest.approx(FLEET_FOCAL_PX)
    assert cal.camera_height_m == pytest.approx(2.6, rel=1e-5)


def test_returns_none_when_pickets_are_all_parallel():
    """Parallel pickets never meet, so there is no vertical vanishing point
    and no focal length to derive -- refuse rather than divide by ~zero."""
    cam = _SyntheticCamera()
    zone = _synthetic_zone(cam)
    flat = ((0.30, 0.40), (0.30, 0.90))
    also_flat = ((0.50, 0.40), (0.50, 0.90))
    parallel = CameraZone(
        fence=zone.fence,
        outside=zone.outside,
        depth_cutoff=zone.depth_cutoff,
        ignore=zone.ignore,
        fence_bottom=zone.fence_bottom,
        fence_height_m=zone.fence_height_m,
        fence_pickets=((flat[0], flat[1]), (also_flat[0], also_flat[1])),
        metric_calibration=True,
    )
    assert calibrate(parallel, WIDTH, HEIGHT) is None


# -- limits ----------------------------------------------------------------


def test_distance_beyond_configured_max_range_is_refused():
    cam = _SyntheticCamera()
    cal = calibrate(_synthetic_zone(cam, metric_max_range_m=15.0), WIDTH, HEIGHT)
    assert cal.max_range_m == pytest.approx(15.0)
    assert cal.distance_m(cam.project((0.0, 12.0, 0.0))) == pytest.approx(12.0, rel=1e-6)
    assert cal.distance_m(cam.project((0.0, 18.0, 0.0))) is None


def test_default_max_range_applies_when_unset():
    cal = calibrate(_synthetic_zone(), WIDTH, HEIGHT)
    assert cal.max_range_m <= DEFAULT_MAX_RANGE_M


def test_points_above_the_horizon_are_refused():
    cal = calibrate(_synthetic_zone(), WIDTH, HEIGHT)
    above = cal.horizon_vp[1] - 5.0
    assert cal.distance_m((WIDTH / 2, above)) is None


def test_near_horizon_rows_are_refused_by_the_precision_limit():
    """One pixel of noise just below the horizon swings the answer wildly, so
    the calibration caps range before that region rather than reporting it."""
    cal = calibrate(_synthetic_zone(), WIDTH, HEIGHT)
    just_below = cal.horizon_vp[1] + 0.5
    assert cal.distance_m((WIDTH / 2, just_below)) is None


def test_absurdly_tall_subjects_are_refused():
    cam = _SyntheticCamera()
    cal = calibrate(_synthetic_zone(cam), WIDTH, HEIGHT)
    base = cam.project((0.0, 12.0, 0.0))
    tall = cam.project((0.0, 12.0, MAX_SUBJECT_HEIGHT_M + 1.5))
    assert cal.height_m(base, tall[1]) is None


def test_row_at_distance_round_trips():
    cal = calibrate(_synthetic_zone(), WIDTH, HEIGHT)
    for target in (5.0, 12.0, 25.0):
        row = cal.row_at_distance(target)
        assert row is not None
        point = cal.base_point_at_row(row)
        assert cal.distance_m(point) == pytest.approx(target, abs=0.05)


def test_row_at_distance_refuses_beyond_max_range():
    cal = calibrate(_synthetic_zone(metric_max_range_m=15.0), WIDTH, HEIGHT)
    assert cal.row_at_distance(20.0) is None


# -- the real camera -------------------------------------------------------


def test_cam06_real_config_calibrates_sanely():
    from src.config import load_cameras_config

    zone = load_cameras_config("config/cameras.yaml").by_id("cam06").zone
    cal = calibrate(zone, WIDTH, HEIGHT)
    assert cal is not None
    # HFOV in the range a fixed security camera actually ships with.
    hfov = 2 * math.degrees(math.atan((WIDTH / 2) / cal.focal_px))
    assert 60.0 < hfov < 100.0
    assert 1.5 < cal.camera_height_m < 6.0
    # Camera height is derived at one reference row; if the model is right it
    # must stay put everywhere else along the fence.
    near = cal.distance_m(cal.base_point_at_row(HEIGHT - 1))
    assert near is not None and 0.5 < near < 3.0
