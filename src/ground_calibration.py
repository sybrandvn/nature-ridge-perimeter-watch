"""Metric ground-plane calibration from a camera's own fence geometry.

Turns an image point into a real-world distance and a tracked box into a real
height, using single-view metrology rather than a per-row pixels-per-metre
ruler.

Why not the `src.zones` ruler: that walks `fence_bottom` accumulating a local
pixels-per-metre scale derived from the picket-tilt interpolation. Measured on
cam06 (2026-09-05) it caps out around 4.7 m and contains an unphysical
discontinuity, because piecewise-interpolated picket tilt is not projectively
consistent -- every real vertical line in a scene must converge on ONE vertical
vanishing point, and interpolated tilts do not.

What this does instead:
  1. the fence's top rail and base line are parallel in the world, so where
     they meet in the image is a horizon vanishing point;
  2. the traced pickets are vertical in the world, so where THEY meet is the
     vertical vanishing point;
  3. those two directions are perpendicular, which fixes the focal length via
     ``f^2 = -(vz - c).(vh - c)`` (principal point assumed at frame centre,
     square pixels, no roll correction needed -- the vanishing points carry it);
  4. with the camera matrix known, forcing the fence's own `fence_height_m`
     between the two lines fixes absolute scale, giving the camera's height
     above the ground plane.

Distance then comes from intersecting a pixel's back-projected ray with that
ground plane, which needs no fence line at all -- so it works for any point in
frame below the horizon, not only points near the fence.

Opt-in per camera (`metric_calibration: true`), because it is only as good as
the traced geometry it is built from and has not been validated on every
camera. See `MAX_REL_ERROR_PER_PIXEL` for why a maximum range is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.config import CameraZone, Point

# Distance goes as roughly 1/(row - horizon_row), so a subject whose feet sit
# near the horizon has a distance that explodes on sub-pixel detector noise --
# an unclamped scan of cam06 produced a reading of 106,739 m from a blob a few
# rows below the horizon. Beyond the row where one pixel of vertical error
# moves the answer by more than this fraction, the estimate is refused.
MAX_REL_ERROR_PER_PIXEL = 0.10

# Fallback cap when a camera does not set `metric_max_range_m`. Deliberately
# well inside the ~69 m that cam06's pixel geometry alone would allow: past a
# few tens of metres these 320x240 IR cameras cannot resolve a subject well
# enough for the "is it standing on the ground plane" assumption to be worth
# much, and that assumption -- not pixel precision -- is the real limit.
DEFAULT_MAX_RANGE_M = 40.0

# A calibration that implies a camera mounted underground or on a mast is a
# mistraced fence, not a real camera -- refuse it rather than emit numbers.
MIN_CAMERA_HEIGHT_M = 0.5
MAX_CAMERA_HEIGHT_M = 30.0

# Motion blobs merge subjects together and smear with IR flare; a "person"
# taller than this is a merged/streaked blob, so report nothing rather than a
# number that looks measured.
MAX_SUBJECT_HEIGHT_M = 4.0

# Same reasoning for width, and it matters MORE: width comes from the ground
# distance between the bbox's two BOTTOM corners, and near the horizon those
# two corners back-project to wildly separated ground points. Left unbounded
# this produced a 17.8m-wide, 65.8 sq m "subject" on cam02/18444 (a guard
# walking inside with a flashlight), which then dominated the ranking model --
# a 20-sigma outlier in a fit with only 20 positive examples.
MAX_SUBJECT_WIDTH_M = 4.0

# Nothing on this terrain outruns a sprint. A higher reading means the tracker
# jumped between two unrelated blobs, so the frame pair is discarded rather
# than contributing a wild speed -- same bounding discipline as height/width.
MAX_SUBJECT_SPEED_MPS = 15.0

# Every camera on this site is the same model, so the intrinsics are identical
# and only the mounting differs (pitch, yaw, roll, height above ground).
# Measured on cam06 -- the one camera with enough traced pickets to derive it
# from its own geometry. Corroborated independently 2026-09-06: the focal
# length that makes all 15 traced cameras' mounting heights most consistent is
# ~180px, within 4% of this. Used only when a camera has a single picket; with
# two or more, that camera's own geometry gives f directly and this is ignored.
FLEET_FOCAL_PX = 187.34

_EPS = 1e-9


def _homog(point) -> np.ndarray:
    return np.array([point[0], point[1], 1.0], dtype=float)


def _join(p, q) -> np.ndarray:
    """The image line through two points."""
    return np.cross(_homog(p), _homog(q))


def _meet(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    """Where two image lines cross, or None if they are parallel."""
    point = np.cross(line_a, line_b)
    if abs(point[2]) < _EPS:
        return None
    return point[:2] / point[2]


def _fit_line(points: np.ndarray) -> np.ndarray:
    """Total-least-squares line through >= 2 pixel points.

    Fitted rather than taken pairwise so a hand-traced polyline's own jitter
    averages out instead of whichever two points happen to be at the ends
    setting the whole calibration.
    """
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre)
    return _join(centre, centre + vt[0])


@dataclass(frozen=True, eq=False)
class GroundCalibration:
    """A camera's metric ground plane. Build via `calibrate`."""

    frame_width: int
    frame_height: int
    focal_px: float
    camera_height_m: float
    horizon_vp: Point
    vertical_vp: Point
    max_range_m: float
    _k_inv: np.ndarray
    _up: np.ndarray
    _plane_d: float
    _base_line: np.ndarray
    _top_line: np.ndarray

    def ground_point(self, point_px) -> np.ndarray | None:
        """Where the ray through `point_px` meets the ground plane, in camera
        coordinates (metres). None if that ray never reaches the ground -- it
        points at or above the horizon."""
        ray = self._k_inv @ _homog(point_px)
        denom = float(ray @ self._up)
        if abs(denom) < _EPS:
            return None
        scale = self._plane_d / denom
        if scale <= 0:
            return None
        return scale * ray

    def distance_m(self, point_px) -> float | None:
        """Horizontal ground distance from the camera to `point_px`, or None
        if it is above the horizon or beyond `max_range_m`."""
        ground = self.ground_point(point_px)
        if ground is None:
            return None
        flat = float(ground @ ground) - self.camera_height_m**2
        distance = float(np.sqrt(max(flat, 0.0)))
        if distance > self.max_range_m:
            return None
        return distance

    def height_m(
        self, base_px, top_row: float, *, enforce_limits: bool = True
    ) -> float | None:
        """Real height of something standing at `base_px` whose top edge is at
        image row `top_row`.

        Assumes the subject stands on the ground plane; a bird on the fence or
        a light on a pole will read as a tall subject standing further away.

        `enforce_limits=False` skips the `MAX_SUBJECT_HEIGHT_M` clamp and the
        `max_range_m` cap, returning the raw geometric answer instead of None
        -- for measuring HOW implausible a reading is (e.g. an
        `implausible_height_fraction` feature), not for reporting a number
        anyone should trust. Still None if the point is off the ground plane
        entirely or the solution is geometrically invalid (behind the camera).
        """
        ground = self.ground_point(base_px)
        if ground is None:
            return None
        if enforce_limits and self.distance_m(base_px) is None:
            return None
        # The subject's top lies on the image line from its base toward the
        # vertical vanishing point, at the box's top row.
        top_px = _meet(
            _join(base_px, self.vertical_vp),
            np.array([0.0, 1.0, -float(top_row)]),
        )
        if top_px is None:
            return None
        ray_top = self._k_inv @ _homog(top_px)
        solution, *_ = np.linalg.lstsq(
            np.column_stack([ray_top, -self._up]), ground, rcond=None
        )
        along_ray, height = float(solution[0]), float(solution[1])
        if along_ray <= 0 or height <= 0:
            return None
        if enforce_limits and height > MAX_SUBJECT_HEIGHT_M:
            return None
        return height

    def base_point_at_row(self, row: float) -> np.ndarray | None:
        """The fence base line's own image point at `row`."""
        return _meet(self._base_line, np.array([0.0, 1.0, -float(row)]))

    def fence_top_above(self, base_px) -> np.ndarray | None:
        """Where the fence's top rail sits directly above a base-line point,
        following the true vertical direction rather than straight up-screen."""
        return _meet(_join(base_px, self.vertical_vp), self._top_line)

    def row_at_distance(self, target_m: float) -> float | None:
        """Image row on the fence base line at ground distance `target_m`, or
        None if that distance is not resolvable within `max_range_m`."""
        if target_m <= 0 or target_m > self.max_range_m:
            return None
        low, high = float(self.horizon_vp[1]), float(self.frame_height)
        for _ in range(80):
            mid = (low + high) / 2
            point = self.base_point_at_row(mid)
            distance = None if point is None else self.distance_m(point)
            if distance is None:
                low = mid
            elif distance > target_m:
                low = mid
            else:
                high = mid
        row = (low + high) / 2
        point = self.base_point_at_row(row)
        distance = None if point is None else self.distance_m(point)
        if distance is None or abs(distance - target_m) > max(0.05, 0.01 * target_m):
            return None
        return row


def _vertical_vanishing_point(
    pickets: tuple[tuple[Point, Point], ...], width: int, height: int
) -> np.ndarray | None:
    """Least-squares meeting point of every traced picket.

    Over-determined on purpose: any two near-parallel pickets pin the point
    only weakly (cam06's pickets 2 and 3 differ by 0.007 in slope and alone
    put it 4x further away than the full fit does), so all of them are fitted
    together and the result is checked for plausibility by the caller.
    """
    normal_sum = np.zeros((2, 2))
    rhs = np.zeros(2)
    for top, base in pickets:
        a = np.array([top[0] * width, top[1] * height], dtype=float)
        b = np.array([base[0] * width, base[1] * height], dtype=float)
        span = b - a
        norm = float(np.linalg.norm(span))
        if norm < _EPS:
            continue
        direction = span / norm
        projector = np.eye(2) - np.outer(direction, direction)
        normal_sum += projector
        rhs += projector @ a
    if abs(np.linalg.det(normal_sum)) < 1e-6:
        return None
    return np.linalg.solve(normal_sum, rhs)


def _vertical_vp_from_known_focal(
    focal: float,
    horizon_vp: np.ndarray,
    picket: tuple[Point, Point],
    centre: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray | None:
    """Vertical vanishing point from ONE picket plus an already-known focal.

    With two pickets, their meeting point gives vz and the conjugacy
    ``(vz-c).(vh-c) = -f^2`` then yields f. With f known up front that same
    equation runs backwards: it constrains vz to a line, and a single traced
    picket supplies the second line, so the two intersect at a point.

    This is what lets a camera with only one picket calibrate at all. The
    picket is still needed -- it carries the camera's roll, which varies from
    0 to ~13 degrees across this fleet and is worth ~20% of absolute scale, so
    assuming zero roll instead is not good enough.
    """
    top, base = picket
    origin = np.array([top[0] * width, top[1] * height])
    direction = np.array([base[0] * width - origin[0], base[1] * height - origin[1]])
    to_horizon = horizon_vp - centre
    denom = float(direction @ to_horizon)
    if abs(denom) < _EPS:
        return None
    along = (-focal * focal - float((origin - centre) @ to_horizon)) / denom
    return origin + along * direction


def _precision_limit_m(
    calibration_probe, frame_width: int, frame_height: int
) -> float:
    """Greatest distance at which one pixel row of error still moves the
    answer by less than `MAX_REL_ERROR_PER_PIXEL`."""
    centre_x = frame_width / 2.0
    limit = 0.0
    for row in range(frame_height, 1, -1):
        near = calibration_probe((centre_x, float(row)))
        far = calibration_probe((centre_x, float(row - 1)))
        if near is None or far is None or near <= 0:
            continue
        if abs(far - near) / near > MAX_REL_ERROR_PER_PIXEL:
            break
        limit = max(limit, far)
    return limit


@lru_cache(maxsize=64)
def calibrate(
    zone: CameraZone, frame_width: int, frame_height: int
) -> GroundCalibration | None:
    """Build a metric ground plane for `zone`, or None if it cannot be built.

    Returns None -- rather than raising -- whenever the camera has not opted
    in, has not got the geometry traced, or the traced geometry implies
    something physically impossible. Callers treat None as "this camera has no
    metric ruler" and fall back to pixel-space behaviour.

    Cached because it runs an SVD and a least-squares solve, and callers ask
    for it once per rendered frame.
    """
    if not zone.metric_calibration:
        return None
    if zone.fence is None or zone.fence_bottom is None:
        return None
    if len(zone.fence) < 2 or len(zone.fence_bottom) < 2:
        return None
    # At least one picket is always required: it is the only thing that carries
    # the camera's roll (see `_vertical_vp_from_known_focal`).
    if not zone.fence_pickets:
        return None

    width, height = float(frame_width), float(frame_height)
    centre = np.array([width / 2.0, height / 2.0])

    top_line = _fit_line(np.array([(p[0] * width, p[1] * height) for p in zone.fence]))
    base_line = _fit_line(
        np.array([(p[0] * width, p[1] * height) for p in zone.fence_bottom])
    )
    horizon_vp = _meet(top_line, base_line)
    if horizon_vp is None:
        return None

    if len(zone.fence_pickets) >= 2:
        vertical_vp = _vertical_vanishing_point(
            zone.fence_pickets, frame_width, frame_height
        )
        if vertical_vp is None:
            return None
        # Perpendicular world directions are conjugate about the image of the
        # absolute conic; with a centred principal point that reduces to this.
        focal_sq = -float((vertical_vp - centre) @ (horizon_vp - centre))
        if not np.isfinite(focal_sq) or focal_sq <= 0:
            return None
        focal = float(np.sqrt(focal_sq))
    else:
        focal = zone.metric_focal_px or FLEET_FOCAL_PX
        vertical_vp = _vertical_vp_from_known_focal(
            focal, horizon_vp, zone.fence_pickets[0], centre, width, height
        )
        if vertical_vp is None:
            return None

    k_inv = np.linalg.inv(
        np.array([[focal, 0, centre[0]], [0, focal, centre[1]], [0, 0, 1.0]])
    )
    up = k_inv @ _homog(vertical_vp)
    norm = float(np.linalg.norm(up))
    if norm < _EPS:
        return None
    up = up / norm

    # Absolute scale: the fence's own known height between the two lines, read
    # at the middle of the base line's traced span (its best-supported part).
    reference_row = float(np.mean([p[1] for p in zone.fence_bottom]) * height)
    base_px = _meet(base_line, np.array([0.0, 1.0, -reference_row]))
    if base_px is None:
        return None
    top_px = _meet(_join(base_px, vertical_vp), top_line)
    if top_px is None:
        return None
    ray_base = k_inv @ _homog(base_px)
    ray_top = k_inv @ _homog(top_px)

    oriented = None
    for sign in (1.0, -1.0):
        solution, *_ = np.linalg.lstsq(
            np.column_stack([ray_top, -ray_base]), zone.fence_height_m * sign * up,
            rcond=None,
        )
        if solution[0] > 0 and solution[1] > 0:
            oriented = (sign * up, solution[1] * ray_base)
            break
    if oriented is None:
        return None
    up, base_world = oriented

    camera_height = float(abs(base_world @ up))
    if not MIN_CAMERA_HEIGHT_M <= camera_height <= MAX_CAMERA_HEIGHT_M:
        return None
    plane_d = -camera_height if float(base_world @ up) < 0 else camera_height

    partial = GroundCalibration(
        frame_width=frame_width,
        frame_height=frame_height,
        focal_px=focal,
        camera_height_m=camera_height,
        horizon_vp=(float(horizon_vp[0]), float(horizon_vp[1])),
        vertical_vp=(float(vertical_vp[0]), float(vertical_vp[1])),
        max_range_m=float("inf"),
        _k_inv=k_inv,
        _up=up,
        _plane_d=plane_d,
        _base_line=base_line,
        _top_line=top_line,
    )

    def raw_distance(point_px):
        ground = partial.ground_point(point_px)
        if ground is None:
            return None
        return float(np.sqrt(max(float(ground @ ground) - camera_height**2, 0.0)))

    configured = zone.metric_max_range_m or DEFAULT_MAX_RANGE_M
    max_range = min(configured, _precision_limit_m(raw_distance, frame_width, frame_height))
    if max_range <= 0:
        return None

    return GroundCalibration(
        frame_width=frame_width,
        frame_height=frame_height,
        focal_px=focal,
        camera_height_m=camera_height,
        horizon_vp=partial.horizon_vp,
        vertical_vp=partial.vertical_vp,
        max_range_m=float(max_range),
        _k_inv=k_inv,
        _up=up,
        _plane_d=plane_d,
        _base_line=base_line,
        _top_line=top_line,
    )
