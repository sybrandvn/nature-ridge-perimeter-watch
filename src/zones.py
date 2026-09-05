"""Polyline-side / depth / ignore-region geometry for a single camera's zone config.

Pure geometry, no video required: this operates on normalized (x, y) points in
[0, 1], x=column fraction (0=left), y=row fraction (0=top, 1=bottom of frame).
Used by the (future) motion/classification pipeline and by scripts/spike.py's
feature extraction once real contours exist.

Side convention: plain screen position, nothing direction-relative. For a query
point, find the fence's x at that same y (interpolating along the polyline) and
compare -- point x larger than the fence's x there is "right", smaller is
"left". Point order in the polyline doesn't matter (reversing it gives the same
answer). This must match config/cameras.yaml's `outside: left|right` field.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.config import CameraZone, Point

# Zone classification outcomes. "ignored" and "ambiguous" take priority over
# "outside"/"inside" because they represent "cannot reliably classify here",
# not a side judgement -- see the fail-safe policy in docs/plan.md.
ZoneClassification = str  # "outside" | "inside" | "ambiguous" | "ignored"


def _fence_x_at_y(y: float, fence: Sequence[Point]) -> float:
    """The fence polyline's x at row `y`, interpolating within its y-range and
    extrapolating along the nearest end segment outside it (only reachable via
    a point above/below every fence point, e.g. above the top of a fence that
    doesn't start at y=0 -- depth_cutoff excludes most of that band already).
    """
    segments = list(zip(fence, fence[1:], strict=False))
    for (ax, ay), (bx, by) in segments:
        if ay == by:
            continue
        t = (y - ay) / (by - ay)
        if 0.0 <= t <= 1.0:
            return ax + t * (bx - ax)
    (ax, ay), (bx, by) = min(segments, key=lambda seg: min(abs(seg[0][1] - y), abs(seg[1][1] - y)))
    if ay == by:
        return ax
    t = (y - ay) / (by - ay)
    return ax + t * (bx - ax)


def signed_side(point: Point, fence: Sequence[Point]) -> float:
    """Signed horizontal distance from the fence: point's x minus the fence's
    x at the same row. Positive => point is to the right (larger x) of the
    fence there, negative => left, zero => exactly on the line.
    """
    if len(fence) < 2:
        raise ValueError("fence needs >= 2 points")

    px, py = point
    return px - _fence_x_at_y(py, fence)


def side_name(point: Point, fence: Sequence[Point]) -> str:
    cross = signed_side(point, fence)
    if cross > 0:
        return "right"
    if cross < 0:
        return "left"
    return "on_line"


def effective_fence(zone: CameraZone) -> tuple[Point, ...] | None:
    """The fence line used for inside/outside classification.

    Prefers `fence_bottom` (the base/ground line) when a camera has one traced,
    falling back to `fence` (the top rail, every camera's original and only
    line) otherwise. `fence_bottom` is the correct reference for the side
    decision: cameras are mounted near the fence top looking down and
    slightly outward, so a subject walking INSIDE the fence still projects to
    the outside of the top-rail line -- see docs/plan.md "Fence base line".
    """
    return zone.fence_bottom if zone.fence_bottom is not None else zone.fence


def is_beyond_depth_cutoff(point: Point, zone: CameraZone) -> bool:
    """True if the point is farther away than the configured depth cutoff.

    Larger y = lower in frame = nearer the camera (cameras look down the fence
    line, so distant objects sit higher in frame). "Beyond" the cutoff means a
    smaller y than depth_cutoff -- too far out for reliable classification.
    """
    _, y = point
    return y < zone.depth_cutoff


def in_ignore_region(point: Point, zone: CameraZone) -> bool:
    return any(_point_in_polygon(point, polygon) for polygon in zone.ignore)


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def classify_zone(point: Point, zone: CameraZone) -> ZoneClassification:
    """Classify a single point against a camera's zone config.

    Ignore regions and depth cutoff are checked first: they mean "we cannot
    reliably classify here", which must never be conflated with a genuine
    inside/outside judgement.
    """
    if in_ignore_region(point, zone):
        return "ignored"
    if is_beyond_depth_cutoff(point, zone):
        return "ambiguous"
    fence = effective_fence(zone)
    if fence is None or zone.outside is None:
        return "ambiguous"
    return "outside" if side_name(point, fence) == zone.outside else "inside"


def outside_pixel_fraction(points: Sequence[Point], zone: CameraZone) -> float:
    """Fraction of `points` (e.g. contour vertices or a sampled blob mask) that
    fall outside the fence, among points that are actually classifiable.

    Ignored/ambiguous points are excluded from both numerator and denominator
    so depth cutoff or ignore regions can't silently dilute the ratio toward
    "not outside" -- a blob that's mostly in an ignore region should have an
    ill-defined fraction (0.0, on an empty set), not a falsely low one.
    """
    classifications = [classify_zone(p, zone) for p in points]
    relevant = [c for c in classifications if c in ("outside", "inside")]
    if not relevant:
        return 0.0
    return sum(1 for c in relevant if c == "outside") / len(relevant)


def track_crosses_fence(track: Sequence[Point], zone: CameraZone) -> bool:
    """True if a track's centroids appear on both sides of the fence line.

    Track-level, unlike `outside_pixel_fraction`, which reads one frame's blob.
    Empirically this is the strongest single separator between the guard (who
    walks along and across the fence line, and whose beam sweeps over it) and
    animal/incident subjects (which stay on one side for the whole clip).
    """
    fence = effective_fence(zone)
    if fence is None or len(track) < 2:
        return False
    sides = [signed_side(p, fence) for p in track]
    return any(s > 0 for s in sides) and any(s < 0 for s in sides)


def median_fence_distance(track: Sequence[Point], zone: CameraZone) -> float:
    """Median absolute horizontal distance (frame-width fraction) from the fence.

    Subjects that matter approach or follow the fence; sensor noise, insects
    near the lens and wind-shaken foliage sit wherever they happen to be, which
    on this footage is typically much farther from the fence line.
    """
    fence = effective_fence(zone)
    if fence is None or not track:
        return 0.0
    distances = sorted(abs(signed_side(p, fence)) for p in track)
    mid = len(distances) // 2
    if len(distances) % 2:
        return distances[mid]
    return (distances[mid - 1] + distances[mid]) / 2


# --------------------------------------------------------------------------
# Fence-pair calibration: top-rail + base line give a per-row pixels-to-metres
# ruler. Only meaningful for cameras with a traced `fence_bottom`; every other
# camera's functions here return None ("uncalibrated"), never a silent 0/guess.
# --------------------------------------------------------------------------

MIN_FENCE_SEPARATION_PX = 4.0


def _line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Point | None:
    """Intersection of infinite line p1-p2 with infinite line p3-p4, or None
    if parallel. Standard two-point-form determinant solution."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    px = (a * (x3 - x4) - (x1 - x2) * b) / denom
    py = (a * (y3 - y4) - (y1 - y2) * b) / denom
    return (px, py)


def _project_along_picket(
    origin: Point, direction: Point, polyline: Sequence[Point]
) -> Point | None:
    """Walk from `origin` in `direction` until it crosses `polyline`, and
    return the nearest such crossing (smallest positive distance along the
    ray), or None if the ray never crosses within any segment.
    """
    far = (origin[0] + direction[0], origin[1] + direction[1])
    best: Point | None = None
    best_t: float | None = None
    for seg_a, seg_b in zip(polyline, polyline[1:], strict=False):
        point = _line_intersection(origin, far, seg_a, seg_b)
        if point is None:
            continue
        # t: how far along `direction` the crossing sits (must be ahead of origin).
        if abs(direction[0]) > abs(direction[1]):
            t = (point[0] - origin[0]) / direction[0]
        elif direction[1] != 0:
            t = (point[1] - origin[1]) / direction[1]
        else:
            continue
        if t <= 1e-9:
            continue
        # s: where the crossing falls along this specific segment (must be within it).
        ax, ay = seg_a
        bx, by = seg_b
        if abs(bx - ax) > abs(by - ay):
            if bx == ax:
                continue
            s = (point[0] - ax) / (bx - ax)
        else:
            if by == ay:
                continue
            s = (point[1] - ay) / (by - ay)
        if not (-1e-6 <= s <= 1 + 1e-6):
            continue
        if best_t is None or t < best_t:
            best_t, best = t, point
    return best


def is_grounded_at_fence(y: float, zone: CameraZone) -> bool:
    """True if row `y` falls within the base line's own traced y-range.

    A subject perched above the fence (e.g. a bird sitting on the top rail)
    has its feet row above where the base line was ever traced -- the ruler
    only exists at rows the base line actually spans; extrapolating past that
    turns a real "not on the ground" case into a silently wrong estimate.
    """
    if zone.fence_bottom is None:
        return False
    ys = [p[1] for p in zone.fence_bottom]
    return min(ys) <= y <= max(ys)


def _picket_direction_at_y(y: float, zone: CameraZone) -> Point | None:
    """The picket's on-screen tilt direction, upright-corrected for depth.

    A real picket's apparent tilt is a near-camera perspective effect: poles
    close to the camera lean the most, and lean progressively less the
    farther away they are, converging to upright (vertical) at the edge of
    where this camera's geometry is trusted at all. `fence_picket` is only
    ever traced at ONE row (wherever the tilt was clearly visible to trace),
    so its raw direction is only valid there -- using it unmodified at every
    row overcorrects distant subjects. This linearly shrinks the horizontal
    (tilt) component from full strength at the picket's own row down to zero
    (purely vertical) at `zone.depth_cutoff` -- the existing "this camera's
    geometry stops being trustworthy here" row, reused rather than inventing
    a second cutoff. Rows nearer than the picket's own keep the full tilt;
    rows at or beyond depth_cutoff are fully upright.
    """
    if zone.fence_picket is None:
        return None
    top_point, base_point = zone.fence_picket
    dx = top_point[0] - base_point[0]
    dy = top_point[1] - base_point[1]
    span = base_point[1] - zone.depth_cutoff
    if span <= 1e-9:
        return (dx, dy)
    fraction = (y - zone.depth_cutoff) / span
    fraction = max(0.0, min(1.0, fraction))
    return (dx * fraction, dy)


def fence_separation_at_y(
    y: float, zone: CameraZone, frame_width: int, frame_height: int
) -> float | None:
    """Pixel separation between the top-rail and base fence lines at row `y`.

    Prefers projecting along `zone.fence_picket`'s own on-screen angle (one
    hand-traced picket's top-to-base edge) when configured: pairing the two
    lines at the SAME row silently assumes a picket renders perfectly
    vertical in frame, which is false whenever the camera looks down the
    fence at an angle -- confirmed to matter in practice (2026-09-05: a
    corpus-wide height-calibration check clustered correctly only on cam06,
    whose base line happens to be near-vertical already). The picket's tilt
    is itself depth-adjusted (see `_picket_direction_at_y`) rather than
    applied uniformly at every row. Falls back to the naive same-row method
    when no picket is traced yet, or the picket's projection never crosses
    the top rail.

    None when the camera has no `fence_bottom` traced yet, `y` falls outside
    the base line's own range (see `is_grounded_at_fence`), or the separation
    is too small to be a reliable ruler (`MIN_FENCE_SEPARATION_PX`).
    """
    if zone.fence is None or zone.fence_bottom is None:
        return None
    if not is_grounded_at_fence(y, zone):
        return None
    origin = (_fence_x_at_y(y, zone.fence_bottom), y)
    direction = _picket_direction_at_y(y, zone)
    if direction is not None:
        crossing = _project_along_picket(origin, direction, zone.fence)
        if crossing is not None:
            dx_px = (crossing[0] - origin[0]) * frame_width
            dy_px = (crossing[1] - origin[1]) * frame_height
            separation = (dx_px * dx_px + dy_px * dy_px) ** 0.5
            if separation < MIN_FENCE_SEPARATION_PX:
                return None
            return separation
    top_x = _fence_x_at_y(y, zone.fence) * frame_width
    separation = abs(top_x - origin[0] * frame_width)
    if separation < MIN_FENCE_SEPARATION_PX:
        return None
    return separation


def pixels_per_metre_at_y(
    y: float, zone: CameraZone, frame_width: int, frame_height: int
) -> float | None:
    """Pixels-per-metre scale at row `y`, from the fence's known real height.

    Frame width AND height must be passed explicitly: normalised x and y are
    not the same scale (frames are 320x240, not square), so pixel separation
    must be computed in real pixels before it's divided into a metres-based
    ruler.
    """
    separation = fence_separation_at_y(y, zone, frame_width, frame_height)
    if separation is None:
        return None
    return separation / zone.fence_height_m


def subject_base_y(bbox: tuple[float, float, float, float], frame_height: int) -> float:
    """Normalised row of a bbox's bottom edge (the subject's feet).

    The feet row, not the bbox centroid, is the correct depth reference for
    fence-based calibration -- a tall subject's centroid sits well above where
    it actually touches the ground.
    """
    _, y, _, h = bbox
    return (y + h) / frame_height


def estimated_height_m(
    bbox_height_px: float, y: float, zone: CameraZone, frame_width: int, frame_height: int
) -> float | None:
    """A tracked bbox's real height in metres at feet-row `y`, or None if
    the camera is uncalibrated there (see `pixels_per_metre_at_y`)."""
    scale = pixels_per_metre_at_y(y, zone, frame_width, frame_height)
    if not scale:
        return None
    return bbox_height_px / scale


def estimated_width_m(
    bbox_width_px: float, y: float, zone: CameraZone, frame_width: int, frame_height: int
) -> float | None:
    """A tracked bbox's real width in metres at feet-row `y`, or None if
    the camera is uncalibrated there (see `pixels_per_metre_at_y`)."""
    scale = pixels_per_metre_at_y(y, zone, frame_width, frame_height)
    if not scale:
        return None
    return bbox_width_px / scale


def estimated_speed_mps(
    distance_px: float,
    dt_seconds: float,
    y: float,
    zone: CameraZone,
    frame_width: int,
    frame_height: int,
) -> float | None:
    """Real-world speed (m/s) for a subject moving `distance_px` in
    `dt_seconds`, at feet-row `y`. None if `dt_seconds` isn't positive or the
    camera is uncalibrated there.
    """
    if dt_seconds <= 0:
        return None
    scale = pixels_per_metre_at_y(y, zone, frame_width, frame_height)
    if not scale:
        return None
    return (distance_px / scale) / dt_seconds


# --------------------------------------------------------------------------
# Fence "band" (the structure itself, between the two lines) -- discovery
# stage only. NOT wired into scripts/backtest.py::classify; docs/plan.md
# requires reporting how often this even occurs in the labelled incident+
# animal set before any rule is built on it.
# --------------------------------------------------------------------------


def in_fence_band(point: Point, zone: CameraZone) -> bool:
    """True if `point` sits between the top-rail and base fence lines at its
    own row -- i.e. on the fence structure itself, not clearly inside or
    outside. Requires both lines and a grounded row (see
    `is_grounded_at_fence`); returns False ("not on the fence") otherwise.
    """
    if zone.fence is None or zone.fence_bottom is None:
        return False
    x, y = point
    if not is_grounded_at_fence(y, zone):
        return False
    top_x = _fence_x_at_y(y, zone.fence)
    bottom_x = _fence_x_at_y(y, zone.fence_bottom)
    lo, hi = sorted((top_x, bottom_x))
    return lo <= x <= hi


def entered_band_from_outside(track: Sequence[Point], zone: CameraZone) -> bool:
    """True if the track has a point outside the fence at some point before a
    later point sitting on the fence band itself -- a candidate "climbing/
    breaching the fence" signature. Track order matters here (unlike
    `track_crosses_fence`), since only OUTSIDE-then-band is a breach attempt;
    band-then-outside (e.g. a guard stepping away from the fence) is not.
    """
    seen_outside = False
    for p in track:
        if seen_outside and in_fence_band(p, zone):
            return True
        if classify_zone(p, zone) == "outside":
            seen_outside = True
    return False
