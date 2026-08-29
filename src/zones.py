"""Polyline-side / depth / ignore-region geometry for a single camera's zone config.

Pure geometry, no video required: this operates on normalized (x, y) points in
[0, 1], x=column fraction (0=left), y=row fraction (0=top, 1=bottom of frame).
Used by the (future) motion/classification pipeline and by scripts/spike.py's
feature extraction once real contours exist.

Side convention: fences are drawn as an ordered polyline from one point to the
next. Walking along that direction in image coordinates (x right, y down),
"right" and "left" match normal compass-facing intuition (facing the direction
of travel, your right hand points to larger-cross-product side). This must
match config/cameras.yaml's `outside: left|right` field.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.config import CameraZone, Point

# Zone classification outcomes. "ignored" and "ambiguous" take priority over
# "outside"/"inside" because they represent "cannot reliably classify here",
# not a side judgement -- see the fail-safe policy in docs/plan.md.
ZoneClassification = str  # "outside" | "inside" | "ambiguous" | "ignored"


def signed_side(point: Point, fence: Sequence[Point]) -> float:
    """Cross-product sign relative to the nearest fence segment.

    Positive => right of the polyline's point order, negative => left, zero =>
    exactly on the line. Uses the *nearest* segment so bends in the polyline
    are handled correctly rather than assuming a single straight line.
    """
    if len(fence) < 2:
        raise ValueError("fence needs >= 2 points")

    px, py = point
    best_dist_sq = float("inf")
    best_cross = 0.0
    for a, b in zip(fence, fence[1:], strict=False):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0.0:
            continue
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
        proj_x, proj_y = ax + t * dx, ay + t * dy
        dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_cross = dx * (py - ay) - dy * (px - ax)
    return best_cross


def side_name(point: Point, fence: Sequence[Point]) -> str:
    cross = signed_side(point, fence)
    if cross > 0:
        return "right"
    if cross < 0:
        return "left"
    return "on_line"


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
    if zone.fence is None or zone.outside is None:
        return "ambiguous"
    return "outside" if side_name(point, zone.fence) == zone.outside else "inside"


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
