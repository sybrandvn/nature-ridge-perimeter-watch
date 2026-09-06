"""Per-blob and per-track feature extraction for the CV spike / classifier.

Frame-level features (aspect_ratio, solidity, saturation_ratio, edge_density,
row_normalised_area) take a single OpenCV contour plus its source frame.
Track-level features (path_length, jitter, persistence) take a sequence of
centroids across a clip's frames, since those need a track rather than a
single blob.

All frame-level functions are testable against synthetic OpenCV-drawn shapes,
without needing real camera footage -- see tests/test_features.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

import cv2
import numpy as np


def aspect_ratio(contour: np.ndarray) -> float:
    """height / width of the contour's upright bounding box."""
    _, _, w, h = cv2.boundingRect(contour)
    if w == 0:
        return 0.0
    return h / w


def solidity(contour: np.ndarray) -> float:
    """contour area / convex hull area; low solidity = ragged/fragmented blob."""
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0:
        return 0.0
    return area / hull_area


def row_normalised_area(contour: np.ndarray, reference_row: float) -> float:
    """Contour area scaled by (reference_row / centroid_row)^2.

    Corrects for perspective: cameras look down the fence line, so an object
    farther away (smaller row) appears smaller at the same real-world size.
    Scaling up by this ratio makes area comparable across depth.
    """
    area = cv2.contourArea(contour)
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return 0.0
    centroid_row = max(m["m01"] / m["m00"], 1.0)
    scale = (reference_row / centroid_row) ** 2
    return area * scale


def saturation_ratio(frame_bgr: np.ndarray, contour: np.ndarray) -> float:
    """Mean HSV saturation inside the contour mask, normalised to [0, 1].

    Discriminates flashlight glare / colour anomalies from IR-greyscale
    content, where saturation should sit near zero.
    """
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    pixels = saturation[mask == 255]
    if pixels.size == 0:
        return 0.0
    return float(pixels.mean()) / 255.0


def color_saturation_fraction(frame_bgr: np.ndarray, *, saturation_threshold: int = 30) -> float:
    """Fraction of the WHOLE frame with HSV saturation above `saturation_threshold`.

    Distinguishes genuine daylight/dusk colour footage from IR-lit night
    footage: real ambient colour (foliage, ground, sky) is spread broadly
    across the frame, whereas true IR content is near-monochrome everywhere
    except a small lit source (flashlight, headlamp). A single contour's
    `saturation_ratio` can't tell those apart -- a green flashlight glow and a
    frame full of green foliage can look identical from inside the blob alone.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return float(np.count_nonzero(saturation > saturation_threshold)) / saturation.size


def green_light_mask(
    frame_bgr: np.ndarray,
    *,
    hue_low: int = 33,
    hue_high: int = 85,
    min_saturation: int = 60,
    min_value: int = 60,
) -> np.ndarray:
    """Boolean mask of pixels reading as the guard's flashlight (green, lit,
    saturated), shared by `green_light_ratio` and any overlay that draws it so
    the two can never disagree about what counts as "the light".

    hue_low was 35 until 2026-09-03: cam01's flashlight measured hue 34-35
    (camera-to-camera sensor/white-balance variance), missing the old bound by
    a single unit. Lowering to 33 recovers it (raw green-pixel fraction on
    cam01/16167's real flashlight frames +19%) with no measurable change on
    known foliage false-positive references (cam03/9066, cam03/8767) or the
    cam08/4306 true-flashlight reference.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (hue >= hue_low) & (hue <= hue_high) & (sat >= min_saturation) & (val >= min_value)


def green_light_ratio(
    frame_bgr: np.ndarray,
    contour: np.ndarray,
    *,
    hue_low: int = 33,
    hue_high: int = 85,
    min_saturation: int = 60,
    min_value: int = 60,
    exclude_mask: np.ndarray | None = None,
) -> float:
    """Fraction of contour pixels whose HSV hue falls in the green band with
    enough saturation/brightness to be a real light source rather than IR noise.

    Site-specific: the guard's flashlight reads as a distinct green in these
    clips, which is a narrower and likely more reliable signal than
    `saturation_ratio` alone (that also fires on any colour anomaly, e.g. a
    reddish insect glare). Hue bounds use OpenCV's 0-179 scale.

    `exclude_mask`, when given (see `ignore_region_mask`), removes pixels from
    both the numerator and denominator -- a known per-camera artifact region
    (a stationary light left permanently in frame, or lens edge/vignette
    colour fringing) would otherwise read identically to the guard's flashlight
    to this hue check alone.
    """
    contour_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, color=255, thickness=-1)
    inside = contour_mask == 255
    if exclude_mask is not None:
        inside &= ~exclude_mask
    if not np.any(inside):
        return 0.0
    green = green_light_mask(
        frame_bgr,
        hue_low=hue_low,
        hue_high=hue_high,
        min_saturation=min_saturation,
        min_value=min_value,
    )
    return float(np.count_nonzero(green[inside])) / int(np.count_nonzero(inside))


# If this much of a tracked box's own area is flashlight-hue pixels, the
# tracked subject is the beam itself, not a person -- shared by the render
# (relabels the box) and the model (a feature) so they never disagree.
FLASHLIGHT_SUBJECT_THRESHOLD = 0.5


def flashlight_bbox_overlap(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    exclude_mask: np.ndarray | None = None,
) -> float:
    """Fraction of a tracked box's own pixels (not just its best-frame
    contour, see `green_light_ratio`) that read as the guard's flashlight.

    Shared by `scripts.render_debug` (marks a box as the flashlight itself,
    not a subject) and `scripts.spike.extract_clip_features` (so the render
    and the model score the same signal the same way) -- checked per tracked
    frame, so a beam that only fills the box briefly still gets counted.

    `exclude_mask` removes a known stationary light (`ignore_region_mask`)
    from the count, same reasoning as `green_light_ratio`.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    mask = green_light_mask(frame_bgr)
    if exclude_mask is not None:
        mask = mask & ~exclude_mask
    region = mask[y : y + h, x : x + w]
    return float(np.count_nonzero(region)) / region.size if region.size else 0.0


def ignore_region_mask(
    frame_width: int, frame_height: int, ignore_polygons: Sequence[Sequence[tuple[float, float]]]
) -> np.ndarray:
    """Boolean pixel mask, True inside any configured ignore polygon.

    `ignore_polygons` are normalised (x, y) in [0, 1] (`CameraZone.ignore`),
    same convention as `src.zones`. Used to keep a known, fixed camera-specific
    artifact region -- a stationary light left in view, or a lens edge/vignette
    colour-fringing band -- out of both motion contour detection and the
    green-light hue check, without it needing to look like a real subject or
    flashlight to either.
    """
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    for polygon in ignore_polygons:
        if len(polygon) < 3:
            continue
        points = np.array(
            [(x * frame_width, y * frame_height) for x, y in polygon], dtype=np.int32
        )
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def detect_stationary_light_mask(
    frames: Sequence[np.ndarray],
    *,
    min_stable_fraction: float = 0.8,
    exclude_region: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean pixel mask, auto-detecting a fixed light left in view for a whole
    clip -- the per-clip alternative to hand-tracing a `zone.ignore` polygon
    for every new stationary light a camera happens to have.

    A pixel counts as a stationary light if it reads as `green_light_mask` in
    at least `min_stable_fraction` of the given frames: a fixed light stays
    lit (and stays in the same place) for the whole clip, while the guard's
    own moving flashlight only lights up wherever it's currently pointed for
    a handful of frames at a time.

    `exclude_region` (e.g. the union of every frame's tracked subject box,
    dilated a little) is subtracted from the result before returning it --
    without this, a guard who stands still pointing a flashlight at one spot
    for most of a short clip would read exactly like a fixed light and get
    incorrectly excluded from its own flashlight scoring. Callers that track
    a subject should always pass this.

    Meaningful mainly on already-dark (non-daylight) footage: daytime foliage
    reads within the same hue band and can be "stable" too, but callers don't
    need to gate on daylight themselves -- every feature this feeds
    (`green_light_ratio`, `flashlight_bbox_overlap`) is already zeroed on a
    daylight clip regardless of what this mask excludes.
    """
    if not frames:
        return np.zeros((0, 0), dtype=bool)
    count = np.zeros(frames[0].shape[:2], dtype=np.int32)
    for frame in frames:
        count += green_light_mask(frame)
    stable = count >= max(1, round(min_stable_fraction * len(frames)))
    if exclude_region is not None:
        stable &= ~exclude_region
    return stable


def edge_density(frame_bgr: np.ndarray, contour: np.ndarray) -> float:
    """Fraction of pixels inside the contour's bounding box that are Canny edges."""
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return 0.0
    roi = frame_bgr[y : y + h, x : x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges)) / edges.size


def path_length(centroids: Sequence[tuple[float, float]]) -> float:
    """Sum of frame-to-frame centroid displacement distances."""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(centroids, centroids[1:], strict=False):
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def jitter(centroids: Sequence[tuple[float, float]]) -> float:
    """Std deviation of frame-to-frame displacement distance.

    High jitter suggests erratic movement (e.g. insects near the lens); low
    jitter suggests a smooth, deliberate track (guard, animal, or intruder).
    """
    if len(centroids) < 3:
        return 0.0
    steps = [
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(centroids, centroids[1:], strict=False)
    ]
    mean = sum(steps) / len(steps)
    variance = sum((s - mean) ** 2 for s in steps) / len(steps)
    return math.sqrt(variance)


def persistence(frames_detected: int, total_frames: int) -> float:
    """Fraction of a clip's frames in which the blob was detected at all."""
    if total_frames == 0:
        return 0.0
    return frames_detected / total_frames


def longest_detection_run(detected_frame_indices: Sequence[int], total_frames: int) -> float:
    """Longest run of *consecutive* detected frames, as a fraction of the clip.

    Complements `persistence`, which counts detected frames wherever they fall:
    a blob seen in 60% of frames scattered one at a time is flicker, the same
    60% in one unbroken run is a subject moving through frame.
    """
    if total_frames == 0 or not detected_frame_indices:
        return 0.0
    longest = current = 1
    for previous, index in zip(
        detected_frame_indices, detected_frame_indices[1:], strict=False
    ):
        current = current + 1 if index == previous + 1 else 1
        longest = max(longest, current)
    return longest / total_frames


def area_stability(areas: Sequence[float]) -> float:
    """Coefficient of variation (std / mean) of the tracked blob's area.

    A real body keeps roughly the same silhouette between frames; a flashlight
    pool, wind-shaken foliage or a near-lens insect swells and collapses. Lower
    is more subject-like.
    """
    if len(areas) < 2:
        return 0.0
    mean = sum(areas) / len(areas)
    if mean <= 0:
        return 0.0
    variance = sum((a - mean) ** 2 for a in areas) / len(areas)
    return math.sqrt(variance) / mean


def depth_progression(distances: Sequence[float]) -> float:
    """Net change in ground-plane distance, divided by the total distance
    travelled back and forth (see `src.ground_calibration`).

    A subject walking steadily along the fence line changes range in one
    direction, so this stays near 1; one that mills in place, or an artifact
    that never really moves in depth, drifts back and forth and stays near 0.
    0.0 with fewer than 2 samples or no measured movement at all.
    """
    if len(distances) < 2:
        return 0.0
    steps = [abs(b - a) for a, b in zip(distances, distances[1:], strict=False)]
    total_variation = sum(steps)
    if total_variation <= 0:
        return 0.0
    return abs(distances[-1] - distances[0]) / total_variation


def normalised_speed(centroids: Sequence[tuple[float, float]], blob_width: float) -> float:
    """Mean per-frame centroid displacement in blob-widths ("body lengths").

    Raw pixel speed conflates a distant subject with a slow one; dividing by
    the blob's own width makes it scale-invariant, so a near insect crossing
    the lens and a distant person walking are directly comparable.
    """
    if len(centroids) < 2 or blob_width <= 0:
        return 0.0
    steps = [
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(centroids, centroids[1:], strict=False)
    ]
    return (sum(steps) / len(steps)) / blob_width


def heading_change(
    centroids: Sequence[tuple[float, float]], *, min_step: float = 0.5
) -> float:
    """Mean absolute turn angle (radians, 0..pi) between successive motion steps.

    Near-zero means a straight, deliberate track; near pi means the blob
    reverses direction every frame, which is what insects, foliage and beam
    flicker do. Steps shorter than `min_step` pixels are dropped -- their
    direction is quantisation noise, not heading.
    """
    angles = [
        math.atan2(y2 - y1, x2 - x1)
        for (x1, y1), (x2, y2) in zip(centroids, centroids[1:], strict=False)
        if math.hypot(x2 - x1, y2 - y1) >= min_step
    ]
    if len(angles) < 2:
        return 0.0
    turns = [
        abs(((b - a + math.pi) % (2 * math.pi)) - math.pi)
        for a, b in zip(angles, angles[1:], strict=False)
    ]
    return sum(turns) / len(turns)


def green_light_flicker(whole_frame_green_ratios: Sequence[float]) -> float:
    """Std deviation of the whole-frame green-hue ratio across a clip's frames.

    Site-specific: the guard sweeps the flashlight rather than holding it
    still, so its green signal spikes up and down between frames rather than
    sitting at one level. Distinct from `green_light_ratio`, which reads a
    single clearest frame -- this catches guard clips where the beam isn't in
    frame at the moment of largest motion, at the cost of also needing a
    high-variance swing rather than just presence.
    """
    if len(whole_frame_green_ratios) < 2:
        return 0.0
    n = len(whole_frame_green_ratios)
    mean = sum(whole_frame_green_ratios) / n
    variance = sum((v - mean) ** 2 for v in whole_frame_green_ratios) / n
    return math.sqrt(variance)


def flare_frames(frame_medians: Sequence[float], *, tolerance: float = 3.0) -> list[bool]:
    """Which frames are corrupted by an IR gain/illuminator step.

    These cameras change IR gain (and switch the illuminator) as a *global*
    exposure change, so the whole-frame median grey level jumps by 12-24 levels
    between adjacent frames. Once settled it moves by 0-2 levels even while a
    subject crosses the frame, because one animal covers too few pixels to
    shift the median. That gap is what makes flare separable from motion at all.

    Both frames either side of a step are marked: the step is a transition, and
    neither end of it can be differenced against a stable background.
    """
    n = len(frame_medians)
    if n < 2:
        return [False] * n
    flagged = [False] * n
    for i in range(1, n):
        if abs(frame_medians[i] - frame_medians[i - 1]) > tolerance:
            flagged[i - 1] = True
            flagged[i] = True
    return flagged


def flare_settle_index(
    frame_medians: Sequence[float],
    *,
    tolerance: float = 3.0,
    max_fraction: float = 0.4,
) -> int:
    """First frame index after the opening IR gain ramp has settled.

    A fixed warmup cut is wrong in both directions on this footage: some clips
    settle by frame 2 and lose usable subject motion, while others are still
    ramping at frame 21 and poison the background model. Measuring the ramp per
    clip fixes both.

    `max_fraction` caps how much of a clip this may discard -- the cameras are
    motion-triggered, so the subject is often already moving during the ramp,
    and on a short clip dropping the ramp can mean dropping the whole event.
    """
    flagged = flare_frames(frame_medians, tolerance=tolerance)
    settle = 0
    for i, is_flare in enumerate(flagged):
        if is_flare:
            settle = i + 1
    return min(settle, int(len(frame_medians) * max_fraction))


def _hhmm_to_minutes(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def time_of_day(
    timestamp_utc: str,
    *,
    window_start: str = "18:00",
    window_end: str = "06:00",
    utc_offset_hours: float = 2.0,
) -> str:
    """"night" if the clip's local time falls inside the configured operating
    window (which wraps midnight, e.g. 18:00-06:00), else "day".

    Parsed straight from the clip's UTC timestamp -- a descriptive tag for
    spike/report analysis, not a manually-entered label and not itself a
    classification input (docs/plan.md keeps a single night-only threshold
    profile; this just makes dusk/dawn-lit clips distinguishable from deep
    night ones when eyeballing separability).
    """
    dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    local = dt + timedelta(hours=utc_offset_hours)
    local_minutes = local.hour * 60 + local.minute
    start_minutes = _hhmm_to_minutes(window_start)
    end_minutes = _hhmm_to_minutes(window_end)
    if start_minutes <= end_minutes:
        in_window = start_minutes <= local_minutes < end_minutes
    else:
        in_window = local_minutes >= start_minutes or local_minutes < end_minutes
    return "night" if in_window else "day"


# Approximate Pretoria (site is on the Highveld, ~25.75S 28.19E) local sunrise/sunset
# by month, rounded to 5 minutes. Not astronomically precise -- good enough to flag the
# cases the fixed 18:00-06:00 window gets wrong at the height of southern-hemisphere
# summer (Nov-Feb), where real dusk/dawn daylight extends well past those clock times.
_PRETORIA_SUN_TIMES_BY_MONTH = {
    1: ("05:30", "18:55"),
    2: ("05:50", "18:35"),
    3: ("06:05", "18:05"),
    4: ("06:20", "17:35"),
    5: ("06:35", "17:15"),
    6: ("06:50", "17:10"),
    7: ("06:50", "17:20"),
    8: ("06:30", "17:40"),
    9: ("05:55", "17:55"),
    10: ("05:25", "18:10"),
    11: ("05:05", "18:30"),
    12: ("05:00", "18:55"),
}


def is_daylight(timestamp_utc: str, *, utc_offset_hours: float = 2.0) -> bool:
    """Whether the clip's local time falls between approximate sunrise and sunset
    for the given month, per `_PRETORIA_SUN_TIMES_BY_MONTH`.

    A rough, sun-aware companion to `time_of_day()`'s fixed window -- use it as an
    extra filter/eyeball column (e.g. in spike CSVs) to catch dusk/dawn clips that
    the fixed window mislabels as "night" during summer, not as a replacement for
    the operating-window logic itself.
    """
    dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    local = dt + timedelta(hours=utc_offset_hours)
    sunrise, sunset = _PRETORIA_SUN_TIMES_BY_MONTH[local.month]
    local_minutes = local.hour * 60 + local.minute
    return _hhmm_to_minutes(sunrise) <= local_minutes < _hhmm_to_minutes(sunset)


def minutes_from_daylight_boundary(timestamp_utc: str, *, utc_offset_hours: float = 2.0) -> float:
    """Signed minutes from whichever of sunrise/sunset (per
    `_PRETORIA_SUN_TIMES_BY_MONTH`) is nearer to the clip's local time.

    `is_daylight` only answers day-or-night; it treats a clip one minute past
    sunset identically to one taken at midnight. This is the continuous
    version underneath it -- negative before the nearer boundary (still dark
    before sunrise, or still light before sunset), positive after (day has
    broken, or dusk has begun). Confirmed necessary 2026-09-04: a cluster of
    `blob_count` false-fires on real guard clips landed 13-56 minutes after
    sunset, all still `is_daylight=False` since they're past the boundary --
    the binary flag can't distinguish "just went dark" from "the middle of
    the night", only this can.
    """
    dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    local = dt + timedelta(hours=utc_offset_hours)
    sunrise, sunset = _PRETORIA_SUN_TIMES_BY_MONTH[local.month]
    local_minutes = local.hour * 60 + local.minute
    from_sunrise = local_minutes - _hhmm_to_minutes(sunrise)
    from_sunset = local_minutes - _hhmm_to_minutes(sunset)
    return from_sunrise if abs(from_sunrise) <= abs(from_sunset) else from_sunset


def is_twilight(
    timestamp_utc: str, *, margin_minutes: float = 60.0, utc_offset_hours: float = 2.0
) -> bool:
    """Whether the clip falls within `margin_minutes` of dawn OR dusk (either
    side of sunrise or sunset), per `minutes_from_daylight_boundary`.

    Dawn and dusk share the same underlying problem for this site's footage:
    residual ambient colour/light that a strict day/night split assigns
    entirely to "night", but that still perturbs colour- and motion-based
    features the same way full daylight does. `margin_minutes=60` matches the
    measured window where `blob_count`'s guard false-fire rate spikes to 38%
    (15-60 minutes after sunset) before dropping back to baseline.
    """
    return (
        abs(minutes_from_daylight_boundary(timestamp_utc, utc_offset_hours=utc_offset_hours))
        <= margin_minutes
    )
