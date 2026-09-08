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


def blob_white_fraction(
    frame_bgr: np.ndarray, contour: np.ndarray, *, brightness_threshold: int = 220
) -> float:
    """Fraction of the contour's own pixels reading near-white/overexposed.

    A bright wind-blown obstruction (vegetation, a web) sitting right against
    the lens saturates the sensor -- genuinely overexposed grey (near 255),
    not just brightly lit. A real subject's skin/clothing/fur rarely reads
    this way even under a flashlight. Measured 2026-09-06 on the labelled
    corpus: animal's max across every clip is 0.182, incident's is 0.310 --
    both well under the 0.4 threshold `classify()` uses, while several
    known-blinding guard clips hit 0.94-1.00.
    """
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask == 255]
    if pixels.size == 0:
        return 0.0
    return float((pixels > brightness_threshold).mean())


def post_flash_red_shift(frames_bgr: Sequence[np.ndarray]) -> float:
    """Whole-frame red shift AFTER the clip's brightness peak, minus before it.

    The signature of a guard shining a flashlight directly into the lens: a
    bright flash, then the camera's own auto-exposure/white-balance overcorrects
    and the WHOLE FRAME settles visibly red for the frames that follow. A
    physical obstruction against the lens (vegetation, a web) is overexposed
    too, but produces no such flash-then-recover transition.

    Must be measured as this before/after TRANSITION, not as a clip-wide average
    R/G -- averaging over all frames dilutes the effect to noise (measured
    2026-09-07: the averaged version gave +0.017 for confirmed flashlight clips
    vs +0.118 for confirmed obstructions, i.e. backwards).

    Measured over the whole labelled corpus, only GUARD clips ever exceed 0.10
    (guard max 0.581; environment max 0.014, animal 0.038, incident 0.001,
    resident 0.020) -- so it is a zero-leak guard/flashlight identifier on this
    corpus, not merely a maintenance filter.
    """
    if len(frames_bgr) < 6:
        return 0.0
    brightness, red_green = [], []
    for frame in frames_bgr:
        brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        _, green, red = cv2.split(frame.astype(np.float32))
        green_mean = green.mean()
        red_green.append(red.mean() / green_mean if green_mean > 1 else 1.0)
    peak = int(np.argmax(brightness))
    if peak < 1 or peak >= len(frames_bgr) - 1:
        return 0.0
    return float(np.mean(red_green[peak + 1 :]) - np.mean(red_green[:peak]))


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


# One definition of "this pixel is the guard's flashlight", shared by
# green_light_mask and therefore by every feature and overlay derived from it.
#
# min_saturation was 60 until 2026-09-08, which admitted essentially all
# daylight foliage: measured over real clips, a flashlight's green components
# run S p50 168 (p90 237) while sunlit grass runs S p50 69 with a p99 of only
# 99 -- saturation separates the two at AUC 0.907, and 60 sits below the whole
# grass distribution. Raising it collapses daylight foliage's peak whole-frame
# green reading from 0.164 to 0.0001 while costing a real flashlight only about
# half its reading (0.053 -> 0.025).
#
# min_blob_area drops connected components below that many pixels. It is the
# smaller effect of the two and does NOT work alone -- daylight grass is not
# speckle, it forms large contiguous regions (median largest-per-frame
# component 827px, bigger than a real flashlight's 580px). It earns its place
# only once the saturation floor has removed the bulk, where it halves what is
# left (daylight non-guard clips reading > 0.02: 13.3% -> 6.7%).
FLASHLIGHT_MIN_SATURATION = 130
FLASHLIGHT_MIN_BLOB_AREA = 8


def green_light_mask(
    frame_bgr: np.ndarray,
    *,
    hue_low: int = 33,
    hue_high: int = 85,
    min_saturation: int = FLASHLIGHT_MIN_SATURATION,
    min_value: int = 60,
    min_blob_area: int = FLASHLIGHT_MIN_BLOB_AREA,
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

    `min_blob_area` drops connected components smaller than that many pixels.
    A flashlight is one contiguous lit patch even at distance; sunlit grass and
    foliage produce scattered single-pixel speckle that satisfies the same
    hue/saturation test. Without it the only defence against daylight foliage
    was the whole-frame `color_fraction` gate, which is a blunt instrument --
    see `scripts.spike.extract_clip_features`. 0 (the default) keeps every
    matching pixel, i.e. the pre-2026-09-08 behaviour.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (hue >= hue_low) & (hue <= hue_high) & (sat >= min_saturation) & (val >= min_value)
    if min_blob_area <= 0 or not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob_area
    return keep[labels]


def green_light_ratio(
    frame_bgr: np.ndarray,
    contour: np.ndarray,
    *,
    hue_low: int = 33,
    hue_high: int = 85,
    min_saturation: int = FLASHLIGHT_MIN_SATURATION,
    min_value: int = 60,
    min_blob_area: int = FLASHLIGHT_MIN_BLOB_AREA,
    exclude_mask: np.ndarray | None = None,
) -> float:
    """Fraction of contour pixels whose HSV hue falls in the green band with
    enough saturation/brightness to be a real light source rather than IR noise.

    Site-specific: the guard's flashlight reads as a distinct green in these
    clips, which is a narrower and likely more reliable signal than
    `saturation_ratio` alone (that also fires on any colour anomaly, e.g. a
    reddish insect glare). Hue bounds use OpenCV's 0-179 scale.

    `min_blob_area` (see `green_light_mask`) requires the matching pixels to
    form a contiguous patch, which is what separates a real light from sunlit
    grass speckle.

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
        min_blob_area=min_blob_area,
    )
    return float(np.count_nonzero(green[inside])) / int(np.count_nonzero(inside))


# If this much of a tracked box's own area is flashlight-hue pixels, the
# tracked subject is the beam itself, not a person -- shared by the render
# (relabels the box) and the model (a feature) so they never disagree.
# These cameras record at 5 fps, but ~2% of the corpus carries corrupt fps
# metadata -- measured values of 1005 and 16000 on a random 400-clip sample.
# Anything derived from fps (real elapsed time, clip duration) is nonsense on
# those clips unless the value is sanity-checked first.
CAMERA_FPS = 5.0
MAX_PLAUSIBLE_FPS = 60.0


def sane_fps(raw_fps: float | None) -> float:
    """A usable frame rate, falling back to `CAMERA_FPS` when the container's
    metadata is missing or impossible.

    Left unchecked this silently scaled `speed_mps` by up to 3200x on the
    affected clips, which is exactly the kind of unbounded outlier a fitted
    ranker latches onto (see the subject_width_m incident, 2026-09-06).
    """
    if not raw_fps or raw_fps <= 0 or raw_fps > MAX_PLAUSIBLE_FPS:
        return CAMERA_FPS
    return float(raw_fps)


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


def photometric_match(source: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Least-squares (gain, offset) so that `gain * source + offset`
    approximates `reference`, one single-channel array at a time.

    This is the per-frame counterpart to `flare_settle_index`'s whole-clip
    view: instead of deciding a frame is unusably corrupted by the IR gain
    ramp and discarding it outright, fit the brightness/contrast transform
    that would make it look like the settled background, so a real per-pixel
    difference against that background becomes possible even during the ramp.
    A single scalar (median ratio, as `_warmup_motion_features` already uses
    for consecutive-frame comparison) only corrects brightness; an affine fit
    also corrects the gain/contrast component of the ramp.

    Falls back to a pure offset (gain=1.0) when `source` has near-zero
    variance (e.g. a blank/saturated frame) -- `np.polyfit` on a
    zero-variance input is a divide-by-zero, not a meaningful fit.
    """
    src = source.astype(np.float64).ravel()
    ref = reference.astype(np.float64).ravel()
    if src.size == 0:
        return 1.0, 0.0
    if float(np.std(src)) < 1e-6:
        return 1.0, float(np.mean(ref) - np.mean(src))
    gain, offset = np.polyfit(src, ref, 1)
    return float(gain), float(offset)


def apply_photometric_match(frame: np.ndarray, gain: float, offset: float) -> np.ndarray:
    """Apply a `photometric_match` (gain, offset) pair, clipped back to uint8."""
    corrected = frame.astype(np.float64) * gain + offset
    return np.clip(corrected, 0, 255).astype(np.uint8)


def photometric_match_color(frame_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
    """Per-channel `photometric_match`, for evening out a flare-lit COLOUR
    frame against a settled reference -- corrects both brightness and colour
    cast (each of B/G/R gets its own gain/offset) rather than only grey level.
    Display/debug use only; detection always works in greyscale.
    """
    channels = cv2.split(frame_bgr)
    ref_channels = cv2.split(reference_bgr)
    corrected = [
        apply_photometric_match(channel, *photometric_match(channel, ref_channel))
        for channel, ref_channel in zip(channels, ref_channels, strict=True)
    ]
    return cv2.merge(corrected)


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


def daylight_hint(timestamp_utc: str | None) -> bool | None:
    """Exogenous "could this clip plausibly have ambient daylight colour in it"
    answer, for `scripts.spike.extract_clip_features(daylight_hint=...)`.

    True for real daylight OR either twilight margin, so a dusk clip with
    genuine residual colour is still treated as colour footage. `None` when
    there is no usable timestamp, which restores the pure image-statistic
    behaviour rather than guessing.

    Every caller of `extract_clip_features` that has a clip timestamp should
    use this, so the render, the screening run and the ranker can never
    disagree about whether a clip was shot at night.
    """
    if not timestamp_utc:
        return None
    try:
        return bool(is_daylight(timestamp_utc) or is_twilight(timestamp_utc))
    except (ValueError, KeyError):
        return None
