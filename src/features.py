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


def green_light_ratio(
    frame_bgr: np.ndarray,
    contour: np.ndarray,
    *,
    hue_low: int = 35,
    hue_high: int = 85,
    min_saturation: int = 60,
    min_value: int = 60,
) -> float:
    """Fraction of contour pixels whose HSV hue falls in the green band with
    enough saturation/brightness to be a real light source rather than IR noise.

    Site-specific: the guard's flashlight reads as a distinct green in these
    clips, which is a narrower and likely more reliable signal than
    `saturation_ratio` alone (that also fires on any colour anomaly, e.g. a
    reddish insect glare). Hue bounds use OpenCV's 0-179 scale.
    """
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    region = hsv[mask == 255]
    if region.size == 0:
        return 0.0
    hue, sat, val = region[:, 0], region[:, 1], region[:, 2]
    green = (hue >= hue_low) & (hue <= hue_high) & (sat >= min_saturation) & (val >= min_value)
    return float(np.count_nonzero(green)) / region.shape[0]


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
