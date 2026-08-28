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
