"""Versioned persistence for expensive clip feature extraction.

The current extractor still lives in :mod:`scripts.spike`; this module owns the
stable cache boundary while that large, heavily-tested implementation is moved
in smaller behaviour-preserving steps. Cached values are the extractor's
JSON-safe feature mapping, not ``ClipDetection`` (which contains raw NumPy
frames, masks, and contours).

Although the database column is named ``motion_fingerprint``, its value here is
an extraction fingerprint: global motion settings plus every per-clip input
that can affect extracted features. This deliberately trades reuse after a zone
edit for correctness until the detector/feature split can cache a smaller,
truly zone-independent payload.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src import db
from src.config import CameraZone

EXTRACTOR_VERSION = "motion-features-v1"
_NO_MOTION_KEY = "__perimeter_watch_no_motion__"


def detect_clip(*args: Any, **kwargs: Any) -> ClipDetection | None:
    """Public detector entry point.

    The implementation remains in ``scripts.spike._detect_clip`` during its
    behaviour-preserving extraction, because its tracking helpers are still
    colocated there.  Keeping callers on this neutral entry point first lets
    those helpers move in small tested slices without another import churn.
    """
    from scripts.spike import _detect_clip

    return _detect_clip(*args, **kwargs)


def largest_contour(mask: np.ndarray, *, max_area: float | None = None) -> np.ndarray | None:
    """Largest external contour in a binary motion mask, or ``None`` if empty."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if max_area is not None:
        contours = [contour for contour in contours if cv2.contourArea(contour) <= max_area]
    return max(contours, key=cv2.contourArea) if contours else None


def _contour_bbox(contour: np.ndarray) -> tuple[int, int, int, int]:
    """Return a contour box in corner form ``(x0, y0, x1, y1)``."""
    x, y, width, height = cv2.boundingRect(contour)
    return (x, y, x + width, y + height)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    intersection_width = max(0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def _size_change_plausible(from_area: float, to_area: float, max_ratio: float) -> bool:
    """Whether two tracked-box areas can plausibly describe one subject."""
    if from_area <= 0 or to_area <= 0:
        return True
    ratio = to_area / from_area
    return (1.0 / max_ratio) <= ratio <= max_ratio


def _size_relative_margin(
    bbox: tuple[int, int, int, int], *, margin_fraction: float, min_margin: float
) -> float:
    """A position-search margin scaled to the tracked object's own size."""
    x0, y0, x1, y1 = bbox
    return max(margin_fraction * max(x1 - x0, y1 - y0), min_margin)


def contour_centroid(contour: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def _bbox_to_rect_contour(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Synthesize a rectangular contour from a corner-form bounding box."""
    x0, y0, x1, y1 = bbox
    return np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32)


def reacquire_by_template(
    gray_frame: np.ndarray,
    template: np.ndarray,
    last_bbox: tuple[int, int, int, int],
    *,
    search_margin: float,
    match_threshold: float,
    velocity: tuple[float, float] = (0.0, 0.0),
) -> tuple[int, int, int, int] | None:
    """Find a prior subject template in a local, velocity-biased window."""
    template_height, template_width = template.shape[:2]
    if template_height == 0 or template_width == 0:
        return None
    x0, y0, x1, y1 = last_bbox
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    velocity_x, velocity_y = velocity
    base_half_width = (x1 - x0) / 2.0 + search_margin
    base_half_height = (y1 - y0) / 2.0 + search_margin
    center_x += velocity_x / 2.0
    center_y += velocity_y / 2.0
    half_width = base_half_width + abs(velocity_x) / 2.0
    half_height = base_half_height + abs(velocity_y) / 2.0
    frame_height, frame_width = gray_frame.shape[:2]
    window_x0 = max(int(center_x - half_width), 0)
    window_y0 = max(int(center_y - half_height), 0)
    window_x1 = min(int(center_x + half_width), frame_width)
    window_y1 = min(int(center_y + half_height), frame_height)
    window = gray_frame[window_y0:window_y1, window_x0:window_x1]
    if window.shape[0] < template_height or window.shape[1] < template_width:
        return None
    result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, maximum, _, location = cv2.minMaxLoc(result)
    if maximum < match_threshold:
        return None
    match_x, match_y = location
    return (
        window_x0 + match_x,
        window_y0 + match_y,
        window_x0 + match_x + template_width,
        window_y0 + match_y + template_height,
    )


@dataclass(frozen=True)
class TrackedObject:
    """One persistently identified object in one frame.

    ``bbox`` is corner-form ``(x0, y0, x1, y1)``. It is intentionally a
    detector-owned, zone-independent DTO: consumers such as feature scoring
    and the debug renderer can share the same observed object without pulling
    tracking implementation back into their module.
    """

    track_id: int
    bbox: tuple[int, int, int, int]
    # Other ids sharing this detector box; empty when this id has its own blob.
    merged_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class FrameDetection:
    """Everything the motion detector observed in one video frame."""

    index: int
    frame: np.ndarray
    mask: np.ndarray
    all_contours: list[np.ndarray]
    blobs: list[np.ndarray]
    largest: np.ndarray | None
    centroid: tuple[float, float] | None
    motion_pixel_fraction: float
    median_grey: float
    is_flare: bool
    recovered: bool = False
    filled_by_reverse: bool = False
    filled_by_anchor: bool = False
    suppressed_light_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ClipDetection:
    """Detector output for one clip, before any zone-specific scoring.

    This is deliberately an in-memory hand-off type, not the cache payload:
    it contains NumPy imagery and contours. Keeping it here establishes the
    detector/feature boundary needed for a later compact raw-track cache.
    """

    frames: list[FrameDetection]
    background: np.ndarray
    frame_width: int
    frame_height: int
    warmup_dropped: int
    total_frames: int
    dropped_frames: list[np.ndarray]
    dropped_frame_boxes: list[tuple[int, int, int, int] | None]
    dropped_frame_box_is_photometric: list[bool] = field(default_factory=list)
    dropped_frame_compensated: list[np.ndarray] = field(default_factory=list)
    multi_tracks: list[list[TrackedObject]] = field(default_factory=list)
    scenery_motion_fraction: float = 0.0
    has_reference_background: bool = False


@dataclass(frozen=True)
class GeometryObservations:
    """Compact, JSON-safe detector evidence needed for fence-side scoring.

    Unlike :class:`ClipDetection`, this deliberately contains no frames,
    masks, or OpenCV contours.  It is the first cacheable slice of detector
    output: enough to replay the fence/depth features after a zone edit, but
    not enough to replay colour, texture, or metric-calibration features.
    Pixel dimensions and multi-object boxes stay in pixel coordinates so a
    replay has exactly the same base-of-box convention as the detector.
    """

    frame_width: int
    frame_height: int
    best_contour_points: tuple[tuple[float, float], ...]
    genuine_contour_points: tuple[tuple[tuple[float, float], ...], ...]
    centroid_track: tuple[tuple[float, float], ...]
    multi_tracks: tuple[tuple[TrackedObject, ...], ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a plain JSON-compatible representation for future storage."""
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "best_contour_points": [list(point) for point in self.best_contour_points],
            "genuine_contour_points": [
                [list(point) for point in contour] for contour in self.genuine_contour_points
            ],
            "centroid_track": [list(point) for point in self.centroid_track],
            "multi_tracks": [
                [
                    {
                        "track_id": obj.track_id,
                        "bbox": list(obj.bbox),
                        "merged_ids": list(obj.merged_ids),
                    }
                    for obj in frame_tracks
                ]
                for frame_tracks in self.multi_tracks
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GeometryObservations:
        """Restore a payload produced by :meth:`to_payload`."""
        return cls(
            frame_width=int(payload["frame_width"]),
            frame_height=int(payload["frame_height"]),
            best_contour_points=tuple(
                (float(x), float(y)) for x, y in payload["best_contour_points"]
            ),
            genuine_contour_points=tuple(
                tuple((float(x), float(y)) for x, y in contour)
                for contour in payload["genuine_contour_points"]
            ),
            centroid_track=tuple((float(x), float(y)) for x, y in payload["centroid_track"]),
            multi_tracks=tuple(
                tuple(
                    TrackedObject(
                        track_id=int(obj["track_id"]),
                        bbox=tuple(int(value) for value in obj["bbox"]),
                        merged_ids=tuple(int(value) for value in obj["merged_ids"]),
                    )
                    for obj in frame_tracks
                )
                for frame_tracks in payload["multi_tracks"]
            ),
        )


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_identity(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def extraction_fingerprint(
    *,
    video_path: str | Path,
    motion_fingerprint: str,
    zone: CameraZone,
    reference_background: np.ndarray | None,
    daylight_hint: bool | None,
) -> str:
    """Hash every input that can change the standard feature-extraction path."""
    identity = {
        "video_sha256": _hash_file(video_path),
        "motion": motion_fingerprint,
        "zone": asdict(zone),
        "reference_background": _array_identity(reference_background),
        "daylight_hint": daylight_hint,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:32]


def get_cached_features(
    conn: Any,
    *,
    channel_id: str,
    message_id: int,
    fingerprint: str,
) -> tuple[bool, dict[str, float] | None]:
    """Return ``(hit, features)``; a cached no-motion result is a real hit."""
    payload = db.get_cached_track(
        conn,
        channel_id=channel_id,
        message_id=message_id,
        extractor_version=EXTRACTOR_VERSION,
        motion_fingerprint=fingerprint,
    )
    if payload is None:
        return False, None
    if payload == {_NO_MOTION_KEY: True}:
        return True, None
    return True, {str(key): float(value) for key, value in payload.items()}


def put_cached_features(
    conn: Any,
    *,
    channel_id: str,
    message_id: int,
    fingerprint: str,
    features: Mapping[str, float] | None,
) -> None:
    payload: dict[str, Any] = (
        {_NO_MOTION_KEY: True} if features is None else dict(features)
    )
    db.put_cached_track(
        conn,
        channel_id=channel_id,
        message_id=message_id,
        extractor_version=EXTRACTOR_VERSION,
        motion_fingerprint=fingerprint,
        features=payload,
    )
