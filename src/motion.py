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

import numpy as np

from src import db
from src.config import CameraZone

EXTRACTOR_VERSION = "motion-features-v1"
_NO_MOTION_KEY = "__perimeter_watch_no_motion__"


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
