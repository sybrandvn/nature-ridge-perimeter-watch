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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src import db
from src.config import CameraZone

EXTRACTOR_VERSION = "motion-features-v1"
_NO_MOTION_KEY = "__perimeter_watch_no_motion__"


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
