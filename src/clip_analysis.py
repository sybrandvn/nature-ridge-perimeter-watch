"""Production clip-analysis path shared by backtests and the live watcher."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.classify import ClassificationResult, classify_detailed, is_blinding_foreground
from src.config import Camera, ThresholdsConfig
from src.features import daylight_hint, is_daylight
from src.motion import extraction_fingerprint, get_cached_features, put_cached_features
from src.reference_bg import era_of, load_reference_image, reference_for
from src.scoring import extract_clip_features

ExtractFn = Callable[..., dict[str, float] | None]


@dataclass(frozen=True)
class ClipAnalysis:
    classification: ClassificationResult
    features: dict[str, float] | None
    blinding_foreground: bool
    cache_hit: bool


def analyze_clip(
    *,
    conn: Any,
    channel_id: str,
    message_id: int,
    video_path: str | Path,
    timestamp: str,
    camera: Camera,
    thresholds: ThresholdsConfig,
    reference_entries: Any = None,
    reference_root: str | Path = "data/reference_bg",
    extract_fn: ExtractFn = extract_clip_features,
    use_cache: bool = True,
    cache_standard: bool | None = None,
) -> ClipAnalysis:
    """Extract and classify one clip with the exact production inputs."""
    thresholds.motion_thresholds()
    reference = None
    if reference_entries:
        entry = reference_for(
            reference_entries,
            camera.id,
            timestamp,
            era=era_of(camera, timestamp),
            daylight=is_daylight(timestamp),
        )
        if entry is not None:
            reference = load_reference_image(Path(reference_root), entry)

    hint = daylight_hint(timestamp)
    zone = camera.zone_at(timestamp)
    fingerprint = None
    cache_hit = False
    features: dict[str, float] | None = None
    standard = use_cache and (
        extract_fn is extract_clip_features if cache_standard is None else cache_standard
    )
    if standard:
        fingerprint = extraction_fingerprint(
            video_path=video_path,
            motion_fingerprint=thresholds.motion_fingerprint(),
            zone=zone,
            reference_background=reference,
            daylight_hint=hint,
        )
        cache_hit, features = get_cached_features(
            conn,
            channel_id=channel_id,
            message_id=message_id,
            fingerprint=fingerprint,
        )
    if not cache_hit:
        kwargs: dict[str, Any] = {}
        if reference is not None:
            kwargs["reference_background"] = reference
        if hint is not None:
            kwargs["daylight_hint"] = hint
        features = extract_fn(str(video_path), zone, **kwargs)
        if standard and fingerprint is not None:
            put_cached_features(
                conn,
                channel_id=channel_id,
                message_id=message_id,
                fingerprint=fingerprint,
                features=features,
            )
    if features is not None:
        features["is_daylight"] = float(is_daylight(timestamp))
    return ClipAnalysis(
        classification=classify_detailed(features),
        features=features,
        blinding_foreground=is_blinding_foreground(features),
        cache_hit=cache_hit,
    )
