"""Environment and YAML configuration loading, with validation.

Motion-extraction settings and classification thresholds are kept in separate
top-level sections of thresholds.yaml so a threshold-only change never
invalidates cached motion features (see ThresholdsConfig.motion_fingerprint,
used later by src/motion.py's feature cache).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from src.errors import ConfigError

Point = tuple[float, float]

_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


# --------------------------------------------------------------------------
# App / environment config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    telegram_api_id: int | None
    telegram_api_hash: str | None
    telegram_session_path: Path
    source_channel: str | None
    telegram_bot_token: str | None
    alert_channel_id: str | None
    ntfy_base_url: str | None
    ntfy_topic: str | None
    ntfy_priority: str
    ntfy_token: str | None
    db_path: Path
    operating_window_start: str
    operating_window_end: str
    bot_trustee_ids: tuple[int, ...]
    bot_security_ids: tuple[int, ...]


def load_app_config(env_path: str | Path = ".env", *, require_telegram: bool = True) -> AppConfig:
    """Load AppConfig from a .env file overlaid with real process env vars."""
    file_values = dotenv_values(env_path)
    merged: dict[str, str] = {k: v for k, v in file_values.items() if v is not None}
    import os

    merged.update({k: v for k, v in os.environ.items() if k in _KNOWN_KEYS})
    return load_app_config_from_mapping(merged, require_telegram=require_telegram)


_KNOWN_KEYS = {
    "TELEGRAM_API_ID",
    "TELEGRAM_API_HASH",
    "TELEGRAM_SESSION_PATH",
    "SOURCE_CHANNEL",
    "TELEGRAM_BOT_TOKEN",
    "ALERT_CHANNEL_ID",
    "NTFY_BASE_URL",
    "NTFY_TOPIC",
    "NTFY_PRIORITY",
    "NTFY_TOKEN",
    "DB_PATH",
    "OPERATING_WINDOW_START",
    "OPERATING_WINDOW_END",
    "BOT_TRUSTEE_IDS",
    "BOT_SECURITY_IDS",
}


def load_app_config_from_mapping(
    env: Mapping[str, str], *, require_telegram: bool = True
) -> AppConfig:
    """Build AppConfig from an explicit mapping (used directly by tests)."""

    def _get(key: str) -> str | None:
        value = env.get(key, "")
        return value.strip() or None

    api_id_raw = _get("TELEGRAM_API_ID")
    api_hash = _get("TELEGRAM_API_HASH")
    source_channel = _get("SOURCE_CHANNEL")

    if require_telegram:
        missing = [
            name
            for name, value in (
                ("TELEGRAM_API_ID", api_id_raw),
                ("TELEGRAM_API_HASH", api_hash),
                ("SOURCE_CHANNEL", source_channel),
            )
            if value is None
        ]
        if missing:
            raise ConfigError(f"Missing required environment variable(s): {', '.join(missing)}")

    api_id: int | None = None
    if api_id_raw is not None:
        try:
            api_id = int(api_id_raw)
        except ValueError as exc:
            raise ConfigError(f"TELEGRAM_API_ID must be an integer, got {api_id_raw!r}") from exc

    start = _get("OPERATING_WINDOW_START") or "18:00"
    end = _get("OPERATING_WINDOW_END") or "06:00"
    for label, value in (("OPERATING_WINDOW_START", start), ("OPERATING_WINDOW_END", end)):
        if not _HHMM_RE.match(value):
            raise ConfigError(f"{label} must be HH:MM (24h), got {value!r}")

    return AppConfig(
        telegram_api_id=api_id,
        telegram_api_hash=api_hash,
        telegram_session_path=Path(_get("TELEGRAM_SESSION_PATH") or "data/session.session"),
        source_channel=source_channel,
        telegram_bot_token=_get("TELEGRAM_BOT_TOKEN"),
        alert_channel_id=_get("ALERT_CHANNEL_ID"),
        ntfy_base_url=_get("NTFY_BASE_URL"),
        ntfy_topic=_get("NTFY_TOPIC"),
        ntfy_priority=_get("NTFY_PRIORITY") or "urgent",
        ntfy_token=_get("NTFY_TOKEN"),
        db_path=Path(_get("DB_PATH") or "data/perimeter_watch.db"),
        operating_window_start=start,
        operating_window_end=end,
        bot_trustee_ids=_parse_id_list("BOT_TRUSTEE_IDS", _get("BOT_TRUSTEE_IDS")),
        bot_security_ids=_parse_id_list("BOT_SECURITY_IDS", _get("BOT_SECURITY_IDS")),
    )


def resolve_channel_ref(value: str) -> int | str:
    """Coerce a numeric channel/chat id string to int for Telethon.

    Telethon's string-based entity lookup treats numeric strings as phone
    numbers (stripping a leading "-"), so negative chat/channel ids must be
    passed as int instead; usernames like "@name" pass through unchanged.
    """
    text = value.strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def _parse_id_list(field_name: str, raw: str | None) -> tuple[int, ...]:
    if not raw:
        return ()
    ids = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.append(int(chunk))
        except ValueError as exc:
            raise ConfigError(f"{field_name} entries must be integers, got {chunk!r}") from exc
    return tuple(ids)


# --------------------------------------------------------------------------
# Camera / zone config (config/cameras.yaml)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraZone:
    fence: tuple[Point, ...] | None
    outside: str | None  # "left" | "right"
    depth_cutoff: float
    ignore: tuple[tuple[Point, ...], ...]
    # Second, optional polyline traced along the fence's BASE (where it meets
    # the ground), not a redefinition of `fence` (still always the top rail).
    # When present, inside/outside classification prefers this line over
    # `fence` (see src/zones.py); paired with `fence_height_m` it also gives a
    # per-row pixels-to-metres ruler. None for every camera without one traced
    # yet -- see docs/plan.md "Fence base line + metric features".
    fence_bottom: tuple[Point, ...] | None = None
    fence_height_m: float = 2.0
    # One or more traced pickets (each a top point, base point pair), tracing
    # their real on-screen tilt at whatever depth each was still clearly
    # visible. The naive ruler pairs `fence`/`fence_bottom` at the SAME row,
    # which silently assumes a picket renders perfectly vertical in frame --
    # false whenever the camera looks down at an angle. When set,
    # `src.zones.fence_separation_at_y` projects along the picket angle
    # interpolated for that row instead of straight up, before falling back
    # to the naive same-row method if no camera has any traced yet.
    # Deliberately plural: measured 2026-09-05 that tilt does NOT taper
    # linearly with depth (drops sharply near the camera, flattens out
    # further away) -- a single point + linear taper gets this wrong in both
    # directions, so multiple real samples are interpolated between instead.
    fence_pickets: tuple[tuple[Point, Point], ...] = ()
    # Opt in to `src.ground_calibration`: real distances/heights from the
    # fence's two lines plus the pickets' vertical vanishing point. Off by
    # default -- it needs `fence`, `fence_bottom` and >= 2 `fence_pickets`
    # traced well, and has only been validated on cam06 so far.
    metric_calibration: bool = False
    # Refuse distances past this many metres. None uses
    # `ground_calibration.DEFAULT_MAX_RANGE_M`. The effective cap is also
    # limited by how much distance one pixel row is worth near the horizon.
    metric_max_range_m: float | None = None
    # Override the shared `ground_calibration.FLEET_FOCAL_PX`. Only consulted
    # when this camera has exactly one traced picket -- with two or more, the
    # focal length comes from the camera's own geometry. Set this only for a
    # camera that is genuinely a different model to the rest of the fleet.
    metric_focal_px: float | None = None


@dataclass(frozen=True)
class Camera:
    id: str
    aliases: tuple[str, ...]
    order: int | None
    zone: CameraZone
    threshold_overrides: Mapping[str, Any]
    # Dated geometry history for cameras that were re-aimed/re-mounted, sorted
    # ascending by effective_from (None sorts first: "applies from the start").
    # Empty for every camera with only ever one geometry (the common case) --
    # `zone` is then always returned regardless of timestamp. See docs/plan.md
    # "Geometry model".
    zone_history: tuple[tuple[datetime | None, CameraZone], ...] = field(default=())

    def zone_at(self, timestamp: datetime | str | None) -> CameraZone:
        """The geometry in effect at `timestamp` (a clip's timestamp), falling
        back to `zone` (the most recent geometry) if there's no dated history
        or the timestamp can't be resolved.
        """
        if not self.zone_history or timestamp is None:
            return self.zone
        if isinstance(timestamp, str):
            try:
                ts = _parse_iso_timestamp(timestamp)
            except ValueError:
                return self.zone
        else:
            ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
        applicable = [
            zone
            for effective_from, zone in self.zone_history
            if effective_from is None or effective_from <= ts
        ]
        return applicable[-1] if applicable else self.zone_history[0][1]


@dataclass(frozen=True)
class CamerasConfig:
    cameras: tuple[Camera, ...]
    unknown_camera_id: str

    def by_id(self, camera_id: str) -> Camera | None:
        return next((c for c in self.cameras if c.id == camera_id), None)

    def resolve_alias(self, text: str) -> Camera | None:
        needle = text.strip().lower()
        for camera in self.cameras:
            if camera.id.lower() == needle:
                return camera
            if any(alias.lower() == needle for alias in camera.aliases):
                return camera
        return None

    def ordered(self) -> list[Camera]:
        return sorted((c for c in self.cameras if c.order is not None), key=lambda c: c.order)


def load_cameras_config(path: str | Path) -> CamerasConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Cameras config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Cameras config is not valid YAML: {path}: {exc}") from exc

    unknown_camera_id = raw.get("unknown_camera_id", "unknown")
    entries = raw.get("cameras") or []
    if not isinstance(entries, list):
        raise ConfigError("cameras.yaml: 'cameras' must be a list")

    cameras: list[Camera] = []
    seen_ids: set[str] = set()
    seen_aliases: dict[str, str] = {}
    seen_orders: dict[int, str] = {}

    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry:
            raise ConfigError(f"cameras.yaml: each camera entry needs an 'id': {entry!r}")
        camera_id = str(entry["id"])
        if camera_id in seen_ids:
            raise ConfigError(f"cameras.yaml: duplicate camera id {camera_id!r}")
        seen_ids.add(camera_id)

        aliases = tuple(str(a) for a in entry.get("aliases") or [])
        for alias in (camera_id, *aliases):
            key = alias.lower()
            if key in seen_aliases and seen_aliases[key] != camera_id:
                raise ConfigError(
                    f"cameras.yaml: alias {alias!r} claimed by both "
                    f"{seen_aliases[key]!r} and {camera_id!r}"
                )
            seen_aliases[key] = camera_id

        order = entry.get("order")
        if order is not None:
            order = int(order)
            if order < 0:
                raise ConfigError(f"cameras.yaml: {camera_id!r} order must be >= 0")
            if order in seen_orders:
                raise ConfigError(
                    f"cameras.yaml: order {order} used by both "
                    f"{seen_orders[order]!r} and {camera_id!r}"
                )
            seen_orders[order] = camera_id

        zone, zone_history = _parse_camera_zones(camera_id, entry)

        overrides = entry.get("threshold_overrides") or {}
        if not isinstance(overrides, dict):
            raise ConfigError(f"cameras.yaml: {camera_id!r} threshold_overrides must be a mapping")
        _validate_numeric_tree(overrides, context=f"cameras.yaml:{camera_id}.threshold_overrides")

        cameras.append(
            Camera(
                id=camera_id,
                aliases=aliases,
                order=order,
                zone=zone,
                threshold_overrides=overrides,
                zone_history=zone_history,
            )
        )

    return CamerasConfig(cameras=tuple(cameras), unknown_camera_id=str(unknown_camera_id))


_ZONE_FIELDS = (
    "fence",
    "outside",
    "depth_cutoff",
    "ignore",
    "fence_bottom",
    "fence_height_m",
    "fence_pickets",
    "metric_calibration",
    "metric_max_range_m",
    "metric_focal_px",
)


def _parse_camera_zones(
    camera_id: str, entry: dict[str, Any]
) -> tuple[CameraZone, tuple[tuple[datetime | None, CameraZone], ...]]:
    """Parse a camera's geometry: either a single flat zone (the common case),
    or -- for a camera that was re-aimed/re-mounted -- a `zones` list of dated
    entries. `zone` is always the current (most recent) geometry; `zone_history`
    is empty unless `zones` was used. See docs/plan.md "Geometry model".
    """
    raw_zones = entry.get("zones")
    if raw_zones is None:
        return _parse_zone(camera_id, entry), ()

    if any(field_name in entry for field_name in _ZONE_FIELDS):
        raise ConfigError(
            f"cameras.yaml: {camera_id!r} must not mix top-level fence/outside/"
            f"depth_cutoff/ignore with a 'zones' list"
        )
    if not isinstance(raw_zones, list) or not raw_zones:
        raise ConfigError(f"cameras.yaml: {camera_id!r} zones must be a non-empty list")

    parsed: list[tuple[datetime | None, CameraZone]] = []
    seen_effective_from: set[datetime | None] = set()
    for item in raw_zones:
        if not isinstance(item, dict):
            raise ConfigError(f"cameras.yaml: {camera_id!r} each zones entry must be a mapping")
        effective_from = _parse_effective_from(camera_id, item.get("effective_from"))
        if effective_from in seen_effective_from:
            raise ConfigError(
                f"cameras.yaml: {camera_id!r} duplicate zones effective_from {effective_from}"
            )
        seen_effective_from.add(effective_from)
        parsed.append((effective_from, _parse_zone(camera_id, item)))

    parsed.sort(key=lambda pair: pair[0] or datetime.min.replace(tzinfo=UTC))
    return parsed[-1][1], tuple(parsed)


def _parse_iso_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _parse_effective_from(camera_id: str, value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return _parse_iso_timestamp(str(value))
    except ValueError as exc:
        raise ConfigError(
            f"cameras.yaml: {camera_id!r} zones effective_from invalid: {value!r}"
        ) from exc


def _parse_zone(camera_id: str, entry: dict[str, Any]) -> CameraZone:
    raw_fence = entry.get("fence")
    fence: tuple[Point, ...] | None = None
    if raw_fence is not None:
        if not isinstance(raw_fence, list) or len(raw_fence) < 2:
            raise ConfigError(f"cameras.yaml: {camera_id!r} fence needs >= 2 points")
        fence = tuple(_parse_point(camera_id, "fence", p) for p in raw_fence)

    outside = entry.get("outside")
    if outside is not None:
        outside = str(outside).lower()
        if outside not in ("left", "right"):
            raise ConfigError(
                f"cameras.yaml: {camera_id!r} outside must be 'left' or 'right', got {outside!r}"
            )
    if fence is not None and outside is None:
        raise ConfigError(f"cameras.yaml: {camera_id!r} has a fence but no outside")

    depth_cutoff = entry.get("depth_cutoff", 0.0)
    depth_cutoff = float(depth_cutoff)
    if not 0.0 <= depth_cutoff <= 1.0:
        raise ConfigError(f"cameras.yaml: {camera_id!r} depth_cutoff must be in [0, 1]")

    raw_ignore = entry.get("ignore") or []
    if not isinstance(raw_ignore, list):
        raise ConfigError(f"cameras.yaml: {camera_id!r} ignore must be a list of polygons")
    ignore = tuple(
        tuple(_parse_point(camera_id, "ignore", p) for p in polygon)
        for polygon in raw_ignore
    )
    for polygon in ignore:
        if len(polygon) < 3:
            raise ConfigError(f"cameras.yaml: {camera_id!r} ignore polygons need >= 3 points")

    raw_fence_bottom = entry.get("fence_bottom")
    fence_bottom: tuple[Point, ...] | None = None
    if raw_fence_bottom is not None:
        if not isinstance(raw_fence_bottom, list) or len(raw_fence_bottom) < 2:
            raise ConfigError(f"cameras.yaml: {camera_id!r} fence_bottom needs >= 2 points")
        fence_bottom = tuple(_parse_point(camera_id, "fence_bottom", p) for p in raw_fence_bottom)

    fence_height_m = float(entry.get("fence_height_m", 2.0))
    if fence_height_m <= 0.0:
        raise ConfigError(f"cameras.yaml: {camera_id!r} fence_height_m must be > 0")

    raw_fence_pickets = entry.get("fence_pickets") or []
    if not isinstance(raw_fence_pickets, list):
        raise ConfigError(f"cameras.yaml: {camera_id!r} fence_pickets must be a list")
    fence_pickets: list[tuple[Point, Point]] = []
    for picket in raw_fence_pickets:
        if not isinstance(picket, list) or len(picket) != 2:
            raise ConfigError(
                f"cameras.yaml: {camera_id!r} each fence_pickets entry needs exactly 2 points"
            )
        top_pt, base_pt = (_parse_point(camera_id, "fence_pickets", p) for p in picket)
        fence_pickets.append((top_pt, base_pt))

    metric_calibration = bool(entry.get("metric_calibration", False))

    raw_max_range = entry.get("metric_max_range_m")
    metric_max_range_m: float | None = None
    if raw_max_range is not None:
        metric_max_range_m = float(raw_max_range)
        if metric_max_range_m <= 0.0:
            raise ConfigError(
                f"cameras.yaml: {camera_id!r} metric_max_range_m must be > 0"
            )

    raw_focal = entry.get("metric_focal_px")
    metric_focal_px: float | None = None
    if raw_focal is not None:
        metric_focal_px = float(raw_focal)
        if metric_focal_px <= 0.0:
            raise ConfigError(f"cameras.yaml: {camera_id!r} metric_focal_px must be > 0")

    return CameraZone(
        fence=fence,
        outside=outside,
        depth_cutoff=depth_cutoff,
        ignore=ignore,
        fence_bottom=fence_bottom,
        fence_height_m=fence_height_m,
        fence_pickets=tuple(fence_pickets),
        metric_calibration=metric_calibration,
        metric_max_range_m=metric_max_range_m,
        metric_focal_px=metric_focal_px,
    )


def _parse_point(camera_id: str, field_name: str, point: Any) -> Point:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ConfigError(
            f"cameras.yaml: {camera_id!r} {field_name} point must be [x, y]: {point!r}"
        )
    x, y = float(point[0]), float(point[1])
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ConfigError(
            f"cameras.yaml: {camera_id!r} {field_name} point out of [0,1] range: {point!r}"
        )
    return (x, y)


# --------------------------------------------------------------------------
# Threshold config (config/thresholds.yaml)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationThresholds:
    """Every tuned number src.classify compares against.

    Flat on purpose: thresholds.yaml nests for readability, but a flat record
    keeps classify()'s call sites short. Field names carry `_min`/`_max` to say
    which side of the bound is acceptable -- they deliberately do NOT encode the
    comparison operator, since some sites use `>` and some `>=`; src/classify.py
    is where each operator lives, and moving a value here must never change one.
    """

    green_light_ratio_min: float
    green_light_flicker_min: float
    warmup_flashlight_ratio_min: float
    blob_count_peak_min: float
    blob_count_median_min: float
    implausible_height_fraction_min: float
    motion_pixel_fraction_median_min: float
    outside_pixel_fraction_min: float
    median_fence_distance_min: float
    median_fence_distance_max: float
    color_fraction_min: float
    row_normalised_area_max: float
    jitter_min: float
    solidity_max: float
    blob_white_fraction_min: float
    long_flare_frames_min: float
    neighbour_subject_height_min: float
    neighbour_subject_height_max: float


# field name -> (yaml section, yaml key). The nesting exists for the humans
# reading thresholds.yaml; this table is the only place the two shapes meet.
_CLASSIFICATION_FIELDS: Mapping[str, tuple[str, str]] = {
    "green_light_ratio_min": ("guard", "green_light_ratio_min"),
    "green_light_flicker_min": ("guard", "green_light_flicker_min"),
    "warmup_flashlight_ratio_min": ("guard", "warmup_flashlight_ratio_min"),
    "blob_count_peak_min": ("environment", "blob_count_peak_min"),
    "blob_count_median_min": ("environment", "blob_count_median_min"),
    "implausible_height_fraction_min": ("environment", "implausible_height_fraction_min"),
    "motion_pixel_fraction_median_min": ("environment", "motion_pixel_fraction_median_min"),
    "outside_pixel_fraction_min": ("outside", "pixel_fraction_min"),
    "median_fence_distance_min": ("outside", "median_fence_distance_min"),
    "median_fence_distance_max": ("outside", "median_fence_distance_max"),
    "color_fraction_min": ("outside", "color_fraction_min"),
    "row_normalised_area_max": ("animal", "row_normalised_area_max"),
    "jitter_min": ("insect", "jitter_min"),
    "solidity_max": ("insect", "solidity_max"),
    "blob_white_fraction_min": ("blinding", "blob_white_fraction_min"),
    "long_flare_frames_min": ("blinding", "long_flare_frames_min"),
    "neighbour_subject_height_min": ("neighbour", "subject_height_min"),
    "neighbour_subject_height_max": ("neighbour", "subject_height_max"),
}


def _classification_thresholds(classification: Mapping[str, Any]) -> ClassificationThresholds:
    """Build the typed record, raising on any missing or unrecognised key.

    Strict in both directions on purpose. A missing key must not silently
    default -- a defaulted threshold is one nobody measured, and it would change
    what the classifier does without anyone noticing. An unrecognised key must
    not be silently ignored either, or a stale entry left over from an older
    file shape (or a typo'd rename) reads as "configured" while the real value
    quietly falls back to something else.
    """
    values: dict[str, Any] = {}
    for field_name, (section, key) in _CLASSIFICATION_FIELDS.items():
        subsection = classification.get(section)
        if not isinstance(subsection, Mapping):
            raise ConfigError(
                f"thresholds.yaml:classification.{section} must be a mapping "
                f"(needed for {field_name})"
            )
        if key not in subsection:
            raise ConfigError(f"thresholds.yaml: missing classification.{section}.{key}")
        values[field_name] = float(subsection[key])

    expected = {(section, key) for section, key in _CLASSIFICATION_FIELDS.values()}
    for section, subsection in classification.items():
        if not isinstance(subsection, Mapping):
            raise ConfigError(f"thresholds.yaml:classification.{section} must be a mapping")
        for key in subsection:
            if (section, key) not in expected:
                raise ConfigError(f"thresholds.yaml: unrecognised classification.{section}.{key}")

    return ClassificationThresholds(**values)


@dataclass(frozen=True)
class ThresholdsConfig:
    motion: Mapping[str, Any]
    classification: Mapping[str, Any]

    def motion_fingerprint(self) -> str:
        """Stable hash of motion-extraction settings only.

        Used to decide whether cached blob tracks can be reused: classifier-only
        threshold edits must never change this value.
        """
        return _stable_hash(self.motion)

    def classification_fingerprint(self) -> str:
        """Stable hash of classification thresholds only.

        Recorded on every backtest run (see src/backtester.py) so two runs that
        classified identically hash identically -- which is why this excludes
        `motion`, whose settings do not affect a classification result.
        """
        return _stable_hash(self.classification)

    def classification_thresholds(self) -> ClassificationThresholds:
        return _classification_thresholds(self.classification)


def load_thresholds_config(path: str | Path) -> ThresholdsConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Thresholds config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Thresholds config is not valid YAML: {path}: {exc}") from exc

    motion = raw.get("motion")
    classification = raw.get("classification")
    if not isinstance(motion, dict):
        raise ConfigError("thresholds.yaml: top-level 'motion' section must be a mapping")
    if not isinstance(classification, dict):
        raise ConfigError("thresholds.yaml: top-level 'classification' section must be a mapping")

    _validate_numeric_tree(motion, context="thresholds.yaml:motion")
    _validate_numeric_tree(classification, context="thresholds.yaml:classification")

    return ThresholdsConfig(motion=motion, classification=classification)


def _validate_numeric_tree(data: Mapping[str, Any], *, context: str) -> None:
    """Recursively require every leaf value to be int/float/bool, so a typo
    (e.g. a string where a threshold number belongs) fails fast instead of
    breaking arithmetic deep inside the classifier."""
    for key, value in data.items():
        path = f"{context}.{key}"
        if isinstance(value, dict):
            _validate_numeric_tree(value, context=path)
        elif not isinstance(value, (int, float, bool)):
            raise ConfigError(f"{path}: expected a number, got {value!r} ({type(value).__name__})")


def _stable_hash(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
