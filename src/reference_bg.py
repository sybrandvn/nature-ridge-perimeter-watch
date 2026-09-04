"""Per-camera reference backgrounds: build, load and align.

A reference background is the median of many *different* clips' own median
backgrounds, from the same camera, geometry era, lighting and rough time of
year. It answers a question a single clip cannot: is what the tracker latched
onto permanent scenery, or a subject?

That distinction is why this exists at all. `detect_clip` already computes a
per-clip median background, but comparing an appearance-recovered frame
against it is circular -- the frame is only recovered *because* background
subtraction found nothing there, so it necessarily resembles that clip's own
median whether it holds a fence rail or a motionless guard. Measured over 60
frozen-recovered runs, real subjects scored 0.54-1.00 against their own clip's
median and known scenery scored 0.87-0.99: no separation at all. Against a
reference built from other clips the same runs separate, because a rail is in
every clip from that camera and a guard is in exactly one.

Bucketing, in order of importance:

- **Era.** A remount invalidates a reference exactly as it invalidates a fence
  polyline, so era boundaries are read straight from `cameras.yaml`'s dated
  `zones` history rather than duplicated here.
- **Lighting.** Night IR and daylight backgrounds share nothing; mixing them
  produces a reference that matches neither.
- **Time of year.** Vegetation moves. Buckets are per calendar quarter, which
  approximates the "nearest clips in time" rule the approach was validated
  with while staying cheap enough to precompute. A quarter with too few clips
  is merged into its neighbours rather than producing a thin, noisy reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import Camera
from src.errors import ConfigError

MIN_CLIPS = 5
MAX_CLIPS = 40
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ReferenceEntry:
    camera_id: str
    daylight: bool
    era: str | None  # ISO effective_from of the geometry era, None = from the start
    start: str
    end: str
    clip_count: int
    path: str

    @property
    def midpoint(self) -> datetime:
        return _parse(self.start) + (_parse(self.end) - _parse(self.start)) / 2


def _parse(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def era_of(camera: Camera | None, timestamp: str) -> str | None:
    """ISO `effective_from` of the geometry era a clip falls in, or None for the
    first/only era. Reuses cameras.yaml rather than a second source of truth."""
    if camera is None or not camera.zone_history:
        return None
    ts = _parse(timestamp)
    applicable = [
        effective_from
        for effective_from, _zone in camera.zone_history
        if effective_from is None or effective_from <= ts
    ]
    if not applicable:
        return None
    latest = applicable[-1]
    return None if latest is None else latest.isoformat().replace("+00:00", "Z")


def quarter_of(timestamp: str) -> tuple[int, int]:
    dt = _parse(timestamp)
    return dt.year, (dt.month - 1) // 3


def clip_median_background(video_path: str | Path) -> np.ndarray | None:
    """One clip's median background, preprocessed exactly as `detect_clip` does
    -- any mismatch here shows up as a similarity penalty later."""
    capture = cv2.VideoCapture(str(video_path))
    grays: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            grays.append(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0))
    finally:
        capture.release()
    if not grays:
        return None
    return np.median(np.stack(grays), axis=0).astype(np.uint8)


def combine(medians: list[np.ndarray]) -> np.ndarray | None:
    """Median across per-clip medians, keeping only the most common frame shape
    -- a camera swapped mid-era can leave odd resolutions in the same bucket."""
    if not medians:
        return None
    shapes = [m.shape for m in medians]
    common = max(set(shapes), key=shapes.count)
    usable = [m for m in medians if m.shape == common]
    if not usable:
        return None
    return np.median(np.stack(usable), axis=0).astype(np.uint8)


def bucket_clips(
    clips: list[tuple[str, str]], camera: Camera | None
) -> dict[tuple[str | None, bool, int, int], list[str]]:
    """Group (timestamp, path) pairs into (era, daylight, year, quarter) buckets.

    Quarters below `MIN_CLIPS` are folded into the nearest populated quarter in
    the same era and lighting, so a sparse camera still gets one usable
    reference instead of several unusable ones.
    """
    from src.features import is_daylight

    buckets: dict[tuple[str | None, bool, int, int], list[str]] = {}
    for timestamp, path in clips:
        year, quarter = quarter_of(timestamp)
        key = (era_of(camera, timestamp), is_daylight(timestamp), year, quarter)
        buckets.setdefault(key, []).append(path)

    sparse = [key for key, paths in buckets.items() if len(paths) < MIN_CLIPS]
    for key in sparse:
        era, daylight, year, quarter = key
        siblings = [
            other
            for other in buckets
            if other != key
            and other[0] == era
            and other[1] == daylight
            and len(buckets[other]) >= MIN_CLIPS
        ]
        if not siblings:
            continue
        nearest = min(siblings, key=lambda o: abs((o[2] * 4 + o[3]) - (year * 4 + quarter)))
        buckets[nearest].extend(buckets.pop(key))
    return buckets


def sample_evenly(paths: list[str], limit: int = MAX_CLIPS) -> list[str]:
    if len(paths) <= limit:
        return paths
    step = len(paths) / limit
    return [paths[int(i * step)] for i in range(limit)]


def load_manifest(root: str | Path) -> list[ReferenceEntry]:
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Reference manifest is not valid JSON: {path}: {exc}") from exc
    return [ReferenceEntry(**entry) for entry in raw]


def reference_for(
    entries: list[ReferenceEntry],
    camera_id: str,
    timestamp: str,
    *,
    era: str | None = None,
    daylight: bool | None = None,
) -> ReferenceEntry | None:
    """The reference covering a clip: same camera, era and lighting, then the
    bucket nearest in time.

    Returns None when nothing matches, which callers must treat as "no
    reference available" rather than an error. That happens legitimately: a
    camera with a handful of clips never gets one (cam11 has two in the whole
    corpus), and a freshly remounted camera has no material for its new era yet
    -- cam12's era starting 2026-03-02 has 2 clips. Falling back to the
    previous era's reference in that case would be worse than having none,
    since the whole scene has moved.
    """
    from src.features import is_daylight

    lit = is_daylight(timestamp) if daylight is None else daylight
    candidates = [
        e for e in entries if e.camera_id == camera_id and e.daylight == lit and e.era == era
    ]
    if not candidates:
        return None
    ts = _parse(timestamp)
    return min(candidates, key=lambda e: abs((e.midpoint - ts).total_seconds()))


def load_reference_image(root: str | Path, entry: ReferenceEntry) -> np.ndarray | None:
    image = cv2.imread(str(Path(root) / entry.path), cv2.IMREAD_GRAYSCALE)
    return image


def align(reference: np.ndarray, clip_background: np.ndarray) -> tuple[np.ndarray, float]:
    """Shift `reference` onto a clip's own background by phase correlation.

    Not optional. Cameras drift on their mounts between clips, and the
    comparison is per-pixel: on cam01/16167 alignment moved two scenery runs
    from 0.847/0.844 to 0.953/0.954, and cameras that shift more (a 36px offset
    measured on cam01a, 27px on cam14) score as false locks without it.
    """
    if reference.shape != clip_background.shape or reference.size == 0:
        return reference, 0.0
    (dx, dy), _response = cv2.phaseCorrelate(
        reference.astype(np.float32), clip_background.astype(np.float32)
    )
    shift = float((dx**2 + dy**2) ** 0.5)
    if shift < 0.5:
        return reference, shift
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        reference,
        matrix,
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return shifted, shift
