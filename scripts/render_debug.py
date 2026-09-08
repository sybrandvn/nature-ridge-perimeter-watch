"""Render a clip as an annotated video showing what the detector actually sees.

The features CSV says a clip scored `aspect_ratio=0.81, jitter=6.8`. It does not
say whether the detector was looking at a porcupine or at a branch. This renders
the detector's own view frame by frame so that can be judged by eye and the
thresholds tuned against something visible.

Drawn per frame:
  - fence polyline, with the OUTSIDE and INSIDE half-frames tinted
  - ignore regions (hatched) and the depth cutoff line
  - every blob passing the area gates in dim cyan, contours that failed a gate
    (too small or too big) in dim grey, and the tracked blob in bright magenta
  - a fading trail: past bounding boxes plus the joined centroid path
  - an IR FLARE banner on frames where the global gain stepped
  - the pre-cutoff warmup frames themselves, first, with an amber banner --
    these are excluded from the background model and never scored, but are
    now visible so flare-cutoff accuracy can be judged by eye
  - a HUD of live per-frame values and the clip's final feature vector

Detection comes from `scripts.spike.detect_clip`, the same function that feeds
`extract_clip_features`, so what is drawn is what scored the clip. The tuning
flags default to the spike's own values and the HUD marks them when overridden.

When selecting clips by `--label`/`--message-id`/`--message-ids-file`, clips
that share an embedded alert timestamp (the same physical camera trigger,
via `scripts.label._event_key`) are collapsed to just the longest one --
otherwise a short startup-only clip and its longer sibling both carrying the
same label would both render, and the short one isn't representative.

Output is a real H.264 .mp4 (via `src.video_encode.Mp4Writer`, not
`cv2.VideoWriter`) so it plays in VS Code's built-in preview, Telegram, and
ntfy-viewing clients alike -- this OpenCV build's own FFmpeg has no libx264.

Run:
    uv run python scripts/render_debug.py --message-id 21520 --out out.mp4
    uv run python scripts/render_debug.py --clip data/history/cam06/21520.mp4 \
        --camera cam06 --out out.mp4
    uv run python scripts/render_debug.py --label incident --label animal \
        --out data/reports/debug
    uv run python scripts/render_debug.py \
        --message-ids-file data/reports/candidates.message_ids --out data/reports/debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts.label import (
    _EVENT_TS_RE,  # noqa: E402
    _event_key,  # noqa: E402
)
from scripts.spike import (  # noqa: E402
    ClipDetection,
    TrackedObject,
    detect_clip,
    extract_clip_features,
)
from src import db  # noqa: E402
from src.config import CameraZone, load_app_config, load_cameras_config  # noqa: E402
from src.features import (  # noqa: E402
    FLASHLIGHT_SUBJECT_THRESHOLD,
    daylight_hint,
    detect_stationary_light_mask,
    flashlight_bbox_overlap,
    green_light_mask,
    ignore_region_mask,
    is_daylight,
    is_twilight,
    sane_fps,
)
from src.ground_calibration import GroundCalibration, calibrate  # noqa: E402
from src.reference_bg import (  # noqa: E402
    era_of,
    load_manifest,
    load_reference_image,
    reference_for,
)
from src.video_encode import Mp4Writer  # noqa: E402
from src.zones import (  # noqa: E402
    _fence_x_at_y,
    effective_fence,
    estimated_height_m,
    side_name,
    subject_base_y,
)

DEFAULTS = {
    "threshold": 18,
    "min_area": 0.0005,
    "max_area": 0.25,
    "flare_tolerance": 3.0,
    "max_flare_fraction": 0.4,
}

COLOR_TRACKED = (255, 0, 255)
COLOR_RECOVERED = (0, 165, 255)
COLOR_REVERSE = (255, 255, 0)
COLOR_BLOB = (200, 200, 0)
COLOR_DISCARDED = (90, 90, 90)
COLOR_FENCE = (0, 255, 255)
COLOR_FENCE_BOTTOM = (0, 140, 255)
COLOR_METRIC = (0, 255, 0)

# Ground-distance marks drawn along the fence base line when a camera opts in
# to metric calibration. Anything past the camera's own max_range_m is simply
# not drawn, so the ruler shows exactly how far the camera can actually measure.
METRIC_RULER_TICKS_M = (2, 5, 10, 15, 20, 25, 30, 40, 50)
COLOR_OUTSIDE = (0, 0, 255)
COLOR_INSIDE = (0, 255, 0)
COLOR_IGNORE = (110, 110, 110)
COLOR_FLARE = (0, 90, 255)
COLOR_WARMUP = (0, 200, 255)
COLOR_LIGHT = (0, 255, 140)
COLOR_STATIONARY_LIGHT = (0, 140, 0)
HUD_BG = (24, 24, 24)
TRAIL_LENGTH = 12


def _to_px(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x, y = point
    return int(round(x * width)), int(round(y * height))


def _text(img, s, org, *, color=(235, 235, 235), scale=0.4, thickness=1) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    dash_length: int = 6,
) -> None:
    """Dashed-outline rectangle -- used for recovered/reverse-filled boxes so
    an inferred (template-dragged) detection reads as visually less certain
    than a genuine bg-diff hit, not just a different colour. A solid box of
    equal weight for every provenance type looks equally confident even when
    the underlying evidence is a fixed-size template match on a stale frame,
    not a fresh measurement."""
    x0, y0 = pt1
    x1, y1 = pt2
    for xa, ya, xb, yb in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
        length = max(abs(xb - xa), abs(yb - ya))
        steps = max(1, length // (dash_length * 2))
        for i in range(int(steps) + 1):
            t0 = (i * 2 * dash_length) / length if length else 0
            t1 = min(1.0, t0 + dash_length / length) if length else 1.0
            px0 = int(round(xa + (xb - xa) * t0))
            py0 = int(round(ya + (yb - ya) * t0))
            px1 = int(round(xa + (xb - xa) * t1))
            py1 = int(round(ya + (yb - ya) * t1))
            cv2.line(img, (px0, py0), (px1, py1), color, thickness)


def _draw_metric_ruler(
    ink: np.ndarray, calib: GroundCalibration | None, scale: int
) -> None:
    """Mark real ground distances along the fence base line.

    Each tick also gets the fence's own height drawn at that range, following
    the true vertical direction, which makes the perspective foreshortening
    visible: if a tick's fence bar looks wrong next to the real fence in the
    footage, the calibration is wrong.
    """
    if calib is None:
        return
    for metres in METRIC_RULER_TICKS_M:
        row = calib.row_at_distance(float(metres))
        if row is None:
            continue
        base = calib.base_point_at_row(row)
        if base is None:
            continue
        base_px = (int(round(base[0] * scale)), int(round(base[1] * scale)))
        top = calib.fence_top_above(base)
        if top is not None:
            cv2.line(
                ink,
                base_px,
                (int(round(top[0] * scale)), int(round(top[1] * scale))),
                COLOR_METRIC,
                1,
            )
        cv2.circle(ink, base_px, 3, COLOR_METRIC, -1)
        _text(ink, f"{metres}m", (base_px[0] + 6, base_px[1] + 4), color=COLOR_METRIC, scale=0.35)


def _zone_layers(
    width: int,
    height: int,
    zone: CameraZone,
    calib: GroundCalibration | None = None,
    scale: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Static zone artwork as (tint, ink) layers, built once per clip.

    `tint` is blended in flat; `ink` is copied where non-black, so the fence
    line and labels stay legible instead of washing out with the tint.
    """
    tint = np.zeros((height, width, 3), dtype=np.uint8)
    ink = np.zeros((height, width, 3), dtype=np.uint8)

    classification_fence = effective_fence(zone)
    if classification_fence is not None:
        # Sample which side each pixel column falls on, rather than assuming the
        # fence runs top-to-bottom.
        ys, xs = np.mgrid[0:height, 0:width]
        norm_x = xs / width
        norm_y = ys / height
        fence_x = np.empty((height, width), dtype=np.float64)
        for row in range(height):
            fence_x[row, :] = _fence_x_at_y(norm_y[row, 0], classification_fence)
        left = norm_x < fence_x
        outside_mask = left if zone.outside == "left" else ~left
        tint[outside_mask] = COLOR_OUTSIDE
        tint[~outside_mask] = COLOR_INSIDE

        # Top-rail line always drawn when present -- it's every camera's
        # original geometry, kept visible even on cameras whose base line
        # (fence_bottom) is what actually drives classification now.
        if zone.fence is not None:
            fence_px = [_to_px(p, width, height) for p in zone.fence]
            for a, b in zip(fence_px, fence_px[1:], strict=False):
                cv2.line(ink, a, b, COLOR_FENCE, 2)
            for p in fence_px:
                cv2.circle(ink, p, 3, COLOR_FENCE, -1)

        # Base/ground line, only traced on some cameras (cam06 as of
        # 2026-09-05) -- drawn in a distinct colour so both lines are visible
        # at once rather than overlapping the top rail's colour.
        if zone.fence_bottom is not None:
            bottom_px = [_to_px(p, width, height) for p in zone.fence_bottom]
            for a, b in zip(bottom_px, bottom_px[1:], strict=False):
                cv2.line(ink, a, b, COLOR_FENCE_BOTTOM, 2)
            for p in bottom_px:
                cv2.circle(ink, p, 3, COLOR_FENCE_BOTTOM, -1)

        _draw_metric_ruler(ink, calib, scale)

        (ax, ay), (bx, by) = classification_fence[0], classification_fence[-1]
        dx, dy = bx - ax, by - ay
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / norm, dx / norm
        mid = ((ax + bx) / 2, (ay + by) / 2)
        for sign in (1, -1):
            probe = (mid[0] + sign * nx * 0.14, mid[1] + sign * ny * 0.14)
            outside = side_name(probe, classification_fence) == zone.outside
            px, py = _to_px(probe, width, height)
            px = max(4, min(width - 74, px))
            py = max(14, min(height - 4, py))
            _text(
                ink,
                "OUTSIDE" if outside else "INSIDE",
                (px, py),
                color=COLOR_OUTSIDE if outside else COLOR_INSIDE,
                scale=0.55,
                thickness=2,
            )

    # zone.ignore regions are deliberately NOT drawn here as a static hatch --
    # every configured ignore region so far exists because of a stationary
    # light (see config/cameras.yaml comments), and a permanent box baked into
    # this per-clip layer drew on every frame regardless of whether the light
    # was actually lit, including in broad daylight where it never is. The
    # region is still masked out of detection/scoring exactly as before; it's
    # only VISUALLY represented when something real is happening there --
    # `_draw_light_mask`'s STATIONARY LIGHT marker (lit) and the
    # `suppressed_light_box` "IGNORED" marker (real motion suppressed there),
    # both already gated per-frame, not a permanent overlay.

    if zone.depth_cutoff > 0:
        y = int(round(zone.depth_cutoff * height))
        cv2.line(ink, (0, y), (width, y), COLOR_IGNORE, 1)
        _text(ink, f"depth cutoff {zone.depth_cutoff:.2f}", (4, max(11, y - 4)), color=COLOR_IGNORE)

    return tint, ink


def _apply_zone(canvas: np.ndarray, tint: np.ndarray, ink: np.ndarray) -> None:
    # Deliberately faint: the footage is near-black IR and a heavier tint hides
    # the subject this tool exists to show.
    cv2.addWeighted(tint, 0.06, canvas, 0.94, 0, canvas)
    drawn = ink.any(axis=2)
    canvas[drawn] = ink[drawn]


def _light_masks(
    frame: np.ndarray, ignore_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Split the raw green-light hue mask into (stationary, moving) pixels.

    `ignore_mask` (same shape as the frame, from `zone.ignore` -- a known
    fixed light left permanently in view, e.g. cam10) marks the stationary
    half; everything else that reads as the flashlight hue is the guard's
    actual moving beam. This never changes what `detect_clip`/
    `extract_clip_features` track or score -- it only labels the same pixels
    for the render.
    """
    mask = green_light_mask(frame)
    if ignore_mask is None:
        return np.zeros_like(mask), mask
    return mask & ignore_mask, mask & ~ignore_mask


def _draw_light_mask(
    canvas: np.ndarray,
    frame: np.ndarray,
    *,
    daylight_gated: bool,
    ignore_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Outline pixels the detector would call flashlight, split into a known
    STATIONARY light (dim green, e.g. a fixed porch/security light already
    fenced off by `zone.ignore`) and the guard's actual MOVING flashlight
    (bright green, same trace as before) -- so a viewer doesn't have to guess
    which one they're looking at. Returns the moving-light mask (raw frame
    resolution) so the caller can check the tracked box against it.

    Suppressed on a daylight/dusk-colour clip (`daylight_gated`), same as
    `green_light_ratio`/`green_light_flicker` are zeroed in
    `extract_clip_features` -- real ambient colour (green foliage covering
    much of the frame) reads the same as the guard's flashlight to a raw hue
    mask, and drawing it anyway made every green-toned daylight clip look
    like it was full of flashlight detections that were never actually
    scored.
    """
    empty = np.zeros(frame.shape[:2], dtype=bool)
    if daylight_gated:
        return empty
    stationary, moving = _light_masks(frame, ignore_mask)
    if np.any(stationary):
        # Dashed box, not a solid contour outline -- marked as a known fixed
        # light rather than drawn as if it were nothing, but still visually
        # distinct from a genuine tracked/flashlight detection (solid boxes).
        mask_scaled = cv2.resize(
            stationary.astype(np.uint8),
            (canvas.shape[1], canvas.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        contours, _ = cv2.findContours(mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        _draw_dashed_rect(
            canvas, (x, y), (x + w, y + h), COLOR_STATIONARY_LIGHT, thickness=1
        )
        _text(
            canvas,
            "STATIONARY LIGHT",
            (x, max(11, y - 4)),
            color=COLOR_STATIONARY_LIGHT,
            scale=0.35,
        )
    if np.any(moving):
        mask_scaled = cv2.resize(
            moving.astype(np.uint8),
            (canvas.shape[1], canvas.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        contours, _ = cv2.findContours(mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, COLOR_LIGHT, 1)
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        _text(canvas, "FLASHLIGHT", (x, max(11, y - 4)), color=COLOR_LIGHT, scale=0.35)
    return moving


def _draw_trail(canvas: np.ndarray, history: list[tuple[np.ndarray, tuple[float, float]]]) -> None:
    """Past boxes and centroid path, older = closer to the background colour."""
    recent = history[-TRAIL_LENGTH:]
    for age, (contour, _) in enumerate(reversed(recent[:-1]), start=1):
        weight = 1.0 - age / (len(recent) + 1)
        color = tuple(int(c * weight + 40 * (1 - weight)) for c in COLOR_TRACKED)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)

    points = [(int(cx), int(cy)) for _, (cx, cy) in recent]
    for age, (a, b) in enumerate(zip(points, points[1:], strict=False)):
        weight = (age + 1) / max(len(points), 1)
        color = tuple(int(c * weight + 40 * (1 - weight)) for c in COLOR_TRACKED)
        cv2.line(canvas, a, b, color, 1)
        cv2.circle(canvas, a, 2, color, -1)


MULTI_TRACK_PALETTE = (
    (255, 0, 0),
    (0, 220, 0),
    (0, 128, 255),
    (255, 0, 200),
    (0, 220, 220),
    (180, 100, 255),
    (255, 140, 0),
    (120, 255, 120),
)


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Stable colour per persistent track id, cycling through a fixed palette
    rather than hashing -- keeps low ids (the common case, few simultaneous
    subjects) visually distinct instead of relying on hash luck."""
    return MULTI_TRACK_PALETTE[track_id % len(MULTI_TRACK_PALETTE)]


def _draw_multi_tracks(
    canvas: np.ndarray,
    tracks: list[TrackedObject],
    *,
    scale: int,
    multi_track_history: dict[int, list[tuple[int, int]]],
) -> None:
    """Outline every persistently-identified subject from
    `ClipDetection.multi_tracks` in its own stable colour, plus a short fading
    trail of past centroids per id -- lets multiple simultaneous subjects (e.g.
    two people) be told apart across the whole clip, not just the single
    `largest`/tracked blob `_draw_trail` already shows.

    Purely additive: drawn UNDER the single-track box/trail/label so the
    feature-scoring track (still the operator's primary reference) stays on
    top and unobscured.
    """
    for track in tracks:
        color = _track_color(track.track_id)
        x0, y0, x1, y1 = (c * scale for c in track.bbox)
        centroid = (int((x0 + x1) / 2), int((y0 + y1) / 2))
        history = multi_track_history.setdefault(track.track_id, [])
        history.append(centroid)
        del history[:-TRAIL_LENGTH]

        points = history[-TRAIL_LENGTH:]
        for age, (a, b) in enumerate(zip(points, points[1:], strict=False)):
            weight = (age + 1) / max(len(points), 1)
            faded = tuple(int(c * weight + 40 * (1 - weight)) for c in color)
            cv2.line(canvas, a, b, faded, 1)

        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1)
        label = f"#{track.track_id}"
        if track.merged_ids:
            label += "+" + "+".join(f"#{i}" for i in track.merged_ids)
        _text(canvas, label, (x0, min(canvas.shape[0] - 2, y1 + 12)), color=color, scale=0.35)


def _draw_hud(
    canvas: np.ndarray,
    *,
    lines: list[tuple[str, str]],
    origin_y: int,
) -> None:
    box = canvas.copy()
    cv2.rectangle(box, (0, origin_y), (canvas.shape[1], canvas.shape[0]), HUD_BG, -1)
    cv2.addWeighted(box, 0.75, canvas, 0.25, 0, canvas)
    y = origin_y + 14
    for key, value in lines:
        _text(canvas, key, (6, y), color=(150, 150, 150), scale=0.36)
        _text(canvas, value, (150, y), color=(235, 235, 235), scale=0.36)
        y += 13


def render_clip(
    video_path: str,
    zone: CameraZone,
    *,
    out_path: str,
    title: str = "",
    scale: int = 2,
    threshold: int = DEFAULTS["threshold"],
    min_blob_area_fraction: float = DEFAULTS["min_area"],
    max_area_fraction: float = DEFAULTS["max_area"],
    flare_tolerance: float = DEFAULTS["flare_tolerance"],
    max_flare_fraction: float = DEFAULTS["max_flare_fraction"],
    reference_background: np.ndarray | None = None,
    timestamp: str | None = None,
) -> str | None:
    """Write an annotated H.264 .mp4 video for one clip. Returns the path, or
    None if the clip has no readable frames.

    Real H.264, not this OpenCV build's mp4v/VP8 fallbacks: those play in
    VS Code's own preview but Telegram and ntfy-viewing clients treat them as
    a generic file, not an inline video. `Mp4Writer` pipes frames to a bundled
    ffmpeg with a genuine libx264, so out_path is always coerced to .mp4.
    """
    out_path = str(Path(out_path).with_suffix(".mp4"))
    detection: ClipDetection | None = detect_clip(
        video_path,
        max_area_fraction=max_area_fraction,
        min_blob_area_fraction=min_blob_area_fraction,
        threshold=threshold,
        flare_tolerance=flare_tolerance,
        max_flare_fraction=max_flare_fraction,
        reference_background=reference_background,
        compensate_warmup=True,
    )
    if detection is None:
        return None

    features = extract_clip_features(
        video_path,
        zone,
        max_area_fraction=max_area_fraction,
        min_blob_area_fraction=min_blob_area_fraction,
        threshold=threshold,
        flare_tolerance=flare_tolerance,
        max_flare_fraction=max_flare_fraction,
        daylight_hint=daylight_hint(timestamp),
    )

    source_fps = sane_fps(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
    width, height = detection.frame_width * scale, detection.frame_height * scale
    # None unless this camera opted in via `metric_calibration` in cameras.yaml.
    calib = calibrate(zone, detection.frame_width, detection.frame_height)
    hud_height = 165
    overrides = [
        f"{name}={value}"
        for name, value in (
            ("threshold", threshold),
            ("min_area", min_blob_area_fraction),
            ("max_area", max_area_fraction),
            ("flare_tolerance", flare_tolerance),
            ("max_flare_fraction", max_flare_fraction),
        )
        if value != DEFAULTS[name]
    ]
    tint, ink = _zone_layers(width, height, zone, calib, scale)
    # Sun-time is an independent check on the colour statistic: a dawn clip can sit
    # under the colour gate and still be broad daylight (cam03 at 05:55 in November).
    # Dusk/dawn TWILIGHT (still not full daylight by the sun-time table, but not deep
    # night either) gets the same treatment -- confirmed 2026-09-04 that a cluster of
    # real guard clips 13-56 minutes after sunset were reading a faint natural-foliage
    # green (all well under the 0.05 guard-candidate threshold) that the overlay drew
    # anyway, since neither the colour gate nor the strict day/night split caught them.
    sun_daylight = timestamp is not None and (is_daylight(timestamp) or is_twilight(timestamp))
    daylight_gated = sun_daylight or (features is not None and features["color_fraction"] > 0.15)
    ignore_mask = (
        ignore_region_mask(detection.frame_width, detection.frame_height, zone.ignore)
        if zone.ignore
        else None
    )
    # Auto-detected stationary lights, same reasoning/mask combination as
    # `extract_clip_features` -- so a fixed light with no hand-traced
    # `zone.ignore` polygon still renders as STATIONARY LIGHT rather than
    # FLASHLIGHT, and the render never disagrees with what was scored.
    tracked_region = np.zeros((detection.frame_height, detection.frame_width), dtype=bool)
    for detected in detection.frames:
        if detected.largest is None:
            continue
        x, y, w, h = cv2.boundingRect(detected.largest)
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(detection.frame_width, x + w + pad), min(
            detection.frame_height, y + h + pad
        )
        tracked_region[y0:y1, x0:x1] = True
    auto_light_mask = detect_stationary_light_mask(
        [d.frame for d in detection.frames], exclude_region=tracked_region
    )
    if np.any(auto_light_mask):
        ignore_mask = auto_light_mask if ignore_mask is None else (ignore_mask | auto_light_mask)

    writer = Mp4Writer(out_path, fps=source_fps, width=width, height=height + hud_height)

    min_blob_area_px = min_blob_area_fraction * detection.frame_height * detection.frame_width
    max_area_px = max_area_fraction * detection.frame_height * detection.frame_width

    for warm_index, raw_frame in enumerate(detection.dropped_frames):
        source_frame = (
            detection.dropped_frame_compensated[warm_index]
            if detection.dropped_frame_compensated
            else raw_frame
        )
        # Naive correction magnitude: mean absolute pixel change from the raw
        # capture, as a percentage of the full 0-255 range -- not a measurement
        # of accuracy, just a reminder that this is NOT the original frame.
        correction_pct = (
            float(
                np.abs(source_frame.astype(np.float64) - raw_frame.astype(np.float64)).mean()
            )
            / 255.0
            * 100.0
            if detection.dropped_frame_compensated
            else 0.0
        )
        canvas = cv2.resize(source_frame, (width, height), interpolation=cv2.INTER_CUBIC)
        _apply_zone(canvas, tint, ink)
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), COLOR_WARMUP, 4)
        _text(
            canvas,
            "IR WARMUP - dropped before background model, not scored",
            (10, 40),
            color=COLOR_WARMUP,
            scale=0.5,
            thickness=2,
        )
        if detection.dropped_frame_compensated:
            _text(
                canvas,
                f"~{correction_pct:.0f}% corrected -- NOT the original frame",
                (10, 60),
                color=COLOR_WARMUP,
                scale=0.45,
                thickness=1,
            )
        traced_box = detection.dropped_frame_boxes[warm_index]
        is_photometric = (
            bool(detection.dropped_frame_box_is_photometric)
            and detection.dropped_frame_box_is_photometric[warm_index]
        )
        if traced_box is not None:
            x0, y0, x1, y1 = (c * scale for c in traced_box)
            label = (
                "TRACKED (real diff, brightness/colour corrected)"
                if is_photometric
                else "INFERRED - TRACKED (reverse trace)"
            )
            if is_photometric:
                cv2.rectangle(
                    canvas, (int(x0), int(y0)), (int(x1), int(y1)), COLOR_REVERSE, thickness=2
                )
            else:
                _draw_dashed_rect(canvas, (x0, y0), (x1, y1), COLOR_REVERSE, thickness=2)
            _text(
                canvas,
                label,
                (x0, max(11, y0 - 4)),
                color=COLOR_REVERSE,
                scale=0.4,
            )
        panel = np.zeros((height + hud_height, width, 3), dtype=np.uint8)
        panel[:height] = canvas
        live = [
            (title, ""),
            (
                "frame",
                f"warmup {warm_index + 1}/{detection.warmup_dropped}"
                f"  (raw frame {warm_index + 1}/{detection.total_frames})",
            ),
            (
                "status",
                "gain/illuminator settling -- excluded from background model & features"
                + (
                    f"; ~{correction_pct:.0f}% corrected to the settled background"
                    if detection.dropped_frame_compensated
                    else ""
                ),
            ),
        ]
        _draw_hud(panel, lines=live, origin_y=height)
        writer.write(panel)

    history: list[tuple[np.ndarray, tuple[float, float]]] = []
    multi_track_history: dict[int, list[tuple[int, int]]] = {}
    flare_total = sum(1 for f in detection.frames if f.is_flare)
    prev_centroid: tuple[float, float] | None = None
    try:
        for detected in detection.frames:
            canvas = cv2.resize(
                detected.frame, (width, height), interpolation=cv2.INTER_CUBIC
            )
            _apply_zone(canvas, tint, ink)
            _draw_light_mask(
                canvas, detected.frame, daylight_gated=daylight_gated, ignore_mask=ignore_mask
            )

            discarded_small = 0
            discarded_large = 0
            for contour in detected.all_contours:
                area = cv2.contourArea(contour)
                if min_blob_area_px <= area <= max_area_px:
                    continue  # already drawn below as a kept blob
                if area < min_blob_area_px:
                    discarded_small += 1
                else:
                    discarded_large += 1
                scaled = (contour * scale).astype(np.int32)
                x, y, w, h = cv2.boundingRect(scaled)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_DISCARDED, 1)

            for contour in detected.blobs:
                scaled = (contour * scale).astype(np.int32)
                x, y, w, h = cv2.boundingRect(scaled)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BLOB, 1)

            if detection.multi_tracks:
                _draw_multi_tracks(
                    canvas,
                    detection.multi_tracks[detected.index],
                    scale=scale,
                    multi_track_history=multi_track_history,
                )

            instant_speed = None
            blob_width_px = 0.0
            light_overlap: float | None = None
            estimated_height: float | None = None
            metric_distance: float | None = None
            metric_height: float | None = None
            if detected.largest is not None and detected.centroid is not None:
                scaled = (detected.largest * scale).astype(np.int32)
                history.append(
                    (scaled, (detected.centroid[0] * scale, detected.centroid[1] * scale))
                )
                _draw_trail(canvas, history)
                x, y, w, h = cv2.boundingRect(scaled)
                light_overlap = (
                    0.0
                    if daylight_gated
                    else flashlight_bbox_overlap(
                        detected.frame,
                        cv2.boundingRect(detected.largest),
                        exclude_mask=ignore_mask,
                    )
                )
                is_flashlight = light_overlap > FLASHLIGHT_SUBJECT_THRESHOLD
                inferred = detected.filled_by_reverse or detected.recovered
                if is_flashlight:
                    box_color = COLOR_LIGHT
                    label = "FLASHLIGHT"
                elif detected.filled_by_reverse:
                    box_color = COLOR_REVERSE
                    label = (
                        "RECOVERED (reverse fill)"
                        if detected.recovered
                        else "TRACKED (reverse fill)"
                    )
                elif detected.recovered:
                    box_color = COLOR_RECOVERED
                    label = "RECOVERED (appearance match)"
                else:
                    box_color = COLOR_TRACKED
                    label = "TRACKED"
                if inferred and not is_flashlight:
                    # Dashed, not solid -- this box is a fixed-size template
                    # dragged along, not a fresh measurement of the subject.
                    _draw_dashed_rect(canvas, (x, y), (x + w, y + h), box_color, thickness=2)
                    label = "INFERRED - " + label
                else:
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, 2)
                cv2.drawContours(canvas, [scaled], -1, box_color, 1)
                _text(canvas, label, (x, max(11, y - 4)), color=box_color, scale=0.4)
                raw_box = cv2.boundingRect(detected.largest)
                estimated_height = estimated_height_m(
                    float(raw_box[3]),
                    subject_base_y(raw_box, detection.frame_height),
                    zone,
                    detection.frame_width,
                    detection.frame_height,
                )
                _text(
                    canvas,
                    f"{estimated_height:.2f}m" if estimated_height is not None else "height n/a",
                    (x, y + h + 12),
                    color=box_color,
                    scale=0.4,
                )
                if calib is not None:
                    # Feet row, not the centroid: the ground plane is what
                    # carries the scale, so the box's base is the only part of
                    # it whose depth is actually known.
                    base_px = (raw_box[0] + raw_box[2] / 2.0, float(raw_box[1] + raw_box[3]))
                    metric_distance = calib.distance_m(base_px)
                    metric_height = calib.height_m(base_px, float(raw_box[1]))
                    if metric_distance is not None:
                        _text(
                            canvas,
                            f"{metric_distance:.1f}m away"
                            + (f", {metric_height:.2f}m tall" if metric_height else ""),
                            (x, y + h + 24),
                            color=COLOR_METRIC,
                            scale=0.4,
                        )
                blob_width_px = float(cv2.boundingRect(detected.largest)[2])
                if prev_centroid is not None and blob_width_px > 0:
                    step = np.hypot(
                        detected.centroid[0] - prev_centroid[0],
                        detected.centroid[1] - prev_centroid[1],
                    )
                    instant_speed = step / blob_width_px
                prev_centroid = detected.centroid

            if detected.suppressed_light_box is not None:
                # Real motion inside a stationary-light region that would
                # otherwise render as nothing -- marked instead of hidden, per
                # the "mark instead of ignore" ask. Independent of the main
                # tracked box above: this is whatever the ignore mask ate,
                # not the subject the tracker is currently following.
                sx, sy, sw, sh = (c * scale for c in detected.suppressed_light_box)
                _draw_dashed_rect(canvas, (sx, sy), (sx + sw, sy + sh), COLOR_IGNORE, thickness=1)
                _text(
                    canvas,
                    "IGNORED (stationary light region)",
                    (sx, max(11, sy - 4)),
                    color=COLOR_IGNORE,
                    scale=0.35,
                )

            if detected.is_flare:
                cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), COLOR_FLARE, 4)
                _text(
                    canvas,
                    "IR FLARE - gain step, frame not differenceable",
                    (10, 40),
                    color=COLOR_FLARE,
                    scale=0.5,
                    thickness=2,
                )

            panel = np.zeros((height + hud_height, width, 3), dtype=np.uint8)
            panel[:height] = canvas
            area = 0.0 if detected.largest is None else cv2.contourArea(detected.largest)
            grey = f"{detected.median_grey:.1f}" + ("   FLARE" if detected.is_flare else "")
            light_frac = float(np.count_nonzero(green_light_mask(detected.frame))) / (
                detected.frame.shape[0] * detected.frame.shape[1]
            )
            recovered_total = sum(1 for f in detection.frames if f.recovered)
            reverse_total = sum(1 for f in detection.frames if f.filled_by_reverse)
            live = [
                (title, ""),
                (
                    "frame",
                    f"{detected.index + 1}/{len(detection.frames)} scored"
                    f"  ({detection.warmup_dropped} warmup frames shown first, above, not scored)",
                ),
                ("median grey", grey),
                (
                    "blobs (kept/discarded)",
                    f"{len(detected.blobs)} kept"
                    f" / {discarded_small} too small, {discarded_large} too big",
                ),
                (
                    "tracked blob area",
                    f"{area:.0f} px"
                    + (" (recovered)" if detected.recovered else "")
                    + (" (reverse fill)" if detected.filled_by_reverse else "")
                    if area
                    else "none",
                ),
                (
                    "estimated height (fence ruler)",
                    f"{estimated_height:.2f}m" if estimated_height is not None else "uncalibrated",
                ),
                (
                    "ground plane (distance / height)",
                    (
                        "camera not opted in"
                        if calib is None
                        else (
                            f"{metric_distance:.1f}m"
                            + (
                                f" / {metric_height:.2f}m"
                                if metric_height is not None
                                else " / height n/a"
                            )
                            if metric_distance is not None
                            else f"out of range (max {calib.max_range_m:.0f}m)"
                        )
                    ),
                ),
                (
                    "instant speed (body/frame)",
                    f"{instant_speed:.2f}" if instant_speed is not None else "n/a",
                ),
                ("frame light_ratio", f"{light_frac:.4f}"),
                (
                    "tracked box flashlight overlap",
                    (
                        f"{light_overlap:.2f}"
                        + (" -> FLASHLIGHT" if light_overlap > FLASHLIGHT_SUBJECT_THRESHOLD else "")
                    )
                    if light_overlap is not None
                    else "n/a",
                ),
                ("motion px fraction", f"{detected.motion_pixel_fraction:.4f}"),
                ("flare frames", f"{flare_total}/{len(detection.frames)}"),
                ("recovered frames (appearance)", f"{recovered_total}/{len(detection.frames)}"),
                ("reverse-filled frames", f"{reverse_total}/{len(detection.frames)}"),
                (
                    "multi-tracks (this frame)",
                    str(len(detection.multi_tracks[detected.index]))
                    if detection.multi_tracks
                    else "n/a",
                ),
            ]
            if features is not None:
                live += [
                    (
                        "clip aspect/solidity",
                        f"{features['aspect_ratio']:.2f} / {features['solidity']:.2f}",
                    ),
                    (
                        "clip outside/crossed",
                        f"{features['outside_pixel_fraction']:.2f} /"
                        f" {features['fence_crossed']:.0f}",
                    ),
                    (
                        "clip jitter/speed",
                        f"{features['jitter']:.1f} / {features['normalised_speed']:.2f}",
                    ),
                    (
                        "clip persist/run",
                        f"{features['persistence']:.2f} /"
                        f" {features['longest_detection_run']:.2f}",
                    ),
                    ("clip blob_count", f"{features['blob_count']:.0f}"),
                    (
                        "clip color_fraction (green gated?)",
                        f"{features['color_fraction']:.2f}"
                        + (
                            " YES"
                            if daylight_gated
                            else " no"
                        )
                        + (" (sun)" if sun_daylight else ""),
                    ),
                ]
            if overrides:
                live.append(("OVERRIDDEN", ", ".join(overrides)))
            _draw_hud(panel, lines=live, origin_y=height)
            writer.write(panel)
    finally:
        writer.release()
    return out_path


def _clip_duration_seconds(file_path: str) -> float:
    cap = cv2.VideoCapture(file_path)
    try:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = sane_fps(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    return frame_count / fps if frame_count > 0 else 0.0


def _prefer_longest_per_event(clips: list[dict]) -> list[dict]:
    """Collapse clips that share an embedded alert timestamp (the same physical
    trigger, per `scripts.label._event_key`) to just the longest one.

    Without this, an event's short startup-only clip and its longer, more
    representative sibling can both carry the same label and both get
    selected -- rendering the short one isn't useful on its own.
    """
    best_by_key: dict[str, dict] = {}
    result: list[dict] = []
    for clip in clips:
        key = _event_key(clip["camera_id"], clip.get("caption"))
        if key is None:
            result.append(clip)
            continue
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = clip
            result.append(clip)
        elif _clip_duration_seconds(clip["file_path"]) > _clip_duration_seconds(
            existing["file_path"]
        ):
            result[result.index(existing)] = clip
            best_by_key[key] = clip
    return result


def _resolve_clips(args, conn) -> list[dict]:
    """Clips to render, from an explicit path or from the database."""
    if args.clip:
        return [
            {
                "camera_id": args.camera,
                "message_id": 0,
                "channel_id": None,
                "caption": None,
                "file_path": args.clip,
                "label": "",
                "startup_state": None,
                "timestamp": None,
            }
        ]

    message_ids: set[int] = set(args.message_id or [])
    if args.message_ids_file:
        for line in Path(args.message_ids_file).read_text().splitlines():
            entry = line.strip()
            if entry and not entry.startswith("#"):
                message_ids.add(int(entry))

    selected: list[dict] = []
    seen: set[int] = set()

    def take(row, label: str) -> None:
        if row["message_id"] in seen:
            return
        if not row["file_path"] or not Path(row["file_path"]).exists():
            return
        seen.add(row["message_id"])
        label_row = db.get_label(conn, row["channel_id"], row["message_id"])
        selected.append(
            {
                "camera_id": row["camera_id"],
                "message_id": row["message_id"],
                "channel_id": row["channel_id"],
                "caption": row["caption"],
                "file_path": row["file_path"],
                "label": label,
                "startup_state": label_row["startup_state"] if label_row is not None else None,
                "timestamp": row["timestamp"],
            }
        )

    for label in args.label or []:
        for row in db.iter_clips_with_label(
            conn, label, camera_id=args.camera, with_file_only=True
        ):
            take(row, label)

    if message_ids:
        labels = {
            mid: lab
            for mid, lab in conn.execute("SELECT message_id, label FROM labels")
            if lab
        }
        for row in db.iter_clips(conn, camera_id=args.camera):
            if row["message_id"] in message_ids:
                take(row, labels.get(row["message_id"], ""))

    selected = _prefer_longest_per_event(selected)
    return selected[: args.limit] if args.limit else selected


def _widen_year(embedded: str) -> str:
    """Captions carry `DD-MM-YY HH:MM:SS`; show the century so a 2024 clip
    can't be misread as 2004 or 1924."""
    date, _, clock = embedded.partition(" ")
    day, month, year = date.split("-")
    return f"{day}-{month}-20{year} {clock}" if len(year) == 2 else embedded


def _reference_background(entries, root, camera, timestamp):
    """Resolve this camera's reference background for the clip's era and
    lighting, or None when no reference covers it (too few clips, or a
    remounted camera with nothing recorded since the move)."""
    if not entries or timestamp is None:
        return None
    entry = reference_for(
        entries,
        camera.id,
        timestamp,
        era=era_of(camera, timestamp),
        daylight=is_daylight(timestamp),
    )
    return None if entry is None else load_reference_image(Path(root), entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("what to render")
    source.add_argument("--clip", help="render one video file directly (needs --camera)")
    source.add_argument("--camera", help="camera id, e.g. cam06")
    source.add_argument("--message-id", type=int, action="append", help="repeatable")
    source.add_argument("--message-ids-file", help="one message_id per line")
    source.add_argument("--label", action="append", help="render all clips with this label")
    source.add_argument("--limit", type=int, help="cap how many clips are rendered")
    parser.add_argument("--out", help="output path (single clip) or directory (batch)")
    parser.add_argument("--scale", type=int, default=2, help="upscale factor (default 2)")

    tuning = parser.add_argument_group("detector tuning (defaults match scripts/spike.py)")
    tuning.add_argument("--threshold", type=int, default=DEFAULTS["threshold"])
    tuning.add_argument("--min-area", type=float, default=DEFAULTS["min_area"])
    tuning.add_argument("--max-area", type=float, default=DEFAULTS["max_area"])
    tuning.add_argument("--flare-tolerance", type=float, default=DEFAULTS["flare_tolerance"])
    tuning.add_argument("--max-flare-fraction", type=float, default=DEFAULTS["max_flare_fraction"])
    tuning.add_argument(
        "--reference-bg",
        default="data/reference_bg",
        help="per-camera reference background directory (build with scripts/build_reference_bg.py)",
    )
    tuning.add_argument(
        "--no-reference-bg",
        action="store_true",
        help="disable the reference-background scenery veto, for before/after comparison",
    )
    args = parser.parse_args(argv)

    if args.clip and not args.camera:
        parser.error("--clip needs --camera so the fence geometry can be loaded")

    cameras = load_cameras_config("config/cameras.yaml")
    reference_entries = [] if args.no_reference_bg else load_manifest(args.reference_bg)
    if args.clip:
        clips = _resolve_clips(args, None)
    else:
        app = load_app_config()
        with db.connect(app.db_path) as conn:
            clips = _resolve_clips(args, conn)
    if not clips:
        print("no clips matched; pass --clip, --message-id, --message-ids-file or --label")
        return 1

    out_dir = Path(args.out) if args.out and len(clips) > 1 else Path("data/reports/debug")
    rendered = 0
    for clip in clips:
        camera = cameras.by_id(clip["camera_id"])
        if camera is None:
            print(f"  skip {clip['camera_id']}/{clip['message_id']}: camera not in cameras.yaml")
            continue
        if args.out and len(clips) == 1:
            out_path = args.out
        else:
            out_path = str(out_dir / f"{clip['camera_id']}_{clip['message_id']}.mp4")
        title = f"{clip['camera_id']}/{clip['message_id']} {clip['label']}".strip()
        ts_match = _EVENT_TS_RE.search(clip.get("caption") or "")
        if ts_match:
            # embedded camera time, site-local (SAST); captions carry a 2-digit year
            title += f" @ {_widen_year(ts_match.group(1))}"
        if clip.get("startup_state"):
            title += f" [startup_state={clip['startup_state']}]"
        result = render_clip(
            clip["file_path"],
            camera.zone_at(clip.get("timestamp")),
            out_path=out_path,
            title=title,
            scale=args.scale,
            threshold=args.threshold,
            min_blob_area_fraction=args.min_area,
            max_area_fraction=args.max_area,
            flare_tolerance=args.flare_tolerance,
            max_flare_fraction=args.max_flare_fraction,
            reference_background=_reference_background(
                reference_entries, args.reference_bg, camera, clip.get("timestamp")
            ),
            timestamp=clip.get("timestamp"),
        )
        if result is None:
            print(f"  skip {title}: no readable frames")
            continue
        rendered += 1
        print(f"  wrote {result}")
    print(f"rendered {rendered}/{len(clips)} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
