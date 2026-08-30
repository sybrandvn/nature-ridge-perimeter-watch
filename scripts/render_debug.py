"""Render a clip as an annotated video showing what the detector actually sees.

The features CSV says a clip scored `aspect_ratio=0.81, jitter=6.8`. It does not
say whether the detector was looking at a porcupine or at a branch. This renders
the detector's own view frame by frame so that can be judged by eye and the
thresholds tuned against something visible.

Drawn per frame:
  - fence polyline, with the OUTSIDE and INSIDE half-frames tinted
  - ignore regions (hatched) and the depth cutoff line
  - every blob passing the area gates in dim cyan, the tracked (largest) blob
    in bright magenta
  - a fading trail: past bounding boxes plus the joined centroid path
  - an IR FLARE banner on frames where the global gain stepped
  - a HUD of live per-frame values and the clip's final feature vector

Detection comes from `scripts.spike.detect_clip`, the same function that feeds
`extract_clip_features`, so what is drawn is what scored the clip. The tuning
flags default to the spike's own values and the HUD marks them when overridden.

Output is .webm (VP8) so it plays in VS Code's built-in preview and any
browser -- this OpenCV build can't write a Chromium-playable H.264 mp4.

Run:
    uv run python scripts/render_debug.py --message-id 21520 --out out.webm
    uv run python scripts/render_debug.py --clip data/history/cam06/21520.mp4 \
        --camera cam06 --out out.webm
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

from scripts.spike import ClipDetection, detect_clip, extract_clip_features  # noqa: E402
from src import db  # noqa: E402
from src.config import CameraZone, load_app_config, load_cameras_config  # noqa: E402
from src.features import green_light_mask  # noqa: E402
from src.zones import _fence_x_at_y, side_name  # noqa: E402

DEFAULTS = {
    "threshold": 18,
    "min_area": 0.0005,
    "max_area": 0.25,
    "flare_tolerance": 3.0,
    "max_flare_fraction": 0.4,
}

COLOR_TRACKED = (255, 0, 255)
COLOR_BLOB = (200, 200, 0)
COLOR_FENCE = (0, 255, 255)
COLOR_OUTSIDE = (0, 0, 255)
COLOR_INSIDE = (0, 255, 0)
COLOR_IGNORE = (110, 110, 110)
COLOR_FLARE = (0, 90, 255)
COLOR_LIGHT = (0, 255, 140)
HUD_BG = (24, 24, 24)
TRAIL_LENGTH = 12


def _to_px(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x, y = point
    return int(round(x * width)), int(round(y * height))


def _text(img, s, org, *, color=(235, 235, 235), scale=0.4, thickness=1) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _zone_layers(width: int, height: int, zone: CameraZone) -> tuple[np.ndarray, np.ndarray]:
    """Static zone artwork as (tint, ink) layers, built once per clip.

    `tint` is blended in flat; `ink` is copied where non-black, so the fence
    line and labels stay legible instead of washing out with the tint.
    """
    tint = np.zeros((height, width, 3), dtype=np.uint8)
    ink = np.zeros((height, width, 3), dtype=np.uint8)

    if zone.fence is not None:
        # Sample which side each pixel column falls on, rather than assuming the
        # fence runs top-to-bottom.
        ys, xs = np.mgrid[0:height, 0:width]
        norm_x = xs / width
        norm_y = ys / height
        fence_x = np.empty((height, width), dtype=np.float64)
        for row in range(height):
            fence_x[row, :] = _fence_x_at_y(norm_y[row, 0], zone.fence)
        left = norm_x < fence_x
        outside_mask = left if zone.outside == "left" else ~left
        tint[outside_mask] = COLOR_OUTSIDE
        tint[~outside_mask] = COLOR_INSIDE

        fence_px = [_to_px(p, width, height) for p in zone.fence]
        for a, b in zip(fence_px, fence_px[1:], strict=False):
            cv2.line(ink, a, b, COLOR_FENCE, 2)
        for p in fence_px:
            cv2.circle(ink, p, 3, COLOR_FENCE, -1)

        (ax, ay), (bx, by) = zone.fence[0], zone.fence[-1]
        dx, dy = bx - ax, by - ay
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / norm, dx / norm
        mid = ((ax + bx) / 2, (ay + by) / 2)
        for sign in (1, -1):
            probe = (mid[0] + sign * nx * 0.14, mid[1] + sign * ny * 0.14)
            outside = side_name(probe, zone.fence) == zone.outside
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

    for region in zone.ignore:
        pts = np.array([_to_px(p, width, height) for p in region], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        hatch = np.zeros_like(ink)
        for offset in range(0, w + h, 8):
            cv2.line(hatch, (x + offset, y), (x + offset - h, y + h), COLOR_IGNORE, 1)
        region_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(region_mask, [pts], 255)
        hatch[region_mask == 0] = 0
        ink[hatch > 0] = COLOR_IGNORE[0]
        cv2.polylines(ink, [pts], True, COLOR_IGNORE, 1)
        _text(ink, "IGNORE", (x + 2, y + 11), color=COLOR_IGNORE, scale=0.35)

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


def _draw_light_mask(canvas: np.ndarray, frame: np.ndarray) -> None:
    """Outline pixels the detector would call flashlight, in the frame's own
    colour rather than a flat tint -- lets the operator judge hue by eye, not
    just the pass/fail of `green_light_ratio`."""
    mask = green_light_mask(frame)
    if not np.any(mask):
        return
    mask_scaled = cv2.resize(
        mask.astype(np.uint8), (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, COLOR_LIGHT, 1)


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
) -> str | None:
    """Write an annotated .webm (VP8) video for one clip. Returns the path, or
    None if the clip has no readable frames.

    VP8/webm, not H.264/mp4: this OpenCV build only has FFmpeg's hardware
    h264_v4l2m2m encoder (no libx264), which fails without a v4l2 device, and
    Chromium (VS Code's built-in preview) can't play the mp4v codec this repo
    used to write. webm+VP8 is natively decodable by both cv2.VideoCapture and
    VS Code's preview, so out_path is always coerced to a .webm suffix.
    """
    out_path = str(Path(out_path).with_suffix(".webm"))
    detection: ClipDetection | None = detect_clip(
        video_path,
        max_area_fraction=max_area_fraction,
        min_blob_area_fraction=min_blob_area_fraction,
        threshold=threshold,
        flare_tolerance=flare_tolerance,
        max_flare_fraction=max_flare_fraction,
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
    )

    source_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS) or 10.0
    width, height = detection.frame_width * scale, detection.frame_height * scale
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
    tint, ink = _zone_layers(width, height, zone)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"VP80"), source_fps, (width, height + hud_height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    history: list[tuple[np.ndarray, tuple[float, float]]] = []
    flare_total = sum(1 for f in detection.frames if f.is_flare)
    prev_centroid: tuple[float, float] | None = None
    try:
        for detected in detection.frames:
            canvas = cv2.resize(
                detected.frame, (width, height), interpolation=cv2.INTER_NEAREST
            )
            _apply_zone(canvas, tint, ink)
            _draw_light_mask(canvas, detected.frame)

            for contour in detected.blobs:
                scaled = (contour * scale).astype(np.int32)
                x, y, w, h = cv2.boundingRect(scaled)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BLOB, 1)

            instant_speed = None
            blob_width_px = 0.0
            if detected.largest is not None and detected.centroid is not None:
                scaled = (detected.largest * scale).astype(np.int32)
                history.append(
                    (scaled, (detected.centroid[0] * scale, detected.centroid[1] * scale))
                )
                _draw_trail(canvas, history)
                x, y, w, h = cv2.boundingRect(scaled)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_TRACKED, 2)
                cv2.drawContours(canvas, [scaled], -1, COLOR_TRACKED, 1)
                _text(canvas, "TRACKED", (x, max(11, y - 4)), color=COLOR_TRACKED, scale=0.4)
                blob_width_px = float(cv2.boundingRect(detected.largest)[2])
                if prev_centroid is not None and blob_width_px > 0:
                    step = np.hypot(
                        detected.centroid[0] - prev_centroid[0],
                        detected.centroid[1] - prev_centroid[1],
                    )
                    instant_speed = step / blob_width_px
                prev_centroid = detected.centroid

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
            live = [
                (title, ""),
                (
                    "frame",
                    f"{detected.index + 1}/{len(detection.frames)}"
                    f"  (+{detection.warmup_dropped} flare frames dropped)",
                ),
                ("median grey", grey),
                ("blobs this frame", f"{len(detected.blobs)}"),
                ("tracked blob area", f"{area:.0f} px" if area else "none"),
                (
                    "instant speed (body/frame)",
                    f"{instant_speed:.2f}" if instant_speed is not None else "n/a",
                ),
                ("frame light_ratio", f"{light_frac:.4f}"),
                ("motion px fraction", f"{detected.motion_pixel_fraction:.4f}"),
                ("flare frames", f"{flare_total}/{len(detection.frames)}"),
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
                ]
            if overrides:
                live.append(("OVERRIDDEN", ", ".join(overrides)))
            _draw_hud(panel, lines=live, origin_y=height)
            writer.write(panel)
    finally:
        writer.release()
    return out_path


def _resolve_clips(args, conn) -> list[dict]:
    """Clips to render, from an explicit path or from the database."""
    if args.clip:
        return [{"camera_id": args.camera, "message_id": 0, "file_path": args.clip, "label": ""}]

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
        selected.append(
            {
                "camera_id": row["camera_id"],
                "message_id": row["message_id"],
                "file_path": row["file_path"],
                "label": label,
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

    return selected[: args.limit] if args.limit else selected


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
    args = parser.parse_args(argv)

    if args.clip and not args.camera:
        parser.error("--clip needs --camera so the fence geometry can be loaded")

    cameras = load_cameras_config("config/cameras.yaml")
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
            out_path = str(out_dir / f"{clip['camera_id']}_{clip['message_id']}.webm")
        title = f"{clip['camera_id']}/{clip['message_id']} {clip['label']}".strip()
        result = render_clip(
            clip["file_path"],
            camera.zone,
            out_path=out_path,
            title=title,
            scale=args.scale,
            threshold=args.threshold,
            min_blob_area_fraction=args.min_area,
            max_area_fraction=args.max_area,
            flare_tolerance=args.flare_tolerance,
            max_flare_fraction=args.max_flare_fraction,
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
