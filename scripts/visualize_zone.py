"""Render one clip's best-motion frame with the fence polyline, outside/inside
labels, and the detector's bounding box overlaid -- a visual sanity check of
what src/zones.py actually classifies, not just a features.csv row.

For a full per-frame video with a live HUD instead of a single still, use
scripts/render_debug.py -- it draws every blob, the tracked blob's trail, and
IR flare frames, not just the clearest one.

Uses `scripts.spike.detect_clip`, the same detector `extract_clip_features`
feeds on, so the box shown here is what actually produced
`outside_pixel_fraction` for this clip.

Run:
    uv run python scripts/visualize_zone.py data/history/cam06/21520.mp4 cam06 \
        --out data/reports/zone_check_cam06_21520.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from scripts.spike import detect_clip  # noqa: E402
from src.config import CameraZone, load_cameras_config  # noqa: E402
from src.zones import effective_fence, outside_pixel_fraction, side_name  # noqa: E402


def _to_px(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x, y = point
    return int(round(x * width)), int(round(y * height))


def render(video_path: str, zone: CameraZone, *, out_path: str) -> str:
    detection = detect_clip(video_path)
    if detection is None:
        raise ValueError(f"No readable frames in {video_path}")
    best = max(
        (f for f in detection.frames if f.largest is not None),
        key=lambda f: cv2.contourArea(f.largest),
        default=None,
    )
    if best is None:
        raise ValueError(f"No motion detected in {video_path}")
    frame, contour = best.frame, best.largest
    height, width = frame.shape[:2]
    canvas = frame.copy()

    if zone.fence is not None:
        fence_px = [_to_px(p, width, height) for p in zone.fence]
        for a, b in zip(fence_px, fence_px[1:], strict=False):
            cv2.line(canvas, a, b, (0, 255, 255), 2)

    # Base/ground line, only traced on some cameras (cam06 as of 2026-09-05) --
    # drawn in a distinct colour, alongside the top rail rather than replacing it.
    if zone.fence_bottom is not None:
        bottom_px = [_to_px(p, width, height) for p in zone.fence_bottom]
        for a, b in zip(bottom_px, bottom_px[1:], strict=False):
            cv2.line(canvas, a, b, (0, 140, 255), 2)

    classification_fence = effective_fence(zone)
    if classification_fence is not None:
        # Perpendicular to the first->last segment, so the outside/inside labels
        # land on the correct side regardless of the fence's own orientation.
        (ax, ay), (bx, by) = classification_fence[0], classification_fence[-1]
        dx, dy = bx - ax, by - ay
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / norm, dx / norm
        mid = ((ax + bx) / 2, (ay + by) / 2)
        for sign, tag_color in ((1, (0, 0, 255)), (-1, (0, 255, 0))):
            probe = (mid[0] + sign * nx * 0.12, mid[1] + sign * ny * 0.12)
            side = side_name(probe, classification_fence)
            tag = "OUTSIDE" if side == zone.outside else "INSIDE"
            px, py = _to_px(probe, width, height)
            px, py = max(5, min(width - 90, px)), max(15, min(height - 5, py))
            cv2.putText(
                canvas, tag, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag_color, 2, cv2.LINE_AA
            )

    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 255), 2)
    centroid = best.centroid
    frac = None
    if centroid is not None and classification_fence is not None and zone.outside is not None:
        frac = outside_pixel_fraction([(centroid[0] / width, centroid[1] / height)], zone)
    label = (
        f"intruder box (outside_pixel_fraction={frac:.2f})"
        if frac is not None
        else "intruder box"
    )
    cv2.putText(
        canvas, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2,
        cv2.LINE_AA,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, canvas)
    return out_path


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("camera_id")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Clip timestamp (ISO 8601), to pick the dated geometry in effect then. "
        "Defaults to the camera's current (most recent) geometry.",
    )
    args = parser.parse_args()

    cameras_cfg = load_cameras_config("config/cameras.yaml")
    camera = cameras_cfg.by_id(args.camera_id)
    if camera is None:
        raise ValueError(f"Unknown camera id {args.camera_id!r}")
    out = render(args.video_path, camera.zone_at(args.timestamp), out_path=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
