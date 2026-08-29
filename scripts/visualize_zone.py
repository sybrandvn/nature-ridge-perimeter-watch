"""Render one clip's best-motion frame with the fence polyline, outside/inside
labels, and the detector's bounding box overlaid -- a visual sanity check of
what src/zones.py actually classifies, not just a features.csv row.

Uses the exact same throwaway detector as scripts/spike.py's
extract_clip_features (background = per-pixel median of non-warmup frames,
largest contour under max_area_fraction), so the box shown here is what that
detector would have fed into outside_pixel_fraction.

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
import numpy as np  # noqa: E402

from scripts.spike import contour_centroid, largest_contour  # noqa: E402
from src.config import CameraZone, load_cameras_config  # noqa: E402
from src.zones import outside_pixel_fraction, side_name  # noqa: E402


def _detect_best_frame(
    video_path: str,
    *,
    warmup_frames: int = 10,
    max_area_fraction: float = 0.25,
    threshold: int = 18,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Same background-subtraction pass as spike.extract_clip_features, but
    returning the best (frame, contour) pair instead of derived features."""
    cap = cv2.VideoCapture(video_path)
    try:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        return None
    considered = frames[warmup_frames:] if len(frames) - warmup_frames >= 5 else frames
    height, width = considered[0].shape[:2]
    max_area = max_area_fraction * height * width

    grays = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in considered]
    background = np.median(np.stack(grays), axis=0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    best_contour, best_frame, best_area = None, None, -1.0
    for frame, gray in zip(considered, grays, strict=True):
        diff = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contour = largest_contour(mask, max_area=max_area)
        if contour is None:
            continue
        area = cv2.contourArea(contour)
        if area > best_area:
            best_area, best_contour, best_frame = area, contour, frame

    if best_contour is None or best_frame is None:
        return None
    return best_frame, best_contour


def _to_px(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x, y = point
    return int(round(x * width)), int(round(y * height))


def render(video_path: str, zone: CameraZone, *, out_path: str) -> str:
    detection = _detect_best_frame(video_path)
    if detection is None:
        raise ValueError(f"No motion detected in {video_path}")
    frame, contour = detection
    height, width = frame.shape[:2]
    canvas = frame.copy()

    if zone.fence is not None:
        fence_px = [_to_px(p, width, height) for p in zone.fence]
        for a, b in zip(fence_px, fence_px[1:], strict=False):
            cv2.line(canvas, a, b, (0, 255, 255), 2)

        # Perpendicular to the first->last segment, so the outside/inside labels
        # land on the correct side regardless of the fence's own orientation.
        (ax, ay), (bx, by) = zone.fence[0], zone.fence[-1]
        dx, dy = bx - ax, by - ay
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / norm, dx / norm
        mid = ((ax + bx) / 2, (ay + by) / 2)
        for sign, tag_color in ((1, (0, 0, 255)), (-1, (0, 255, 0))):
            probe = (mid[0] + sign * nx * 0.12, mid[1] + sign * ny * 0.12)
            side = side_name(probe, zone.fence)
            tag = "OUTSIDE" if side == zone.outside else "INSIDE"
            px, py = _to_px(probe, width, height)
            px, py = max(5, min(width - 90, px)), max(15, min(height - 5, py))
            cv2.putText(
                canvas, tag, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag_color, 2, cv2.LINE_AA
            )

    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 255), 2)
    centroid = contour_centroid(contour)
    frac = None
    if centroid is not None and zone.fence is not None and zone.outside is not None:
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
    args = parser.parse_args()

    cameras_cfg = load_cameras_config("config/cameras.yaml")
    camera = cameras_cfg.by_id(args.camera_id)
    if camera is None:
        raise ValueError(f"Unknown camera id {args.camera_id!r}")
    out = render(args.video_path, camera.zone, out_path=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
