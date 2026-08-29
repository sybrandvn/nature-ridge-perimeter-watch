"""Phase 0c: CV feasibility spike (gate 2).

Extracts candidate discriminating features to a flat CSV for a small,
hand-labelled clip sample, so a human can check whether any threshold
combination separates guard/environment from animal/incident before the rest
of the pipeline (motion.py, classify.py, backtester.py) gets built on top of a
false premise.

This deliberately does NOT do motion detection/background subtraction itself
-- that is Phase 1's src/motion.py, built only after this gate passes. Instead
it takes an already-detected blob per sampled frame (bounding contour + centroid)
so the feature math (src/features.py, src/zones.py) can be validated against
real footage using a throwaway per-clip detector, without committing to a
production motion pipeline before knowing it's worth building.

Workflow (see docs/plan.md Phase 0c):
  1. Pick 2-3 cameras with the richest incident history (needs user input --
     see the "which cameras" question in README.md).
  2. Download a clip subset for those cameras with scripts/download_clips.py
     (requires live Telegram access; writes clips.file_path automatically).
  3. Hand-enter fence polylines for those cameras in config/cameras.yaml.
  4. Hand-label ~150 clips with scripts/label.py.
  5. Run this script to extract features to a CSV.
  6. Manually inspect the CSV (or a notebook) for separability. If nothing
     separates guard/environment from animal/incident, stop and revisit scope
     per docs/plan.md rather than proceeding to Phase 1.

Run:
    uv run python scripts/spike.py --camera cam_north --out data/reports/spike_cam_north.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import db  # noqa: E402
from src.config import CamerasConfig, CameraZone, load_app_config, load_cameras_config  # noqa: E402
from src.features import (  # noqa: E402
    aspect_ratio,
    edge_density,
    green_light_flicker,
    green_light_ratio,
    jitter,
    path_length,
    persistence,
    row_normalised_area,
    saturation_ratio,
    solidity,
    time_of_day,
)
from src.zones import outside_pixel_fraction  # noqa: E402

FEATURE_COLUMNS = (
    "channel_id",
    "message_id",
    "camera_id",
    "label",
    "time_of_day",
    "outside_pixel_fraction",
    "aspect_ratio",
    "solidity",
    "saturation_ratio",
    "green_light_ratio",
    "green_light_flicker",
    "row_normalised_area",
    "edge_density",
    "path_length",
    "jitter",
    "persistence",
    "motion_pixel_fraction",
    "blob_count",
)


def largest_contour(mask: np.ndarray, *, max_area: float | None = None) -> np.ndarray | None:
    """Largest external contour in a binary motion mask, or None if empty.

    `max_area` discards blobs bigger than that, which on these cameras means a
    whole-frame IR gain/flicker change rather than a subject.

    This is the "throwaway per-clip detector" referenced in the module
    docstring: good enough to validate feature separability, not a claim
    about the eventual production motion pipeline.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if max_area is not None:
        contours = [c for c in contours if cv2.contourArea(c) <= max_area]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_centroid(contour: np.ndarray) -> tuple[float, float] | None:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def normalized_contour_points(contour: np.ndarray, frame_width: int, frame_height: int) -> list:
    return [(float(x) / frame_width, float(y) / frame_height) for [[x, y]] in contour]


def _whole_frame_contour(frame_width: int, frame_height: int) -> np.ndarray:
    w, h = frame_width - 1, frame_height - 1
    return np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)


def extract_clip_features(
    video_path: str,
    zone: CameraZone,
    *,
    reference_row: float | None = None,
    warmup_frames: int = 10,
    max_area_fraction: float = 0.25,
    min_blob_area_fraction: float = 0.0005,
    threshold: int = 18,
) -> dict[str, float] | None:
    """Run a simple background-subtraction detector over one clip and compute
    features for its largest track. Returns None if no motion was detected.

    Consecutive-frame differencing was tried first and failed on the real
    footage: these cameras spend roughly the first two seconds of every clip
    settling their IR gain, and that whole-frame brightness swing is a far
    bigger inter-frame delta than an actual animal. It hid a porcupine on
    cam15/15454 entirely. So warmup frames are dropped, the background is the
    per-pixel median of what remains (the subject moves, the fence doesn't),
    and blobs larger than `max_area_fraction` of the frame are rejected as
    residual illumination changes rather than subjects.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        return None

    # Keep the warmup frames if dropping them would leave too little to model a
    # background from -- short blank "startup" clips are shorter than the warmup.
    considered = frames[warmup_frames:] if total_frames - warmup_frames >= 5 else frames

    frame_height, frame_width = considered[0].shape[:2]
    max_area = max_area_fraction * frame_height * frame_width
    min_blob_area = min_blob_area_fraction * frame_height * frame_width

    grays = [
        cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in considered
    ]
    background = np.median(np.stack(grays), axis=0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    centroids: list[tuple[float, float]] = []
    whole_frame_green_ratios: list[float] = []
    frames_detected = 0
    best_contour: np.ndarray | None = None
    best_frame: np.ndarray | None = None
    best_area = -1.0
    whole_frame = _whole_frame_contour(frame_width, frame_height)
    motion_pixel_fraction = 0.0
    blob_count = 0

    for frame, gray in zip(considered, grays, strict=True):
        whole_frame_green_ratios.append(green_light_ratio(frame, whole_frame))
        diff = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Peak-frame readings, not an average -- a storm/wind frame with motion
        # scattered across many small blobs (bushes, branches) reads very
        # differently from a single compact subject even at the same threshold.
        motion_pixel_fraction = max(
            motion_pixel_fraction, float(np.count_nonzero(mask)) / mask.size
        )
        frame_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob_count = max(
            blob_count,
            sum(1 for c in frame_contours if min_blob_area <= cv2.contourArea(c) <= max_area),
        )
        contour = largest_contour(mask, max_area=max_area)
        if contour is None:
            continue
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        frames_detected += 1
        # Shape features describe the subject at its clearest, not whichever
        # frame happened to be last -- tracks often end on a fading speck.
        if area > best_area:
            best_area = area
            best_contour = contour
            best_frame = frame
        centroid = contour_centroid(contour)
        if centroid is not None:
            centroids.append(centroid)

    if best_contour is None or best_frame is None:
        return None

    ref_row = reference_row if reference_row is not None else float(frame_height)
    points = normalized_contour_points(best_contour, frame_width, frame_height)

    return {
        "outside_pixel_fraction": outside_pixel_fraction(points, zone),
        "aspect_ratio": aspect_ratio(best_contour),
        "solidity": solidity(best_contour),
        "saturation_ratio": saturation_ratio(best_frame, best_contour),
        "green_light_ratio": green_light_ratio(best_frame, best_contour),
        "green_light_flicker": green_light_flicker(whole_frame_green_ratios),
        "row_normalised_area": row_normalised_area(best_contour, ref_row),
        "edge_density": edge_density(best_frame, best_contour),
        "path_length": path_length(centroids),
        "jitter": jitter(centroids),
        "persistence": persistence(frames_detected, len(considered)),
        "motion_pixel_fraction": motion_pixel_fraction,
        "blob_count": float(blob_count),
    }


def iter_labelled_clips_with_files(conn, *, camera_id: str) -> Iterator[dict[str, Any]]:
    """Labelled clips for one camera that have a local file_path to extract from."""
    for row in db.iter_clips(conn, camera_id=camera_id):
        if not row["file_path"]:
            continue
        label_row = db.get_label(conn, row["channel_id"], row["message_id"])
        if label_row is None or label_row["label"] is None:
            continue  # event's class not known yet (may only carry a startup_state)
        yield {
            "channel_id": row["channel_id"],
            "message_id": row["message_id"],
            "camera_id": row["camera_id"],
            "file_path": row["file_path"],
            "label": label_row["label"],
            "timestamp": row["timestamp"],
        }


def run_spike(
    conn,
    cameras: CamerasConfig,
    *,
    camera_id: str,
    operating_window_start: str = "18:00",
    operating_window_end: str = "06:00",
) -> list[dict[str, Any]]:
    camera = cameras.by_id(camera_id)
    if camera is None:
        raise ValueError(f"Unknown camera_id: {camera_id!r}")

    rows: list[dict[str, Any]] = []
    for clip in iter_labelled_clips_with_files(conn, camera_id=camera_id):
        features = extract_clip_features(clip["file_path"], camera.zone)
        if features is None:
            continue
        rows.append(
            {
                "channel_id": clip["channel_id"],
                "message_id": clip["message_id"],
                "camera_id": clip["camera_id"],
                "label": clip["label"],
                "time_of_day": time_of_day(
                    clip["timestamp"],
                    window_start=operating_window_start,
                    window_end=operating_window_end,
                ),
                **features,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:  # pragma: no cover - requires real labelled footage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="Camera id from config/cameras.yaml")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    cameras_cfg = load_cameras_config("config/cameras.yaml")
    conn = db.connect(app_cfg.db_path)

    rows = run_spike(
        conn,
        cameras_cfg,
        camera_id=args.camera,
        operating_window_start=app_cfg.operating_window_start,
        operating_window_end=app_cfg.operating_window_end,
    )
    conn.close()

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} labelled feature rows to {args.out}")


if __name__ == "__main__":
    main()
