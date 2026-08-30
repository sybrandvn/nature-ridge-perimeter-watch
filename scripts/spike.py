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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import db  # noqa: E402
from src.config import CamerasConfig, CameraZone, load_app_config, load_cameras_config  # noqa: E402
from src.features import (  # noqa: E402
    area_stability,
    aspect_ratio,
    edge_density,
    flare_frames,
    flare_settle_index,
    green_light_flicker,
    green_light_ratio,
    heading_change,
    jitter,
    longest_detection_run,
    normalised_speed,
    path_length,
    persistence,
    row_normalised_area,
    saturation_ratio,
    solidity,
    time_of_day,
)
from src.zones import (  # noqa: E402
    median_fence_distance,
    outside_pixel_fraction,
    track_crosses_fence,
)

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
    "longest_detection_run",
    "area_stability",
    "normalised_speed",
    "heading_change",
    "fence_crossed",
    "median_fence_distance",
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


def _contour_bbox(contour: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x + w, y + h)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def track_contour(
    candidates: list[np.ndarray],
    track_bbox: tuple[int, int, int, int] | None,
    *,
    max_jump_distance: float,
) -> np.ndarray | None:
    """Pick the candidate contour that continues an existing track, instead of
    always re-selecting the frame's largest blob independently.

    With no active track (nothing tracked yet, or the track was dropped after
    too many consecutive misses), this falls back to the largest candidate --
    the original behaviour. With an active track, it prefers the candidate
    with the highest bounding-box overlap (IoU) against the last known
    position; if nothing overlaps, it falls back to the nearest centroid
    within `max_jump_distance`. This is what keeps the "tracked" identity from
    flipping between two co-occurring subjects (e.g. two people) just because
    their relative blob sizes swap from one frame to the next.
    """
    if not candidates:
        return None
    if track_bbox is None:
        return max(candidates, key=cv2.contourArea)

    boxes = [_contour_bbox(c) for c in candidates]
    ious = [_bbox_iou(track_bbox, box) for box in boxes]
    best_iou_idx = max(range(len(candidates)), key=lambda i: ious[i])
    if ious[best_iou_idx] > 0:
        return candidates[best_iou_idx]

    track_center = _bbox_center(track_bbox)
    distances = [
        ((cx - track_center[0]) ** 2 + (cy - track_center[1]) ** 2) ** 0.5
        for cx, cy in (_bbox_center(box) for box in boxes)
    ]
    best_dist_idx = min(range(len(candidates)), key=lambda i: distances[i])
    if distances[best_dist_idx] <= max_jump_distance:
        return candidates[best_dist_idx]

    # No plausible continuation of the existing track -- treat this as a fresh
    # acquisition rather than force-matching an unrelated blob.
    return max(candidates, key=cv2.contourArea)


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


@dataclass(frozen=True)
class FrameDetection:
    """Everything the detector saw in one frame, before it is reduced to features."""

    index: int
    frame: np.ndarray
    mask: np.ndarray
    all_contours: list[np.ndarray]  # every contour found, before either area gate
    blobs: list[np.ndarray]  # contours passing both the min and max area gates
    largest: np.ndarray | None  # largest contour under max_area, no min gate
    centroid: tuple[float, float] | None
    motion_pixel_fraction: float
    median_grey: float
    is_flare: bool


@dataclass(frozen=True)
class ClipDetection:
    """Per-frame detector output for a whole clip."""

    frames: list[FrameDetection]
    background: np.ndarray
    frame_width: int
    frame_height: int
    warmup_dropped: int
    total_frames: int
    dropped_frames: list[np.ndarray]  # raw frames before the flare-settle cutoff, undetected


def detect_clip(
    video_path: str,
    *,
    max_area_fraction: float = 0.25,
    min_blob_area_fraction: float = 0.0005,
    threshold: int = 18,
    flare_tolerance: float = 3.0,
    max_flare_fraction: float = 0.4,
    max_track_jump_fraction: float = 0.2,
    max_track_miss_frames: int = 5,
) -> ClipDetection | None:
    """Run the background-subtraction detector over one clip, keeping per-frame
    detail. Returns None if the clip has no readable frames.

    Consecutive-frame differencing was tried first and failed on the real
    footage: these cameras change IR gain/illuminator state as a global
    exposure step, and that step is a far bigger inter-frame delta than an
    actual animal. It hid a porcupine on cam15/15454 entirely. Rather than
    drop a fixed warmup window -- flare settles anywhere from frame 1 to
    frame 21 depending on the clip, and can recur mid-clip -- the cutoff is
    measured per clip from `flare_settle_index` and everything up to it is
    dropped before the background is modelled. Blobs larger than
    `max_area_fraction` of the frame are additionally rejected as residual
    illumination change rather than a subject.

    The per-frame `largest`/`centroid` is not simply "the biggest blob this
    frame" -- it is a persistent track (see `track_contour`) that prefers
    continuing the previous frame's identity by bounding-box overlap, then by
    nearest centroid within `max_track_jump_fraction` of the frame diagonal.
    This is what keeps a tracked subject from flipping to a second person/
    animal that happens to have a larger blob in a later frame. The track
    tolerates up to `max_track_miss_frames` consecutive frames with no
    matching candidate (e.g. the subject briefly blends into the background)
    before it is dropped and the next frame re-acquires on the largest blob,
    same as the original stateless behaviour.

    Kept separate from `extract_clip_features` so overlays and diagnostics can
    render exactly what scored a clip rather than a lookalike reimplementation.
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

    all_grays = [
        cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in frames
    ]
    all_medians = [float(np.median(g)) for g in all_grays]
    drop = flare_settle_index(
        all_medians, tolerance=flare_tolerance, max_fraction=max_flare_fraction
    )
    considered = frames[drop:]
    grays = all_grays[drop:]
    medians = all_medians[drop:]

    frame_height, frame_width = considered[0].shape[:2]
    max_area = max_area_fraction * frame_height * frame_width
    min_blob_area = min_blob_area_fraction * frame_height * frame_width

    background = np.median(np.stack(grays), axis=0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    flares = flare_frames(medians, tolerance=flare_tolerance)
    max_jump_distance = max_track_jump_fraction * (frame_width**2 + frame_height**2) ** 0.5

    detections: list[FrameDetection] = []
    track_bbox: tuple[int, int, int, int] | None = None
    track_miss = 0
    for frame_index, (frame, gray) in enumerate(zip(considered, grays, strict=True)):
        diff = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        frame_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in frame_contours if min_blob_area <= cv2.contourArea(c) <= max_area]
        candidates = [c for c in frame_contours if 0 < cv2.contourArea(c) <= max_area]
        contour = track_contour(candidates, track_bbox, max_jump_distance=max_jump_distance)
        if contour is None:
            track_miss += 1
            if track_miss > max_track_miss_frames:
                track_bbox = None
        else:
            track_bbox = _contour_bbox(contour)
            track_miss = 0
        detections.append(
            FrameDetection(
                index=frame_index,
                frame=frame,
                mask=mask,
                all_contours=list(frame_contours),
                blobs=blobs,
                largest=contour,
                centroid=None if contour is None else contour_centroid(contour),
                motion_pixel_fraction=float(np.count_nonzero(mask)) / mask.size,
                median_grey=medians[frame_index],
                is_flare=flares[frame_index],
            )
        )

    return ClipDetection(
        frames=detections,
        background=background,
        frame_width=frame_width,
        frame_height=frame_height,
        warmup_dropped=drop,
        total_frames=total_frames,
        dropped_frames=frames[:drop],
    )


def extract_clip_features(
    video_path: str,
    zone: CameraZone,
    *,
    reference_row: float | None = None,
    max_area_fraction: float = 0.25,
    min_blob_area_fraction: float = 0.0005,
    threshold: int = 18,
    flare_tolerance: float = 3.0,
    max_flare_fraction: float = 0.4,
) -> dict[str, float] | None:
    """Run the detector over one clip and compute features for its largest
    track. Returns None if no motion was detected.
    """
    detection = detect_clip(
        video_path,
        max_area_fraction=max_area_fraction,
        min_blob_area_fraction=min_blob_area_fraction,
        threshold=threshold,
        flare_tolerance=flare_tolerance,
        max_flare_fraction=max_flare_fraction,
    )
    if detection is None:
        return None

    frame_width, frame_height = detection.frame_width, detection.frame_height
    considered = detection.frames

    centroids: list[tuple[float, float]] = []
    blob_areas: list[float] = []
    detected_indices: list[int] = []
    whole_frame_green_ratios: list[float] = []
    frames_detected = 0
    best_contour: np.ndarray | None = None
    best_frame: np.ndarray | None = None
    best_area = -1.0
    whole_frame = _whole_frame_contour(frame_width, frame_height)
    motion_pixel_fraction = 0.0
    blob_count = 0

    for detected in considered:
        whole_frame_green_ratios.append(green_light_ratio(detected.frame, whole_frame))
        # Peak-frame readings, not an average -- a storm/wind frame with motion
        # scattered across many small blobs (bushes, branches) reads very
        # differently from a single compact subject even at the same threshold.
        motion_pixel_fraction = max(motion_pixel_fraction, detected.motion_pixel_fraction)
        blob_count = max(blob_count, len(detected.blobs))
        contour = detected.largest
        if contour is None:
            continue
        area = cv2.contourArea(contour)
        frames_detected += 1
        detected_indices.append(detected.index)
        blob_areas.append(area)
        # Shape features describe the subject at its clearest, not whichever
        # frame happened to be last -- tracks often end on a fading speck.
        if area > best_area:
            best_area = area
            best_contour = contour
            best_frame = detected.frame
        if detected.centroid is not None:
            centroids.append(detected.centroid)

    if best_contour is None or best_frame is None:
        return None

    ref_row = reference_row if reference_row is not None else float(frame_height)
    points = normalized_contour_points(best_contour, frame_width, frame_height)
    track = [(x / frame_width, y / frame_height) for x, y in centroids]
    best_width = float(cv2.boundingRect(best_contour)[2])

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
        "longest_detection_run": longest_detection_run(detected_indices, len(considered)),
        "area_stability": area_stability(blob_areas),
        "normalised_speed": normalised_speed(centroids, best_width),
        "heading_change": heading_change(centroids),
        "fence_crossed": float(track_crosses_fence(track, zone)),
        "median_fence_distance": median_fence_distance(track, zone),
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
