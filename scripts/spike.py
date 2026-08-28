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
  2. Download a clip subset for those cameras (requires live Telegram access;
     not implemented here -- clips are expected to already exist locally,
     referenced via clips.file_path, e.g. from a manual export).
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
    jitter,
    path_length,
    persistence,
    row_normalised_area,
    saturation_ratio,
    solidity,
)
from src.zones import far_side_pixel_fraction  # noqa: E402

FEATURE_COLUMNS = (
    "channel_id",
    "message_id",
    "camera_id",
    "label",
    "far_side_pixel_fraction",
    "aspect_ratio",
    "solidity",
    "saturation_ratio",
    "row_normalised_area",
    "edge_density",
    "path_length",
    "jitter",
    "persistence",
)


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    """Largest external contour in a binary motion mask, or None if empty.

    This is the "throwaway per-clip detector" referenced in the module
    docstring: good enough to validate feature separability, not a claim
    about the eventual production motion pipeline.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def extract_clip_features(
    video_path: str, zone: CameraZone, *, reference_row: float | None = None
) -> dict[str, float] | None:
    """Run a simple frame-differencing detector over one clip and compute
    features for its largest track. Returns None if no motion was detected.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        prev_gray: np.ndarray | None = None
        centroids: list[tuple[float, float]] = []
        frames_detected = 0
        total_frames = 0
        last_contour: np.ndarray | None = None
        last_frame: np.ndarray | None = None
        frame_height = frame_width = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1
            frame_height, frame_width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                contour = largest_contour(mask)
                if contour is not None and cv2.contourArea(contour) > 0:
                    frames_detected += 1
                    last_contour = contour
                    last_frame = frame
                    centroid = contour_centroid(contour)
                    if centroid is not None:
                        centroids.append(centroid)
            prev_gray = gray

        if last_contour is None or last_frame is None or total_frames == 0:
            return None

        ref_row = reference_row if reference_row is not None else float(frame_height)
        points = normalized_contour_points(last_contour, frame_width, frame_height)

        return {
            "far_side_pixel_fraction": far_side_pixel_fraction(points, zone),
            "aspect_ratio": aspect_ratio(last_contour),
            "solidity": solidity(last_contour),
            "saturation_ratio": saturation_ratio(last_frame, last_contour),
            "row_normalised_area": row_normalised_area(last_contour, ref_row),
            "edge_density": edge_density(last_frame, last_contour),
            "path_length": path_length(centroids),
            "jitter": jitter(centroids),
            "persistence": persistence(frames_detected, total_frames),
        }
    finally:
        cap.release()


def iter_labelled_clips_with_files(conn, *, camera_id: str) -> Iterator[dict[str, Any]]:
    """Labelled clips for one camera that have a local file_path to extract from."""
    for row in db.iter_clips(conn, camera_id=camera_id):
        if not row["file_path"]:
            continue
        label_row = db.get_label(conn, row["channel_id"], row["message_id"])
        if label_row is None:
            continue
        yield {
            "channel_id": row["channel_id"],
            "message_id": row["message_id"],
            "camera_id": row["camera_id"],
            "file_path": row["file_path"],
            "label": label_row["label"],
        }


def run_spike(conn, cameras: CamerasConfig, *, camera_id: str) -> list[dict[str, Any]]:
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

    rows = run_spike(conn, cameras_cfg, camera_id=args.camera)
    conn.close()

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} labelled feature rows to {args.out}")


if __name__ == "__main__":
    main()
