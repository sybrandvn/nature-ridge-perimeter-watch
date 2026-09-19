"""Phase 0c: CV feasibility spike (gate 2).

Extracts candidate discriminating features to a flat CSV for a small,
hand-labelled clip sample, so a human can check whether any threshold
combination separates guard/environment from animal/incident before the rest
of the pipeline (motion.py, classify.py, backtester.py) gets built on top of a
false premise.

This began as a throwaway per-clip detector for feasibility work. Detection and
tracking now live in ``src.motion`` and reusable scoring in ``src.scoring``;
this module owns the labelled-corpus report schema and CSV command.

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

from src import db  # noqa: E402
from src.config import (  # noqa: E402
    CamerasConfig,
    load_app_config,
    load_cameras_config,
)
from src.features import is_daylight, time_of_day  # noqa: E402
from src.scoring import extract_clip_features  # noqa: E402

FEATURE_COLUMNS = (
    "channel_id",
    "message_id",
    "camera_id",
    "label",
    "time_of_day",
    "is_daylight",
    "scored_motion_present",
    "outside_pixel_fraction",
    "zone_classifiable_fraction",
    "outside_frame_fraction",
    "multi_object_outside_fraction_weighted",
    "multi_object_dominant_outside_fraction",
    "multi_object_count",
    "multi_object_flashlight_track_count",
    "multi_object_max_flashlight_ratio",
    "multi_object_dominant_excl_flashlight_outside_fraction",
    "multi_object_dominant_excl_flashlight_has_evidence",
    "multi_object_person_track_count",
    "multi_object_animal_track_count",
    "multi_object_artifact_track_count",
    "multi_object_type_has_evidence",
    "aspect_ratio",
    "solidity",
    "saturation_ratio",
    "color_fraction",
    "green_light_ratio",
    "green_light_flicker",
    "whole_frame_green_ratio",
    "warmup_flashlight_ratio",
    "warmup_outside_fraction",
    "warmup_dynamic_frame_fraction",
    "warmup_dynamic_outside_fraction",
    "flashlight_subject_fraction",
    "row_normalised_area",
    "edge_density",
    "blob_frame_fraction",
    "blob_white_fraction",
    "blob_black_white_balance",
    "rectangular_black_white_balance",
    "global_camera_shift_score",
    "long_flare_frames",
    "post_flash_red_shift",
    "path_length",
    "jitter",
    "persistence",
    "motion_pixel_fraction",
    "blob_count",
    "motion_pixel_fraction_median",
    "blob_count_median",
    "longest_detection_run",
    "area_stability",
    "normalised_speed",
    "heading_change",
    "flow_direction_coherence",
    "flow_direction_coherence_has_evidence",
    "fence_crossed",
    "median_fence_distance",
    "recovered_fraction",
    "reverse_filled_fraction",
    "terminal_reverse_seed",
    "scenery_motion_fraction",
    "has_reference_background",
    "implausible_height_fraction",
    "off_plane_fraction",
    "height_consistency",
    "depth_progression",
    "depth_range_m",
    "subject_height_m",
    "subject_width_m",
    "subject_area_m2",
    "metric_aspect",
    "distance_median_m",
    "speed_mps",
    "uncalibrated",
)


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
        features = extract_clip_features(clip["file_path"], camera.zone_at(clip["timestamp"]))
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
                "is_daylight": is_daylight(clip["timestamp"]),
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
