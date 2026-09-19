"""Reproduce the 2026-09-19 code-review findings without footage or credentials.

Run with ``uv run python scripts/audit_detection_assumptions.py``. Each probe
reports its expected invariant and observed result; ``holds: false`` identifies
a counterexample, not an automated recommendation to change a threshold.
Add ``--backtest-csv /tmp/labelled.csv`` to measure sibling policies against a
fresh labelled backtest and all its siblings, opening the source DB read-only.
No production configuration, database, labels, or media are changed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts import check_incident_regression as regression  # noqa: E402
from scripts.backtest import _reference_background, summarize_labelled  # noqa: E402
from scripts.rank_candidates import clip_duration_seconds  # noqa: E402
from src import db  # noqa: E402
from src.classify import classify_detailed, classify_event  # noqa: E402
from src.config import CameraZone, load_cameras_config  # noqa: E402
from src.event_keys import event_key as _event_key  # noqa: E402
from src.features import daylight_hint, flare_settle_index, is_daylight  # noqa: E402
from src.motion import (  # noqa: E402
    ClipDetection,
    FrameDetection,
    TrackedObject,
    _bbox_to_rect_contour,
    _contour_bbox,
    reacquire_by_template,
    track_multiple_objects,
)
from src.reference_bg import load_manifest  # noqa: E402
from src.scoring import (  # noqa: E402
    _frame_is_merged,
    extract_clip_features,
    features_from_detection,
    normalized_contour_points,
    rasterized_zone_fractions,
)
from src.zones import outside_pixel_fraction  # noqa: E402


def rectangle(x0, y0, x1, y1):
    return np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], np.int32)


def base_features(**overrides):
    return {
        "green_light_ratio": 0.0,
        "green_light_flicker": 0.0,
        "outside_pixel_fraction": 0.9,
        "median_fence_distance": 0.2,
        "color_fraction": 0.0,
        "blob_count": 1.0,
        "persistence": 0.5,
        "jitter": 1.0,
        "solidity": 0.9,
        **overrides,
    }


def regression_output(*, missing=False, category="insect_candidate"):
    """Exercise the actual regression main with an isolated fixture and DB."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE clips (camera_id TEXT, message_id INTEGER, file_path TEXT)")
    if not missing:
        conn.execute("INSERT INTO clips VALUES ('cam06', 1, 'unused.mp4')")
    entry = {
        "camera_id": "cam06",
        "message_id": 1,
        "event": "synthetic-incident",
        "label": "incident",
        "timestamp": "2026-07-21T20:31:00Z",
    }
    output = io.StringIO()
    exit_code = 0
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "fixture.jsonl"
        fixture.write_text(json.dumps(entry) + "\n")
        media = Path(directory) / "unused.mp4"
        media.touch()
        if not missing:
            conn.execute("UPDATE clips SET file_path=?", (str(media),))
        with (
            patch.object(regression, "FIXTURE", fixture),
            patch.object(regression, "load_app_config", return_value=SimpleNamespace(db_path="")),
            patch.object(regression.db, "connect", return_value=conn),
            patch.object(regression, "load_manifest", return_value=[]),
            patch.object(regression, "extract_clip_features", return_value=None),
            patch.object(regression, "classify", return_value=category),
            contextlib.redirect_stdout(output),
        ):
            try:
                regression.main()
            except SystemExit as error:
                exit_code = error.code
    return {"exit_code": exit_code, "output": output.getvalue().strip()}


def run_probes():
    probes = []

    def record(name, expected, observed, holds):
        probes.append({"name": name, "expected": expected, "observed": observed, "holds": holds})

    zone = CameraZone(((0.5, 0.0), (0.5, 1.0)), "right", 0.0, ())
    contour = rectangle(40, 20, 90, 80)
    vertices = normalized_contour_points(contour, 100, 100)
    mask = np.zeros((100, 100), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    raster_fraction = np.count_nonzero(mask[:, 51:]) / np.count_nonzero(mask)
    vertex_fraction = outside_pixel_fraction(vertices, zone)
    # Inserting collinear vertices preserves the filled polygon exactly.
    resampled = vertices[:2] + [(0.9, 0.3), (0.9, 0.4)] + vertices[2:]
    resampled_fraction = outside_pixel_fraction(resampled, zone)
    area_fraction, _ = rasterized_zone_fractions(vertices, 100, 100, zone)
    resampled_area_fraction, _ = rasterized_zone_fractions(resampled, 100, 100, zone)
    record(
        "outside_area_is_independent_of_contour_encoding",
        "Identical filled shapes have identical area-based fence-side evidence",
        {"four_vertices": vertex_fraction, "six_vertices": resampled_fraction,
         "filled_pixels": raster_fraction,
         "area_feature_four_vertices": area_fraction,
         "area_feature_six_vertices": resampled_area_fraction,
         "four_vertex_category": classify_detailed(base_features(
             outside_pixel_fraction=vertex_fraction,
         )).category,
         "six_vertex_category": classify_detailed(base_features(
             outside_pixel_fraction=resampled_fraction,
         )).category},
        area_fraction == resampled_area_fraction,
    )

    background = np.random.default_rng(1).integers(40, 100, (100, 100), dtype=np.uint8)
    dropped = []
    for x in (10, 25):
        frame = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        frame[10:25, x:x + 12] = 210
        dropped.append(frame)
    settled = FrameDetection(
        0, cv2.cvtColor(background, cv2.COLOR_GRAY2BGR), np.zeros((100, 100), np.uint8),
        [], [], None, None, 0.0, 70.0, False,
    )
    detection = ClipDetection([settled], background, 100, 100, 2, 3, dropped, [None, None])
    for name, ambiguous_zone in (
        ("missing_geometry", replace(zone, fence=None, outside=None)),
        ("above_depth_cutoff", replace(zone, depth_cutoff=0.8)),
    ):
        features = features_from_detection(detection, ambiguous_zone, fps=5, daylight_hint=False)
        result = classify_detailed(features)
        record(
            f"warmup_{name}_is_not_inside_evidence",
            "Unclassifiable warmup motion must not establish an inside guard",
            {"category": result.category, "reason": result.reason,
             "contributing": dict(result.contributing)},
            result.reason != "warmup_dynamic_inside",
        )

    box = (10, 20, 20, 30)
    round_trip = _contour_bbox(_bbox_to_rect_contour(box))
    record("bbox_round_trip", box, round_trip, box == round_trip)

    match = reacquire_by_template(
        np.full((100, 100), 200, np.uint8), np.full((10, 10), 30, np.uint8),
        (30, 30, 40, 40), search_margin=10, match_threshold=0.99,
    )
    record(
        "flat_template_has_no_localization_evidence",
        "No confident location for a textureless template on an unrelated flat image",
        match, match is None,
    )

    def small_box(x):
        return rectangle(x, 20, x + 3, 23)

    tracks = track_multiple_objects(
        [[small_box(10)], [small_box(16)], [], [], [small_box(34)]],
        max_jump_distance=100, max_track_miss_frames=5,
    )
    identities = [[track.track_id for track in frame] for frame in tracks]
    record(
        "constant_velocity_identity_survives_two_misses",
        "The returning object retains ID 0 within the five-frame miss budget",
        identities, identities[-1] == [0],
    )
    tracks = track_multiple_objects(
        [[small_box(10)], [], [small_box(10)]],
        max_jump_distance=100, max_track_miss_frames=5, confirm_frames=2,
    )
    record(
        "confirmation_requires_consecutive_frames",
        "A hit, miss, hit sequence has not supplied two consecutive observations",
        [[track.track_id for track in frame] for frame in tracks], not tracks[-1],
    )

    unrelated_merge = replace(
        detection,
        frames=[replace(settled, largest=rectangle(70, 60, 90, 80), centroid=(80, 70))],
        multi_tracks=[[
            TrackedObject(0, (70, 60, 91, 81)),
            TrackedObject(1, (5, 5, 10, 10), (2,)),
            TrackedObject(2, (6, 5, 11, 10), (1,)),
        ]],
    )
    marked_merged = _frame_is_merged(unrelated_merge, 0)
    record(
        "unrelated_merge_does_not_disqualify_main_subject",
        "Two objects merging elsewhere do not mark the main contour as merged",
        marked_merged, not marked_merged,
    )

    drop = flare_settle_index([30.0] * 18 + [60.0] * 2)
    record(
        "late_flash_does_not_discard_settled_prefix",
        "An already-settled prefix is not opening warmup",
        {"dropped_prefix_frames": drop, "first_change_frame": 18}, drop == 0,
    )
    for category in ("insect_candidate", "neighbour_candidate"):
        outcome = regression_output(category=category)
        record(
            f"regression_rejects_{category}", "Non-alert incident must fail",
            outcome, outcome["exit_code"] != 0,
        )
    outcome = regression_output(missing=True)
    record(
        "regression_requires_fixture_coverage", "Missing protected footage must fail",
        outcome, outcome["exit_code"] != 0,
    )

    before = classify_detailed(base_features())
    after = classify_detailed(base_features(
        multi_object_max_flashlight_ratio=0.1,
        multi_object_dominant_excl_flashlight_has_evidence=1.0,
        multi_object_dominant_excl_flashlight_outside_fraction=1.0,
    ))
    record(
        "unvalidated_coexistence_does_not_override_flashlight",
        "Outside coexistence stays reporting-only until labelled jointly occurring subjects exist",
        {"before": before.category, "after": after.category, "reason": after.reason},
        after.category == "guard_candidate",
    )

    conn = db.connect(":memory:")
    identity = {
        "channel_id": "synthetic", "message_id": 1, "camera_id": "cam06",
        "timestamp": "2026-07-21T20:31:00Z", "caption": "cam06", "source": "backfill",
    }
    db.upsert_clip(conn, **identity, file_path="data/history/cam06/1.mp4")
    # Exactly the metadata-only upsert that src.backfill.run_backfill performs.
    db.upsert_clip(conn, **identity, file_path=None)
    path = db.get_clip(conn, "synthetic", 1)["file_path"]
    conn.close()
    record(
        "metadata_replay_preserves_downloaded_media",
        "data/history/cam06/1.mp4", path, path == "data/history/cam06/1.mp4",
    )
    return probes


def audit_corpus(csv_path, db_path):
    """Compare event policies, including unlabelled siblings of labelled clips.

    The CSV must be a current normal labelled backtest from this checkout.
    Missing sibling features are extracted uncached; the source DB opens read-only.
    Conflicting event labels and unknowns are reported separately, never resolved
    by picking whichever label happens to appear first.
    """
    with Path(csv_path).open(newline="") as source:
        rows = list(csv.DictReader(source))
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        metadata = [dict(row) for row in conn.execute(
            "SELECT c.*, l.label, l.startup_state FROM clips c "
            "LEFT JOIN labels l USING(channel_id, message_id)"
        )]
    finally:
        conn.close()
    by_id = {(str(row["channel_id"]), str(row["message_id"])): row for row in metadata}
    reported = {(row["channel_id"], row["message_id"]): row for row in rows}

    def key(row):
        return (str(row["channel_id"]), _event_key(row["camera_id"], row["caption"])
                or f"message:{row['message_id']}")

    selected_events = set()
    for row_id, row in reported.items():
        meta = by_id[row_id]
        if row["label"] != meta["label"]:
            raise ValueError(f"CSV and live label disagree for {meta['camera_id']}/{row_id[1]}")
        selected_events.add(key(meta))
    groups = defaultdict(list)
    for row in metadata:
        if key(row) in selected_events:
            groups[key(row)].append(row)

    cameras = load_cameras_config("config/cameras.yaml")
    entries = load_manifest("data/reference_bg")
    alert = {"animal_candidate", "incident_candidate"}
    positive = {"animal", "incident"}
    counts = defaultdict(Counter)
    metrics = {policy: Counter(tp=0, fp=0, fn=0, tn=0) for policy in (
        "any_sibling", "longest", "longest_guard_override",
    )}
    changes, misses, conflicts, extras = [], [], [], []
    for (_channel, event), siblings in groups.items():
        labels = {row["label"] for row in siblings if row["label"] is not None}
        label = next(iter(labels)) if len(labels) == 1 else "conflict"
        analyzed = []
        for meta in siblings:
            row_id = (str(meta["channel_id"]), str(meta["message_id"]))
            if row_id in reported:
                category, reason = reported[row_id]["category"], reported[row_id]["reason"]
            else:
                if not meta["file_path"] or not Path(meta["file_path"]).is_file():
                    raise ValueError(f"Missing sibling media: {meta['camera_id']}/{row_id[1]}")
                camera = cameras.by_id(meta["camera_id"])
                if camera is None:
                    raise ValueError(f"Unknown camera {meta['camera_id']}")
                features = extract_clip_features(
                    meta["file_path"], camera.zone_at(meta["timestamp"]),
                    daylight_hint=daylight_hint(meta["timestamp"]),
                    reference_background=_reference_background(
                        entries, "data/reference_bg", camera, meta["timestamp"],
                    ),
                )
                if features is not None:
                    features["is_daylight"] = float(is_daylight(meta["timestamp"]))
                result = classify_detailed(features)
                category, reason = result.category, result.reason
                extras.append(f"{meta['camera_id']}/{meta['message_id']}")
            analyzed.append({
                "camera_id": meta["camera_id"], "message_id": meta["message_id"],
                "label": meta["label"], "category": category, "reason": reason,
                "duration": clip_duration_seconds(meta["file_path"]),
            })
        union = classify_event(row["category"] for row in analyzed)
        longest = max(analyzed, key=lambda row: (row["duration"], row["message_id"]))
        policies = {
            "any_sibling": union,
            "longest": longest["category"],
            "longest_guard_override": (
                "guard_candidate" if longest["category"] == "guard_candidate" else union
            ),
        }
        counts[label]["events"] += 1
        for policy, category in policies.items():
            counts[label][f"{policy}_alerts"] += category in alert
            if label not in {"unknown", "conflict"}:
                truth, fired = label in positive, category in alert
                bucket = ("tp" if truth else "fp") if fired else ("fn" if truth else "tn")
                metrics[policy][bucket] += 1
        event_report = {"event": event, "label": label, "clips": analyzed}
        if label == "conflict":
            conflicts.append(event_report)
        if label in positive and union not in alert:
            misses.append(event_report)
        if len({category in alert for category in policies.values()}) > 1:
            changes.append({**event_report, "policies": policies})

    return {
        "n_labelled_clips": len(rows), "n_events": len(groups),
        "additional_siblings_extracted": extras,
        "known_clip_metrics": summarize_labelled([r for r in rows if r["label"] != "unknown"]),
        "event_counts": counts, "known_event_confusion": metrics,
        "conflicting_events": conflicts, "missed_protected_events": misses,
        "policy_changes": changes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-csv", help="Current normal labelled backtest CSV")
    parser.add_argument("--db", default="data/perimeter_watch.db")
    args = parser.parse_args()
    report = {"probes": run_probes()}
    if args.backtest_csv:
        report["corpus"] = audit_corpus(args.backtest_csv, args.db)
    print(json.dumps(report, indent=2))
