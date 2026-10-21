"""Try per-object vegetation evidence offline, preserving production and labels.

Build independent references from earlier, same-era/lighting events outside the
entire evaluation corpus. Extract through the production path without its cache,
then measure all startup/scored objects separately using the independent image.
No database, live reference background, threshold or delivery record is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2

from scripts.check_environment_hypothesis import (
    APPROVED_ALERTS,
    URGENT,
    evaluate_production,
    integrity_report,
)
from src import db, scoring
from src.clip_analysis import analyze_clip
from src.config import load_cameras_config, load_thresholds_config
from src.event_keys import message_event_key
from src.features import is_daylight
from src.reference_bg import (
    clip_median_background,
    combine,
    era_of,
    load_manifest,
    sample_evenly,
)
from src.vegetation_probe import ProbeSettings, probe_detection

VARIANTS = ("main_reference_only", "reference_with_subject_veto",
            "bounded_reference_with_subject_veto", "fully_accounted_scene")


def past_reference_pool(
    pool: list[dict[str, Any]], target: dict[str, Any], forbidden_events: set[str],
    *, lookback_days: int = 180,
) -> list[dict[str, Any]]:
    """One clip per earlier event; no same-event, future, era or lighting leakage."""
    timestamp = datetime.fromisoformat(target["timestamp"].replace("Z", "+00:00"))
    earliest = timestamp - timedelta(days=lookback_days)
    events = {}
    for row in pool:
        if (row["camera_id"] != target["camera_id"] or row["era"] != target["era"]
                or row["daylight"] != target["daylight"]
                or row["event_key"] == target["event_key"]
                or row["event_key"] in forbidden_events):
            continue
        observed = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        if earliest <= observed < timestamp:
            previous = events.get(row["event_key"])
            if previous is None or row["timestamp"] > previous["timestamp"]:
                events[row["event_key"]] = row
    return sorted(events.values(), key=lambda row: row["timestamp"])


def hypothesis_matches(probe: dict[str, Any] | None, variant: str) -> bool:
    """Proposals only: missing imagery or unregistered references never match."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    if not probe or not probe["registration"]["usable"]:
        return False
    if not probe["main_reference_confirmed"]:
        return False
    if variant == "main_reference_only":
        return True
    # A linked identity can contain both scenery and a briefly merged subject.
    # Novel observations on the selected track must veto just as other IDs do.
    if probe["novel_appearance_tracks"]:
        return False
    if variant == "bounded_reference_with_subject_veto":
        return bool(probe["main_bounded_reference"])
    if variant == "fully_accounted_scene":
        return (probe["other_unknown_tracks"] == 0 and not probe["startup_unmeasured"]
                and probe["oversized_warmup_objects"] == 0
                and probe["unknown_reference_observations"] == 0
                and probe["novel_observations"] == 0)
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    cv2.setNumThreads(1)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.baseline.read_text().splitlines() if line.strip()]
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/incident_regression.jsonl"
    required = APPROVED_ALERTS | frozenset(
        (r["camera_id"], r["message_id"])
        for line in fixture.read_text().splitlines() for r in [json.loads(line)]
    )
    baseline_check = evaluate_production(rows, required_clips=required)
    if not baseline_check["production"]["regression_passed"]:
        raise ValueError("Current production has new regressions against the frozen baseline")
    cameras = load_cameras_config("config/cameras.yaml")
    thresholds = load_thresholds_config("config/thresholds.yaml")
    original_references = load_manifest("data/reference_bg")
    selected = [r for r in rows if r["camera_id"] in args.camera]
    if args.limit:
        selected = selected[:args.limit]
    forbidden = {r["event_key"] for r in rows}
    conn = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        db.export_labels_jsonl(conn, out / "labels-before.jsonl")
        pool = []
        for raw in conn.execute("SELECT * FROM clips WHERE file_path IS NOT NULL"):
            row = dict(raw)
            if row["camera_id"] not in args.camera or not Path(row["file_path"]).is_file():
                continue
            camera = cameras.by_id(row["camera_id"])
            row.update(era=era_of(camera, row["timestamp"]),
                       daylight=is_daylight(row["timestamp"]),
                       event_key=message_event_key(row["camera_id"], row["caption"],
                                                   row["message_id"]))
            pool.append(row)
    finally:
        conn.close()
    settings = ProbeSettings()
    method = {"source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip(), "camera_scope": args.camera,
        "source": str(args.baseline),
        "source_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "settings": asdict(settings), "reference_lookback_days": 180,
        "reference_min_distinct_events": 5, "reference_max_events": 20,
        "all_evaluation_events_excluded_from_references": True,
        "future_reference_clips_excluded": True, "variants": list(VARIANTS),
        "cache_used": False, "production_modified": False,
        "prototype_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path("src/vegetation_probe.py"))
        },
        "interpretation": "Exploratory discovery-corpus evaluation; no held-out performance claim"}
    (out / "method.json").write_text(json.dumps(method, indent=2) + "\n")

    @lru_cache(maxsize=512)
    def median(path: str):
        return clip_median_background(path)

    observations = []
    diagnostic_variations = []
    for index, row in enumerate(selected, 1):
        camera = cameras.by_id(row["camera_id"])
        target = {**row, "era": era_of(camera, row["timestamp"]),
                  "daylight": is_daylight(row["timestamp"])}
        eligible = past_reference_pool(pool, target, forbidden)
        by_path = {r["file_path"]: r for r in eligible}
        chosen = sample_evenly(list(by_path), limit=20)
        backgrounds = [(by_path[path], median(path)) for path in chosen]
        usable = [(meta, image) for meta, image in backgrounds if image is not None]
        ref = combine([image for _, image in usable]) if len(usable) >= 5 else None
        if ref is not None:
            usable = [(meta, image) for meta, image in usable if image.shape == ref.shape]
            if len(usable) < 5:
                ref = None
        provenance = {"eligible_events": len(eligible),
            "source_events": [{k: meta[k] for k in ("camera_id", "message_id", "event_key",
                              "timestamp", "era", "daylight", "file_path")}
                              for meta, _ in usable],
            "usable": ref is not None}
        if ref is not None:
            digest = hashlib.sha256(ref.tobytes()).hexdigest()
            (out / "references").mkdir(exist_ok=True)
            ref_path = out / "references" / f"{digest}.png"
            cv2.imwrite(str(ref_path), ref)
            provenance.update(sha256=digest, path=str(ref_path))
        observed = {}
        original_detect = scoring.detect_clip

        def capture(*positional, _observed=observed, _detect=original_detect, **keywords):
            _observed["detection"] = _detect(*positional, **keywords)
            return _observed["detection"]

        with patch.object(scoring, "detect_clip", side_effect=capture):
            result = analyze_clip(
                conn=None, channel_id=row["channel_id"], message_id=row["message_id"],
                video_path=row["file_path"], timestamp=row["timestamp"], camera=camera,
                thresholds=thresholds, reference_entries=original_references, use_cache=False,
            )
        if asdict(result.classification) != row["production"]:
            raise ValueError(f"Production drift at {row['camera_id']}/{row['message_id']}")
        if result.features != row["features"]:
            if result.features is None or row["features"] is None:
                raise ValueError(f"Feature availability drift at {row['message_id']}")
            changed = {k: [v, result.features.get(k)] for k, v in row["features"].items()
                       if v != result.features.get(k)}
            if set(changed) - {"global_camera_shift_score"}:
                raise ValueError(f"Decision feature drift: {row['message_id']}: {changed}")
            diagnostic_variations.append({"message_id": row["message_id"], "changes": changed})
        detection = observed.get("detection")
        probe = None if detection is None else probe_detection(
            detection, camera.zone_at(row["timestamp"]), ref, settings,
            threshold=thresholds.motion_thresholds().threshold,
        )
        observations.append({"camera_id": row["camera_id"], "channel_id": row["channel_id"],
            "message_id": row["message_id"], "event_key": row["event_key"],
            "label": row["label"], "classification": asdict(result.classification),
            "reference": provenance, "probe": probe})
        # Flush progress so partial evidence survives an interrupted long replay.
        with (out / "objects.jsonl").open("a" if index > 1 else "w") as output:
            output.write(json.dumps(observations[-1]) + "\n")
        if index % 10 == 0 or index == len(selected):
            print(json.dumps({"processed": index, "total": len(selected)}), flush=True)
    probes = {(r["channel_id"], r["message_id"]): r["probe"] for r in observations}
    records = [{**r, "baseline": r["production"]} for r in rows]
    variants = {}
    for name in VARIANTS:
        def predict(row, variant=name):
            before = row["baseline"]["category"]
            probe = probes.get((row["channel_id"], row["message_id"]))
            return "environment_candidate" if before in URGENT and hypothesis_matches(
                probe, variant
            ) else before
        variants[name] = integrity_report(records, predict)
    conn = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        db.export_labels_jsonl(conn, out / "labels-after.jsonl")
    finally:
        conn.close()
    labels_before = (out / "labels-before.jsonl").read_bytes()
    if labels_before != (out / "labels-after.jsonl").read_bytes():
        raise ValueError("Labels changed during the investigation")
    report = {"method": method, "corpus_clips": len(rows),
        "probed_clips": len(observations), "complete_camera_scope": len(selected) == sum(
            r["camera_id"] in args.camera for r in rows),
        "reference_coverage": dict(Counter("usable" if r["reference"]["usable"]
                                          else "missing" for r in observations)),
        "registered_reference_coverage": sum(bool(r["probe"] and r["probe"]["registration"]
                                                  ["usable"]) for r in observations),
        "labels_preserved": True, "labels_sha256": hashlib.sha256(labels_before).hexdigest(),
        "production_baseline": baseline_check["production"],
        "diagnostic_variations": diagnostic_variations,
        "variants": variants, "production_changed": False}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, result in variants.items():
        print(json.dumps({"variant": name, "changed_clips": len(result["changes"]),
            "regressions": len(result["violations"]), "regression_passed": result[
                "regression_passed"], "environment_events_removed": result[
                "historical_environment_urgent_events_before"] - result[
                "historical_environment_urgent_events_after"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("data/perimeter_watch.db"))
    parser.add_argument("--camera", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.camera = args.camera or ["cam07", "cam12"]
    run(args)


if __name__ == "__main__":
    main()
