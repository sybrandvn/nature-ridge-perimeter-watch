"""Replay an experimental environment gate against a frozen classifier baseline.

This is an offline experiment, not a production classifier change. Snapshot rows
contain clip identity, label/startup metadata, baseline category/reason, and cold
features augmented with genuine_detection_count/scored_frame_count. Counts must
come from detector observations, never reconstructed from rounded persistence.

Run:
    python -m scripts.check_environment_hypothesis --baseline snapshot.jsonl --out result.json

The cam15-only hypothesis is fixed before evaluating historical labels. Broader
ablations show why mixed activity or sparse detections alone are insufficient.
All changes are reported; a surviving sibling cannot conceal a protected clip
loss. No labels, production records, thresholds, or delivery state are written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.classify import classify_detailed, classify_event

URGENT = frozenset({"animal_candidate", "incident_candidate"})
PROTECTED_LABELS = frozenset({"animal", "incident"})
# Explicit user decisions recorded in docs/next_agent_handoff.md.
APPROVED_ALERTS = frozenset({("cam05", 9694), ("cam05", 9695), ("cam07", 19288),
                             ("cam12", 9162), ("cam13", 7778), ("cam07", 4487)})
VARIANTS = (
    "mixed_only", "weak_only", "combined_all_cameras",
    "combined_uncalibrated", "combined_cam15",
)


def hypothesis_matches(row: Mapping[str, Any], variant: str) -> bool:
    """Evaluate the preregistered conjunction; missing evidence never matches."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown hypothesis: {variant}")
    features = row.get("features") or {}
    mixed = (
        features.get("multi_object_count", 0) >= 2
        and 0 < features.get("multi_object_outside_fraction_weighted", 1) < 0.5
    )
    weak = (
        features.get("genuine_detection_count", -1) == 1
        and features.get("scored_frame_count", 0) >= 3
        and features.get("recovered_fraction", 0) >= 2 / 3 - 1e-9
    )
    flare = features.get("long_flare_frames", 0) >= 2
    if variant == "mixed_only":
        return mixed
    if variant == "weak_only":
        return weak
    combined = mixed and weak and flare
    if variant == "combined_all_cameras":
        return combined
    if variant == "combined_uncalibrated":
        return combined and features.get("uncalibrated", 0) == 1
    return combined and row["camera_id"] == "cam15"


def integrity_report(
    records: Sequence[Mapping[str, Any]],
    predict: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    """Compare clips and events, retaining every regression and coverage error."""
    changes: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    baseline_issues: list[dict[str, Any]] = []
    events: dict[str, list[tuple[Mapping[str, Any], str]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    labelled = Counter()
    positive_clips = positive_clips_retained = 0
    for row in records:
        identity = (row["channel_id"], row["message_id"])
        if identity in seen:
            raise ValueError(f"Duplicate clip in snapshot: {identity}")
        seen.add(identity)
        before = row["baseline"]["category"]
        after = predict(row)
        events[row["event_key"]].append((row, after))
        historical = not row.get("review_case", False)
        if historical:
            labelled[str(row.get("label"))] += 1
        entry = {"camera_id": row["camera_id"], "message_id": row["message_id"],
                 "event_key": row["event_key"], "label": row.get("label"),
                 "before": before, "after": after, "historical": historical}
        if after != before:
            changes.append(entry)
        if historical and row.get("label") in PROTECTED_LABELS and before in URGENT:
            positive_clips += 1
            positive_clips_retained += int(after in URGENT)
            if after not in URGENT:
                violations.append({**entry, "violation": "protected_clip_lost"})
        if historical and row.get("label") != "environment" and after != before:
            violations.append({**entry, "violation": "tagged_non_environment_changed"})
        if historical and before not in URGENT and after in URGENT:
            violations.append({**entry, "violation": "new_historical_urgent_alert"})
        if (row["camera_id"], row["message_id"]) in APPROVED_ALERTS:
            if before not in URGENT:
                baseline_issues.append({**entry, "issue": "approved_alert_missing_in_baseline"})
            elif after not in URGENT:
                violations.append({**entry, "violation": "approved_alert_not_preserved"})

    event_changes = []
    protected_events = protected_events_retained = 0
    environment_events_before = environment_events_after = 0
    for key, siblings in events.items():
        before = classify_event(row["baseline"]["category"] for row, _ in siblings)
        after = classify_event(category for _, category in siblings)
        labels = {row.get("label") for row, _ in siblings if row.get("label")}
        historical = all(not row.get("review_case", False) for row, _ in siblings)
        if historical and labels & PROTECTED_LABELS and before in URGENT:
            protected_events += 1
            protected_events_retained += int(after in URGENT)
            if after not in URGENT:
                violations.append({"event_key": key, "violation": "protected_event_lost"})
        if historical and labels == {"environment"}:
            environment_events_before += int(before in URGENT)
            environment_events_after += int(after in URGENT)
        if before != after:
            event_changes.append({"event_key": key, "labels": sorted(labels),
                                  "before": before, "after": after, "historical": historical})
    return {
        "passed": not violations and not baseline_issues,
        "regression_passed": not violations,
        "baseline_issues": baseline_issues,
        "historical_tags": dict(labelled),
        "protected_alerting_clips": positive_clips,
        "protected_alerting_clips_retained": positive_clips_retained,
        "protected_alerting_events": protected_events,
        "protected_alerting_events_retained": protected_events_retained,
        "historical_environment_urgent_events_before": environment_events_before,
        "historical_environment_urgent_events_after": environment_events_after,
        "changes": changes, "event_changes": event_changes, "violations": violations,
    }


def evaluate(
    records: Sequence[Mapping[str, Any]],
    *,
    required_clips: frozenset[tuple[str, int]] = APPROVED_ALERTS,
) -> dict[str, Any]:
    """Check baseline reproducibility before trusting any experimental result."""
    if not records:
        raise ValueError("The baseline must contain clips")
    present = {(row["camera_id"], row["message_id"]) for row in records}
    missing = required_clips - present
    if missing:
        raise ValueError(f"Missing protected clips in baseline: {sorted(missing)}")
    mismatches = []
    for row in records:
        result = classify_detailed(row.get("features"))
        if (result.category, result.reason) != (
            row["baseline"]["category"], row["baseline"]["reason"]
        ):
            mismatches.append({"camera_id": row["camera_id"], "message_id": row["message_id"],
                               "baseline": row["baseline"], "current_category": result.category,
                               "current_reason": result.reason})
    if mismatches:
        raise ValueError(f"Production classifier differs from baseline: {mismatches}")
    reports = {}
    for variant in VARIANTS:
        def predict(row: Mapping[str, Any], variant: str = variant) -> str:
            category = row["baseline"]["category"]
            return (
                "environment_candidate"
                if category in URGENT and hypothesis_matches(row, variant) else category
            )
        reports[variant] = integrity_report(records, predict)
    return {"clip_count": len(records), "baseline_reproduced": True,
            "selected_variant": "combined_cam15", "variants": reports,
            "production_changed_by_tool": False}


def evaluate_production(
    records: Sequence[Mapping[str, Any]],
    *,
    required_clips: frozenset[tuple[str, int]] = APPROVED_ALERTS,
) -> dict[str, Any]:
    """Verify that the implemented camera-scoped rule matches the frozen experiment."""
    experiment = evaluate(records, required_clips=required_clips)
    predictions = {}
    mismatches = []
    for row in records:
        result = classify_detailed(row.get("features"), camera_id=row["camera_id"])
        predictions[(row["channel_id"], row["message_id"])] = result.category
        before = row["baseline"]["category"]
        expected = (
            "environment_candidate"
            if before in URGENT and hypothesis_matches(row, "combined_cam15") else before
        )
        expected_reason = (
            "cam15_mixed_environment" if expected != before else row["baseline"]["reason"]
        )
        if (result.category, result.reason) != (expected, expected_reason):
            mismatches.append({"camera_id": row["camera_id"], "message_id": row["message_id"],
                               "expected_category": expected, "expected_reason": expected_reason,
                               "category": result.category, "reason": result.reason})
    production = integrity_report(
        records, lambda row: predictions[(row["channel_id"], row["message_id"])],
    )
    production["hypothesis_matches_production"] = not mismatches
    production["implementation_mismatches"] = mismatches
    if mismatches:
        production["regression_passed"] = production["passed"] = False
    return {"clip_count": len(records), "baseline_reproduced_without_camera_scope": True,
            "production": production, "experiment": experiment,
            "production_changed_by_tool": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--production", action="store_true", help="check the implemented rule")
    parser.add_argument(
        "--protected-fixture", type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/incident_regression.jsonl",
    )
    args = parser.parse_args()
    records = [json.loads(line) for line in args.baseline.read_text().splitlines() if line.strip()]
    required = APPROVED_ALERTS | frozenset(
        (row["camera_id"], row["message_id"])
        for line in args.protected_fixture.read_text().splitlines() if line.strip()
        for row in [json.loads(line)]
    )
    report = (
        evaluate_production(records, required_clips=required)
        if args.production else evaluate(records, required_clips=required)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    results = {"production": report["production"]} if args.production else report["variants"]
    for name, result in results.items():
        print(f"{name}: {'PASS' if result['passed'] else 'REJECT'}; "
              f"{len(result['changes'])} changed clips; {len(result['violations'])} regressions; "
              f"{len(result['baseline_issues'])} baseline issues")
    selected = report["production"] if args.production else results[report["selected_variant"]]
    raise SystemExit(0 if selected["passed"] else 1)


if __name__ == "__main__":
    main()
