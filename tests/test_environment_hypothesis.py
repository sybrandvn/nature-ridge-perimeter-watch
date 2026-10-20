"""Keep the hypothesis experiment from masking classifier regressions."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.check_environment_hypothesis import (
    evaluate,
    evaluate_production,
    hypothesis_matches,
    integrity_report,
)
from src.classify import classify, classify_detailed


def _record(message_id=1, *, camera="cam15", label="animal", category="incident_candidate"):
    return {
        "channel_id": "source", "message_id": message_id, "camera_id": camera,
        "event_key": f"{camera}|event", "label": label,
        "baseline": {"category": category, "reason": "outside_no_colour"},
        "features": {
            "multi_object_count": 5, "multi_object_outside_fraction_weighted": 0.1,
            "genuine_detection_count": 1, "scored_frame_count": 4,
            "recovered_fraction": 0.75, "long_flare_frames": 2,
        },
    }


def test_clip_regression_fails_even_when_sibling_preserves_event():
    first, second = _record(1), _record(2)
    report = integrity_report(
        [first, second],
        lambda row: "environment_candidate" if row["message_id"] == 1 else "incident_candidate",
    )

    assert report["passed"] is False
    assert report["protected_alerting_events_retained"] == 1
    assert report["protected_alerting_clips_retained"] == 1
    assert any(v["violation"] == "protected_clip_lost" for v in report["violations"])


def test_entire_protected_event_loss_is_reported_separately():
    report = integrity_report([_record()], lambda row: "environment_candidate")

    assert report["passed"] is False
    assert any(v["violation"] == "protected_event_lost" for v in report["violations"])


def test_false_alarm_improvement_does_not_require_relabeling():
    record = _record(label="environment")
    original = deepcopy(record)
    report = integrity_report([record], lambda row: "environment_candidate")

    assert report["passed"] is True
    assert report["historical_environment_urgent_events_before"] == 1
    assert report["historical_environment_urgent_events_after"] == 0
    assert record == original


def test_new_historical_alert_is_a_failure():
    report = integrity_report(
        [_record(label="environment", category="environment_candidate")],
        lambda row: "incident_candidate",
    )

    assert report["passed"] is False
    assert any(v["violation"] == "new_historical_urgent_alert" for v in report["violations"])


def test_startup_only_tags_are_checked_for_behavior_changes():
    report = integrity_report([_record(label=None)], lambda row: "environment_candidate")

    assert report["passed"] is False
    assert any(v["violation"] == "tagged_non_environment_changed" for v in report["violations"])


def test_approved_ambiguous_alert_is_protected_even_with_guard_label():
    report = integrity_report(
        [_record(message_id=7778, camera="cam13", label="guard")],
        lambda row: "environment_candidate",
    )

    assert report["passed"] is False
    assert any(v["violation"] == "approved_alert_not_preserved" for v in report["violations"])


def test_preexisting_approved_alert_gap_stays_visible_without_becoming_a_new_regression():
    report = integrity_report(
        [_record(message_id=7778, camera="cam13", label="guard", category="unclassified")],
        lambda row: "unclassified",
    )

    assert report["passed"] is False
    assert report["regression_passed"] is True
    assert report["violations"] == []
    assert report["baseline_issues"][0]["message_id"] == 7778


def test_unknown_review_case_is_not_counted_as_a_confirmed_false_alarm():
    row = {**_record(label=None), "review_case": True}
    report = integrity_report([row], lambda row: "environment_candidate")

    assert report["historical_tags"] == {}
    assert report["historical_environment_urgent_events_before"] == 0
    assert report["changes"][0]["historical"] is False


def test_missing_protected_coverage_fails():
    with pytest.raises(ValueError, match="Missing protected clips"):
        evaluate([_record()], required_clips=frozenset({("cam15", 123)}))


def test_duplicate_snapshot_rows_fail_instead_of_inflating_coverage():
    with pytest.raises(ValueError, match="Duplicate clip"):
        integrity_report([_record(), _record()], lambda row: "incident_candidate")


def test_changed_production_baseline_cannot_be_mistaken_for_hypothesis_effect():
    record = _record()
    record["features"] = None
    with pytest.raises(ValueError, match="differs from baseline"):
        evaluate([record], required_clips=frozenset())


def test_targeted_hypothesis_does_not_apply_to_another_camera():
    assert hypothesis_matches(_record(camera="cam15"), "combined_cam15")
    assert not hypothesis_matches(_record(camera="cam01"), "combined_cam15")


@pytest.mark.parametrize(
    ("field", "value"),
    [("multi_object_count", 1), ("multi_object_outside_fraction_weighted", 0),
     ("multi_object_outside_fraction_weighted", 0.5), ("genuine_detection_count", 2),
     ("scored_frame_count", 2), ("recovered_fraction", 0.5), ("long_flare_frames", 1)],
)
def test_combined_hypothesis_requires_all_independent_evidence(field, value):
    record = _record()
    record["features"][field] = value

    assert not hypothesis_matches(record, "combined_cam15")


def test_missing_detection_count_is_not_evidence_for_suppression():
    record = _record()
    del record["features"]["genuine_detection_count"]

    assert not hypothesis_matches(record, "combined_cam15")


def test_mixed_motion_with_several_real_detections_remains_eligible():
    record = _record()
    record["features"].update(
        genuine_detection_count=8, scored_frame_count=43,
        multi_object_outside_fraction_weighted=0.39, recovered_fraction=0.674,
    )

    assert hypothesis_matches(record, "mixed_only")
    assert not hypothesis_matches(record, "combined_cam15")


def test_frozen_real_clips_preserve_animals_and_expose_unsafe_ablations():
    path = Path(__file__).parent / "fixtures/environment_hypothesis.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    report = evaluate(records)
    selected = report["variants"]["combined_cam15"]

    assert report["baseline_reproduced"]
    assert selected["regression_passed"]
    assert selected["passed"] is False  # Existing cam13/7778 expectation gap remains visible.
    assert {issue["message_id"] for issue in selected["baseline_issues"]} == {7778}
    assert {change["message_id"] for change in selected["changes"]} == {24395, 24398, 24680}
    assert not any(change["historical"] for change in selected["changes"])
    assert any(
        change["message_id"] == 15454 and change["label"] == "animal"
        for change in report["variants"]["mixed_only"]["changes"]
    )
    assert not report["variants"]["weak_only"]["regression_passed"]


def _real_clips():
    path = Path(__file__).parent / "fixtures/environment_hypothesis.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_production_matches_experiment_on_frozen_real_clips():
    report = evaluate_production(_real_clips())
    result = report["production"]

    assert result["hypothesis_matches_production"]
    assert result["regression_passed"]
    assert not result["passed"]  # Keep the existing cam13 discrepancy visible.
    assert {change["message_id"] for change in result["changes"]} == {24395, 24398, 24680}


@pytest.mark.parametrize("camera", [None, "cam01", "cam01b", "cam16"])
def test_production_rule_requires_explicit_cam15_context(camera):
    row = next(row for row in _real_clips() if row["message_id"] == 24395)

    assert classify(row["features"], camera_id=camera) == "incident_candidate"
    assert classify(row["features"], camera_id="cam15") == "environment_candidate"


@pytest.mark.parametrize("field", ["genuine_detection_count", "scored_frame_count"])
def test_production_rule_preserves_alert_when_new_count_evidence_is_missing(field):
    row = next(row for row in _real_clips() if row["message_id"] == 24395)
    features = {key: value for key, value in row["features"].items() if key != field}

    assert classify(features, camera_id="cam15") == "incident_candidate"


def test_verified_flashlight_keeps_priority_over_mixed_scene_evidence():
    row = next(row for row in _real_clips() if row["message_id"] == 24395)
    features = {**row["features"], "green_light_ratio": 0.2}

    result = classify_detailed(features, camera_id="cam15")

    assert result.category == "guard_candidate"
    assert result.reason == "green_light"
