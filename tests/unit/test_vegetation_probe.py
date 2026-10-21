"""Offline evidence must not manufacture background confidence or lose subjects."""

import cv2
import numpy as np
import pytest

from scripts.analysis.probe_vegetation import hypothesis_matches, past_reference_pool
from src.config import CameraZone
from src.motion import ClipDetection, FrameDetection, _bbox_to_rect_contour
from src.scoring import WarmupMotionAnalysis, WarmupMotionObject
from src.vegetation_probe import (
    align_reference,
    patch_similarity,
    probe_detection,
    summarize_object,
    trajectory_evidence,
)


def _observation(index, x, *, similarity=0.98, phase="scored", selected=True):
    return {"index": index, "bbox": (x, 5, x + 8, 13), "similarity": similarity,
            "phase": phase, "selected_direct": selected}


def test_reference_resemblance_is_invariant_to_brightness_scale():
    reference = np.random.default_rng(7).integers(20, 100, (30, 40), dtype=np.uint8)
    current = (reference.astype(float) * 1.5 + 20).astype(np.uint8)

    assert patch_similarity(current, reference, (5, 5, 20, 20),
                            np.ones_like(reference, dtype=bool)) > 0.99


@pytest.mark.parametrize("bbox", [(0, 0, 3, 3), (-1, 0, 8, 8), (0, 0, 41, 30)])
def test_tiny_and_out_of_frame_patches_are_unknown(bbox):
    image = np.random.default_rng(7).integers(0, 255, (30, 40), dtype=np.uint8)

    assert patch_similarity(image, image, bbox, np.ones_like(image, dtype=bool)) is None


def test_equal_black_patches_are_not_confident_scenery():
    image = np.zeros((30, 40), np.uint8)

    assert patch_similarity(image, image, (0, 0, 10, 10),
                            np.ones_like(image, dtype=bool)) is None


def test_alignment_border_has_no_reference_evidence():
    image = np.random.default_rng(7).integers(0, 255, (30, 40), dtype=np.uint8)
    valid = np.ones_like(image, dtype=bool)
    valid[:, :3] = False

    assert patch_similarity(image, image, (0, 0, 10, 10), valid) is None


def test_missing_and_flat_references_do_not_register():
    image = np.zeros((30, 40), np.uint8)

    assert not align_reference(None, image)[2]["usable"]
    assert not align_reference(image, image)[2]["usable"]


def test_single_observation_and_gaps_cannot_establish_swaying():
    assert not trajectory_evidence([_observation(0, 0)])["has_evidence"]
    assert not trajectory_evidence([_observation(i * 2, i * 3) for i in range(5)])[
        "has_evidence"
    ]


def test_merged_identity_predictions_do_not_establish_a_returning_trajectory():
    observations = [_observation(i, x) for i, x in enumerate([0, 8, 0, 8, 0])]
    observations[2]["merged"] = True

    assert not trajectory_evidence(observations)["has_evidence"]
    assert summarize_object(observations)["observed_frames"] == 4


def test_monotone_subject_and_returning_motion_are_distinct():
    walking = trajectory_evidence([_observation(i, i * 4) for i in range(5)])
    returning = trajectory_evidence([_observation(i, x) for i, x in enumerate([0, 8, 0, 8, 0])])

    assert walking["has_evidence"] and not walking["bounded_return"]
    assert returning["bounded_return"] and returning["reversals"] == 3


def test_unknown_reference_observations_cannot_inflate_match_fraction():
    observations = [_observation(i, 5, similarity=0.99 if i < 3 else None) for i in range(5)]

    assert not summarize_object(observations)["reference_confirmed"]


def test_scenery_reference_does_not_imply_a_vegetation_species():
    summary = summarize_object([_observation(i, 5) for i in range(5)])

    assert summary["reference_confirmed"]
    assert summary["trajectory"]["stationary"]
    assert summary["bounded_reference_track"]


def _pool_row(message, event, **changes):
    return {"camera_id": "cam07", "message_id": message, "event_key": event,
            "timestamp": "2026-01-01T12:00:00Z", "era": None, "daylight": True,
            **changes}


def test_reference_selection_excludes_evaluation_events_siblings_future_and_other_eras():
    target = _pool_row(99, "target", timestamp="2026-02-01T12:00:00Z")
    pool = [_pool_row(1, "independent"), _pool_row(2, "independent"),
            _pool_row(3, "target"), _pool_row(4, "another_test_event"),
            _pool_row(5, "future", timestamp="2026-02-02T12:00:00Z"),
            _pool_row(6, "other_era", era="new"),
            _pool_row(7, "night", daylight=False),
            _pool_row(8, "other_camera", camera_id="cam12"),
            _pool_row(9, "too_old", timestamp="2025-01-01T12:00:00Z")]

    result = past_reference_pool(pool, target, {"another_test_event"})

    assert len(result) == 1 and result[0]["event_key"] == "independent"


def test_subject_veto_preserves_an_independent_object_even_when_main_matches_scenery():
    probe = {"registration": {"usable": True}, "main_reference_confirmed": True,
             "main_bounded_reference": True, "novel_appearance_tracks": 1}

    assert hypothesis_matches(probe, "main_reference_only")
    assert not hypothesis_matches(probe, "reference_with_subject_veto")
    assert not hypothesis_matches(probe, "bounded_reference_with_subject_veto")
    assert not hypothesis_matches(probe, "fully_accounted_scene")


def test_unknown_secondary_objects_block_the_strict_scene_proposal():
    probe = {"registration": {"usable": True}, "main_reference_confirmed": True,
             "novel_appearance_tracks": 0, "other_unknown_tracks": 1,
             "startup_unmeasured": False, "oversized_warmup_objects": 0}

    assert not hypothesis_matches(probe, "fully_accounted_scene")


def _frame(index, image, contours, largest=None, *, recovered=False):
    centroid = None if largest is None else (9.0, 9.0)
    return FrameDetection(index, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                          np.zeros_like(image), contours, contours, largest,
                          centroid, 0.0, 100.0, False, recovered=recovered)


def test_recovered_boxes_never_establish_observed_motion():
    image = np.random.default_rng(7).integers(20, 200, (48, 80), dtype=np.uint8)
    contour = _bbox_to_rect_contour((5, 5, 13, 13))
    frames = [_frame(0, image, [contour], contour)] + [
        _frame(i, image, [], contour, recovered=True) for i in range(1, 7)
    ]
    detection = ClipDetection(frames, image, 80, 48, 0, 7, [], [])
    zone = CameraZone(((0.5, 0), (0.5, 1)), "right", 0.0, ())

    probe = probe_detection(detection, zone, image)

    assert probe["selected_direct_observations"] == 1
    assert sum(obj["observed_frames"] for obj in probe["objects"]) == 1
    assert not probe["main_reference_confirmed"]


def test_startup_subject_remains_separate_from_later_scenery_track(monkeypatch):
    background = np.random.default_rng(4).integers(20, 200, (48, 80), dtype=np.uint8)
    leaf_box = (5, 5, 13, 13)
    person_box = (55, 25, 65, 35)
    corrected = [background.copy() for _ in range(3)]
    for frame in corrected[1:]:
        x0, y0, x1, y1 = person_box
        frame[y0:y1, x0:x1] = 255 - frame[y0:y1, x0:x1]
    warmup_objects = tuple(
        WarmupMotionObject(i, box, 64.0, "not_selected")
        for i in (1, 2) for box in (leaf_box, person_box)
    )
    monkeypatch.setattr("src.vegetation_probe.warmup_motion_analysis", lambda *_a, **_k:
        WarmupMotionAnalysis({}, tuple(corrected), warmup_objects))
    leaf = _bbox_to_rect_contour(leaf_box)
    frames = [_frame(i, background, [leaf], leaf) for i in range(3)]
    detection = ClipDetection(frames, cv2.GaussianBlur(background, (5, 5), 0), 80, 48,
                              3, 6, [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                                     for f in corrected], [None] * 3)
    zone = CameraZone(((0.5, 0), (0.5, 1)), "right", 0.0, ())

    probe = probe_detection(detection, zone, detection.background.copy())

    assert probe["main_reference_confirmed"]
    assert probe["cross_boundary_tracks"] == 1
    assert probe["warmup_novel_appearance_tracks"] == 1
    assert probe["other_novel_appearance_tracks"] == 1
    assert probe["mixed_scene"]
    assert not hypothesis_matches(probe, "reference_with_subject_veto")
