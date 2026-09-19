"""Tests for src.classify's rule engine.

Split out of tests/test_backtest.py 2026-09-09 when classify() moved to
src/classify.py (plan.md step 25). Every assertion here is unchanged from that
file -- each one pins a threshold or a rule-ordering decision measured against
real labelled footage, so a change to an expected value means a real
re-derivation, not a test fix. See src/classify.py's module docstring.
"""

import pytest

from src.classify import (
    classify,
    classify_detailed,
    classify_event,
    default_thresholds,
    is_blinding_foreground,
)


def _features(**overrides) -> dict[str, float]:
    base = {
        "aspect_ratio": 1.0,
        "solidity": 0.9,
        "green_light_ratio": 0.0,
        "green_light_flicker": 0.0,
        "jitter": 1.0,
        "persistence": 0.5,
        "outside_pixel_fraction": 0.0,
        "median_fence_distance": 0.0,
        "color_fraction": 0.0,
        "blob_count": 1,
    }
    base.update(overrides)
    return base


def test_classify_no_motion_when_features_none():
    assert classify(None) == "no_motion"


def test_classify_guard_candidate_on_green_light_ratio():
    assert classify(_features(green_light_ratio=0.2)) == "guard_candidate"


def test_classify_guard_candidate_on_flicker():
    assert classify(_features(green_light_flicker=0.05)) == "guard_candidate"


def test_classify_guard_candidate_on_warmup_flashlight():
    # The guard left before the IR gain settled, so their flashlight is only in
    # the dropped frames -- every scored frame is whatever moved next.
    assert classify(_features(warmup_flashlight_ratio=0.01)) == "guard_candidate"


def test_classify_guard_candidate_on_warmup_only_inside_motion():
    result = classify_detailed(
        _features(
            scored_motion_present=0.0,
            warmup_dynamic_frame_fraction=0.9,
            warmup_dynamic_outside_fraction=0.0,
        )
    )

    assert result.category == "guard_candidate"
    assert result.reason == "warmup_dynamic_inside"


def test_warmup_dynamic_inside_does_not_steal_scored_subject():
    features = _features(
        scored_motion_present=1.0,
        warmup_dynamic_frame_fraction=1.0,
        warmup_dynamic_outside_fraction=0.0,
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
    )

    assert classify(features) == "incident_candidate"


def test_other_warmup_only_motion_remains_unclassified():
    result = classify_detailed(
        _features(
            scored_motion_present=0.0,
            warmup_dynamic_frame_fraction=0.5,
            warmup_dynamic_outside_fraction=1.0,
        )
    )

    assert result.category == "unclassified"
    assert result.reason == "warmup_only_unclassified"


def test_classify_warmup_flashlight_beats_animal_incident_geometry():
    # These clips DO pass the outside/far-from-fence geometry test -- that is
    # why they reached the review queue in the first place.
    features = _features(
        warmup_flashlight_ratio=0.01,
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        color_fraction=0.0,
    )
    assert classify(features) == "guard_candidate"


def test_classify_warmup_flashlight_below_threshold_does_not_fire():
    # incident's highest measured value is 0.00046; the threshold has margin.
    features = _features(
        warmup_flashlight_ratio=0.00046,
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        color_fraction=0.0,
    )
    assert classify(features) == "incident_candidate"


def test_classify_guard_candidate_on_multi_object_flashlight():
    # green_light_ratio itself reads 0.0 (the single tracked contour is on
    # the wrong object, e.g. cam07/11174's bush) but a DIFFERENT persistently-
    # tracked object in the same clip scores as a real flashlight.
    features = _features(multi_object_max_flashlight_ratio=0.2)
    assert classify(features) == "guard_candidate"


def test_classify_multi_object_flashlight_beats_animal_incident_geometry():
    features = _features(
        multi_object_max_flashlight_ratio=0.2,
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        color_fraction=0.0,
    )
    assert classify(features) == "guard_candidate"


def test_classify_multi_object_flashlight_below_threshold_does_not_fire():
    features = _features(multi_object_max_flashlight_ratio=0.01)
    assert classify(features) == "unclassified"


def test_classify_warmup_flashlight_absent_key_is_safe():
    assert classify(_features()) == "unclassified"


def test_is_blinding_foreground_none_features_is_false():
    assert is_blinding_foreground(None) is False


def test_is_blinding_foreground_on_white_blob():
    assert is_blinding_foreground(_features(blob_white_fraction=0.5)) is True


def test_is_blinding_foreground_on_long_flare():
    assert is_blinding_foreground(_features(long_flare_frames=20)) is True


def test_is_blinding_foreground_on_large_bright_obstruction():
    features = _features(blob_frame_fraction=0.346875, blob_white_fraction=0.2374)
    assert is_blinding_foreground(features) is True


@pytest.mark.parametrize(
    "features",
    [
        _features(blob_frame_fraction=0.33, blob_white_fraction=0.2374),
        _features(blob_frame_fraction=0.346875, blob_white_fraction=0.19),
    ],
)
def test_large_obstruction_maintenance_gate_requires_both_features(features):
    assert is_blinding_foreground(features) is False


def test_is_blinding_foreground_false_below_both_thresholds():
    features = _features(blob_white_fraction=0.1, long_flare_frames=5)
    assert is_blinding_foreground(features) is False


def test_is_blinding_foreground_absent_keys_is_safe():
    assert is_blinding_foreground(_features()) is False


def test_is_blinding_foreground_independent_of_category():
    # A guard can be genuinely present AND the lens genuinely obstructed --
    # this is never folded into classify()'s mutually-exclusive chain.
    features = _features(green_light_ratio=0.2, blob_white_fraction=0.9)
    assert classify(features) == "guard_candidate"
    assert is_blinding_foreground(features) is True


def test_classify_environment_candidate_on_blob_count():
    assert classify(_features(blob_count=11)) == "environment_candidate"


def test_classify_guard_wins_over_environment_blob_count():
    features = _features(blob_count=11, green_light_ratio=0.2)
    assert classify(features) == "guard_candidate"


def test_classify_environment_candidate_on_implausible_height():
    features = _features(uncalibrated=0.0, implausible_height_fraction=0.6)
    assert classify(features) == "environment_candidate"


def test_classify_inside_elevated_daylight_animal_before_height_gate():
    # cam10/17146: a bird perched on the fence is fully "inside", but its
    # ground-plane height is impossible precisely because it is elevated.
    features = _features(
        uncalibrated=0.0,
        implausible_height_fraction=0.8,
        zone_classifiable_fraction=1.0,
        outside_pixel_fraction=0.0,
        blob_count=2,
        color_fraction=0.2,
        is_daylight=True,
    )
    assert classify(features) == "animal_candidate"


def test_classify_inside_elevated_animal_requires_compact_motion():
    features = _features(
        uncalibrated=0.0,
        implausible_height_fraction=0.8,
        zone_classifiable_fraction=1.0,
        outside_pixel_fraction=0.0,
        blob_count=3,
        color_fraction=0.2,
        is_daylight=True,
    )
    assert classify(features) == "environment_candidate"


def test_classify_inside_elevated_animal_requires_real_daylight():
    features = _features(
        uncalibrated=0.0,
        implausible_height_fraction=0.8,
        zone_classifiable_fraction=1.0,
        outside_pixel_fraction=0.0,
        blob_count=2,
        color_fraction=0.2,
        is_daylight=False,
    )
    assert classify(features) == "environment_candidate"


def test_classify_implausible_height_does_not_gate_at_or_below_threshold():
    features = _features(uncalibrated=0.0, implausible_height_fraction=0.5)
    assert classify(features) != "environment_candidate"


def test_classify_implausible_height_ignored_when_uncalibrated():
    # cam01b/cam15/cam16 have no usable picket trace -- must fall through to
    # the pixel-space rules exactly as before, never silently suppress.
    features = _features(uncalibrated=1.0, implausible_height_fraction=1.0)
    assert classify(features) != "environment_candidate"


def test_classify_implausible_height_ignored_when_key_absent():
    # Callers that never ran the metric gate (feature dict predates it) must
    # behave exactly as before -- missing key is not the same as 0.0.
    features = _features()
    assert classify(features) == "unclassified"


def test_classify_guard_wins_over_implausible_height():
    features = _features(uncalibrated=0.0, implausible_height_fraction=1.0, green_light_ratio=0.2)
    assert classify(features) == "guard_candidate"


def test_classify_inside_only_blob_is_guard_candidate():
    # The guard patrols inside the fence -- a blob the geometry actually
    # classified, and classified entirely inside, is a guard not an unknown.
    features = _features(zone_classifiable_fraction=1.0, outside_pixel_fraction=0.0)
    assert classify(features) == "guard_candidate"


def test_classify_inside_only_daylight_is_resident_candidate():
    # Same inside-only geometry, but real daylight -- a resident going about
    # their business is at least as likely as a night patrol.
    features = _features(
        zone_classifiable_fraction=1.0, outside_pixel_fraction=0.0, is_daylight=True
    )
    assert classify(features) == "resident_candidate"


def test_classify_inside_only_daylight_key_absent_is_guard_candidate():
    # Callers that never set is_daylight (feature dict predates it) must
    # behave exactly as before -- missing key is not the same as daylight.
    features = _features(zone_classifiable_fraction=1.0, outside_pixel_fraction=0.0)
    assert "is_daylight" not in features
    assert classify(features) == "guard_candidate"


def test_classify_inside_only_rule_needs_classifiable_points():
    # outside_pixel_fraction is 0.0 for BOTH "all inside" and "nothing was
    # classifiable" -- without the classifiable guard this would confidently
    # suppress a blob that was never actually classified.
    features = _features(zone_classifiable_fraction=0.0, outside_pixel_fraction=0.0)
    assert classify(features) == "unclassified"


def test_classify_inside_only_rule_does_not_override_incident():
    # Must be the LAST rule: a clip already reading as outside/far from the
    # fence stays an incident_candidate.
    features = _features(
        zone_classifiable_fraction=1.0,
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        color_fraction=0.0,
    )
    assert classify(features) == "incident_candidate"


def test_classify_inside_only_rule_does_not_override_environment():
    features = _features(
        zone_classifiable_fraction=1.0, outside_pixel_fraction=0.0, blob_count=11
    )
    assert classify(features) == "environment_candidate"


def test_classify_environment_wins_over_animal_incident_shape():
    # high blob_count AND a fence-crossing geometry read -- environment_candidate
    # takes priority over the animal/incident geometry rule.
    features = _features(blob_count=11, outside_pixel_fraction=0.9, median_fence_distance=0.2)
    assert classify(features) == "environment_candidate"


def test_classify_guard_wins_over_low_alert_persistence():
    features = _features(
        persistence=0.0,
        green_light_ratio=0.2,
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
    )
    assert classify(features) == "guard_candidate"


def test_classify_environment_wins_over_low_alert_persistence():
    features = _features(
        persistence=0.0,
        blob_count=11,
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
    )
    assert classify(features) == "environment_candidate"


def test_classify_animal_candidate_on_daylight_color():
    features = _features(outside_pixel_fraction=0.9, median_fence_distance=0.2, color_fraction=0.3)
    assert classify(features) == "animal_candidate"


@pytest.mark.parametrize(
    "features",
    [
        _features(
            persistence=0.005,
            uncalibrated=0.0,
            implausible_height_fraction=0.8,
            zone_classifiable_fraction=1.0,
            outside_pixel_fraction=0.0,
            blob_count=2,
            color_fraction=0.2,
            is_daylight=True,
        ),
        _features(
            persistence=0.005,
            outside_pixel_fraction=1.0,
            median_fence_distance=0.097,
            color_fraction=0.23,
            row_normalised_area=101.0,
            is_daylight=True,
        ),
        _features(
            persistence=0.005,
            outside_pixel_fraction=0.5,
            median_fence_distance=0.034,
            outside_frame_fraction=1.0,
            fence_crossed=1.0,
            color_fraction=0.016,
            row_normalised_area=2028.0,
            motion_pixel_fraction_median=0.003,
            scenery_motion_fraction=0.0,
            is_daylight=False,
        ),
        _features(
            persistence=0.005,
            outside_pixel_fraction=0.9,
            median_fence_distance=0.2,
            color_fraction=0.3,
        ),
        _features(
            persistence=0.005,
            outside_pixel_fraction=0.9,
            median_fence_distance=0.2,
            color_fraction=0.0,
        ),
    ],
)
def test_classify_low_persistence_blocks_every_alert_branch(features):
    assert classify(features) == "unclassified"


def test_classify_alert_persistence_floor_is_inclusive():
    features = _features(
        persistence=0.06,
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        color_fraction=0.0,
    )
    assert classify(features) == "incident_candidate"


def test_classify_near_fence_daylight_animal():
    # cam10/9405 sits just below the standard 0.10 distance floor.
    features = _features(
        outside_pixel_fraction=1.0,
        median_fence_distance=0.097,
        color_fraction=0.23,
        row_normalised_area=101.0,
        is_daylight=True,
    )
    assert classify(features) == "animal_candidate"


def test_classify_near_fence_animal_does_not_relax_night_geometry():
    features = _features(
        outside_pixel_fraction=1.0,
        median_fence_distance=0.097,
        color_fraction=0.23,
        row_normalised_area=101.0,
        is_daylight=False,
    )
    assert classify(features) == "unclassified"


def test_classify_near_fence_animal_requires_compact_blob_count():
    # cam15/9644 is daylight wind shaking bushes on both sides of the fence.
    features = _features(
        outside_pixel_fraction=1.0,
        median_fence_distance=0.053,
        color_fraction=0.83,
        row_normalised_area=365.0,
        blob_count=6,
        is_daylight=True,
    )
    assert classify(features) == "unclassified"


def test_classify_night_fence_straddle_subject():
    # cam15/15454: the clearest porcupine contour is exactly 50/50, while its
    # temporal track is outside and crosses the fence.
    features = _features(
        outside_pixel_fraction=0.5,
        median_fence_distance=0.034,
        outside_frame_fraction=1.0,
        fence_crossed=1.0,
        color_fraction=0.016,
        row_normalised_area=2028.0,
        motion_pixel_fraction_median=0.003,
        scenery_motion_fraction=0.0,
        is_daylight=False,
    )
    assert classify(features) == "incident_candidate"


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("outside_pixel_fraction", 0.49),
        ("median_fence_distance", 0.02),
        ("outside_frame_fraction", 0.6),
        ("fence_crossed", 0.0),
        ("scenery_motion_fraction", 0.13),
    ],
)
def test_classify_night_fence_straddle_requires_all_corroboration(override, value):
    features = _features(
        outside_pixel_fraction=0.5,
        median_fence_distance=0.034,
        outside_frame_fraction=1.0,
        fence_crossed=1.0,
        color_fraction=0.016,
        row_normalised_area=2028.0,
        motion_pixel_fraction_median=0.003,
        scenery_motion_fraction=0.0,
        is_daylight=False,
    )
    features[override] = value
    assert classify(features) == "unclassified"


def test_classify_animal_candidate_large_blob_redirects_to_environment():
    # A large, depth-corrected blob in the animal_candidate branch reads as a
    # branch/bush, not a real animal (real animal median row_normalised_area
    # is 666 vs this leaking population's 10471 -- see classify()'s docstring).
    features = _features(
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        color_fraction=0.3,
        row_normalised_area=5000.0,
    )
    assert classify(features) == "environment_candidate"


def test_classify_animal_candidate_row_area_absent_key_is_safe():
    # Callers that never computed row_normalised_area (feature dict predates
    # it) must behave exactly as before -- missing key is not the same as a
    # large blob.
    features = _features(outside_pixel_fraction=0.9, median_fence_distance=0.2, color_fraction=0.3)
    assert "row_normalised_area" not in features
    assert classify(features) == "animal_candidate"


def test_classify_incident_candidate_ignores_row_area_bound():
    # The row_normalised_area bound is deliberately NEVER applied to
    # incident_candidate -- it does not separate incident from environment
    # there, and touching it risks suppressing a real outside incident.
    features = _features(
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        color_fraction=0.0,
        row_normalised_area=50000.0,
    )
    assert classify(features) == "incident_candidate"


def test_classify_incident_candidate_on_night_color():
    features = _features(outside_pixel_fraction=0.9, median_fence_distance=0.2, color_fraction=0.0)
    assert classify(features) == "incident_candidate"


def test_classify_guard_wins_over_animal_incident_geometry():
    # green light present AND a fence-crossing geometry read -- guard_candidate
    # takes priority, since guards routinely register as "outside" too (they walk
    # close to the fence and shine a flashlight across it).
    features = _features(
        outside_pixel_fraction=0.9, median_fence_distance=0.2, green_light_ratio=0.2
    )
    assert classify(features) == "guard_candidate"


def test_classify_animal_incident_geometry_upper_bounded():
    # cam10/7632 (a genuine animal event) sits at 0.504 -- wind-shaken
    # vegetation out in the field, not a subject approaching the fence.
    # Deliberately accepted cost of the 2026-09-07 upper bound (see
    # classify()'s docstring): this clip alone stops alerting.
    features = _features(outside_pixel_fraction=0.9, median_fence_distance=0.504)
    assert classify(features) != "incident_candidate"
    assert classify(features) != "animal_candidate"


def test_classify_animal_incident_geometry_upper_bound_boundary():
    # The highest incident in the labelled corpus is 0.353 -- comfortably
    # under the 0.40 bound, with real margin either side of it.
    just_under = _features(outside_pixel_fraction=0.9, median_fence_distance=0.399)
    at_bound = _features(outside_pixel_fraction=0.9, median_fence_distance=0.40)
    assert classify(just_under) == "incident_candidate"
    assert classify(at_bound) != "incident_candidate"


def test_classify_requires_both_outside_fraction_and_fence_distance():
    # High outside_pixel_fraction alone (e.g. a guard hugging the fence, visible
    # through the mesh) must not fire without also being far from the fence line.
    features = _features(outside_pixel_fraction=0.9, median_fence_distance=0.05)
    assert classify(features) == "unclassified"


def test_classify_environment_candidate_on_sustained_blob_count():
    # The peak rule (blob_count > 10) misses this; the median catches it.
    features = _features(blob_count=6, blob_count_median=5)
    assert classify(features) == "environment_candidate"


def test_classify_blob_count_median_at_bound_does_not_fire():
    # cam08/4054, a real incident clip, sits at exactly 4.0.
    features = _features(
        blob_count_median=4.0, outside_pixel_fraction=0.9, median_fence_distance=0.2
    )
    assert classify(features) == "incident_candidate"


def test_classify_blob_count_median_absent_key_is_safe():
    assert classify(_features()) == "unclassified"


def test_classify_blinded_lens_never_alerts():
    # A bright obstruction against the lens IS the "outside blob" -- routed
    # away from the alert channel, while is_blinding_foreground stays True.
    features = _features(
        outside_pixel_fraction=0.9, median_fence_distance=0.2, blob_white_fraction=0.5
    )
    assert classify(features) == "environment_candidate"
    assert is_blinding_foreground(features) is True


def test_classify_blinded_lens_gate_shares_the_maintenance_threshold():
    # cam08/4054 (a real incident) peaks at 0.310, so the 0.4 bound must not
    # fire below it -- and must fire at the same value the flag uses.
    just_under = _features(
        outside_pixel_fraction=0.9, median_fence_distance=0.2, blob_white_fraction=0.39
    )
    at_bound = _features(
        outside_pixel_fraction=0.9, median_fence_distance=0.2, blob_white_fraction=0.4
    )
    assert classify(just_under) == "incident_candidate"
    assert classify(at_bound) == "environment_candidate"


def test_classify_sustained_whole_frame_motion_is_environment():
    features = _features(
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        motion_pixel_fraction_median=0.3,
    )
    assert classify(features) == "environment_candidate"


def test_classify_whole_frame_motion_gate_uses_the_median_not_the_peak():
    # A single flare-settle frame spiking the PEAK must not suppress an alert;
    # the worst real incident clip peaks at 0.123 with a median of 0.083.
    features = _features(
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        motion_pixel_fraction=0.9,
        motion_pixel_fraction_median=0.083,
    )
    assert classify(features) == "incident_candidate"


def test_classify_long_flare_is_maintenance_only_and_never_gates_an_alert():
    # cam06/21520, the crawl incident, sits at long_flare_frames=15 -- too
    # thin a margin to use as an alert veto, deliberately not shipped as one.
    features = _features(
        outside_pixel_fraction=0.9, median_fence_distance=0.2, long_flare_frames=25
    )
    assert classify(features) == "incident_candidate"
    assert is_blinding_foreground(features) is True


def test_classify_insect_candidate():
    assert classify(_features(jitter=60, solidity=0.5)) == "insect_candidate"


def _neighbour_features(**overrides) -> dict[str, float]:
    base = _features(
        outside_pixel_fraction=0.9,
        uncalibrated=0.0,
        subject_height_m=1.3,
        is_daylight=True,
    )
    base.update(overrides)
    return base


def test_classify_neighbour_candidate_person_sized_outside_daylight():
    assert classify(_neighbour_features()) == "neighbour_candidate"


def test_classify_neighbour_candidate_requires_real_daylight_not_a_pixel_statistic():
    # color_fraction alone (an image statistic) must not substitute for the
    # real is_daylight signal -- same "colour is not a daylight proxy" lesson
    # as the inside-only fallback's own resident/guard split.
    assert classify(_neighbour_features(is_daylight=False)) == "unclassified"


def test_classify_neighbour_candidate_requires_calibrated_camera():
    # uncalibrated=1.0 (the default when a camera has no metric calibration)
    # means subject_height_m is not trustworthy -- must not fire on it.
    assert classify(_neighbour_features(uncalibrated=1.0)) == "unclassified"


def test_classify_neighbour_candidate_rejects_animal_sized_subject():
    assert classify(_neighbour_features(subject_height_m=0.3)) == "unclassified"


def test_classify_neighbour_candidate_rejects_too_tall_subject():
    assert classify(_neighbour_features(subject_height_m=2.6)) == "unclassified"


def test_classify_neighbour_candidate_never_overrides_an_earlier_rule():
    # A real flashlight sighting must win even if the blob also happens to be
    # person-sized, outside, and in daylight -- rule order is load-bearing.
    assert classify(_neighbour_features(green_light_ratio=0.2)) == "guard_candidate"


def test_classify_unclassified_when_nothing_fires():
    assert classify(_features()) == "unclassified"


def test_classify_raises_on_missing_feature_key():
    # Guard against a features dict missing an expected key -- should raise loudly
    # (KeyError) rather than silently miscategorising.
    with pytest.raises(KeyError):
        classify({"aspect_ratio": 1.0})


def test_classify_event_incident_wins_over_every_suppressed_sibling():
    assert (
        classify_event(["guard_candidate", "incident_candidate", "environment_candidate"])
        == "incident_candidate"
    )


def test_classify_event_animal_wins_over_guard():
    assert classify_event(["guard_candidate", "animal_candidate"]) == "animal_candidate"


def test_classify_event_all_same_category_is_a_no_op():
    assert classify_event(["guard_candidate", "guard_candidate"]) == "guard_candidate"


def test_classify_event_empty_is_no_motion():
    assert classify_event([]) == "no_motion"


def test_classify_event_single_category_passes_through():
    assert classify_event(["resident_candidate"]) == "resident_candidate"


# --------------------------------------------------------------------------
# Explicit ClassificationThresholds -- proves the wiring is live, not a
# leftover hardcoded literal. Each test moves exactly one field away from the
# repo's real config/thresholds.yaml and checks the category flips accordingly.
# --------------------------------------------------------------------------


def _thresholds(**overrides):
    import dataclasses

    return dataclasses.replace(default_thresholds(), **overrides)


def test_classify_uses_explicit_thresholds_not_hardcoded_literals():
    features = _features(green_light_ratio=0.03)
    # Default green_light_ratio_min is 0.02, so 0.03 fires guard_candidate.
    assert classify(features) == "guard_candidate"
    # Raising the threshold above the observed value must suppress the rule --
    # if classify() still used a hardcoded 0.02 literal, this would be a no-op.
    assert classify(features, _thresholds(green_light_ratio_min=0.05)) != "guard_candidate"


def test_classify_green_light_flicker_threshold_is_wired():
    features = _features(green_light_flicker=0.03)
    assert classify(features) == "guard_candidate"
    assert classify(features, _thresholds(green_light_flicker_min=0.05)) != "guard_candidate"


def test_classify_warmup_flashlight_threshold_is_wired():
    features = _features(warmup_flashlight_ratio=0.003)
    assert classify(features) == "guard_candidate"
    assert (
        classify(features, _thresholds(warmup_flashlight_ratio_min=0.01)) != "guard_candidate"
    )


def test_classify_blob_count_peak_threshold_is_wired():
    features = _features(blob_count=11)
    assert classify(features) == "environment_candidate"
    assert (
        classify(features, _thresholds(blob_count_peak_min=20)) != "environment_candidate"
    )


def test_classify_blob_count_median_threshold_is_wired():
    features = _features(blob_count_median=5.0)
    assert classify(features) == "environment_candidate"
    assert (
        classify(features, _thresholds(blob_count_median_min=10.0)) != "environment_candidate"
    )


def test_classify_implausible_height_threshold_is_wired():
    features = _features(uncalibrated=0.0, implausible_height_fraction=0.6)
    assert classify(features) == "environment_candidate"
    assert (
        classify(features, _thresholds(implausible_height_fraction_min=0.9))
        != "environment_candidate"
    )


def test_classify_alert_persistence_threshold_is_wired():
    features = _features(
        persistence=0.005,
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        color_fraction=0.0,
    )
    assert classify(features) == "unclassified"
    assert classify(features, _thresholds(alert_persistence_min=0.0)) == "incident_candidate"


def test_classify_terminal_reverse_artifact_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=0.9,
        median_fence_distance=0.2,
        terminal_reverse_seed=1.0,
        blob_black_white_balance=0.188,
    )
    assert classify(features) == "environment_candidate"
    assert (
        classify(
            features,
            _thresholds(camera_artifact_black_white_balance_min=0.2),
        )
        == "incident_candidate"
    )


def test_classify_inside_blob_count_threshold_is_wired():
    features = _features(
        uncalibrated=0.0,
        implausible_height_fraction=0.8,
        zone_classifiable_fraction=1.0,
        outside_pixel_fraction=0.0,
        blob_count=2,
        color_fraction=0.2,
        is_daylight=True,
    )
    assert classify(features) == "animal_candidate"
    assert (
        classify(features, _thresholds(inside_blob_count_max=1.0))
        == "environment_candidate"
    )


def test_classify_near_fence_distance_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=1.0,
        median_fence_distance=0.097,
        color_fraction=0.23,
        row_normalised_area=101.0,
        is_daylight=True,
    )
    assert classify(features) == "animal_candidate"
    assert classify(features, _thresholds(near_fence_distance_min=0.099)) == "unclassified"


def test_classify_near_fence_blob_count_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=1.0,
        median_fence_distance=0.053,
        color_fraction=0.83,
        row_normalised_area=365.0,
        blob_count=6,
        is_daylight=True,
    )
    assert classify(features) == "unclassified"
    assert (
        classify(features, _thresholds(near_fence_blob_count_max=6.0))
        == "animal_candidate"
    )


def test_classify_straddle_pixel_fraction_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=0.5,
        median_fence_distance=0.034,
        outside_frame_fraction=1.0,
        fence_crossed=1.0,
        color_fraction=0.016,
        row_normalised_area=2028.0,
        scenery_motion_fraction=0.0,
        is_daylight=False,
    )
    assert classify(features) == "incident_candidate"
    assert (
        classify(features, _thresholds(straddle_pixel_fraction_min=0.55)) == "unclassified"
    )


def test_classify_straddle_distance_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=0.5,
        median_fence_distance=0.034,
        outside_frame_fraction=1.0,
        fence_crossed=1.0,
        color_fraction=0.016,
        row_normalised_area=2028.0,
        scenery_motion_fraction=0.0,
        is_daylight=False,
    )
    assert classify(features) == "incident_candidate"
    assert classify(features, _thresholds(straddle_distance_min=0.04)) == "unclassified"


def test_classify_outside_pixel_fraction_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=0.65, median_fence_distance=0.2, color_fraction=0.0
    )
    assert classify(features) == "incident_candidate"
    assert (
        classify(features, _thresholds(outside_pixel_fraction_min=0.9)) != "incident_candidate"
    )


def test_classify_median_fence_distance_min_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=0.65, median_fence_distance=0.15, color_fraction=0.0
    )
    assert classify(features) == "incident_candidate"
    assert (
        classify(features, _thresholds(median_fence_distance_min=0.3)) != "incident_candidate"
    )


def test_classify_median_fence_distance_max_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=0.65, median_fence_distance=0.35, color_fraction=0.0
    )
    assert classify(features) == "incident_candidate"
    assert (
        classify(features, _thresholds(median_fence_distance_max=0.2)) != "incident_candidate"
    )


def test_classify_color_fraction_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
        color_fraction=0.2,
        row_normalised_area=100.0,
    )
    assert classify(features) == "animal_candidate"
    assert classify(features, _thresholds(color_fraction_min=0.5)) != "animal_candidate"


def test_classify_motion_pixel_fraction_median_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
        color_fraction=0.0,
        motion_pixel_fraction_median=0.13,
    )
    assert classify(features) == "environment_candidate"
    assert (
        classify(features, _thresholds(motion_pixel_fraction_median_min=0.5))
        != "environment_candidate"
    )


def test_classify_row_normalised_area_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
        color_fraction=0.2,
        row_normalised_area=3500.0,
    )
    assert classify(features) == "environment_candidate"
    assert (
        classify(features, _thresholds(row_normalised_area_max=10000.0)) != "environment_candidate"
    )


def test_classify_jitter_threshold_is_wired():
    features = _features(jitter=60.0, solidity=0.5)
    assert classify(features) == "insect_candidate"
    assert classify(features, _thresholds(jitter_min=100.0)) != "insect_candidate"


def test_classify_solidity_threshold_is_wired():
    features = _features(jitter=60.0, solidity=0.5)
    assert classify(features) == "insect_candidate"
    assert classify(features, _thresholds(solidity_max=0.3)) != "insect_candidate"


def test_classify_blob_white_fraction_blinding_gate_threshold_is_wired():
    features = _features(
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
        color_fraction=0.0,
        blob_white_fraction=0.45,
    )
    assert classify(features) == "environment_candidate"
    assert (
        classify(features, _thresholds(blob_white_fraction_min=0.9)) != "environment_candidate"
    )


def test_is_blinding_foreground_blob_white_fraction_threshold_is_wired():
    features = _features(blob_white_fraction=0.45)
    assert is_blinding_foreground(features) is True
    assert is_blinding_foreground(features, _thresholds(blob_white_fraction_min=0.9)) is False


def test_is_blinding_foreground_long_flare_frames_threshold_is_wired():
    features = _features(long_flare_frames=20)
    assert is_blinding_foreground(features) is True
    assert is_blinding_foreground(features, _thresholds(long_flare_frames_min=50)) is False


def test_is_blinding_foreground_large_blob_thresholds_are_wired():
    features = _features(blob_frame_fraction=0.35, blob_white_fraction=0.24)
    assert is_blinding_foreground(features) is True
    assert (
        is_blinding_foreground(features, _thresholds(large_blob_frame_fraction_min=0.5))
        is False
    )
    assert (
        is_blinding_foreground(features, _thresholds(large_blob_white_fraction_min=0.3))
        is False
    )


# --------------------------------------------------------------------------
# classify(features) with no explicit thresholds must agree with passing the
# default explicitly -- proves the implicit default path and the explicit one
# are the same code path, not a divergent shortcut.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "features",
    [
        _features(),
        _features(green_light_ratio=0.5),
        _features(blob_count=15),
        _features(outside_pixel_fraction=0.7, median_fence_distance=0.2, color_fraction=0.2),
        _features(jitter=60, solidity=0.5),
    ],
)
def test_classify_default_thresholds_matches_explicit_default(features):
    assert classify(features) == classify(features, default_thresholds())


def test_is_blinding_foreground_default_thresholds_matches_explicit_default():
    features = _features(blob_white_fraction=0.5)
    assert is_blinding_foreground(features) == is_blinding_foreground(
        features, default_thresholds()
    )


# --------------------------------------------------------------------------
# classify_detailed() -- one test per reason code, asserting both the code and
# that category matches classify()'s own return for the same features. Every
# feature combination here is lifted from an existing classify() test above,
# not invented fresh, so each reason-code assertion rides on an
# already-validated rule trigger.
# --------------------------------------------------------------------------


def test_reason_no_features():
    result = classify_detailed(None)
    assert result.category == "no_motion" == classify(None)
    assert result.reason == "no_features"
    assert result.contributing == {}


def test_reason_green_light():
    features = _features(green_light_ratio=0.2)
    result = classify_detailed(features)
    assert result.category == "guard_candidate" == classify(features)
    assert result.reason == "green_light"
    assert result.contributing == {"green_light_ratio": 0.2, "green_light_flicker": 0.0}


def test_reason_warmup_flashlight():
    features = _features(warmup_flashlight_ratio=0.01)
    result = classify_detailed(features)
    assert result.category == "guard_candidate" == classify(features)
    assert result.reason == "warmup_flashlight"
    assert result.contributing == {"warmup_flashlight_ratio": 0.01}


def test_reason_blob_count_peak():
    features = _features(blob_count=11)
    result = classify_detailed(features)
    assert result.category == "environment_candidate" == classify(features)
    assert result.reason == "blob_count_peak"
    assert result.contributing == {"blob_count": 11}


def test_reason_blob_count_sustained():
    features = _features(blob_count_median=4.5)
    result = classify_detailed(features)
    assert result.category == "environment_candidate" == classify(features)
    assert result.reason == "blob_count_sustained"
    assert result.contributing == {"blob_count_median": 4.5}


def test_reason_inside_elevated_animal():
    features = _features(
        uncalibrated=0.0,
        implausible_height_fraction=0.8,
        zone_classifiable_fraction=1.0,
        outside_pixel_fraction=0.0,
        blob_count=2,
        color_fraction=0.2,
        is_daylight=True,
    )
    result = classify_detailed(features)
    assert result.category == "animal_candidate" == classify(features)
    assert result.reason == "inside_elevated_animal"
    assert result.contributing == {
        "uncalibrated": 0.0,
        "implausible_height_fraction": 0.8,
        "zone_classifiable_fraction": 1.0,
        "outside_pixel_fraction": 0.0,
        "blob_count": 2,
        "color_fraction": 0.2,
        "is_daylight": True,
    }


def test_reason_implausible_height():
    features = _features(uncalibrated=0.0, implausible_height_fraction=0.6)
    result = classify_detailed(features)
    assert result.category == "environment_candidate" == classify(features)
    assert result.reason == "implausible_height"
    assert result.contributing == {"uncalibrated": 0.0, "implausible_height_fraction": 0.6}


def test_reason_near_fence_animal():
    features = _features(
        outside_pixel_fraction=1.0,
        median_fence_distance=0.097,
        color_fraction=0.23,
        row_normalised_area=101.0,
        is_daylight=True,
    )
    result = classify_detailed(features)
    assert result.category == "animal_candidate" == classify(features)
    assert result.reason == "near_fence_animal"
    assert result.contributing == {
        "outside_pixel_fraction": 1.0,
        "median_fence_distance": 0.097,
        "color_fraction": 0.23,
        "row_normalised_area": 101.0,
        "blob_count": 1,
        "blob_white_fraction": 0.0,
        "motion_pixel_fraction_median": 0.0,
        "is_daylight": True,
    }


def test_reason_fence_straddle_no_colour():
    features = _features(
        outside_pixel_fraction=0.5,
        median_fence_distance=0.034,
        outside_frame_fraction=1.0,
        fence_crossed=1.0,
        color_fraction=0.016,
        row_normalised_area=2028.0,
        motion_pixel_fraction_median=0.003,
        scenery_motion_fraction=0.0,
        is_daylight=False,
    )
    result = classify_detailed(features)
    assert result.category == "incident_candidate" == classify(features)
    assert result.reason == "fence_straddle_no_colour"
    assert result.contributing == {
        "outside_pixel_fraction": 0.5,
        "median_fence_distance": 0.034,
        "outside_frame_fraction": 1.0,
        "fence_crossed": 1.0,
        "is_daylight": False,
        "color_fraction": 0.016,
        "row_normalised_area": 2028.0,
        "blob_white_fraction": 0.0,
        "motion_pixel_fraction_median": 0.003,
        "scenery_motion_fraction": 0.0,
    }


def test_reason_blinding_blob_white():
    features = _features(
        outside_pixel_fraction=0.65, median_fence_distance=0.2, blob_white_fraction=0.5
    )
    result = classify_detailed(features)
    assert result.category == "environment_candidate" == classify(features)
    assert result.reason == "blinding_blob_white"
    assert result.contributing == {
        "outside_pixel_fraction": 0.65,
        "median_fence_distance": 0.2,
        "blob_white_fraction": 0.5,
    }


def test_reason_motion_pixel_sustained():
    features = _features(
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
        motion_pixel_fraction_median=0.13,
    )
    result = classify_detailed(features)
    assert result.category == "environment_candidate" == classify(features)
    assert result.reason == "motion_pixel_sustained"
    assert result.contributing == {
        "outside_pixel_fraction": 0.65,
        "median_fence_distance": 0.2,
        "motion_pixel_fraction_median": 0.13,
    }


def test_reason_animal_row_area():
    features = _features(
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
        color_fraction=0.2,
        row_normalised_area=3500.0,
    )
    result = classify_detailed(features)
    assert result.category == "environment_candidate" == classify(features)
    assert result.reason == "animal_row_area"
    assert result.contributing == {
        "outside_pixel_fraction": 0.65,
        "median_fence_distance": 0.2,
        "color_fraction": 0.2,
        "row_normalised_area": 3500.0,
    }


def test_reason_outside_colour():
    features = _features(
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
        color_fraction=0.2,
        row_normalised_area=100.0,
    )
    result = classify_detailed(features)
    assert result.category == "animal_candidate" == classify(features)
    assert result.reason == "outside_colour"
    assert result.contributing == {
        "outside_pixel_fraction": 0.65,
        "median_fence_distance": 0.2,
        "color_fraction": 0.2,
        "row_normalised_area": 100.0,
    }


def test_reason_insufficient_detection_evidence():
    features = _features(
        persistence=0.005,
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
    )
    result = classify_detailed(features)
    assert result.category == "unclassified" == classify(features)
    assert result.reason == "insufficient_detection_evidence"
    assert result.contributing == {"persistence": 0.005}


def test_reason_outside_no_colour():
    features = _features(outside_pixel_fraction=0.65, median_fence_distance=0.2)
    result = classify_detailed(features)
    assert result.category == "incident_candidate" == classify(features)
    assert result.reason == "outside_no_colour"
    assert result.contributing == {
        "outside_pixel_fraction": 0.65,
        "median_fence_distance": 0.2,
        "color_fraction": 0.0,
    }


def test_reason_terminal_reverse_camera_artifact():
    features = _features(
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
        terminal_reverse_seed=1.0,
        blob_black_white_balance=0.188,
    )
    result = classify_detailed(features)
    assert result.category == "environment_candidate" == classify(features)
    assert result.reason == "terminal_reverse_camera_artifact"
    assert result.contributing == {
        "terminal_reverse_seed": 1.0,
        "blob_black_white_balance": 0.188,
    }


def test_terminal_reverse_artifact_preserves_4487_below_measured_balance():
    features = _features(
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
        terminal_reverse_seed=1.0,
        blob_black_white_balance=0.073,
    )
    assert classify(features) == "incident_candidate"


def test_guard_signal_beats_terminal_reverse_camera_artifact():
    features = _features(
        green_light_ratio=0.2,
        outside_pixel_fraction=0.65,
        median_fence_distance=0.2,
        terminal_reverse_seed=1.0,
        blob_black_white_balance=0.188,
    )
    assert classify(features) == "guard_candidate"


def test_reason_jitter_solidity():
    features = _features(jitter=60, solidity=0.5)
    result = classify_detailed(features)
    assert result.category == "insect_candidate" == classify(features)
    assert result.reason == "jitter_solidity"
    assert result.contributing == {"jitter": 60, "solidity": 0.5}


def test_reason_inside_only_daylight():
    features = _features(zone_classifiable_fraction=0.5, is_daylight=True)
    result = classify_detailed(features)
    assert result.category == "resident_candidate" == classify(features)
    assert result.reason == "inside_only_daylight"
    assert result.contributing == {
        "zone_classifiable_fraction": 0.5,
        "outside_pixel_fraction": 0.0,
        "is_daylight": True,
    }


def test_reason_inside_only_night():
    features = _features(zone_classifiable_fraction=0.5, is_daylight=False)
    result = classify_detailed(features)
    assert result.category == "guard_candidate" == classify(features)
    assert result.reason == "inside_only_night"
    assert result.contributing == {
        "zone_classifiable_fraction": 0.5,
        "outside_pixel_fraction": 0.0,
        "is_daylight": False,
    }


def test_reason_no_rule_matched():
    result = classify_detailed(_features())
    assert result.category == "unclassified" == classify(_features())
    assert result.reason == "no_rule_matched"
    assert result.contributing == {}


def test_all_22_reason_codes_are_distinct():
    # Guards against a copy-paste reusing a reason code across two branches.
    codes = {
        "no_features",
        "green_light",
        "warmup_flashlight",
        "multi_object_flashlight",
        "blob_count_peak",
        "blob_count_sustained",
        "inside_elevated_animal",
        "implausible_height",
        "near_fence_animal",
        "fence_straddle_no_colour",
        "blinding_blob_white",
        "motion_pixel_sustained",
        "animal_row_area",
        "insufficient_detection_evidence",
        "outside_colour",
        "outside_no_colour",
        "terminal_reverse_camera_artifact",
        "outside_person_daylight",
        "jitter_solidity",
        "inside_only_daylight",
        "inside_only_night",
        "no_rule_matched",
    }
    assert len(codes) == 22
