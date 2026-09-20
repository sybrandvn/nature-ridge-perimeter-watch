from pathlib import Path

import pytest

from src.config import (
    load_app_config_from_mapping,
    load_cameras_config,
    load_thresholds_config,
    resolve_channel_ref,
)
from src.errors import ConfigError

# --------------------------------------------------------------------------
# AppConfig
# --------------------------------------------------------------------------


def test_load_app_config_minimal_valid_mapping():
    cfg = load_app_config_from_mapping(
        {
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "abcdef",
            "SOURCE_CHANNEL": "@nature_ridge_cams",
        }
    )
    assert cfg.telegram_api_id == 12345
    assert cfg.source_channel == "@nature_ridge_cams"
    assert cfg.operating_window_start == "18:00"
    assert cfg.operating_window_end == "06:00"
    assert cfg.ntfy_priority == "urgent"
    assert cfg.bot_trustee_ids == ()
    assert cfg.event_wait_seconds == 300.0
    assert cfg.delivery_retry_base_seconds == 30.0
    assert cfg.delivery_retry_max_seconds == 900.0
    assert cfg.watcher_poll_seconds == 5.0
    assert cfg.media_retention_interval_seconds == 3600.0


def test_live_runtime_durations_validate_and_parse():
    cfg = load_app_config_from_mapping(
        {
            "TELEGRAM_API_ID": "1",
            "TELEGRAM_API_HASH": "x",
            "SOURCE_CHANNEL": "x",
            "EVENT_WAIT_SECONDS": "12.5",
            "DELIVERY_RETRY_BASE_SECONDS": "2",
            "DELIVERY_RETRY_MAX_SECONDS": "8",
            "WATCHER_POLL_SECONDS": "1",
        }
    )
    assert cfg.event_wait_seconds == 12.5
    assert cfg.delivery_retry_max_seconds == 8.0


def test_live_runtime_rejects_nonpositive_duration():
    with pytest.raises(ConfigError, match="EVENT_WAIT_SECONDS"):
        load_app_config_from_mapping(
            {
                "TELEGRAM_API_ID": "1",
                "TELEGRAM_API_HASH": "x",
                "SOURCE_CHANNEL": "x",
                "EVENT_WAIT_SECONDS": "0",
            }
        )


def test_missing_required_var_raises():
    with pytest.raises(ConfigError, match="TELEGRAM_API_HASH"):
        load_app_config_from_mapping({"TELEGRAM_API_ID": "1", "SOURCE_CHANNEL": "x"})


def test_require_telegram_false_allows_empty():
    cfg = load_app_config_from_mapping({}, require_telegram=False)
    assert cfg.telegram_api_id is None


def test_resolve_channel_ref_numeric_ids_become_int():
    assert resolve_channel_ref("-510921049") == -510921049
    assert resolve_channel_ref("-1001004276399") == -1001004276399
    assert resolve_channel_ref("12345") == 12345


def test_resolve_channel_ref_username_passes_through():
    assert resolve_channel_ref("@nature_ridge_cams") == "@nature_ridge_cams"


def test_non_integer_api_id_raises():
    with pytest.raises(ConfigError, match="TELEGRAM_API_ID"):
        load_app_config_from_mapping(
            {"TELEGRAM_API_ID": "not-a-number", "TELEGRAM_API_HASH": "x", "SOURCE_CHANNEL": "x"}
        )


@pytest.mark.parametrize("field", ["OPERATING_WINDOW_START", "OPERATING_WINDOW_END"])
def test_bad_operating_window_format_raises(field):
    with pytest.raises(ConfigError, match=field):
        load_app_config_from_mapping(
            {
                "TELEGRAM_API_ID": "1",
                "TELEGRAM_API_HASH": "x",
                "SOURCE_CHANNEL": "x",
                field: "not-a-time",
            }
        )


def test_bot_id_list_parses_and_skips_blanks():
    cfg = load_app_config_from_mapping(
        {
            "TELEGRAM_API_ID": "1",
            "TELEGRAM_API_HASH": "x",
            "SOURCE_CHANNEL": "x",
            "BOT_TRUSTEE_IDS": "111, 222,,333",
        }
    )
    assert cfg.bot_trustee_ids == (111, 222, 333)


def test_bot_id_list_rejects_non_integer():
    with pytest.raises(ConfigError, match="BOT_SECURITY_IDS"):
        load_app_config_from_mapping(
            {
                "TELEGRAM_API_ID": "1",
                "TELEGRAM_API_HASH": "x",
                "SOURCE_CHANNEL": "x",
                "BOT_SECURITY_IDS": "abc",
            }
        )


# --------------------------------------------------------------------------
# CamerasConfig
# --------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_load_cameras_config_empty_is_valid(tmp_path):
    path = _write(tmp_path / "cameras.yaml", "cameras: []\n")
    cfg = load_cameras_config(path)
    assert cfg.cameras == ()
    assert cfg.unknown_camera_id == "unknown"


def test_repo_cam12_selects_pre_remount_retrace_and_remounted_geometry():
    cfg = load_cameras_config(Path(__file__).parents[1] / "config" / "cameras.yaml")
    cam12 = cfg.by_id("cam12")
    assert cam12 is not None

    before_9162 = cam12.zone_at("2024-11-19T03:35:39Z")
    at_9162 = cam12.zone_at("2024-11-23T03:52:25Z")
    remounted = cam12.zone_at("2026-03-02T16:13:54Z")

    assert before_9162.fence[0] == (0.4959, 0.2021)
    assert at_9162.fence == before_9162.fence
    assert at_9162.fence_bottom[-1] == (0.4472, 0.999)
    assert at_9162.fence_pickets == (((0.3613, 0.4635), (0.4322, 0.999)),)
    assert remounted.fence[0] == (0.4455, 0.1958)


def test_load_cameras_config_valid_with_zone(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        """
        cameras:
          - id: cam01
            aliases: ["Camera 1", "CAM-01"]
            order: 0
            fence: [[0.1, 0.9], [0.6, 0.2]]
            outside: right
            depth_cutoff: 0.15
            ignore:
              - [[0.0, 0.0], [0.05, 0.0], [0.05, 0.05]]
        """,
    )
    cfg = load_cameras_config(path)
    assert len(cfg.cameras) == 1
    cam = cfg.cameras[0]
    assert cam.order == 0
    assert cam.zone.outside == "right"
    assert cam.zone.depth_cutoff == 0.15
    assert len(cam.zone.ignore) == 1
    assert cfg.resolve_alias("camera 1") is cam
    assert cfg.resolve_alias("cam01") is cam
    assert cfg.by_id("cam01") is cam


def test_load_cameras_config_with_fence_bottom(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        """
        cameras:
          - id: cam06
            fence: [[0.45, 0.14], [0.18, 0.99]]
            fence_bottom: [[0.5, 0.07], [0.49, 0.99]]
            outside: right
            depth_cutoff: 0.05
            fence_height_m: 1.8
        """,
    )
    cam = load_cameras_config(path).by_id("cam06")
    assert cam.zone.fence == ((0.45, 0.14), (0.18, 0.99))
    assert cam.zone.fence_bottom == ((0.5, 0.07), (0.49, 0.99))
    assert cam.zone.fence_height_m == 1.8


def test_fence_bottom_defaults_to_none_and_height_to_2m(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    fence: [[0.1, 0.9], [0.6, 0.2]]\n    outside: right\n",
    )
    cam = load_cameras_config(path).by_id("cam01")
    assert cam.zone.fence_bottom is None
    assert cam.zone.fence_height_m == 2.0


def test_fence_bottom_needs_at_least_two_points(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    fence_bottom: [[0.1, 0.1]]\n",
    )
    with pytest.raises(ConfigError, match="fence_bottom needs >= 2 points"):
        load_cameras_config(path)


def test_fence_height_m_must_be_positive(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    fence_height_m: 0\n",
    )
    with pytest.raises(ConfigError, match="fence_height_m must be > 0"):
        load_cameras_config(path)


def test_load_cameras_config_with_fence_pickets(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        """
        cameras:
          - id: cam06
            fence: [[0.45, 0.14], [0.18, 0.99]]
            fence_bottom: [[0.5, 0.07], [0.49, 0.99]]
            outside: right
            fence_pickets:
              - [[0.6, 0.4], [0.65, 0.5]]
              - [[0.55, 0.3], [0.58, 0.38]]
        """,
    )
    cam = load_cameras_config(path).by_id("cam06")
    assert cam.zone.fence_pickets == (
        ((0.6, 0.4), (0.65, 0.5)),
        ((0.55, 0.3), (0.58, 0.38)),
    )


def test_fence_pickets_defaults_to_empty(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    fence: [[0.1, 0.9], [0.6, 0.2]]\n    outside: right\n",
    )
    cam = load_cameras_config(path).by_id("cam01")
    assert cam.zone.fence_pickets == ()


def test_fence_pickets_entry_needs_exactly_two_points(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    fence_pickets:\n"
        "      - [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]\n",
    )
    with pytest.raises(ConfigError, match="fence_pickets entry needs exactly 2 points"):
        load_cameras_config(path)


def test_metric_calibration_is_off_by_default(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    fence: [[0.1, 0.9], [0.6, 0.2]]\n    outside: right\n",
    )
    zone = load_cameras_config(path).by_id("cam01").zone
    assert zone.metric_calibration is False
    assert zone.metric_max_range_m is None


def test_metric_calibration_opt_in_and_range_parse(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        """
        cameras:
          - id: cam06
            fence: [[0.45, 0.14], [0.18, 0.99]]
            outside: right
            metric_calibration: true
            metric_max_range_m: 35
        """,
    )
    zone = load_cameras_config(path).by_id("cam06").zone
    assert zone.metric_calibration is True
    assert zone.metric_max_range_m == 35.0


def test_metric_max_range_must_be_positive(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    metric_max_range_m: 0\n",
    )
    with pytest.raises(ConfigError, match="metric_max_range_m must be > 0"):
        load_cameras_config(path)


def test_metric_focal_px_override_parses(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    fence: [[0.1, 0.9], [0.6, 0.2]]\n"
        "    outside: right\n    metric_focal_px: 210.5\n",
    )
    assert load_cameras_config(path).by_id("cam01").zone.metric_focal_px == 210.5


def test_metric_focal_px_must_be_positive(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    metric_focal_px: -1\n",
    )
    with pytest.raises(ConfigError, match="metric_focal_px must be > 0"):
        load_cameras_config(path)


def test_zones_dated_history_picks_geometry_by_timestamp(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        """
        cameras:
          - id: cam01a
            zones:
              - fence: [[0.1, 0.9], [0.6, 0.2]]
                outside: right
                depth_cutoff: 0.05
              - effective_from: "2026-08-05T17:22:15Z"
                fence: [[0.4, 0.1], [0.1, 0.99]]
                outside: right
                depth_cutoff: 0.05
        """,
    )
    cam = load_cameras_config(path).by_id("cam01a")
    # zone (no history awareness) is always the most recent entry.
    assert cam.zone.fence == ((0.4, 0.1), (0.1, 0.99))
    # before the remount: old geometry.
    assert cam.zone_at("2026-08-05T17:00:00Z").fence == ((0.1, 0.9), (0.6, 0.2))
    # exactly at / after the remount: new geometry.
    assert cam.zone_at("2026-08-05T17:22:15Z").fence == ((0.4, 0.1), (0.1, 0.99))
    assert cam.zone_at("2026-08-26T21:39:41Z").fence == ((0.4, 0.1), (0.1, 0.99))
    # no timestamp / unparseable timestamp falls back to the current geometry.
    assert cam.zone_at(None).fence == ((0.4, 0.1), (0.1, 0.99))
    assert cam.zone_at("not-a-timestamp").fence == ((0.4, 0.1), (0.1, 0.99))


def test_zones_and_flat_fence_bottom_together_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        """
        cameras:
          - id: cam01a
            fence_bottom: [[0.1, 0.9], [0.6, 0.2]]
            zones:
              - fence: [[0.1, 0.9], [0.6, 0.2]]
                outside: right
        """,
    )
    with pytest.raises(ConfigError, match="must not mix"):
        load_cameras_config(path)


def test_zones_and_flat_fields_together_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        """
        cameras:
          - id: cam01a
            fence: [[0.1, 0.9], [0.6, 0.2]]
            outside: right
            zones:
              - fence: [[0.1, 0.9], [0.6, 0.2]]
                outside: right
        """,
    )
    with pytest.raises(ConfigError, match="must not mix"):
        load_cameras_config(path)


def test_zones_duplicate_effective_from_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        """
        cameras:
          - id: cam01a
            zones:
              - effective_from: "2026-08-05T17:22:15Z"
                fence: [[0.1, 0.9], [0.6, 0.2]]
                outside: right
              - effective_from: "2026-08-05T17:22:15Z"
                fence: [[0.4, 0.1], [0.1, 0.99]]
                outside: right
        """,
    )
    with pytest.raises(ConfigError, match="duplicate zones effective_from"):
        load_cameras_config(path)


def test_zones_empty_list_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01a\n    zones: []\n",
    )
    with pytest.raises(ConfigError, match="non-empty list"):
        load_cameras_config(path)


def test_no_zones_history_is_empty_and_zone_at_ignores_timestamp(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    fence: [[0.1, 0.9], [0.6, 0.2]]\n    outside: right\n",
    )
    cam = load_cameras_config(path).by_id("cam01")
    assert cam.zone_history == ()
    assert cam.zone_at("2020-01-01T00:00:00Z") is cam.zone


def test_duplicate_camera_id_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n  - id: cam01\n",
    )
    with pytest.raises(ConfigError, match="duplicate camera id"):
        load_cameras_config(path)


def test_alias_collision_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        """
        cameras:
          - id: cam01
            aliases: ["North Gate"]
          - id: cam02
            aliases: ["North Gate"]
        """,
    )
    with pytest.raises(ConfigError, match="alias"):
        load_cameras_config(path)


def test_duplicate_order_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    order: 0\n  - id: cam02\n    order: 0\n",
    )
    with pytest.raises(ConfigError, match="order 0"):
        load_cameras_config(path)


def test_fence_without_outside_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    fence: [[0.1, 0.1], [0.2, 0.2]]\n",
    )
    with pytest.raises(ConfigError, match="outside"):
        load_cameras_config(path)


def test_fence_point_out_of_range_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    outside: left\n    fence: [[1.5, 0.1], [0.2, 0.2]]\n",
    )
    with pytest.raises(ConfigError, match="out of \\[0,1\\]"):
        load_cameras_config(path)


def test_invalid_outside_value_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    outside: up\n    fence: [[0.1, 0.1], [0.2, 0.2]]\n",
    )
    with pytest.raises(ConfigError, match="outside"):
        load_cameras_config(path)


def test_ignore_polygon_too_short_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    ignore:\n      - [[0.0, 0.0], [0.1, 0.1]]\n",
    )
    with pytest.raises(ConfigError, match=">= 3 points"):
        load_cameras_config(path)


def test_threshold_override_non_numeric_leaf_raises(tmp_path):
    path = _write(
        tmp_path / "cameras.yaml",
        "cameras:\n  - id: cam01\n    threshold_overrides:\n"
        "      saturation:\n        max_ratio: high\n",
    )
    with pytest.raises(ConfigError, match="expected a number"):
        load_cameras_config(path)


def test_missing_cameras_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_cameras_config(tmp_path / "does-not-exist.yaml")


# --------------------------------------------------------------------------
# ThresholdsConfig
# --------------------------------------------------------------------------


def test_load_thresholds_config_from_repo_default():
    cfg = load_thresholds_config(Path(__file__).parents[1] / "config" / "thresholds.yaml")
    assert "tracking" in cfg.motion
    assert "guard" in cfg.classification


def test_repo_motion_thresholds_match_validated_detector_defaults():
    thresholds = load_thresholds_config(_REPO_THRESHOLDS).motion_thresholds()
    assert thresholds.__dict__ == {
        "max_area_fraction": 0.25,
        "min_blob_area_fraction": 0.0005,
        "threshold": 18,
        "flare_tolerance": 3.0,
        "max_flare_fraction": 0.4,
        "max_track_jump_fraction": 0.2,
        "max_track_miss_frames": 5,
        "template_match_threshold": 0.55,
        "flare_match_relax": 0.1,
        "track_search_margin_fraction": 0.75,
        "min_track_search_margin": 6.0,
        "fragment_close_kernel_size": 9,
        "max_recovered_streak": 12,
        "max_size_change_ratio": 4.0,
        "anchor_refine": True,
        "max_anchor_streak": 4,
        "min_reacquire_area": 20.0,
        "max_scenery_streak": 2,
        "scenery_correlation": 0.94,
        "compensate_warmup": False,
        "prefer_flashlight_candidate": False,
        "multi_track_confirm_frames": 1,
    }


def test_motion_config_matches_both_operational_extractor_signatures():
    """Every configured value must reach the real extraction path unchanged."""
    import dataclasses
    import inspect

    from src.motion import detect_clip
    from src.scoring import extract_clip_features

    thresholds = load_thresholds_config(_REPO_THRESHOLDS).motion_thresholds()
    detect_params = inspect.signature(detect_clip).parameters
    feature_params = inspect.signature(extract_clip_features).parameters
    for field in dataclasses.fields(thresholds):
        expected = getattr(thresholds, field.name)
        assert detect_params[field.name].default == expected
        assert feature_params[field.name].default == expected


def test_motion_thresholds_reject_missing_and_unknown_keys():
    cfg = load_thresholds_config(_REPO_THRESHOLDS)
    missing = {section: dict(values) for section, values in cfg.motion.items()}
    del missing["tracking"]["max_miss_frames"]
    with pytest.raises(ConfigError, match=r"missing motion\.tracking\.max_miss_frames"):
        type(cfg)(motion=missing, classification=cfg.classification).motion_thresholds()

    unknown = {section: dict(values) for section, values in cfg.motion.items()}
    unknown["tracking"]["stale_setting"] = 1
    with pytest.raises(ConfigError, match=r"unrecognised motion\.tracking\.stale_setting"):
        type(cfg)(motion=unknown, classification=cfg.classification).motion_thresholds()


def test_motion_thresholds_reject_wrong_scalar_types():
    cfg = load_thresholds_config(_REPO_THRESHOLDS)
    wrong_int = {section: dict(values) for section, values in cfg.motion.items()}
    wrong_int["tracking"]["max_miss_frames"] = 5.5
    with pytest.raises(ConfigError, match=r"max_miss_frames: expected an integer"):
        type(cfg)(motion=wrong_int, classification=cfg.classification).motion_thresholds()

    wrong_bool = {section: dict(values) for section, values in cfg.motion.items()}
    wrong_bool["anchor"]["enabled"] = 1
    with pytest.raises(ConfigError, match=r"anchor\.enabled: expected a boolean"):
        type(cfg)(motion=wrong_bool, classification=cfg.classification).motion_thresholds()


def test_thresholds_missing_section_raises(tmp_path):
    path = _write(tmp_path / "thresholds.yaml", "motion:\n  mog2:\n    history: 500\n")
    with pytest.raises(ConfigError, match="classification"):
        load_thresholds_config(path)


def test_thresholds_non_numeric_leaf_raises(tmp_path):
    path = _write(
        tmp_path / "thresholds.yaml",
        "motion:\n  mog2:\n    history: 500\nclassification:\n  saturation:\n    max_ratio: high\n",
    )
    with pytest.raises(ConfigError, match="expected a number"):
        load_thresholds_config(path)


def test_motion_fingerprint_stable_and_sensitive(tmp_path):
    a = _write(
        tmp_path / "a.yaml",
        "motion:\n  mog2:\n    history: 500\nclassification:\n  saturation:\n    max_ratio: 0.3\n",
    )
    b_same_motion = _write(
        tmp_path / "b.yaml",
        "motion:\n  mog2:\n    history: 500\nclassification:\n  saturation:\n    max_ratio: 0.9\n",
    )
    c_diff_motion = _write(
        tmp_path / "c.yaml",
        "motion:\n  mog2:\n    history: 999\nclassification:\n  saturation:\n    max_ratio: 0.3\n",
    )
    cfg_a = load_thresholds_config(a)
    cfg_b = load_thresholds_config(b_same_motion)
    cfg_c = load_thresholds_config(c_diff_motion)

    # classification-only change must not alter the motion fingerprint (cache reuse).
    assert cfg_a.motion_fingerprint() == cfg_b.motion_fingerprint()
    # motion-setting change must alter it (cache invalidation).
    assert cfg_a.motion_fingerprint() != cfg_c.motion_fingerprint()


# --------------------------------------------------------------------------
# ClassificationThresholds
# --------------------------------------------------------------------------

_REPO_THRESHOLDS = Path(__file__).parents[1] / "config" / "thresholds.yaml"

# The validated operating point of every rule in src.classify, as measured
# against real labelled footage. See src/classify.py's module docstring for the
# derivation of each one and docs/handoff.md for the sessions that produced them.
#
# This is a GOLDEN test, not a restatement of the config file: it exists so that
# editing config/thresholds.yaml cannot quietly move a threshold nobody
# re-measured. If a value here has to change, that means a real re-derivation
# happened -- record it in docs/handoff.md, re-run
# scripts/check_incident_regression.py and a labelled-corpus backtest diff, and
# update this table in the same commit.
_GOLDEN_THRESHOLDS = {
    "green_light_ratio_min": 0.02,
    "green_light_flicker_min": 0.02,
    "warmup_flashlight_ratio_min": 0.0019,
    "warmup_dynamic_frame_fraction_min": 0.8,
    "warmup_dynamic_outside_fraction_max": 0.25,
    "blob_count_peak_min": 10.0,
    "blob_count_median_min": 4.0,
    "implausible_height_fraction_min": 0.5,
    "motion_pixel_fraction_median_min": 0.12,
    "alert_persistence_min": 0.06,
    "camera_artifact_black_white_balance_min": 0.10,
    "outside_pixel_fraction_min": 0.6,
    "median_fence_distance_min": 0.1,
    "median_fence_distance_max": 0.40,
    "corroborated_median_fence_distance_max": 0.45,
    "corroborated_multi_object_fraction_min": 0.90,
    "color_fraction_min": 0.15,
    "row_normalised_area_max": 3000.0,
    "far_outside_subject_height_min": 0.35,
    "far_outside_subject_height_max": 0.90,
    "far_outside_edge_density_max": 0.10,
    "near_fence_distance_min": 0.05,
    "near_fence_blob_count_max": 2.0,
    "straddle_pixel_fraction_min": 0.5,
    "straddle_distance_min": 0.02,
    "inside_blob_count_max": 2.0,
    "jitter_min": 50.0,
    "solidity_max": 0.85,
    "blob_white_fraction_min": 0.4,
    "large_blob_frame_fraction_min": 0.34,
    "large_blob_white_fraction_min": 0.20,
    "long_flare_frames_min": 18.0,
    # neighbour_candidate, added 2026-09-09 (docs/detection_improvement_review.md
    # section 2.2): person-sized (real-world height) + outside + real daylight.
    # 4/5 labelled neighbour clips fire (3/3 events); see src/classify.py's
    # docstring for the full derivation and the one miss's explanation.
    "neighbour_subject_height_min": 0.9,
    "neighbour_subject_height_max": 2.2,
}


def test_classification_thresholds_match_validated_values():
    thresholds = load_thresholds_config(_REPO_THRESHOLDS).classification_thresholds()
    actual = {name: getattr(thresholds, name) for name in _GOLDEN_THRESHOLDS}
    assert actual == _GOLDEN_THRESHOLDS


def test_classification_thresholds_covers_every_field():
    # Guards against a field being added to the dataclass but never mapped to a
    # yaml key, which would otherwise surface as a confusing TypeError.
    import dataclasses

    from src.config import ClassificationThresholds

    fields = {f.name for f in dataclasses.fields(ClassificationThresholds)}
    assert fields == set(_GOLDEN_THRESHOLDS)


def test_flashlight_candidate_ratio_stays_coupled_to_classify_threshold():
    """src.features.FLASHLIGHT_CANDIDATE_MIN_RATIO is deliberately the same bar
    as the classifier's green_light_ratio_min -- one "is this a real flashlight,
    not noise" standard, so a candidate is never held to a different standard
    than a finished clip's own best_contour (see src/features.py's comment).

    That coupling used to be visible as two adjacent constants; now that the
    classify side lives in config/thresholds.yaml it is invisible, so this test
    is what makes a desync fail loudly instead of silently changing which
    flashlight candidates the tracker prefers.
    """
    from src.features import FLASHLIGHT_CANDIDATE_MIN_RATIO

    thresholds = load_thresholds_config(_REPO_THRESHOLDS).classification_thresholds()
    assert FLASHLIGHT_CANDIDATE_MIN_RATIO == thresholds.green_light_ratio_min


def test_classification_thresholds_missing_key_raises(tmp_path):
    raw = _REPO_THRESHOLDS.read_text().replace("    color_fraction_min: 0.15\n", "")
    path = _write(tmp_path / "thresholds.yaml", raw)
    with pytest.raises(ConfigError, match=r"missing classification\.outside\.color_fraction_min"):
        load_thresholds_config(path).classification_thresholds()


def test_classification_thresholds_unrecognised_key_raises(tmp_path):
    raw = _REPO_THRESHOLDS.read_text().replace(
        "  animal:\n", "  animal:\n    aspect_ratio_upright_min: 1.6\n"
    )
    path = _write(tmp_path / "thresholds.yaml", raw)
    with pytest.raises(ConfigError, match=r"unrecognised classification\.animal\."):
        load_thresholds_config(path).classification_thresholds()


def test_classification_thresholds_missing_section_raises(tmp_path):
    raw = _REPO_THRESHOLDS.read_text().replace("  insect:\n", "  unused_section:\n")
    path = _write(tmp_path / "thresholds.yaml", raw)
    with pytest.raises(ConfigError, match=r"classification\.insect"):
        load_thresholds_config(path).classification_thresholds()


def test_classification_fingerprint_ignores_motion(tmp_path):
    raw = _REPO_THRESHOLDS.read_text()
    baseline = load_thresholds_config(_write(tmp_path / "a.yaml", raw))
    motion_changed = load_thresholds_config(
        _write(tmp_path / "b.yaml", raw.replace("history: 500", "history: 999"))
    )
    classification_changed = load_thresholds_config(
        _write(tmp_path / "c.yaml", raw.replace("jitter_min: 50", "jitter_min: 55"))
    )

    # A motion edit must not change what a backtest run records as its
    # thresholds hash -- it did not change any classification result.
    assert baseline.classification_fingerprint() == motion_changed.classification_fingerprint()
    assert (
        baseline.classification_fingerprint() != classification_changed.classification_fingerprint()
    )
