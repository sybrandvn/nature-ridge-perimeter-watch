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
    assert "mog2" in cfg.motion
    assert "saturation" in cfg.classification


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
