import os
from pathlib import Path

import pytest

from scripts.container_healthcheck import (
    check_heartbeat,
    check_media_round_trip,
    check_readiness,
    main,
)

ROOT = Path(__file__).parents[1]


def test_readiness_initializes_and_reopens_persistent_database(tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "perimeter_watch.db"

    first = check_readiness(
        data_dir=data_dir,
        db_path=db_path,
        cameras_path=ROOT / "config/cameras.yaml",
        thresholds_path=ROOT / "config/thresholds.yaml",
    )
    second = check_readiness(
        data_dir=data_dir,
        db_path=db_path,
        cameras_path=ROOT / "config/cameras.yaml",
        thresholds_path=ROOT / "config/thresholds.yaml",
    )

    assert first["ready"] is True
    assert first["schema_version"] == second["schema_version"]
    assert db_path.is_file()
    assert (data_dir / "history").is_dir()
    assert (data_dir / "reference_bg").is_dir()


def test_heartbeat_requires_a_fresh_real_file(tmp_path):
    heartbeat = tmp_path / "watcher.heartbeat"
    heartbeat.write_text("ready\n")
    os.utime(heartbeat, (900.0, 900.0))

    assert check_heartbeat(heartbeat, max_age_seconds=120.0, now=1000.0) == 100.0
    with pytest.raises(RuntimeError, match="stale"):
        check_heartbeat(heartbeat, max_age_seconds=50.0, now=1000.0)
    with pytest.raises(RuntimeError, match="missing"):
        check_heartbeat(tmp_path / "missing", max_age_seconds=120.0, now=1000.0)


def test_media_round_trip_produces_readable_h264(tmp_path):
    result = check_media_round_trip(work_dir=tmp_path)

    assert result["media_smoke"] is True
    assert result["bytes"] > 0
    assert result["decoded_shape"] == [64, 96, 3]


def test_main_returns_nonzero_and_json_for_failed_readiness(tmp_path, capsys):
    exit_code = main(
        [
            "--readiness",
            "--data-dir",
            str(tmp_path / "data"),
            "--db-path",
            str(tmp_path / "data/db.sqlite"),
            "--cameras",
            str(tmp_path / "missing-cameras.yaml"),
            "--thresholds",
            str(ROOT / "config/thresholds.yaml"),
        ]
    )

    assert exit_code == 1
    assert '"ready": false' in capsys.readouterr().out


def test_docker_configuration_is_non_root_locked_and_persistent():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "compose.yaml").read_text()
    dockerignore = (ROOT / ".dockerignore").read_text()

    assert "uv sync --frozen --no-dev" in dockerfile
    assert "USER perimeter" in dockerfile
    assert 'PYTHONPATH="/app"' in dockerfile
    assert 'ENTRYPOINT ["/usr/bin/tini", "--"]' in dockerfile
    assert "./data:/app/data" in compose
    assert "read_only: true" in compose
    assert "no-new-privileges:true" in compose
    assert "session-bootstrap:" in compose
    assert "watcher:" in compose
    assert 'command: ["python", "-m", "scripts.watch"]' in compose
    assert "restart: unless-stopped" in compose
    assert '"--heartbeat"' in compose
    assert ".env" in dockerignore
    assert "data" in dockerignore
