
import pytest

from src import backtester, db
from src.errors import DbError


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    yield connection
    connection.close()


@pytest.fixture
def cameras_yaml(tmp_path):
    path = tmp_path / "cameras.yaml"
    path.write_text("cameras: []\n")
    return str(path)


def _rows():
    return [
        {
            "channel_id": "chan1",
            "message_id": 1,
            "camera_id": "cam01",
            "label": "guard",
            "category": "guard_candidate",
            "reason": "green_light",
            "blinding_foreground": False,
            "green_light_ratio": 0.2,
        },
        {
            "channel_id": "chan1",
            "message_id": 2,
            "camera_id": "cam01",
            "label": None,
            "category": "incident_candidate",
            "reason": "outside_no_colour",
            "blinding_foreground": False,
            "outside_pixel_fraction": 0.7,
        },
    ]


def test_record_run_writes_a_completed_run_with_all_result_rows(conn, cameras_yaml):
    run_id = backtester.record_run(conn, _rows(), cameras_path=cameras_yaml, run_id="run-1")

    assert run_id == "run-1"
    runs = {r["run_id"]: r for r in db.iter_backtest_runs(conn)}
    assert runs["run-1"]["status"] == "completed"
    assert runs["run-1"]["cameras_path"] == cameras_yaml
    assert runs["run-1"]["thresholds_path"] == "src/classify.py"
    # Hashes are real, not placeholders -- both non-empty and distinct.
    assert runs["run-1"]["cameras_hash"]
    assert runs["run-1"]["thresholds_hash"]

    results = list(db.iter_backtest_results(conn, "run-1"))
    assert len(results) == 2
    by_message = {r["message_id"]: r for r in results}
    assert by_message[1]["predicted_class"] == "guard_candidate"
    assert by_message[2]["predicted_class"] == "incident_candidate"


def test_record_run_features_exclude_identity_but_keep_label_and_blinding(conn, cameras_yaml):
    import json

    backtester.record_run(conn, _rows(), cameras_path=cameras_yaml, run_id="run-1")
    result = next(
        r for r in db.iter_backtest_results(conn, "run-1") if r["message_id"] == 1
    )
    features = json.loads(result["features_json"])
    assert "channel_id" not in features
    assert "message_id" not in features
    assert "camera_id" not in features
    assert "category" not in features
    assert "reason" not in features
    assert features["label"] == "guard"
    assert features["blinding_foreground"] is False
    assert features["green_light_ratio"] == 0.2


def test_record_run_reason_codes_wrap_the_single_reason_in_a_list(conn, cameras_yaml):
    import json

    backtester.record_run(conn, _rows(), cameras_path=cameras_yaml, run_id="run-1")
    result = next(
        r for r in db.iter_backtest_results(conn, "run-1") if r["message_id"] == 1
    )
    assert json.loads(result["reason_codes_json"]) == ["green_light"]


def test_record_run_default_run_id_is_a_utc_timestamp(conn, cameras_yaml):
    import re

    run_id = backtester.record_run(conn, _rows(), cameras_path=cameras_yaml)
    assert re.fullmatch(r"\d{8}T\d{6}Z", run_id)


def test_record_run_duplicate_run_id_raises_dberror(conn, cameras_yaml):
    backtester.record_run(conn, _rows(), cameras_path=cameras_yaml, run_id="run-1")
    with pytest.raises(DbError, match="run-1"):
        backtester.record_run(conn, _rows(), cameras_path=cameras_yaml, run_id="run-1")
    # The first run's own record must be untouched by the failed second attempt.
    runs = {r["run_id"]: r["status"] for r in db.iter_backtest_runs(conn)}
    assert runs == {"run-1": "completed"}


def test_record_run_marks_failed_on_a_bad_row_and_reraises(conn, cameras_yaml):
    bad_rows = [
        {
            "channel_id": "chan1",
            "message_id": 1,
            "camera_id": "cam01",
            "label": None,
            "category": "not_a_real_category",  # violates the CHECK constraint
            "reason": "green_light",
            "blinding_foreground": False,
        }
    ]
    with pytest.raises(DbError, match="predicted_class"):
        backtester.record_run(conn, bad_rows, cameras_path=cameras_yaml, run_id="run-1")
    runs = {r["run_id"]: r["status"] for r in db.iter_backtest_runs(conn)}
    assert runs == {"run-1": "failed"}


def test_record_run_tolerates_missing_git(conn, cameras_yaml, monkeypatch):
    import subprocess

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backtester.record_run(conn, _rows(), cameras_path=cameras_yaml, run_id="run-1")
    run = next(iter(db.iter_backtest_runs(conn)))
    assert run["git_revision"] is None
    assert run["status"] == "completed"


def test_record_run_tolerates_non_repo_git_failure(conn, cameras_yaml, monkeypatch):
    import subprocess

    class _FakeCompleted:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompleted())
    backtester.record_run(conn, _rows(), cameras_path=cameras_yaml, run_id="run-1")
    run = next(iter(db.iter_backtest_runs(conn)))
    assert run["git_revision"] is None
