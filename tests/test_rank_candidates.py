from pathlib import Path

import numpy as np
import pytest

from scripts import rank_candidates as rc
from src import db
from src.config import CameraZone


def test_auc_perfect_separation():
    pos = np.array([0.9, 0.8, 0.7])
    neg = np.array([0.1, 0.2, 0.3])
    assert rc.auc(pos, neg) == pytest.approx(1.0)


def test_auc_ties_count_as_half():
    pos = np.array([0.5])
    neg = np.array([0.5])
    assert rc.auc(pos, neg) == pytest.approx(0.5)


def test_auc_no_positives_is_nan():
    assert np.isnan(rc.auc(np.array([]), np.array([0.1])))


def test_fit_logistic_separates_a_trivial_dataset():
    rng = np.random.default_rng(0)
    x_pos = rng.normal(loc=3.0, scale=0.3, size=(20, 2))
    x_neg = rng.normal(loc=-3.0, scale=0.3, size=(20, 2))
    x = np.vstack([x_pos, x_neg])
    y = np.array([1.0] * 20 + [0.0] * 20)

    weights = rc.fit_logistic(x, y, iterations=1000)
    scores = rc.predict_proba(x, weights)

    assert rc.auc(scores[y == 1], scores[y == 0]) > 0.95


def test_leave_one_out_auc_on_separable_data():
    rng = np.random.default_rng(1)
    x_pos = rng.normal(loc=3.0, scale=0.3, size=(15, 2))
    x_neg = rng.normal(loc=-3.0, scale=0.3, size=(15, 2))
    x = np.vstack([x_pos, x_neg])
    y = np.array([1.0] * 15 + [0.0] * 15)

    assert rc.leave_one_out_auc(x, y, iterations=500) > 0.9


def test_stratified_top_n_caps_per_camera_not_globally():
    rows = [
        {"camera_id": "cam01", "score": s} for s in (0.9, 0.8, 0.7, 0.6)
    ] + [{"camera_id": "cam02", "score": s} for s in (0.5, 0.4)]

    queue = rc.stratified_top_n(rows, top_n=2)

    assert sum(1 for r in queue if r["camera_id"] == "cam01") == 2
    assert sum(1 for r in queue if r["camera_id"] == "cam02") == 2
    # sorted overall by score descending
    assert [r["score"] for r in queue] == [0.9, 0.8, 0.5, 0.4]


def test_stratified_top_n_sorts_within_camera_first():
    rows = [{"camera_id": "cam01", "score": s} for s in (0.1, 0.9, 0.5)]
    queue = rc.stratified_top_n(rows, top_n=2)
    assert [r["score"] for r in queue] == [0.9, 0.5]


def _detected_row(camera_id, message_id, label, **feature_overrides):
    row = {
        "channel_id": "chan1",
        "message_id": message_id,
        "camera_id": camera_id,
        "timestamp": "2026-01-01T00:00:00Z",
        "label": label,
        "detected": True,
    }
    for feat in rc.NUMERIC_FEATURES:
        row[feat] = 0.0
    row.update(feature_overrides)
    return row


def test_rank_and_write_scores_and_writes_only_unlabelled(tmp_path: Path):
    rows = [
        _detected_row("cam01", 1, "animal", aspect_ratio=0.5, jitter=1.0),
        _detected_row("cam01", 2, "guard", aspect_ratio=1.2, green_light_ratio=0.3),
        _detected_row("cam01", 3, None, aspect_ratio=0.5, jitter=1.0),  # unlabelled, animal-like
        _detected_row("cam01", 4, None, aspect_ratio=1.2, green_light_ratio=0.3),  # guard-like
    ]
    out_path = tmp_path / "candidates.csv"

    queue = rc.rank_and_write(rows, top_per_camera=10, out_path=str(out_path))

    assert {r["message_id"] for r in queue} == {3, 4}
    assert out_path.exists()
    ids_path = out_path.with_suffix(".message_ids")
    assert ids_path.exists()
    assert set(ids_path.read_text().split()) == {"3", "4"}


def test_rank_and_write_raises_without_labelled_rows(tmp_path: Path):
    rows = [_detected_row("cam01", 1, None)]
    with pytest.raises(ValueError, match="no labelled"):
        rc.rank_and_write(rows, top_per_camera=10, out_path=str(tmp_path / "out.csv"))


def test_collect_features_uses_injected_extract_fn(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=1,
        camera_id="cam01",
        timestamp="2026-01-01T00:00:00Z",
        caption=None,
        file_path="clip1.mp4",
        source="backfill",
    )
    db.upsert_label(conn, channel_id="chan1", message_id=1, label="guard")

    class _Cameras:
        def by_id(self, camera_id):
            return _Camera()

    class _Camera:
        zone = CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=())

    def fake_extract(file_path, zone):
        return {"aspect_ratio": 1.0}

    rows = rc.collect_features(conn, _Cameras(), extract_fn=fake_extract)

    assert len(rows) == 1
    assert rows[0]["label"] == "guard"
    assert rows[0]["detected"] is True
    assert rows[0]["aspect_ratio"] == 1.0
