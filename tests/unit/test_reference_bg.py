import json

import numpy as np
import pytest

from src.config import Camera, CameraZone
from src.errors import ConfigError
from src.reference_bg import (
    ReferenceEntry,
    align,
    bucket_clips,
    combine,
    era_of,
    load_manifest,
    quarter_of,
    reference_for,
    sample_evenly,
)


def _zone():
    return CameraZone(fence=None, outside=None, depth_cutoff=0.05, ignore=())


def _entry(camera_id, daylight, era, start, end, path="x.png"):
    return ReferenceEntry(
        camera_id=camera_id,
        daylight=daylight,
        era=era,
        start=start,
        end=end,
        clip_count=10,
        path=path,
    )


def test_era_of_returns_none_for_a_camera_with_no_dated_history():
    camera = Camera(id="cam05", aliases=(), order=1, zone=_zone(), threshold_overrides={})
    assert era_of(camera, "2026-03-02T16:13:54Z") is None
    assert era_of(None, "2026-03-02T16:13:54Z") is None


def test_era_of_splits_on_the_remount_timestamp():
    from datetime import UTC, datetime

    remount = datetime(2026, 3, 2, 16, 13, 54, tzinfo=UTC)
    camera = Camera(
        id="cam12",
        aliases=(),
        order=1,
        zone=_zone(),
        threshold_overrides={},
        zone_history=((None, _zone()), (remount, _zone())),
    )
    assert era_of(camera, "2025-02-21T16:15:17Z") is None
    assert era_of(camera, "2026-03-02T16:18:11Z") == "2026-03-02T16:13:54Z"


def test_quarter_of_buckets_by_calendar_quarter():
    assert quarter_of("2026-01-05T00:00:00Z") == (2026, 0)
    assert quarter_of("2026-03-31T23:59:00Z") == (2026, 0)
    assert quarter_of("2026-04-01T00:00:00Z") == (2026, 1)


def test_bucket_clips_folds_a_sparse_quarter_into_the_nearest_populated_one():
    # Six clips in one night quarter, two in the next -- the pair alone would
    # make a thin, noisy reference, so they join their neighbour instead.
    clips = [(f"2026-01-0{i + 1}T22:00:00Z", f"q1_{i}.mp4") for i in range(6)]
    clips += [(f"2026-04-0{i + 1}T22:00:00Z", f"q2_{i}.mp4") for i in range(2)]

    buckets = bucket_clips(clips, None)

    assert len(buckets) == 1
    assert len(next(iter(buckets.values()))) == 8


def test_bucket_clips_keeps_day_and_night_apart():
    clips = [(f"2026-01-0{i + 1}T22:00:00Z", f"n{i}.mp4") for i in range(6)]
    clips += [(f"2026-01-0{i + 1}T10:00:00Z", f"d{i}.mp4") for i in range(6)]

    buckets = bucket_clips(clips, None)

    assert {key[1] for key in buckets} == {True, False}


def test_sample_evenly_spans_the_bucket_rather_than_taking_a_prefix():
    paths = [str(i) for i in range(100)]
    chosen = sample_evenly(paths, limit=4)
    assert chosen == ["0", "25", "50", "75"]


def test_combine_drops_odd_frame_sizes_rather_than_failing():
    common = [np.full((8, 8), 100, np.uint8) for _ in range(3)]
    odd = np.full((6, 6), 200, np.uint8)
    result = combine([*common, odd])
    assert result is not None
    assert result.shape == (8, 8)


def test_combine_returns_none_without_input():
    assert combine([]) is None


def test_reference_for_picks_the_bucket_nearest_in_time():
    entries = [
        _entry("cam05", False, None, "2024-01-01T00:00:00Z", "2024-03-31T00:00:00Z", "old.png"),
        _entry("cam05", False, None, "2026-01-01T00:00:00Z", "2026-03-31T00:00:00Z", "new.png"),
    ]
    chosen = reference_for(entries, "cam05", "2026-02-10T22:00:00Z")
    assert chosen is not None
    assert chosen.path == "new.png"


def test_reference_for_will_not_reuse_a_previous_eras_reference():
    # A remounted camera with no material yet must get nothing, not the old
    # scene -- the whole frame has moved, so the stale reference is worse than
    # having none at all.
    entries = [
        _entry("cam12", False, None, "2024-01-01T00:00:00Z", "2025-12-31T00:00:00Z"),
    ]
    chosen = reference_for(
        entries, "cam12", "2026-03-02T16:18:11Z", era="2026-03-02T16:13:54Z"
    )
    assert chosen is None


def test_reference_for_will_not_cross_lighting():
    entries = [_entry("cam05", True, None, "2026-01-01T00:00:00Z", "2026-03-31T00:00:00Z")]
    assert reference_for(entries, "cam05", "2026-02-10T22:00:00Z", daylight=False) is None


def test_align_leaves_an_already_registered_reference_alone():
    image = np.zeros((40, 40), np.uint8)
    image[10:20, 10:20] = 200
    aligned, shift = align(image, image)
    assert shift < 0.5
    assert np.array_equal(aligned, image)


def test_align_shifts_a_drifted_reference_back_onto_the_clip():
    reference = np.zeros((64, 64), np.float32)
    reference[20:30, 20:30] = 255
    drifted = np.roll(reference, 5, axis=1)
    aligned, shift = align(reference.astype(np.uint8), drifted.astype(np.uint8))
    assert shift > 1.0
    # The bright block should now sit where the clip has it, not where the
    # reference originally did.
    assert aligned[20:30, 25:35].mean() > aligned[20:30, 20:25].mean()


def test_align_declines_on_a_size_mismatch():
    reference = np.zeros((40, 40), np.uint8)
    other = np.zeros((30, 30), np.uint8)
    aligned, shift = align(reference, other)
    assert shift == 0.0
    assert aligned.shape == (40, 40)


def test_load_manifest_is_empty_when_nothing_has_been_built(tmp_path):
    assert load_manifest(tmp_path) == []


def test_load_manifest_reports_a_corrupt_file(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_manifest(tmp_path)


def test_load_manifest_round_trips_entries(tmp_path):
    entry = _entry("cam05", False, None, "2026-01-01T00:00:00Z", "2026-03-31T00:00:00Z")
    (tmp_path / "manifest.json").write_text(json.dumps([entry.__dict__]))
    assert load_manifest(tmp_path) == [entry]
