from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import spike
from src import db
from src.config import Camera, CamerasConfig, CameraZone

_ZONE = CameraZone(fence=((0.0, 0.5), (1.0, 0.5)), outside="left", depth_cutoff=0.0, ignore=())


def _blank_frame(size: int = 60, value: int = 0) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _frame_with_square(pos: int, size: int = 60) -> np.ndarray:
    frame = _blank_frame(size)
    cv2.rectangle(frame, (pos, pos), (pos + 8, pos + 8), (255, 255, 255), thickness=-1)
    return frame


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self._idx = 0

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame

    def get(self, _prop) -> float:
        return 10.0  # matches extract_clip_features' own "or 10.0" fallback

    def release(self) -> None:
        pass


def test_largest_contour_returns_none_for_empty_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    assert spike.largest_contour(mask) is None


def test_largest_contour_finds_blob():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    contour = spike.largest_contour(mask)
    assert contour is not None
    assert cv2.contourArea(contour) > 0


def test_largest_contour_rejects_blob_over_max_area():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[0:35, 0:35] = 255  # whole-frame illumination change
    mask[37:39, 37:39] = 255  # small real subject
    unfiltered = spike.largest_contour(mask)
    filtered = spike.largest_contour(mask, max_area=100)
    assert cv2.contourArea(unfiltered) > 100
    assert filtered is not None
    assert cv2.contourArea(filtered) <= 100


def test_largest_contour_returns_none_when_all_blobs_over_max_area():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[0:35, 0:35] = 255
    assert spike.largest_contour(mask, max_area=100) is None


def test_contour_centroid_of_square():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4:10, 4:10] = 255
    contour = spike.largest_contour(mask)
    centroid = spike.contour_centroid(contour)
    assert centroid == pytest.approx((6.5, 6.5), abs=1.0)


def test_normalized_contour_points_scales_to_unit_range():
    contour = np.array([[[0, 0]], [[50, 0]], [[50, 100]], [[0, 100]]], dtype=np.int32)
    points = spike.normalized_contour_points(contour, frame_width=100, frame_height=100)
    assert points == [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]


def _square_contour(x: int, y: int, size: int) -> np.ndarray:
    mask = np.zeros((y + size + 5, x + size + 5), dtype=np.uint8)
    mask[y : y + size, x : x + size] = 255
    return spike.largest_contour(mask)


def _rect_contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros((y + h + 5, x + w + 5), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return spike.largest_contour(mask)


def test_track_contour_returns_largest_when_no_active_track():
    small = _square_contour(0, 0, 4)
    big = _square_contour(40, 40, 10)
    assert spike.track_contour([small, big], None, max_jump_distance=100) is big


def test_track_contour_prefers_overlap_over_size():
    # A previous track near (5, 5)-(15, 15): a small candidate that overlaps it
    # should win over a much bigger candidate that is elsewhere in frame -- this
    # is what stops the tracked identity flipping between two co-occurring
    # subjects just because their relative blob sizes swap between frames. A
    # generous size-change ratio here isolates this from the size-change cap
    # (covered separately below) so this purely tests overlap preference.
    track_bbox = (5, 5, 15, 15)
    overlapping_small = _square_contour(5, 5, 4)
    far_big = _square_contour(40, 40, 10)
    result = spike.track_contour(
        [overlapping_small, far_big],
        track_bbox,
        max_jump_distance=100,
        max_size_change_ratio=10.0,
    )
    assert result is overlapping_small


def test_track_contour_falls_back_to_nearest_centroid_without_overlap():
    # No candidate overlaps the previous box, but one is much closer to it --
    # a generous size margin and size-change ratio here isolate this from the
    # size-relative and size-change caps (covered separately below) so this
    # purely tests nearest-centroid choice.
    track_bbox = (5, 5, 15, 15)
    nearby = _square_contour(20, 20, 4)
    far = _square_contour(80, 80, 4)
    result = spike.track_contour(
        [nearby, far],
        track_bbox,
        max_jump_distance=50,
        min_size_margin=50,
        max_size_change_ratio=10.0,
    )
    assert result is nearby


def test_track_contour_returns_none_when_track_is_lost_rather_than_reacquiring_blindly():
    # Neither candidate is within max_jump_distance of the old track -- report
    # a miss (None) instead of force-matching onto something unrelated; the
    # caller's appearance-recovery / miss-tolerance machinery gets the chance
    # to reacquire properly, or drop the track, before anything jumps.
    track_bbox = (5, 5, 15, 15)
    small = _square_contour(100, 100, 4)
    big = _square_contour(140, 140, 10)
    result = spike.track_contour([small, big], track_bbox, max_jump_distance=5)
    assert result is None


def test_track_contour_returns_none_for_no_candidates():
    assert spike.track_contour([], (0, 0, 10, 10), max_jump_distance=10) is None


def test_track_contour_size_relative_cap_beats_generous_max_jump_distance():
    # A tiny (4x4) track shouldn't be allowed to jump 21px to a candidate just
    # because max_jump_distance (a generous, frame-wide ceiling) allows it --
    # the effective jump distance is also capped near the track's own size, so
    # this candidate is rejected. Reports a miss (None) rather than jumping to
    # whatever else is in frame.
    track_bbox = (5, 5, 9, 9)
    nearby = _square_contour(20, 20, 4)  # ~21px from the track centre
    big = _square_contour(100, 100, 10)
    result = spike.track_contour([nearby, big], track_bbox, max_jump_distance=50)
    assert result is None


def test_track_contour_rejects_overlapping_candidate_with_implausible_size_change():
    # A candidate that overlaps the track's last box but whose area collapses
    # by far more than max_size_change_ratio (both ways) is not a plausible
    # continuation -- e.g. a large residual-illumination blob shrinking away
    # while a much smaller, unrelated blob happens to sit inside it. Falls
    # through to nearest-centroid, which also fails the same size check here,
    # so this reports a miss (None) rather than "handing off" onto the tiny
    # blob under the same identity.
    track_bbox = (0, 0, 20, 20)  # area 400
    tiny_overlap = _square_contour(0, 0, 2)  # area 4, ratio 100x -- way outside cap
    result = spike.track_contour(
        [tiny_overlap], track_bbox, max_jump_distance=50, max_size_change_ratio=4.0
    )
    assert result is None


def test_track_contour_accepts_overlapping_candidate_within_size_change_cap():
    # Same shape of overlap as above, but the candidate's area is within the
    # default cap of the track's own size -- a normal, gradual size change
    # (e.g. the subject turning or moving slightly closer/further) should
    # still be accepted as a continuation.
    track_bbox = (0, 0, 20, 20)  # area 400
    modest_shrink = _square_contour(0, 0, 12)  # area 144, ratio ~2.8x -- within cap
    result = spike.track_contour(
        [modest_shrink], track_bbox, max_jump_distance=50, max_size_change_ratio=4.0
    )
    assert result is modest_shrink


def test_track_contour_fresh_start_rejects_candidates_below_min_reacquire_area():
    # No active track (track_bbox=None) and every candidate is a noise-speck
    # size -- report a miss instead of picking the "largest" tiny speck, same
    # family as the size/distance implausibility checks above but for a fresh
    # (re)acquisition with nothing established yet to compare against.
    speck = _square_contour(10, 10, 3)  # area 9
    other_speck = _square_contour(50, 50, 4)  # area 16
    result = spike.track_contour(
        [speck, other_speck], None, max_jump_distance=100, min_reacquire_area=20.0
    )
    assert result is None


def test_track_contour_fresh_start_still_picks_largest_candidate_clearing_the_floor():
    speck = _square_contour(10, 10, 3)  # area 9, below floor
    real = _square_contour(50, 50, 10)  # area 100, clears floor
    result = spike.track_contour(
        [speck, real], None, max_jump_distance=100, min_reacquire_area=20.0
    )
    assert result is real


def test_track_contour_min_reacquire_area_defaults_on_for_fresh_starts():
    # track_contour's own default (20.0) applies even when callers don't pass
    # it explicitly -- a lone noise-speck candidate with no active track is a
    # miss, not a pick, unless a caller explicitly opts out with 0.0.
    speck = _square_contour(10, 10, 3)  # area 9
    assert spike.track_contour([speck], None, max_jump_distance=100) is None
    assert (
        spike.track_contour([speck], None, max_jump_distance=100, min_reacquire_area=0.0)
        is speck
    )


def test_track_multiple_objects_assigns_stable_ids_to_two_independent_subjects():
    # Two subjects, far enough apart to never share a candidate, should each
    # keep the same id across every frame they appear in.
    frame0 = [_square_contour(10, 10, 4), _square_contour(60, 10, 4)]
    frame1 = [_square_contour(12, 10, 4), _square_contour(58, 10, 4)]
    frame2 = [_square_contour(14, 10, 4), _square_contour(56, 10, 4)]
    results = spike.track_multiple_objects(
        [frame0, frame1, frame2], max_jump_distance=50, max_track_miss_frames=2
    )
    ids_by_frame = [{t.track_id for t in frame} for frame in results]
    assert ids_by_frame == [{0, 1}] * 3
    assert all(t.merged_ids == () for frame in results for t in frame)


def test_track_multiple_objects_spawns_new_id_for_later_unmatched_candidate():
    frame0 = [_square_contour(10, 10, 4)]
    frame1 = [_square_contour(12, 10, 4), _square_contour(80, 80, 4)]
    results = spike.track_multiple_objects(
        [frame0, frame1], max_jump_distance=50, max_track_miss_frames=2
    )
    assert {t.track_id for t in results[0]} == {0}
    assert {t.track_id for t in results[1]} == {0, 1}


def test_track_multiple_objects_drops_track_after_miss_tolerance():
    frame0 = [_square_contour(10, 10, 4)]
    empty: list[np.ndarray] = []
    # Reappearing well after the miss tolerance should be a fresh id, not a
    # continuation of the dropped one.
    results = spike.track_multiple_objects(
        [frame0, empty, empty, empty, frame0], max_jump_distance=50, max_track_miss_frames=1
    )
    assert {t.track_id for t in results[0]} == {0}
    assert results[1] == [] and results[2] == [] and results[3] == []
    assert {t.track_id for t in results[4]} == {1}


def test_track_multiple_objects_keeps_both_ids_alive_through_a_merge_and_resplits():
    # Two subjects converge into one background-diff blob for a frame, then
    # separate again -- both ids should survive the merge (flagged via
    # merged_ids) and reattach to the correct side once they split back out.
    frame0 = [_square_contour(10, 10, 4), _square_contour(30, 10, 4)]
    frame1 = [_square_contour(12, 10, 4), _square_contour(28, 10, 4)]
    merged = [_square_contour(12, 10, 20)]  # spans both subjects' last boxes
    frame3 = [_square_contour(16, 10, 4), _square_contour(24, 10, 4)]
    results = spike.track_multiple_objects(
        [frame0, frame1, merged, frame3], max_jump_distance=50, max_track_miss_frames=2
    )

    merge_frame = results[2]
    assert {t.track_id for t in merge_frame} == {0, 1}
    for t in merge_frame:
        assert t.merged_ids == (1 - t.track_id,)

    split_frame = results[3]
    assert {t.track_id for t in split_frame} == {0, 1}
    split_by_id = {t.track_id: t.bbox for t in split_frame}
    left_box, right_box = spike._contour_bbox(frame3[0]), spike._contour_bbox(frame3[1])
    # id 0 was on the left before the merge and should reattach to the left
    # candidate after the split, not the right one.
    assert split_by_id[0] == left_box
    assert split_by_id[1] == right_box


def _draw_textured_patch(img: np.ndarray, x: int, y: int, outer: int, inner: int) -> None:
    # Solid-fill blocks have zero internal variance, which makes normalised
    # cross-correlation undefined -- give the patch real internal structure so
    # a template match has something to lock onto.
    img[y : y + 8, x : x + 8] = outer
    img[y + 2 : y + 6, x + 2 : x + 6] = inner


def test_reacquire_by_template_finds_shifted_low_contrast_patch():
    # Same shape/pattern as the template, shifted a few pixels and at much
    # lower absolute contrast -- simulates a subject that has visually blended
    # into the background (too small a brightness delta to diff-detect) but
    # hasn't changed shape or moved far.
    template_frame = np.zeros((40, 40), dtype=np.uint8)
    _draw_textured_patch(template_frame, 10, 10, 200, 100)
    template = template_frame[10:18, 10:18]

    frame = np.zeros((40, 40), dtype=np.uint8)
    _draw_textured_patch(frame, 15, 12, 15, 8)  # low contrast, shifted by (5, 2)

    match = spike.reacquire_by_template(
        frame, template, last_bbox=(10, 10, 18, 18), search_margin=15, match_threshold=0.5
    )

    assert match is not None
    x0, y0, _x1, _y1 = match
    assert (x0, y0) == (15, 12)


def test_reacquire_by_template_returns_none_when_nothing_matches():
    frame = np.zeros((40, 40), dtype=np.uint8)  # nothing resembling the template anywhere
    template = np.zeros((8, 8), dtype=np.uint8)
    template[2:6, 2:6] = 200

    match = spike.reacquire_by_template(
        frame, template, last_bbox=(10, 10, 18, 18), search_margin=15, match_threshold=0.9
    )

    assert match is None


def test_reacquire_by_template_returns_none_when_search_window_too_small():
    frame = np.zeros((10, 10), dtype=np.uint8)
    template = np.zeros((8, 8), dtype=np.uint8)
    template[2:6, 2:6] = 200

    # Near the corner with almost no margin -- the clipped window ends up
    # smaller than the template itself.
    match = spike.reacquire_by_template(
        frame, template, last_bbox=(8, 8, 10, 10), search_margin=1, match_threshold=0.1
    )

    assert match is None


def test_reacquire_by_template_ignores_far_patch_without_velocity():
    # A tight, size-relative margin (no direction bias) shouldn't reach a
    # patch shifted 22px away from the last known position.
    template_frame = np.zeros((60, 60), dtype=np.uint8)
    _draw_textured_patch(template_frame, 10, 10, 200, 100)
    template = template_frame[10:18, 10:18]

    frame = np.zeros((60, 60), dtype=np.uint8)
    _draw_textured_patch(frame, 32, 10, 15, 8)  # shifted +22 in x

    match = spike.reacquire_by_template(
        frame, template, last_bbox=(10, 10, 18, 18), search_margin=5, match_threshold=0.5
    )

    assert match is None


def test_reacquire_by_template_velocity_extends_search_in_direction_of_travel():
    # Same scene as above, but with a velocity matching the subject's actual
    # displacement -- the search window should stretch forward enough to find
    # it, without needing a larger base margin.
    template_frame = np.zeros((60, 60), dtype=np.uint8)
    _draw_textured_patch(template_frame, 10, 10, 200, 100)
    template = template_frame[10:18, 10:18]

    frame = np.zeros((60, 60), dtype=np.uint8)
    _draw_textured_patch(frame, 32, 10, 15, 8)  # shifted +22 in x

    match = spike.reacquire_by_template(
        frame,
        template,
        last_bbox=(10, 10, 18, 18),
        search_margin=5,
        match_threshold=0.5,
        velocity=(20.0, 0.0),
    )

    assert match is not None
    x0, y0, _x1, _y1 = match
    assert (x0, y0) == (32, 10)


def test_reacquire_by_template_velocity_still_finds_reversal_near_last_position():
    # A subject that doubles back (moves opposite to its last velocity)
    # should still be found close to its last known position -- the trailing
    # edge of the window isn't shrunk by a forward velocity bias.
    template_frame = np.zeros((60, 60), dtype=np.uint8)
    _draw_textured_patch(template_frame, 10, 10, 200, 100)
    template = template_frame[10:18, 10:18]

    frame = np.zeros((60, 60), dtype=np.uint8)
    _draw_textured_patch(frame, 7, 10, 15, 8)  # shifted -3 in x, opposite the velocity

    match = spike.reacquire_by_template(
        frame,
        template,
        last_bbox=(10, 10, 18, 18),
        search_margin=5,
        match_threshold=0.5,
        velocity=(20.0, 0.0),
    )

    assert match is not None
    x0, y0, _x1, _y1 = match
    assert (x0, y0) == (7, 10)


def _exemplar_detection(index, bbox, *, recovered=False):
    x0, y0, x1, y1 = bbox
    contour = spike._bbox_to_rect_contour((x0, y0, x1, y1))
    return spike.FrameDetection(
        index=index,
        frame=np.zeros((40, 40, 3), dtype=np.uint8),
        mask=np.zeros((40, 40), dtype=np.uint8),
        all_contours=[],
        blobs=[],
        largest=contour,
        centroid=spike.contour_centroid(contour),
        motion_pixel_fraction=0.0,
        median_grey=0.0,
        is_flare=False,
        recovered=recovered,
    )


def test_anchor_exemplar_prefers_median_sized_real_detection():
    # The oversized frame 2 box is the kind of residual-illumination blob that
    # would drag every anchor-filled box across the clip up to its own size.
    detections = [
        _exemplar_detection(0, (10, 10, 20, 20)),
        _exemplar_detection(1, (10, 10, 21, 21)),
        _exemplar_detection(2, (0, 0, 39, 39)),
    ]

    assert spike._anchor_exemplar_index(detections) == 1


def test_anchor_exemplar_ignores_appearance_recovered_boxes():
    # A recovered box is not independent evidence of the subject's appearance;
    # seeding from one would entrench whatever the first match latched onto.
    detections = [
        _exemplar_detection(0, (10, 10, 20, 20), recovered=True),
        _exemplar_detection(1, (12, 12, 22, 22)),
    ]

    assert spike._anchor_exemplar_index(detections) == 1


def test_anchor_exemplar_returns_none_without_any_real_detection():
    detections = [_exemplar_detection(0, (10, 10, 20, 20), recovered=True)]

    assert spike._anchor_exemplar_index(detections) is None


def test_anchor_trace_fills_frames_on_both_sides_of_the_anchor():
    # Subject drifts right by 2px per frame; the anchor sits in the middle, so
    # only a bidirectional sweep can reach both ends.
    grays = []
    for i in range(5):
        frame = np.zeros((40, 60), dtype=np.uint8)
        _draw_textured_patch(frame, 10 + 2 * i, 10, 200, 100)
        grays.append(frame)
    anchor_template = grays[2][10:18, 14:22]

    boxes = spike._anchor_trace(
        grays,
        2,
        (14, 10, 22, 18),
        anchor_template,
        search_margin=10,
        match_threshold=0.5,
        max_streak=10,
    )

    assert [None if b is None else b[0] for b in boxes] == [10, 12, 14, 16, 18]


def test_anchor_trace_stops_at_first_unmatched_frame():
    # Frame 3 holds nothing resembling the subject, so the forward sweep must
    # stop there rather than keep guessing on down a cold trail.
    grays = []
    for i in range(5):
        frame = np.zeros((40, 60), dtype=np.uint8)
        if i != 3:
            _draw_textured_patch(frame, 10, 10, 200, 100)
        grays.append(frame)
    anchor_template = grays[1][10:18, 10:18]

    boxes = spike._anchor_trace(
        grays, 1, (10, 10, 18, 18), anchor_template, search_margin=6, match_threshold=0.9,
        max_streak=10,
    )

    assert boxes[0] is not None
    assert boxes[2] is not None
    assert boxes[3] is None
    assert boxes[4] is None


def test_anchor_trace_stops_after_max_streak_even_while_still_matching():
    # Every frame holds the same unmoving patch, so the match never fails on
    # its own -- exactly what a static background feature that happens to
    # resemble the exemplar looks like once the real subject has left frame.
    # Without a cap this would sweep the whole clip.
    grays = []
    for _ in range(6):
        frame = np.zeros((40, 60), dtype=np.uint8)
        _draw_textured_patch(frame, 10, 10, 200, 100)
        grays.append(frame)
    anchor_template = grays[0][10:18, 10:18]

    boxes = spike._anchor_trace(
        grays, 0, (10, 10, 18, 18), anchor_template, search_margin=6, match_threshold=0.5,
        max_streak=2,
    )

    assert boxes[0] is not None
    assert boxes[1] is not None
    assert boxes[2] is not None
    assert boxes[3] is None
    assert boxes[4] is None
    assert boxes[5] is None


def test_run_track_pass_drops_track_stuck_on_static_texture_after_recovered_streak():
    # A small, unchanging textured patch (e.g. a wire/vine) gets tracked
    # first, then stops registering as a bg-diff candidate (as if it settled
    # into the background model) -- but its pixels are still physically
    # there, so appearance-recovery trivially keeps re-matching it forever. A
    # much bigger candidate is present the whole time but far outside the
    # tiny track's search margin, so it's never picked while the recovery
    # streak is under the cap. After the cap, the track should drop and the
    # next frame should pick up the real, bigger candidate instead.
    wire = _square_contour(5, 5, 8)
    big = _square_contour(40, 40, 20)

    def _gray(_frame_index: int) -> np.ndarray:
        g = np.zeros((80, 80), dtype=np.uint8)
        _draw_textured_patch(g, 5, 5, 200, 100)  # wire's real pixels, unchanging
        return g

    grays = [_gray(i) for i in range(7)]
    candidates_per_frame = [
        [wire],  # frame 0: only the wire registers
        [wire],  # frame 1: still overlaps -> plain continuation
        [big],  # frame 2+: wire no longer a bg-diff candidate, big appears
        [big],
        [big],
        [big],
        [big],
    ]

    results = spike._run_track_pass(
        grays,
        candidates_per_frame,
        max_jump_distance=200,
        max_track_miss_frames=5,
        template_match_threshold=0.5,
        max_recovered_streak=2,
    )

    recovered_flags = [r for _c, r in results]
    assert recovered_flags[0] is False
    assert recovered_flags[1] is False
    # Frames 2-3: still within the streak cap, sustained by recovery alone.
    assert recovered_flags[2] is True
    assert recovered_flags[3] is True
    # Once the streak exceeds the cap, the track drops and re-acquires on the
    # only remaining real candidate (the big, distant one) instead of
    # perpetually re-matching the static wire texture.
    last_contour, last_recovered = results[-1]
    assert last_recovered is False
    assert spike._contour_bbox(last_contour) == spike._contour_bbox(big)


def test_run_track_pass_min_reacquire_area_blocks_noise_speck_after_track_drop():
    # Real subject tracked for 2 frames, then leaves for good (its pixels stop
    # being present at all, so appearance-recovery can't find anything either
    # -- a genuine drop, not a recovery streak). Once dropped, only a
    # noise-speck-sized candidate is available -- the default floor should
    # report a miss instead of confidently latching onto the speck, matching
    # the cam13/4101 "guard leaves, tracker locks onto a 9-16px speck" finding.
    real = _square_contour(10, 10, 20)  # area 400
    speck = _square_contour(60, 60, 3)  # area 9

    def _gray(i):
        g = np.zeros((80, 80), dtype=np.uint8)
        if i < 2:
            _draw_textured_patch(g, 10, 10, 200, 100)  # real subject's pixels, frames 0-1 only
        return g

    grays = [_gray(i) for i in range(6)]
    candidates_per_frame = [[real], [real], [], [], [speck], [speck]]

    default_results = spike._run_track_pass(
        grays,
        candidates_per_frame,
        max_jump_distance=200,
        max_track_miss_frames=1,
        template_match_threshold=0.5,
    )
    assert default_results[4][0] is None
    assert default_results[5][0] is None

    permissive_results = spike._run_track_pass(
        grays,
        candidates_per_frame,
        max_jump_distance=200,
        max_track_miss_frames=1,
        template_match_threshold=0.5,
        min_reacquire_area=0.0,
    )
    assert permissive_results[4][0] is speck


def _scenery_pass(reference, **kwargs):
    # A fence rail: bg-diff sees it once, then never again, but its pixels stay
    # physically present so appearance-recovery re-matches the same frozen box
    # forever. Exactly the cam02/4302 / cam05/4695 failure.
    rail = _square_contour(20, 20, 12)

    def _gray(_index):
        g = np.zeros((80, 80), dtype=np.uint8)
        _draw_textured_patch(g, 20, 20, 200, 100)
        return g

    grays = [_gray(i) for i in range(6)]
    return spike._run_track_pass(
        grays,
        [[rail]] + [[] for _ in range(5)],
        max_jump_distance=200,
        max_track_miss_frames=5,
        template_match_threshold=0.5,
        reference_background=reference,
        **kwargs,
    )


def test_run_track_pass_drops_frozen_recovery_that_matches_the_reference_background():
    # The reference (built from other clips of this camera) contains the rail
    # at the same coordinates, because the rail is always there.
    reference = np.zeros((80, 80), dtype=np.uint8)
    _draw_textured_patch(reference, 20, 20, 200, 100)

    results = _scenery_pass(reference, max_scenery_streak=2)

    # Frame 0 is a real bg-diff hit and is kept; the frozen recovered run that
    # follows is removed retroactively rather than reported as a confident,
    # motionless detection.
    assert results[0][0] is not None
    assert all(contour is None for contour, _recovered in results[1:])


def test_run_track_pass_keeps_frozen_recovery_absent_from_the_reference_background():
    # Same frozen, appearance-recovered run, but the reference has nothing at
    # those coordinates -- a subject that stopped moving, not scenery. Holding
    # still must not be enough on its own to lose the track.
    reference = np.zeros((80, 80), dtype=np.uint8)
    _draw_textured_patch(reference, 60, 60, 200, 100)

    results = _scenery_pass(reference, max_scenery_streak=2)

    assert sum(1 for contour, _recovered in results if contour is not None) == len(results)


def test_run_track_pass_scenery_check_is_off_without_a_reference_background():
    # Cameras with too few clips to build a reference (cam11 has two in the
    # whole corpus) must fall through to the previous behaviour untouched.
    assert all(contour is not None for contour, _r in _scenery_pass(None, max_scenery_streak=2))


def test_detect_clip_scenery_motion_fraction_high_when_blobs_match_reference(monkeypatch):
    # A blob visits three different spots across the clip -- unlike the frozen
    # single-location veto tests above, this exercises the whole-clip
    # scenery-motion feature, which scores every blob wherever it moves. The
    # reference has matching texture at all three, standing in for "this
    # camera's typical scenery covers this range of positions" (e.g. a bush
    # swaying in the wind).
    monkeypatch.setattr(spike, "_aligned_reference", lambda ref, _bg: ref)
    positions = [(10, 10), (30, 30), (50, 10)]
    frames = []
    for x, y in positions:
        frame = np.zeros((70, 70, 3), dtype=np.uint8)
        for c in range(3):
            _draw_textured_patch(frame[:, :, c], x, y, 200, 100)
        frames.append(frame)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    reference = np.zeros((70, 70), dtype=np.uint8)
    for x, y in positions:
        _draw_textured_patch(reference, x, y, 200, 100)
    # Blurred the same way `src.reference_bg.clip_median_background` blurs a
    # real reference -- an unblurred hard-edged synthetic reference compared
    # against the (blurred) clip frames underscores real texture, not scenery.
    reference = cv2.GaussianBlur(reference, (5, 5), 0)

    detection = spike.detect_clip("clip.mp4", threshold=18, reference_background=reference)

    assert detection is not None
    assert detection.has_reference_background is True
    assert detection.scenery_motion_fraction > 0.9


def test_detect_clip_scenery_motion_fraction_low_when_blobs_dont_match_reference(monkeypatch):
    # Same moving blob, but the reference has nothing there at any of its
    # positions -- a subject visiting a place with no known static scenery.
    monkeypatch.setattr(spike, "_aligned_reference", lambda ref, _bg: ref)
    positions = [(10, 10), (30, 30), (50, 10)]
    frames = []
    for x, y in positions:
        frame = np.zeros((70, 70, 3), dtype=np.uint8)
        for c in range(3):
            _draw_textured_patch(frame[:, :, c], x, y, 200, 100)
        frames.append(frame)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    reference = np.zeros((70, 70), dtype=np.uint8)

    detection = spike.detect_clip("clip.mp4", threshold=18, reference_background=reference)

    assert detection is not None
    assert detection.has_reference_background is True
    assert detection.scenery_motion_fraction == pytest.approx(0.0)


def test_detect_clip_scenery_motion_fraction_zero_without_a_reference(monkeypatch):
    # No reference supplied at all (e.g. too few clips to build one) --
    # has_reference_background must say so rather than silently reading the
    # same as "found no scenery motion".
    frames = [_frame_with_square(pos) for pos in (5, 10, 15, 20, 25)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    detection = spike.detect_clip("clip.mp4", threshold=18)

    assert detection is not None
    assert detection.has_reference_background is False
    assert detection.scenery_motion_fraction == pytest.approx(0.0)


def test_extract_clip_features_surfaces_scenery_motion_fraction(monkeypatch):
    monkeypatch.setattr(spike, "_aligned_reference", lambda ref, _bg: ref)
    positions = [(10, 10), (30, 30), (50, 10)]
    frames = []
    for x, y in positions:
        frame = np.zeros((70, 70, 3), dtype=np.uint8)
        for c in range(3):
            _draw_textured_patch(frame[:, :, c], x, y, 200, 100)
        frames.append(frame)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    reference = np.zeros((70, 70), dtype=np.uint8)
    for x, y in positions:
        _draw_textured_patch(reference, x, y, 200, 100)
    reference = cv2.GaussianBlur(reference, (5, 5), 0)

    result = spike.extract_clip_features(
        "clip.mp4", _ZONE, threshold=18, reference_background=reference
    )

    assert result is not None
    assert result["has_reference_background"] == 1.0
    assert result["scenery_motion_fraction"] > 0.9


def test_extract_clip_features_scenery_motion_fraction_zero_without_reference(monkeypatch):
    frames = [_frame_with_square(pos) for pos in (5, 10, 15, 20, 25)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features("clip.mp4", _ZONE, threshold=18)

    assert result is not None
    assert result["has_reference_background"] == 0.0
    assert result["scenery_motion_fraction"] == pytest.approx(0.0)


def _green_lit_frames():
    """A moving subject on a heavily green-cast frame -- what cam01b's and
    cam16's night footage actually looks like."""
    frames = []
    for pos in (5, 12, 19, 26, 33, 40):
        frame = _frame_with_square(pos)
        frame[:, :, 1] = np.clip(frame[:, :, 1].astype(int) + 90, 0, 255).astype(np.uint8)
        cv2.rectangle(frame, (pos, 30), (pos + 8, 38), (40, 255, 40), thickness=-1)
        frames.append(frame)
    return frames


def test_daylight_hint_false_overrules_the_colour_gate(monkeypatch):
    frames = _green_lit_frames()
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    gated = spike.extract_clip_features("clip.mp4", _ZONE)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    hinted = spike.extract_clip_features("clip.mp4", _ZONE, daylight_hint=False)

    assert gated is not None and hinted is not None
    assert gated["color_fraction"] > 0.15  # the image statistic says "daylight"
    assert gated["green_light_ratio"] == pytest.approx(0.0)  # ...so it was zeroed
    assert hinted["green_light_ratio"] > 0.0  # the clock says night, so it isn't


def test_daylight_hint_true_leaves_the_colour_gate_alone(monkeypatch):
    frames = _green_lit_frames()
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    result = spike.extract_clip_features("clip.mp4", _ZONE, daylight_hint=True)

    assert result is not None
    assert result["green_light_ratio"] == pytest.approx(0.0)
    assert result["green_light_flicker"] == pytest.approx(0.0)


def test_daylight_hint_none_is_the_pre_existing_behaviour(monkeypatch):
    frames = _green_lit_frames()
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    default = spike.extract_clip_features("clip.mp4", _ZONE)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    explicit_none = spike.extract_clip_features("clip.mp4", _ZONE, daylight_hint=None)

    assert default == explicit_none


def test_daylight_hint_does_not_invent_colour_where_there_is_none(monkeypatch):
    # Overruling the gate must not manufacture a flashlight reading on a clip
    # that has no green in it at all.
    frames = [_frame_with_square(pos) for pos in (5, 12, 19, 26, 33, 40)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    result = spike.extract_clip_features("clip.mp4", _ZONE, daylight_hint=False)

    assert result is not None
    assert result["green_light_ratio"] == pytest.approx(0.0)


def test_detect_clip_recovers_track_via_appearance_when_bg_diff_finds_nothing(monkeypatch):
    # A single textured subject moves across frame, well-detected by
    # background-subtraction everywhere except one frame where it's drawn at
    # much lower contrast (diff stays under `threshold`) -- the appearance
    # (shape/texture) hasn't changed, only how visible it is against the
    # background model. Moves slowly relative to its own 8px size, well within
    # the size-relative jump cap, so only the dip frame needs recovery.
    positions = [5, 9, 13, 17, 21, 25]
    contrasts = [(200, 100)] * 6
    contrasts[2] = (15, 8)  # frame index 2 (post any warmup): low-contrast dip
    frames = []
    for pos, (outer, inner) in zip(positions, contrasts, strict=True):
        frame = np.zeros((70, 70, 3), dtype=np.uint8)
        _draw_textured_patch(frame[:, :, 0], pos, 30, outer, inner)
        _draw_textured_patch(frame[:, :, 1], pos, 30, outer, inner)
        _draw_textured_patch(frame[:, :, 2], pos, 30, outer, inner)
        frames.append(frame)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    detection = spike.detect_clip("clip.mp4", threshold=18)

    assert detection is not None
    dip_frame = detection.frames[2]
    assert dip_frame.largest is not None
    assert dip_frame.recovered is True
    # Every other frame should be a real bg-diff detection, not a recovery.
    assert all(not f.recovered for i, f in enumerate(detection.frames) if i != 2)


def test_extract_clip_features_excludes_recovered_frames_from_motion_stats(monkeypatch):
    # Same scenario as test_detect_clip_recovers_track_via_appearance_when_bg_diff_finds_nothing
    # (one frame recovered via appearance match, not a real bg-diff hit): the
    # track-derived motion features should only count the 5 genuine frames,
    # not treat the recovered one as an equally-trustworthy detection.
    positions = [5, 9, 13, 17, 21, 25]
    contrasts = [(200, 100)] * 6
    contrasts[2] = (15, 8)  # frame index 2: low-contrast dip, recovered via appearance
    frames = []
    for pos, (outer, inner) in zip(positions, contrasts, strict=True):
        frame = np.zeros((70, 70, 3), dtype=np.uint8)
        for c in range(3):
            _draw_textured_patch(frame[:, :, c], pos, pos, outer, inner)
        frames.append(frame)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features("clip.mp4", _ZONE, threshold=18)

    assert result is not None
    # 5 of 6 frames are genuine bg-diff hits -- the recovered one is excluded.
    assert result["persistence"] == pytest.approx(5 / 6)
    assert result["recovered_fraction"] == pytest.approx(1 / 6)


def test_extract_clip_features_recovered_fraction_zero_when_all_genuine(monkeypatch):
    positions = [5, 9, 13, 17, 21, 25]
    frames = []
    for pos in positions:
        frame = np.zeros((70, 70, 3), dtype=np.uint8)
        for c in range(3):
            _draw_textured_patch(frame[:, :, c], pos, pos, 200, 100)
        frames.append(frame)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features("clip.mp4", _ZONE, threshold=18)

    assert result is not None
    assert result["recovered_fraction"] == pytest.approx(0.0)
    assert result["persistence"] == pytest.approx(1.0)


def _fake_frame_detection(index: int, contour: np.ndarray) -> "spike.FrameDetection":
    return spike.FrameDetection(
        index=index,
        frame=_blank_frame(),
        mask=np.zeros((60, 60), dtype=np.uint8),
        all_contours=[contour],
        blobs=[contour],
        largest=contour,
        centroid=spike.contour_centroid(contour),
        motion_pixel_fraction=0.0,
        median_grey=0.0,
        is_flare=False,
    )


def test_extract_clip_features_excludes_merged_blob_frame_from_best_contour(monkeypatch):
    # Frame 0: a single small square subject, unmerged. Frame 1: a much larger
    # wide rectangle -- as if the multi-object tracker's own accounting says
    # two subjects are sharing one blob there. Even though frame 1's contour
    # has the bigger area, the merged frame should be excluded from the
    # "clearest frame" pick in favour of the smaller, genuinely single-subject
    # frame 0.
    square = _rect_contour(5, 5, 10, 10)  # area 100, aspect_ratio 1.0
    merged_rect = _rect_contour(5, 5, 40, 8)  # area 320, aspect_ratio 0.2
    frames = [_fake_frame_detection(0, square), _fake_frame_detection(1, merged_rect)]
    clip_detection = spike.ClipDetection(
        frames=frames,
        background=_blank_frame()[:, :, 0],
        frame_width=60,
        frame_height=60,
        warmup_dropped=0,
        total_frames=2,
        dropped_frames=[],
        dropped_frame_boxes=[],
        multi_tracks=[
            [],
            [
                spike.TrackedObject(track_id=0, bbox=(5, 5, 25, 13), merged_ids=(1,)),
                spike.TrackedObject(track_id=1, bbox=(25, 5, 45, 13), merged_ids=(0,)),
            ],
        ],
    )
    monkeypatch.setattr(spike, "detect_clip", lambda *_a, **_k: clip_detection)

    result = spike.extract_clip_features("clip.mp4", _ZONE)

    assert result is not None
    assert result["aspect_ratio"] == pytest.approx(1.0, abs=0.05)


def test_extract_clip_features_falls_back_to_merged_frame_when_no_alternative(monkeypatch):
    # Both frames are merged -- there's no unmerged alternative, so the merged
    # (larger) frame should still be used rather than returning no features.
    small_merge = _rect_contour(5, 5, 10, 10)  # area 100, aspect_ratio 1.0
    big_merge = _rect_contour(5, 5, 40, 8)  # area 320, aspect_ratio 0.2
    frames = [_fake_frame_detection(0, small_merge), _fake_frame_detection(1, big_merge)]
    merged_tracks = [
        spike.TrackedObject(track_id=0, bbox=(5, 5, 25, 13), merged_ids=(1,)),
        spike.TrackedObject(track_id=1, bbox=(25, 5, 45, 13), merged_ids=(0,)),
    ]
    clip_detection = spike.ClipDetection(
        frames=frames,
        background=_blank_frame()[:, :, 0],
        frame_width=60,
        frame_height=60,
        warmup_dropped=0,
        total_frames=2,
        dropped_frames=[],
        dropped_frame_boxes=[],
        multi_tracks=[merged_tracks, merged_tracks],
    )
    monkeypatch.setattr(spike, "detect_clip", lambda *_a, **_k: clip_detection)

    result = spike.extract_clip_features("clip.mp4", _ZONE)

    assert result is not None
    assert result["aspect_ratio"] == pytest.approx(0.2, abs=0.05)


def test_detect_clip_fills_gap_before_track_first_locks_on_via_backward_pass(monkeypatch):
    # The subject is only visible faintly (below `threshold`) for the first
    # couple of frames, so the forward pass has no candidate -- and no track
    # yet -- to work with there. A backward scan starting from the frame
    # where it does clear the threshold should fill those leading frames in.
    # Moves slowly relative to its own 8px size, well within the
    # size-relative jump cap, so bg-diff frames continue directly.
    positions = [5, 9, 13, 17, 21, 25]
    contrasts = [(15, 8), (15, 8), (200, 100), (200, 100), (200, 100), (200, 100)]
    frames = []
    for pos, (outer, inner) in zip(positions, contrasts, strict=True):
        frame = np.zeros((70, 70, 3), dtype=np.uint8)
        for c in range(3):
            _draw_textured_patch(frame[:, :, c], pos, 30, outer, inner)
        frames.append(frame)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    detection = spike.detect_clip("clip.mp4", threshold=18)

    assert detection is not None
    assert detection.frames[0].largest is not None
    assert detection.frames[0].filled_by_reverse is True
    assert detection.frames[0].recovered is True
    assert detection.frames[1].filled_by_reverse is True
    # The frames where bg-diff genuinely fired should not be marked as fills.
    assert all(not f.filled_by_reverse for f in detection.frames[2:])


def test_detect_clip_traces_track_backward_into_dropped_flare_frames(monkeypatch):
    # Same textured subject, visible throughout the opening IR-gain ramp
    # (dropped before the background model / feature scoring, per
    # flare_settle_index) at the same spot the first scored frame finds it --
    # the anchor established there should trace back through those raw,
    # never-scored frames via appearance alone.
    def _frame(value: int, pos: int) -> np.ndarray:
        frame = _blank_frame(value=value)
        for c in range(3):
            _draw_textured_patch(frame[:, :, c], pos, 30, 200, 100)
        return frame

    ramp = [_frame(v, pos=5) for v in (220, 150, 80, 20, 0)]  # excluded from background model
    positions = (5, 15, 25, 35, 45)
    trailing = [_frame(0, pos) for pos in positions]  # settled, but still part of the ramp clip
    subject = [_frame(0, pos) for pos in positions]
    monkeypatch.setattr(
        spike.cv2, "VideoCapture", lambda _path: FakeCapture(ramp + trailing + subject)
    )

    detection = spike.detect_clip("clip.mp4", threshold=18)

    assert detection is not None
    assert detection.warmup_dropped == 5
    assert detection.frames[0].largest is not None
    assert detection.frames[0].recovered is False
    assert len(detection.dropped_frame_boxes) == 5
    # The trace should reach at least the ramp frames closest to the scored
    # boundary -- how far back it gets before the ramp's brightness washes out
    # the appearance match is a real limit, not something to pin exactly here.
    assert detection.dropped_frame_boxes[-1] is not None
    assert sum(box is not None for box in detection.dropped_frame_boxes) >= 2


def test_detect_clip_compensate_warmup_off_by_default(monkeypatch):
    # Same ramp+subject shape as the trace test above -- default behaviour
    # (compensate_warmup unset) must leave the new fields untouched.
    def _frame(value: int, pos: int) -> np.ndarray:
        frame = _blank_frame(value=value)
        for c in range(3):
            _draw_textured_patch(frame[:, :, c], pos, 30, 200, 100)
        return frame

    ramp = [_frame(v, pos=5) for v in (220, 150, 80, 20, 0)]
    positions = (5, 15, 25, 35, 45)
    trailing = [_frame(0, pos) for pos in positions]
    subject = [_frame(0, pos) for pos in positions]
    monkeypatch.setattr(
        spike.cv2, "VideoCapture", lambda _path: FakeCapture(ramp + trailing + subject)
    )

    detection = spike.detect_clip("clip.mp4", threshold=18)

    assert detection is not None
    assert detection.dropped_frame_compensated == []
    assert detection.dropped_frame_box_is_photometric == [False] * detection.warmup_dropped


def test_detect_clip_compensate_warmup_tracks_real_motion_in_flare_window(monkeypatch):
    # The warmup ramp frames carry a real moving subject at a screen position
    # (column 8) the settled background never sees -- distinct from the
    # subject's later tracked path (columns 30-50) -- so a real per-pixel diff
    # against the (photometrically compensated) background can only find it
    # by actually looking, not by re-confirming the appearance trace's own
    # guess. A checkerboard base (not a flat fill) gives the gain/offset fit
    # real spatial structure to correlate against, closer to a real camera's
    # micro-texture than a blank frame (which has none, see photometric_match
    # tests for the degenerate flat case).
    size = 60

    def _checkerboard(value: int) -> np.ndarray:
        # 10px blocks, not single pixels -- a single-pixel checkerboard is
        # smoothed away almost entirely by detect_clip's own Gaussian blur,
        # leaving no real structure for the gain/offset fit to correlate
        # against (checked directly before picking this block size).
        grid = (np.indices((size, size)) // 10).sum(axis=0) % 2
        return np.clip(value + grid * 40, 0, 255).astype(np.uint8)

    def _frame(value: int, pos: int) -> np.ndarray:
        base = _checkerboard(value)
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        for c in range(3):
            frame[:, :, c] = base
            _draw_textured_patch(frame[:, :, c], pos, 30, 220, 20)
        return frame

    ramp = [_frame(v, pos=8) for v in (150, 100, 50, 10, 0)]  # excluded from background model
    positions = (30, 35, 40, 45, 50)
    trailing = [_frame(0, pos) for pos in positions]  # settled, still part of the ramp clip
    subject = [_frame(0, pos) for pos in positions]
    monkeypatch.setattr(
        spike.cv2, "VideoCapture", lambda _path: FakeCapture(ramp + trailing + subject)
    )

    detection = spike.detect_clip("clip.mp4", threshold=18, compensate_warmup=True)

    assert detection is not None
    assert detection.warmup_dropped >= 5
    assert len(detection.dropped_frame_compensated) == detection.warmup_dropped
    assert any(detection.dropped_frame_box_is_photometric)
    # At least one photometric box should land near column 8 (its real
    # warmup-window position), not column 30+ (where the subject ends up once
    # scoring starts) -- confirms this is real tracking, not the later
    # position carried backward.
    photometric_boxes = [
        box
        for box, is_photo in zip(
            detection.dropped_frame_boxes, detection.dropped_frame_box_is_photometric, strict=True
        )
        if is_photo and box is not None
    ]
    assert any(box[0] < 20 for box in photometric_boxes)


def test_detect_clip_reverse_trace_seeds_from_plausible_size_not_frame_zero(monkeypatch):
    # Frame 0's own contour is a residual-illumination-sized outlier -- a big
    # patch swallows the real, much smaller subject's own true appearance --
    # while every other tracked frame is that same small subject at a
    # consistent size, spaced far enough apart frame to frame that the
    # per-pixel median background used for bg-diff stays clean. Seeding the
    # backward trace from frame 0 as-is would carry that oversized box in; it
    # should instead seed from the first later frame whose size is plausible
    # against the clip's typical tracked size, and use that same trace to
    # correct frame 0's own box too.
    size = 260

    def _frame(pos: int, *, big: bool = False) -> np.ndarray:
        frame = _blank_frame(size=size)
        for c in range(3):
            if big:
                frame[:, :, c][8:48, 8:48] = 200
            frame[:, :, c][20:28, pos : pos + 8] = 200
            frame[:, :, c][22:26, pos + 2 : pos + 6] = 100
        return frame

    positions = [10, 40, 70, 100, 130, 160, 190, 220]
    frames = [_frame(positions[0], big=True)] + [_frame(p) for p in positions[1:]]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    detection = spike.detect_clip(
        "clip.mp4", threshold=18, max_track_miss_frames=3, min_track_search_margin=35
    )

    assert detection is not None
    assert detection.frames[0].largest is not None
    x0, y0, x1, y1 = spike._contour_bbox(detection.frames[0].largest)
    assert max(x1 - x0, y1 - y0) < 20
    assert detection.frames[0].filled_by_reverse is True


def _frame_with_split_subject(
    pos: int, gap: int, *, width: int = 140, height: int = 60, patch: int = 6
):
    # Two solid patches, exactly `gap` background pixels apart -- simulates a
    # low-contrast subject that background-subtraction only picks up as
    # disconnected fragments.
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[20 : 20 + patch, pos : pos + patch] = 100
    right = pos + patch + gap
    frame[20 : 20 + patch, right : right + patch] = 100
    return frame


def test_detect_clip_bridges_fragmented_low_contrast_blob_by_default(monkeypatch):
    # A small gap between two fragments of the same subject should be bridged
    # by the default MORPH_CLOSE step into one contour spanning both, instead
    # of only ever finding one half of the animal.
    positions = (5, 15, 25, 35, 45)
    frames = [_frame_with_split_subject(pos, gap=5) for pos in positions]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    detection = spike.detect_clip("clip.mp4", threshold=18)

    assert detection is not None
    last = detection.frames[-1]
    assert last.largest is not None
    x0, _y0, x1, _y1 = spike._contour_bbox(last.largest)
    assert (x1 - x0) >= 17  # spans both 6px patches plus the 5px gap between them


def test_detect_clip_leaves_fragments_separate_when_closing_disabled(monkeypatch):
    # A gap wide enough to survive the pre-morphology Gaussian blur unmerged,
    # so disabling MORPH_CLOSE is what's actually being exercised here.
    positions = (5, 15, 25, 35, 45)
    frames = [_frame_with_split_subject(pos, gap=20) for pos in positions]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    detection = spike.detect_clip("clip.mp4", threshold=18, fragment_close_kernel_size=0)

    assert detection is not None
    last = detection.frames[-1]
    assert len(last.blobs) == 2


def test_detect_clip_populates_multi_tracks_for_two_independent_subjects(monkeypatch):
    positions = (5, 15, 25, 35, 45)
    frames = []
    for pos in positions:
        frame = np.zeros((70, 140, 3), dtype=np.uint8)
        for c in range(3):
            _draw_textured_patch(frame[:, :, c], pos, 30, 200, 100)
            _draw_textured_patch(frame[:, :, c], pos + 80, 30, 200, 100)
        frames.append(frame)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    detection = spike.detect_clip("clip.mp4", threshold=18)

    assert detection is not None
    assert len(detection.multi_tracks) == len(detection.frames)
    assert all(len(frame_tracks) == 2 for frame_tracks in detection.multi_tracks)


def test_extract_clip_features_returns_none_without_motion(monkeypatch, tmp_path):
    frames = [_blank_frame() for _ in range(5)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is None


def test_extract_clip_features_computes_all_features_with_motion(monkeypatch, tmp_path):
    frames = [_frame_with_square(pos) for pos in (5, 10, 15, 20, 25)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    for key in (
        "outside_pixel_fraction",
        "aspect_ratio",
        "solidity",
        "saturation_ratio",
        "green_light_flicker",
        "flashlight_subject_fraction",
        "row_normalised_area",
        "edge_density",
        "path_length",
        "jitter",
        "persistence",
        "implausible_height_fraction",
        "off_plane_fraction",
        "height_consistency",
        "depth_progression",
        "depth_range_m",
        "subject_height_m",
        "subject_width_m",
        "subject_area_m2",
        "metric_aspect",
        "distance_median_m",
        "speed_mps",
        "uncalibrated",
    ):
        assert key in result
    assert result["persistence"] > 0


def test_extract_clip_features_uncalibrated_when_zone_has_no_pickets(monkeypatch, tmp_path):
    # _ZONE has no fence_bottom/fence_pickets/metric_calibration -- the physics
    # gate must report "uncalibrated" rather than a misleadingly clean 0.0.
    frames = [_frame_with_square(pos) for pos in (5, 10, 15, 20, 25)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result["uncalibrated"] == 1.0
    assert result["implausible_height_fraction"] == 0.0
    assert result["off_plane_fraction"] == 0.0
    assert result["height_consistency"] == 0.0
    assert result["depth_progression"] == 0.0
    assert result["depth_range_m"] == 0.0
    assert result["subject_height_m"] == 0.0
    assert result["subject_width_m"] == 0.0
    assert result["subject_area_m2"] == 0.0
    assert result["metric_aspect"] == 0.0
    assert result["distance_median_m"] == 0.0
    assert result["speed_mps"] == 0.0


# cam06's real, already-validated geometry (src/ground_calibration.py's tests
# cover calibrate() correctness in isolation) -- reused here just to get a
# non-None GroundCalibration, so this test can check _metric_track_features'
# own aggregation over frames, not the calibration math itself.
_CALIBRATED_ZONE = CameraZone(
    fence=((0.4532, 0.1427), (0.3171, 0.5688), (0.1809, 0.9948)),
    outside="right",
    depth_cutoff=0.05,
    ignore=(),
    fence_bottom=(
        (0.5028, 0.0688),
        (0.4992, 0.3003),
        (0.4953, 0.5318),
        (0.4914, 0.7633),
        (0.4875, 0.9948),
    ),
    fence_pickets=(
        ((0.3453, 0.5711), (0.4649, 0.9969)),
        ((0.3907, 0.4188), (0.4762, 0.8445)),
        ((0.4102, 0.366), (0.4832, 0.7392)),
    ),
    metric_calibration=True,
)


def test_extract_clip_features_depth_progression_higher_for_steady_travel(monkeypatch, tmp_path):
    steady = [_frame_with_square(pos) for pos in (5, 12, 19, 26, 33, 40)]
    oscillating = [_frame_with_square(pos) for pos in (5, 20, 8, 22, 6, 24)]

    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(steady))
    steady_result = spike.extract_clip_features(str(tmp_path / "a.mp4"), _CALIBRATED_ZONE)
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(oscillating))
    oscillating_result = spike.extract_clip_features(str(tmp_path / "b.mp4"), _CALIBRATED_ZONE)

    assert steady_result["uncalibrated"] == 0.0
    assert oscillating_result["uncalibrated"] == 0.0
    assert steady_result["depth_progression"] > oscillating_result["depth_progression"]


def test_extract_clip_features_computes_scale_invariant_metric_size(monkeypatch, tmp_path):
    frames = [_frame_with_square(pos) for pos in (5, 12, 19, 26, 33, 40)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _CALIBRATED_ZONE)

    assert result["uncalibrated"] == 0.0
    # A real, positive real-world size and range -- not asserting exact values
    # (this 8x8px square isn't a real subject), just that the ground-plane
    # geometry produced sane, usable numbers rather than 0.0 defaults.
    assert result["subject_height_m"] > 0.0
    assert result["subject_width_m"] > 0.0
    # median(height*width) vs median(height)*median(width) -- close but not
    # exactly equal, so a loose proportionality check, not exact equality.
    assert result["subject_area_m2"] == pytest.approx(
        result["subject_height_m"] * result["subject_width_m"], rel=0.2
    )
    assert result["metric_aspect"] > 0.0
    assert result["distance_median_m"] > 0.0
    assert result["speed_mps"] > 0.0


def test_extract_clip_features_ignores_ir_warmup_brightness_swing(monkeypatch, tmp_path):
    # Mirrors the real cam15 failure: the first frames swing globally as the IR
    # gain settles, which dwarfs the actual subject's motion. Ramps down to the
    # same base level (0) the subject frames sit on, so there's no artificial
    # second step once the ramp ends -- only the opening ramp should be flagged.
    warmup = [_blank_frame(value=v) for v in (220, 150, 80, 20, 0, 0, 0, 0, 0, 0)]
    # Smooth there-and-back motion (same 7px step size throughout) rather than a
    # teleport back to the start -- a big jump-back exceeds the tracker's own
    # size/jump plausibility caps and gets filled in via the reverse pass
    # instead of a genuine bg-diff hit, which would defeat this test's own
    # persistence assertion for reasons unrelated to IR warmup handling.
    subject = [_frame_with_square(pos) for pos in (5, 12, 19, 26, 33, 40, 33, 26, 19, 12)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(warmup + subject))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    # The 8x8 square, not the 60x60 flare -- a whole-frame blob would be far from square.
    assert result["aspect_ratio"] == pytest.approx(1.0, abs=0.3)
    assert result["persistence"] > 0.5


def test_extract_clip_features_keeps_warmup_frames_on_short_clips(monkeypatch, tmp_path):
    # A short clip with no illumination step at all should have nothing dropped.
    frames = [_frame_with_square(pos) for pos in (5, 12, 19, 26, 33, 40)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    assert result["persistence"] > 0


def test_extract_clip_features_flags_swinging_flashlight(monkeypatch, tmp_path):
    # A beam that sweeps in and out of frame, not just present in one frame.
    def _frame_with_square_and_green(pos: int, green: bool) -> np.ndarray:
        frame = _frame_with_square(pos)
        if green:
            cv2.rectangle(frame, (0, 0), (30, 30), (0, 255, 0), thickness=-1)
        return frame

    positions = (5, 12, 19, 26, 33, 40, 5, 12, 19, 26)
    frames = [
        _frame_with_square_and_green(pos, green=i % 2 == 0) for i, pos in enumerate(positions)
    ]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    assert result["green_light_flicker"] > 0.1


def test_extract_clip_features_zeroes_green_light_in_broad_daylight_colour(monkeypatch, tmp_path):
    # Dusk/daytime footage with real ambient colour (green foliage covering
    # most of the frame) can pass the same hue/saturation/value check as the
    # guard's flashlight if the subject itself picks up a green cast -- the
    # broad-frame colour_fraction gate should suppress both green features
    # here even though the un-gated per-contour check would fire.
    def _frame_with_green_square(pos: int, size: int = 60) -> np.ndarray:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :] = (0, 150, 0)  # broad saturated ambient green background
        cv2.rectangle(frame, (pos, pos), (pos + 8, pos + 8), (0, 220, 0), thickness=-1)
        return frame

    positions = (5, 12, 19, 26, 33, 40)
    frames = [_frame_with_green_square(pos) for pos in positions]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    assert result["color_fraction"] > 0.15
    assert result["green_light_ratio"] == 0.0
    assert result["green_light_flicker"] == 0.0


def test_extract_clip_features_flags_the_tracked_box_itself_as_the_flashlight(
    monkeypatch, tmp_path
):
    # The tracked square IS the beam here (solid green), not a separate blob
    # elsewhere in frame -- distinct from `green_light_ratio`, which only
    # checks the single clearest frame; this should catch it persisting
    # across most of the clip.
    def _green_square(pos: int) -> np.ndarray:
        frame = _blank_frame()
        cv2.rectangle(frame, (pos, pos), (pos + 8, pos + 8), (0, 255, 0), thickness=-1)
        return frame

    positions = (5, 12, 19, 26, 33, 40)
    frames = [_green_square(pos) for pos in positions]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    assert result["flashlight_subject_fraction"] > 0.5


def test_extract_clip_features_flashlight_subject_fraction_zero_for_real_subject(
    monkeypatch, tmp_path
):
    frames = [_frame_with_square(pos) for pos in (5, 10, 15, 20, 25)]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    assert result["flashlight_subject_fraction"] == pytest.approx(0.0)


def test_extract_clip_features_flashlight_subject_fraction_zeroed_in_daylight(
    monkeypatch, tmp_path
):
    # Same broad-frame daylight-colour gate as green_light_ratio/flicker --
    # ambient green foliage covering the frame shouldn't get tagged FLASHLIGHT.
    def _frame_with_green_square(pos: int, size: int = 60) -> np.ndarray:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :] = (0, 150, 0)  # broad saturated ambient green background
        cv2.rectangle(frame, (pos, pos), (pos + 8, pos + 8), (0, 220, 0), thickness=-1)
        return frame

    positions = (5, 12, 19, 26, 33, 40)
    frames = [_frame_with_green_square(pos) for pos in positions]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    assert result["flashlight_subject_fraction"] == 0.0


def test_detect_clip_ignore_polygons_mask_out_a_stationary_light(monkeypatch, tmp_path):
    # A stationary "light" that flickers slightly frame-to-frame (like a real
    # IR-lit fixture left in view, not perfectly static) still diffs against
    # the median background and would otherwise count as its own blob every
    # frame, alongside the real tracked subject.
    def _frame_with_subject_and_light(pos: int, light_on: bool) -> np.ndarray:
        frame = _frame_with_square(pos)
        if light_on:
            cv2.rectangle(frame, (2, 2), (10, 10), (0, 255, 0), thickness=-1)
        return frame

    positions = (5, 12, 19, 26, 33, 40)
    frames = [
        _frame_with_subject_and_light(pos, light_on=i % 2 == 0) for i, pos in enumerate(positions)
    ]

    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    unmasked = spike.detect_clip(str(tmp_path / "clip.mp4"), threshold=18)
    assert unmasked is not None
    assert any(len(d.blobs) >= 2 for d in unmasked.frames)  # subject + flickering light

    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    ignore_polygons = (((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)),)
    masked = spike.detect_clip(
        str(tmp_path / "clip.mp4"), threshold=18, ignore_polygons=ignore_polygons
    )
    assert masked is not None
    assert all(len(d.blobs) <= 1 for d in masked.frames)  # light masked out, subject left


def test_extract_clip_features_ignore_region_suppresses_light_blob_count_and_flicker(
    monkeypatch, tmp_path
):
    def _frame_with_subject_and_light(pos: int, light_on: bool) -> np.ndarray:
        frame = _frame_with_square(pos)
        if light_on:
            cv2.rectangle(frame, (2, 2), (10, 10), (0, 255, 0), thickness=-1)
        return frame

    positions = (5, 12, 19, 26, 33, 40)
    frames = [
        _frame_with_subject_and_light(pos, light_on=i % 2 == 0) for i, pos in enumerate(positions)
    ]

    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    result_unmasked = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    ignore_zone = CameraZone(
        fence=_ZONE.fence,
        outside=_ZONE.outside,
        depth_cutoff=_ZONE.depth_cutoff,
        ignore=(((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)),),
    )
    result_masked = spike.extract_clip_features(str(tmp_path / "clip.mp4"), ignore_zone)

    assert result_unmasked is not None
    assert result_masked is not None
    assert result_masked["blob_count"] < result_unmasked["blob_count"]
    assert result_unmasked["green_light_flicker"] > 0.005
    assert result_masked["green_light_flicker"] == pytest.approx(0.0)


def test_extract_clip_features_median_counterparts_ignore_a_single_spike(monkeypatch, tmp_path):
    # The peak blob_count/motion_pixel_fraction fire on one noisy frame; their
    # median counterparts only rise when the scattered motion persists.
    positions = (5, 12, 19, 26, 33, 40)
    frames = [_frame_with_square(pos) for pos in positions]
    noisy = frames[1].copy()
    for x in range(2, 60, 6):
        cv2.rectangle(noisy, (x, 2), (x + 3, 5), (255, 255, 255), thickness=-1)
    frames[1] = noisy

    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    assert result["blob_count_median"] <= result["blob_count"]
    assert result["motion_pixel_fraction_median"] <= result["motion_pixel_fraction"]
    assert "blob_count_median" in spike.FEATURE_COLUMNS
    assert "motion_pixel_fraction_median" in spike.FEATURE_COLUMNS


def test_detect_clip_records_a_suppressed_light_box_instead_of_nothing(monkeypatch, tmp_path):
    # A stationary light masked out by an ignore polygon shouldn't just vanish
    # without a trace -- FrameDetection.suppressed_light_box is the additive,
    # non-scoring diagnostic a render can use to show something was there.
    def _frame_with_subject_and_light(pos: int, light_on: bool) -> np.ndarray:
        frame = _frame_with_square(pos)
        if light_on:
            cv2.rectangle(frame, (2, 2), (10, 10), (0, 255, 0), thickness=-1)
        return frame

    positions = (5, 12, 19, 26, 33, 40)
    frames = [
        _frame_with_subject_and_light(pos, light_on=i % 2 == 0) for i, pos in enumerate(positions)
    ]

    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    ignore_polygons = (((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)),)
    masked = spike.detect_clip(
        str(tmp_path / "clip.mp4"), threshold=18, ignore_polygons=ignore_polygons
    )

    assert masked is not None
    # The light was on for the even-indexed frames -- those should carry a box;
    # the tracked subject itself is untouched (still has its own `largest`).
    assert any(d.suppressed_light_box is not None for d in masked.frames)
    assert all(d.largest is not None for d in masked.frames)


def test_detect_clip_suppressed_light_box_is_none_without_an_ignore_region(monkeypatch, tmp_path):
    positions = (5, 12, 19, 26, 33, 40)
    frames = [_frame_with_square(pos) for pos in positions]
    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))

    detection = spike.detect_clip(str(tmp_path / "clip.mp4"), threshold=18)

    assert detection is not None
    assert all(d.suppressed_light_box is None for d in detection.frames)


def test_extract_clip_features_auto_detects_a_stationary_light_without_zone_ignore(
    monkeypatch, tmp_path
):
    # No hand-traced zone.ignore configured at all -- a light that's on in
    # every frame at a fixed spot away from the tracked subject should still
    # get excluded from flashlight scoring via auto-detection.
    def _frame_with_subject_and_fixed_light(pos: int) -> np.ndarray:
        frame = _frame_with_square(pos)
        cv2.rectangle(frame, (2, 2), (10, 10), (0, 255, 0), thickness=-1)
        return frame

    positions = (5, 12, 19, 26, 33, 40)
    frames = [_frame_with_subject_and_fixed_light(pos) for pos in positions]

    monkeypatch.setattr(spike.cv2, "VideoCapture", lambda _path: FakeCapture(frames))
    result = spike.extract_clip_features(str(tmp_path / "clip.mp4"), _ZONE)

    assert result is not None
    assert result["green_light_flicker"] == pytest.approx(0.0)


def test_iter_labelled_clips_with_files_requires_both_file_and_label(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=1,
        camera_id="cam01",
        timestamp="2026-01-01T20:00:00Z",
        caption=None,
        file_path="data/history/cam01/1.mp4",
        source="backfill",
    )
    db.upsert_label(conn, channel_id="chan1", message_id=1, label="guard")

    db.upsert_clip(  # no file_path -- should be skipped
        conn,
        channel_id="chan1",
        message_id=2,
        camera_id="cam01",
        timestamp="2026-01-01T20:01:00Z",
        caption=None,
        file_path=None,
        source="backfill",
    )
    db.upsert_label(conn, channel_id="chan1", message_id=2, label="guard")

    db.upsert_clip(  # not labelled -- should be skipped
        conn,
        channel_id="chan1",
        message_id=3,
        camera_id="cam01",
        timestamp="2026-01-01T20:02:00Z",
        caption=None,
        file_path="data/history/cam01/3.mp4",
        source="backfill",
    )

    rows = list(spike.iter_labelled_clips_with_files(conn, camera_id="cam01"))

    assert [r["message_id"] for r in rows] == [1]
    conn.close()


def test_run_spike_raises_for_unknown_camera(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    cameras = CamerasConfig(cameras=(), unknown_camera_id="unknown")
    with pytest.raises(ValueError, match="Unknown camera_id"):
        spike.run_spike(conn, cameras, camera_id="missing")
    conn.close()


def test_run_spike_assembles_rows_and_skips_undetected(monkeypatch, tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=1,
        camera_id="cam01",
        timestamp="2026-01-01T20:00:00Z",
        caption=None,
        file_path="clip1.mp4",
        source="backfill",
    )
    db.upsert_label(conn, channel_id="chan1", message_id=1, label="guard")
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=2,
        camera_id="cam01",
        timestamp="2026-01-01T20:01:00Z",
        caption=None,
        file_path="clip2.mp4",
        source="backfill",
    )
    db.upsert_label(conn, channel_id="chan1", message_id=2, label="animal")

    camera = Camera(id="cam01", aliases=(), order=0, zone=_ZONE, threshold_overrides={})
    cameras = CamerasConfig(cameras=(camera,), unknown_camera_id="unknown")

    def fake_extract(file_path, zone, *, reference_row=None):
        if file_path == "clip1.mp4":
            return {"outside_pixel_fraction": 0.9}
        return None  # simulate no motion detected in clip2

    monkeypatch.setattr(spike, "extract_clip_features", fake_extract)

    rows = spike.run_spike(conn, cameras, camera_id="cam01")

    assert len(rows) == 1
    assert rows[0]["message_id"] == 1
    assert rows[0]["label"] == "guard"
    assert rows[0]["outside_pixel_fraction"] == 0.9
    conn.close()


def test_write_csv_round_trip(tmp_path: Path):
    rows = [
        {
            "channel_id": "chan1",
            "message_id": 1,
            "camera_id": "cam01",
            "label": "guard",
            "outside_pixel_fraction": 0.1,
            "aspect_ratio": 1.5,
            "solidity": 0.9,
            "saturation_ratio": 0.0,
            "row_normalised_area": 100.0,
            "edge_density": 0.2,
            "path_length": 10.0,
            "jitter": 0.5,
            "persistence": 0.8,
        }
    ]
    out_path = tmp_path / "features.csv"

    spike.write_csv(rows, str(out_path))

    content = out_path.read_text()
    assert "outside_pixel_fraction" in content
    assert "guard" in content
