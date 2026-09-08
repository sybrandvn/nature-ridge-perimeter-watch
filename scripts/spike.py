"""Phase 0c: CV feasibility spike (gate 2).

Extracts candidate discriminating features to a flat CSV for a small,
hand-labelled clip sample, so a human can check whether any threshold
combination separates guard/environment from animal/incident before the rest
of the pipeline (motion.py, classify.py, backtester.py) gets built on top of a
false premise.

This deliberately does NOT do motion detection/background subtraction itself
-- that is Phase 1's src/motion.py, built only after this gate passes. Instead
it takes an already-detected blob per sampled frame (bounding contour + centroid)
so the feature math (src/features.py, src/zones.py) can be validated against
real footage using a throwaway per-clip detector, without committing to a
production motion pipeline before knowing it's worth building.

Workflow (see docs/plan.md Phase 0c):
  1. Pick 2-3 cameras with the richest incident history (needs user input --
     see the "which cameras" question in README.md).
  2. Download a clip subset for those cameras with scripts/download_clips.py
     (requires live Telegram access; writes clips.file_path automatically).
  3. Hand-enter fence polylines for those cameras in config/cameras.yaml.
  4. Hand-label ~150 clips with scripts/label.py.
  5. Run this script to extract features to a CSV.
  6. Manually inspect the CSV (or a notebook) for separability. If nothing
     separates guard/environment from animal/incident, stop and revisit scope
     per docs/plan.md rather than proceeding to Phase 1.

Run:
    uv run python scripts/spike.py --camera cam_north --out data/reports/spike_cam_north.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import db  # noqa: E402
from src.config import CamerasConfig, CameraZone, load_app_config, load_cameras_config  # noqa: E402
from src.features import (  # noqa: E402
    FLASHLIGHT_CANDIDATE_MIN_RATIO,
    FLASHLIGHT_SUBJECT_THRESHOLD,
    apply_photometric_match,
    area_stability,
    aspect_ratio,
    blob_white_fraction,
    color_saturation_fraction,
    depth_progression,
    detect_stationary_light_mask,
    edge_density,
    flare_frames,
    flare_settle_index,
    flashlight_bbox_overlap,
    green_light_flicker,
    green_light_ratio,
    heading_change,
    ignore_region_mask,
    is_daylight,
    jitter,
    longest_detection_run,
    normalised_speed,
    path_length,
    persistence,
    photometric_match,
    photometric_match_color,
    post_flash_red_shift,
    row_normalised_area,
    sane_fps,
    saturation_ratio,
    solidity,
    time_of_day,
)
from src.ground_calibration import (  # noqa: E402
    MAX_SUBJECT_HEIGHT_M,
    MAX_SUBJECT_SPEED_MPS,
    MAX_SUBJECT_WIDTH_M,
    calibrate,
)
from src.zones import (  # noqa: E402
    classify_zone,
    median_fence_distance,
    outside_pixel_fraction,
    track_crosses_fence,
    zone_classifiable_fraction,
)

FEATURE_COLUMNS = (
    "channel_id",
    "message_id",
    "camera_id",
    "label",
    "time_of_day",
    "is_daylight",
    "outside_pixel_fraction",
    "zone_classifiable_fraction",
    "outside_frame_fraction",
    "aspect_ratio",
    "solidity",
    "saturation_ratio",
    "color_fraction",
    "green_light_ratio",
    "green_light_flicker",
    "whole_frame_green_ratio",
    "warmup_flashlight_ratio",
    "warmup_outside_fraction",
    "flashlight_subject_fraction",
    "row_normalised_area",
    "edge_density",
    "blob_white_fraction",
    "long_flare_frames",
    "post_flash_red_shift",
    "path_length",
    "jitter",
    "persistence",
    "motion_pixel_fraction",
    "blob_count",
    "motion_pixel_fraction_median",
    "blob_count_median",
    "longest_detection_run",
    "area_stability",
    "normalised_speed",
    "heading_change",
    "fence_crossed",
    "median_fence_distance",
    "recovered_fraction",
    "scenery_motion_fraction",
    "has_reference_background",
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
)


def largest_contour(mask: np.ndarray, *, max_area: float | None = None) -> np.ndarray | None:
    """Largest external contour in a binary motion mask, or None if empty.

    `max_area` discards blobs bigger than that, which on these cameras means a
    whole-frame IR gain/flicker change rather than a subject.

    This is the "throwaway per-clip detector" referenced in the module
    docstring: good enough to validate feature separability, not a claim
    about the eventual production motion pipeline.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if max_area is not None:
        contours = [c for c in contours if cv2.contourArea(c) <= max_area]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _contour_bbox(contour: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x + w, y + h)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def _size_change_plausible(from_area: float, to_area: float, max_ratio: float) -> bool:
    """Whether `to_area` is within `max_ratio` of `from_area`, both growing and
    shrinking -- a same-subject bounding box shouldn't balloon or collapse by
    many times its own area from one tracked frame to the next; that's a sign
    a track has latched onto (or is handing off to) an unrelated blob that
    merely overlaps or sits near the last known position, not the same
    subject changing size gradually.
    """
    if from_area <= 0 or to_area <= 0:
        return True
    ratio = to_area / from_area
    return (1.0 / max_ratio) <= ratio <= max_ratio


def _size_relative_margin(
    bbox: tuple[int, int, int, int], *, margin_fraction: float, min_margin: float
) -> float:
    """Search/jump slack scaled to the tracked object's own size rather than a
    fixed fraction of the frame -- a frame-relative radius lets a small
    subject's track latch onto an unrelated blob or lookalike patch far
    outside where it could plausibly have moved by the next frame.
    """
    x0, y0, x1, y1 = bbox
    return max(margin_fraction * max(x1 - x0, y1 - y0), min_margin)


def track_contour(
    candidates: list[np.ndarray],
    track_bbox: tuple[int, int, int, int] | None,
    *,
    max_jump_distance: float,
    size_margin_fraction: float = 0.75,
    min_size_margin: float = 6.0,
    max_size_change_ratio: float = 4.0,
    min_reacquire_area: float = 20.0,
    flashlight_scores: list[float] | None = None,
) -> np.ndarray | None:
    """Pick the candidate contour that continues an existing track, instead of
    always re-selecting the frame's largest blob independently.

    With no active track (nothing tracked yet, or the track was dropped after
    too many consecutive misses), this falls back to the largest candidate --
    the original behaviour. With an active track, it prefers the candidate
    with the highest bounding-box overlap (IoU) against the last known
    position; if nothing overlaps, it falls back to the nearest centroid,
    within whichever is smaller of `max_jump_distance` (a hard, frame-sized
    ceiling) and a margin scaled to the track's own size -- a fixed
    frame-relative radius let a small subject's track jump to an unrelated
    blob far outside where it could plausibly have moved. This is what keeps
    the "tracked" identity from flipping between two co-occurring subjects
    (e.g. two people) just because their relative blob sizes swap from one
    frame to the next.

    Both of those checks (overlap, and nearest-centroid-within-margin) only
    ever consider position -- a candidate that overlaps or sits near the last
    known box but whose area balloons or collapses by more than
    `max_size_change_ratio` (both ways) is treated as not a plausible
    continuation either, same as if it were too far away. Without this, a
    track can silently "hand off" from a real subject onto a much larger or
    smaller co-located blob (e.g. a residual-illumination blob shrinking away
    while the real, much smaller subject happens to sit inside it) and keep
    going under the same identity, at the wrong size, for the rest of the
    clip.

    When an active track has no plausible continuation this frame, this
    returns None rather than grabbing the largest blob in the frame -- the
    caller's miss-tolerance / appearance-recovery machinery gets a chance to
    reacquire first, and only a track that's fully dropped (track_bbox is
    None again) starts fresh unconstrained.

    A fresh/unconstrained start (`track_bbox is None`) still requires at least
    `min_reacquire_area` -- without a floor here, a track force-dropped after
    `max_recovered_streak` (or one that was never established) latches onto
    whatever tiny contour happens to be the largest AVAILABLE one that frame,
    which on a quiet clip can be a single noise-speck/sensor-grain contour a
    few pixels across (observed on cam13/4101: a 9-16px contour tracked for 16
    frames after the real subject left). Below this floor, no candidate
    counts as plausible and this reports a miss instead, same as an active
    track with nothing to continue onto.

    `flashlight_scores`, one value per `candidates` entry (see `green_light_
    ratio`), is opt-in evidence for the SAME fresh/unconstrained pick above --
    largest-area alone has a real, confirmed failure mode: a static bright
    blob (a sunlit branch, an illuminated bush) that is bigger in frame than
    the guard's own flashlight wins the pick outright, and every downstream
    colour feature then describes the wrong blob (confirmed on cam07/11174:
    a real flashlight sat in its own 1785px contour while a 3025px bush
    contour in the same frame won the largest-area vote). When any eligible
    candidate's score clears `FLASHLIGHT_CANDIDATE_MIN_RATIO` (the same threshold
    `classify()`'s own green-light rule uses), the largest SUCH candidate
    wins instead of the largest candidate overall -- still preferring size
    among genuine flashlight hits, just no longer blind to colour. Pass
    `None` (the default) for the original area-only behaviour.
    """
    if not candidates:
        return None
    if track_bbox is None:
        eligible = [
            i for i, c in enumerate(candidates) if cv2.contourArea(c) >= min_reacquire_area
        ]
        if not eligible:
            return None
        if flashlight_scores is not None:
            lit = [i for i in eligible if flashlight_scores[i] > FLASHLIGHT_CANDIDATE_MIN_RATIO]
            if lit:
                eligible = lit
        return candidates[max(eligible, key=lambda i: cv2.contourArea(candidates[i]))]

    boxes = [_contour_bbox(c) for c in candidates]
    track_area = _bbox_area(track_bbox)
    size_ok = [
        _size_change_plausible(track_area, _bbox_area(box), max_size_change_ratio)
        for box in boxes
    ]
    ious = [
        _bbox_iou(track_bbox, box) if ok else 0.0 for box, ok in zip(boxes, size_ok, strict=True)
    ]
    best_iou_idx = max(range(len(candidates)), key=lambda i: ious[i])
    if ious[best_iou_idx] > 0:
        return candidates[best_iou_idx]

    track_center = _bbox_center(track_bbox)
    distances = [
        ((cx - track_center[0]) ** 2 + (cy - track_center[1]) ** 2) ** 0.5 if ok else float("inf")
        for ok, (cx, cy) in zip(size_ok, (_bbox_center(box) for box in boxes), strict=True)
    ]
    best_dist_idx = min(range(len(candidates)), key=lambda i: distances[i])
    allowed = min(
        max_jump_distance,
        _size_relative_margin(
            track_bbox, margin_fraction=size_margin_fraction, min_margin=min_size_margin
        ),
    )
    if distances[best_dist_idx] <= allowed:
        return candidates[best_dist_idx]

    # No plausible continuation -- do NOT blindly grab the frame's largest
    # blob, that's exactly the "teleport across the frame" behaviour this cap
    # exists to prevent. Report a miss instead, so the caller's existing
    # appearance-recovery / miss-tolerance machinery gets a chance, and only a
    # track that is fully dropped (track_bbox is None) starts fresh anywhere.
    return None


def _bbox_to_rect_contour(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Synthesize a rectangular contour from a bbox, for tracks recovered by
    appearance matching rather than a real background-subtraction blob."""
    x0, y0, x1, y1 = bbox
    return np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32)


def reacquire_by_template(
    gray_frame: np.ndarray,
    template: np.ndarray,
    last_bbox: tuple[int, int, int, int],
    *,
    search_margin: float,
    match_threshold: float,
    velocity: tuple[float, float] = (0.0, 0.0),
) -> tuple[int, int, int, int] | None:
    """Look for the last-known subject's appearance directly in the raw frame,
    for the frames where background-subtraction finds no candidate at all
    (e.g. the subject now blends into the background in brightness terms, even
    though it hasn't moved or changed shape). Returns the matched bbox, sized
    like `template`, or None if nothing in the search window around
    `last_bbox` clears `match_threshold`.

    This is a template match (normalised cross-correlation) of the last known
    crop against a window around the last known position, not a full-frame
    search -- keeps it cheap and stops it latching onto an unrelated lookalike
    elsewhere in frame. `search_margin` should already be scaled to the
    subject's own size (see `_size_relative_margin`) rather than a fixed
    frame-relative radius, or a small subject can match a lookalike patch far
    from anywhere it could plausibly be.

    `velocity` (last observed per-frame centre displacement) shifts and
    stretches the window toward the direction of travel: the edge trailing
    the motion stays exactly `search_margin` from the last known position
    (so a subject that doubles back is still covered), while the leading edge
    extends out to roughly the last position plus `velocity` -- a moving
    subject is more likely to have continued than reversed, but not so much
    more likely that a genuine reversal gets missed.

    The match is fixed-scale: the returned box is always the size of the
    template it was given. Trying several scales per frame and keeping the
    best was measured on 40 labelled clips and was worse on every proxy --
    coverage barely moved (30 to 28 boxless frames) while frame-to-frame box
    size jitter rose from 0.229 to 0.263 and centre-path jerk from 13.0 to
    13.8, because normalised cross-correlation picks a slightly different
    best scale each frame and the box flickers between them.
    """
    th, tw = template.shape[:2]
    if th == 0 or tw == 0:
        return None
    x0, y0, x1, y1 = last_bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    vx, vy = velocity
    base_half_w, base_half_h = (x1 - x0) / 2.0 + search_margin, (y1 - y0) / 2.0 + search_margin
    center_x, center_y = cx + vx / 2.0, cy + vy / 2.0
    half_w, half_h = base_half_w + abs(vx) / 2.0, base_half_h + abs(vy) / 2.0
    frame_height, frame_width = gray_frame.shape[:2]
    wx0 = max(int(center_x - half_w), 0)
    wy0 = max(int(center_y - half_h), 0)
    wx1 = min(int(center_x + half_w), frame_width)
    wy1 = min(int(center_y + half_h), frame_height)
    window = gray_frame[wy0:wy1, wx0:wx1]
    if window.shape[0] < th or window.shape[1] < tw:
        return None
    result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < match_threshold:
        return None
    match_x, match_y = max_loc
    return (wx0 + match_x, wy0 + match_y, wx0 + match_x + tw, wy0 + match_y + th)


# Below this grey-level spread a patch has no texture for normalised
# cross-correlation to work with, and NCC degenerates into matching anything.
_FLAT_PATCH_STD = 1e-3
# Mean absolute grey difference under which two textureless patches count as
# the same thing, standing in for the NCC that can't be computed on them.
_FLAT_PATCH_TOLERANCE = 2.0
# Centre movement (px) under which a recovered box counts as not having moved.
_SCENERY_DRIFT_TOLERANCE = 1.0


def _bbox_crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def _aligned_reference(
    reference: np.ndarray | None, background: np.ndarray
) -> np.ndarray | None:
    """Register a reference background onto this clip's own median, or drop it
    if the two don't even describe the same frame size (a camera swapped
    resolution mid-era)."""
    if reference is None or reference.shape != background.shape:
        return None
    from src.reference_bg import align

    aligned, _shift = align(reference, background)
    return aligned


def _patch_similarity(patch: np.ndarray, other: np.ndarray) -> float:
    """Normalised cross-correlation between two equal-sized grey patches.

    Falls back to a direct level comparison when either patch is flat: NCC is
    undefined at zero variance and reports a spurious perfect match against
    anything, so a featureless patch would otherwise look identical to every
    other featureless patch.
    """
    if patch.shape != other.shape or patch.size == 0:
        return 0.0
    a, b = patch.astype(np.float32), other.astype(np.float32)
    if float(a.std()) < _FLAT_PATCH_STD or float(b.std()) < _FLAT_PATCH_STD:
        return 1.0 if float(np.abs(a - b).mean()) <= _FLAT_PATCH_TOLERANCE else 0.0
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


def _run_track_pass(
    grays: list[np.ndarray],
    candidates_per_frame: list[list[np.ndarray]],
    *,
    max_jump_distance: float,
    max_track_miss_frames: int,
    template_match_threshold: float,
    search_margin_fraction: float = 0.75,
    min_search_margin: float = 6.0,
    max_recovered_streak: int = 12,
    max_size_change_ratio: float = 4.0,
    min_reacquire_area: float = 20.0,
    enforce_min_area: list[bool] | None = None,
    reference_background: np.ndarray | None = None,
    max_scenery_streak: int = 2,
    scenery_correlation: float = 0.94,
    flashlight_scores_per_frame: list[list[float]] | None = None,
) -> list[tuple[np.ndarray | None, bool]]:
    """Run the track_contour + reacquire_by_template state machine once over a
    sequence of frames, in whatever order they're given -- forward, or
    reversed for a backward scan. Returns (contour, recovered) per frame.

    A single direction can only ever pick a track up once bg-diff first finds
    a candidate for it (or, within the miss tolerance, once an appearance
    template exists to reacquire against) -- it has nothing to offer the
    frames before that point. Running the same state machine again over the
    reversed sequence lets a track that starts late from the front fill in
    those earlier frames from the back, and vice versa for one that ends
    early.

    Appearance reacquisition searches within `search_margin_fraction` of the
    track's own current size (floored at `min_search_margin` pixels), not a
    fixed frame-relative radius, biased toward the last observed direction of
    travel (see `reacquire_by_template`) -- otherwise a small, slow subject's
    track can bounce to an unrelated lookalike patch well outside where it
    could plausibly have moved in one frame.

    A track sustained for more than `max_recovered_streak` consecutive frames
    purely by appearance recovery, with no real background-subtraction hit in
    between, is dropped and re-acquired fresh. Appearance matching can't tell
    a genuine subject that's briefly blended into the background from a
    static, high-texture background feature (e.g. a wire or vine) that was
    mistakenly picked up once -- both trivially keep re-matching their own
    unchanging template forever. Without this cap, whichever one is anchored
    first wins permanently, even while a much larger, genuinely moving
    candidate persists elsewhere in the same frames.

    `max_size_change_ratio` (see `track_contour`) rejects a continuation whose
    area balloons or collapses by more than that factor from the track's own
    last size, both growing and shrinking.

    `min_reacquire_area` (see `track_contour`) applies once a track has been
    established at least once in this pass, then dropped and needs a fresh
    re-acquisition -- e.g. right after the `max_recovered_streak` force-drop
    above -- keeping that reset from landing on a noise-speck-sized contour
    just because it's the largest one present. It deliberately does NOT apply
    to a track's very first-ever acquisition (nothing established yet to drop
    from): several confirmed real subjects in this corpus are only ever a
    handful of pixels from their first frame onward (e.g. cam10/9405's
    genuine animal starts at 8px) -- gating a clip's very first pickup on a
    size floor would silently erase those rather than protect anything, since
    there's no prior real track it could be wrongly replacing yet.

    That "nothing established yet" check is only accurate in chronological
    (forward) order -- a backward scan starts at the END of the clip, so its
    own first pickup can be chronologically AFTER a real subject came and
    went (e.g. cam13/4101: guard leaves, then a noise speck appears for the
    rest of the clip) and still look like a first-ever acquisition from the
    backward scan's own point of view, defeating the floor entirely for
    exactly the frames it exists to protect. `enforce_min_area`, one bool per
    frame in THIS call's own order, lets a caller supply chronological
    knowledge the scan direction itself can't derive (see `detect_clip`, which
    runs the forward pass first and tells the backward pass which of its
    frames already had a real subject established somewhere earlier in real
    time). The floor applies if EITHER this scan's own `ever_tracked` or the
    supplied flag says so -- whichever direction learned about a real subject
    first still protects every frame after it.

    `reference_background` (see `src.reference_bg`) catches what
    `max_recovered_streak` above is too blunt for: the track latching onto a
    static piece of scenery -- a mounting pole, a fence rail, a strand of razor
    wire -- and holding it for the rest of the clip. A run of
    appearance-recovered frames whose box does not move and whose contents
    correlate above `scenery_correlation` with the SAME COORDINATES in a
    background built from OTHER clips of this camera is scenery, and is removed
    retroactively rather than reported as a confident, motionless, wrong
    detection.

    The reference has to come from other clips. Comparing against this clip's
    own median instead is circular and was measured to be worthless: a frame is
    appearance-recovered precisely BECAUSE background subtraction found nothing
    there, so it always resembles its own clip's median. Over 60 frozen runs
    that gave real subjects 0.54-1.00 and known scenery 0.87-0.99 -- no
    separation. A rail appears in every clip from its camera; a guard appears
    in one.

    Holding still is not itself penalised. A stationary guard also goes
    recovered-and-frozen, but does not look like the scene behind them, so the
    correlation test fails and the track survives. Pass
    `reference_background=None` (the default) to disable this entirely, which
    is what happens for any camera with too few clips to build a reference
    from.

    `flashlight_scores_per_frame`, one list of scores parallel to each
    frame's own `candidates_per_frame` entry, is forwarded to `track_contour`
    unchanged -- see its own docstring. `None` (the default) is the original
    largest-area-only behaviour.
    """
    results: list[tuple[np.ndarray | None, bool]] = []
    track_bbox: tuple[int, int, int, int] | None = None
    track_miss = 0
    recovered_streak = 0
    scenery_streak = 0
    template: np.ndarray | None = None
    velocity = (0.0, 0.0)
    ever_tracked = False
    for index, (gray, candidates) in enumerate(zip(grays, candidates_per_frame, strict=True)):
        externally_enforced = enforce_min_area is not None and enforce_min_area[index]
        floor_active = ever_tracked or bool(externally_enforced)
        contour = track_contour(
            candidates,
            track_bbox,
            max_jump_distance=max_jump_distance,
            size_margin_fraction=search_margin_fraction,
            min_size_margin=min_search_margin,
            max_size_change_ratio=max_size_change_ratio,
            min_reacquire_area=min_reacquire_area if floor_active else 0.0,
            flashlight_scores=(
                flashlight_scores_per_frame[index]
                if flashlight_scores_per_frame is not None
                else None
            ),
        )
        recovered = False
        if contour is None and track_bbox is not None and template is not None:
            search_margin = min(
                max_jump_distance,
                _size_relative_margin(
                    track_bbox, margin_fraction=search_margin_fraction, min_margin=min_search_margin
                ),
            )
            reacquired_bbox = reacquire_by_template(
                gray,
                template,
                track_bbox,
                search_margin=search_margin,
                match_threshold=template_match_threshold,
                velocity=velocity,
            )
            if reacquired_bbox is not None:
                contour = _bbox_to_rect_contour(reacquired_bbox)
                recovered = True
        if contour is None:
            track_miss += 1
            recovered_streak = 0
            scenery_streak = 0
            if track_miss > max_track_miss_frames:
                track_bbox = None
                template = None
                velocity = (0.0, 0.0)
        else:
            ever_tracked = True
            new_bbox = _contour_bbox(contour)
            drift = float("inf")
            if track_bbox is not None:
                old_center, new_center = _bbox_center(track_bbox), _bbox_center(new_bbox)
                velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                drift = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
            track_bbox = new_bbox
            track_miss = 0
            recovered_streak = recovered_streak + 1 if recovered else 0
            if (
                recovered
                and reference_background is not None
                and drift <= _SCENERY_DRIFT_TOLERANCE
                and _patch_similarity(
                    _bbox_crop(gray, new_bbox), _bbox_crop(reference_background, new_bbox)
                )
                >= scenery_correlation
            ):
                scenery_streak += 1
            else:
                scenery_streak = 0
            if scenery_streak > max_scenery_streak:
                for offset in range(1, scenery_streak):
                    results[-offset] = (None, False)
                contour, recovered = None, False
                track_bbox = None
                template = None
                velocity = (0.0, 0.0)
                recovered_streak = 0
                scenery_streak = 0
            elif recovered_streak > max_recovered_streak:
                track_bbox = None
                template = None
                velocity = (0.0, 0.0)
                recovered_streak = 0
            if not recovered and track_bbox is not None:
                x0, y0, x1, y1 = track_bbox
                crop = gray[y0:y1, x0:x1]
                if crop.size > 0:
                    template = crop
        results.append((contour, recovered))
    return results


@dataclass(frozen=True)
class TrackedObject:
    """One persistently-identified subject in one frame, from the multi-object
    tracker (see `track_multiple_objects`). Kept separate from `FrameDetection`
    so it's purely additive diagnostic detail, not fed into feature scoring."""

    track_id: int
    bbox: tuple[int, int, int, int]
    # other track ids currently sharing this same bbox (background-subtraction
    # can no longer tell them apart), empty when this id has its own blob.
    merged_ids: tuple[int, ...] = ()


def track_multiple_objects(
    candidates_per_frame: list[list[np.ndarray]],
    *,
    max_jump_distance: float,
    max_track_miss_frames: int,
    size_margin_fraction: float = 0.75,
    min_size_margin: float = 6.0,
) -> list[list[TrackedObject]]:
    """Greedy multi-object tracker: gives every distinct subject its own
    persistent id across frames, instead of `track_contour`'s single "the"
    track. Candidates are assigned to existing tracks by bounding-box overlap
    first, then by nearest centroid within the same size-relative cap as
    `track_contour` (see `_size_relative_margin`); an unmatched candidate
    spawns a new id, and a track is dropped after `max_track_miss_frames`
    consecutive frames with nothing assigned to it.

    When two or more active tracks' last positions both fall inside a single
    candidate blob (e.g. two people and a backpack walk close enough that
    background-subtraction can no longer separate them), all of them are kept
    alive against that same blob (`TrackedObject.merged_ids`) rather than one
    being silently dropped -- this is what lets a merge be shown (and survived)
    as "still 2 tracks, temporarily sharing one box" instead of collapsing to
    a single identity that then has to be re-acquired as if it were new once
    the subjects separate again. While merged, each track's own internal
    position estimate is dead-reckoned forward by its last observed velocity
    rather than snapped to the shared blob, so that when the blob splits back
    into separate candidates, each id's drifted position is still closest to
    its own actual half of the split rather than an arbitrary pick between
    two now-identical candidate scores.
    """
    next_id = 0
    tracks: dict[int, dict] = {}
    results: list[list[TrackedObject]] = []

    for candidates in candidates_per_frame:
        boxes = [_contour_bbox(c) for c in candidates]
        claims: dict[int, list[int]] = {}
        for tid, track in tracks.items():
            if not boxes:
                continue
            ious = [_bbox_iou(track["bbox"], box) for box in boxes]
            best_iou_idx = max(range(len(boxes)), key=lambda i: ious[i])
            if ious[best_iou_idx] > 0:
                claims.setdefault(best_iou_idx, []).append(tid)
                continue
            allowed = min(
                max_jump_distance,
                _size_relative_margin(
                    track["bbox"], margin_fraction=size_margin_fraction, min_margin=min_size_margin
                ),
            )
            track_center = _bbox_center(track["bbox"])
            distances = [
                ((cx - track_center[0]) ** 2 + (cy - track_center[1]) ** 2) ** 0.5
                for cx, cy in (_bbox_center(box) for box in boxes)
            ]
            nearest_idx = min(range(len(boxes)), key=lambda i: distances[i])
            if distances[nearest_idx] <= allowed:
                claims.setdefault(nearest_idx, []).append(tid)

        frame_tracks: list[TrackedObject] = []
        matched_ids: set[int] = set()
        for cand_idx, tids in claims.items():
            bbox = boxes[cand_idx]
            if len(tids) == 1:
                tid = tids[0]
                old_center, new_center = _bbox_center(tracks[tid]["bbox"]), _bbox_center(bbox)
                tracks[tid]["velocity"] = (
                    new_center[0] - old_center[0],
                    new_center[1] - old_center[1],
                )
                tracks[tid]["bbox"] = bbox
                tracks[tid]["miss"] = 0
                frame_tracks.append(TrackedObject(track_id=tid, bbox=bbox))
                matched_ids.add(tid)
                continue
            for tid in tids:
                vx, vy = tracks[tid].get("velocity", (0.0, 0.0))
                x0, y0, x1, y1 = tracks[tid]["bbox"]
                tracks[tid]["bbox"] = (x0 + vx, y0 + vy, x1 + vx, y1 + vy)
                tracks[tid]["miss"] = 0
                merged_ids = tuple(sorted(set(tids) - {tid}))
                frame_tracks.append(TrackedObject(track_id=tid, bbox=bbox, merged_ids=merged_ids))
                matched_ids.add(tid)

        for tid in list(tracks.keys()):
            if tid in matched_ids:
                continue
            tracks[tid]["miss"] += 1
            if tracks[tid]["miss"] > max_track_miss_frames:
                del tracks[tid]

        claimed_candidate_idxs = set(claims.keys())
        for cand_idx, box in enumerate(boxes):
            if cand_idx in claimed_candidate_idxs:
                continue
            tid = next_id
            next_id += 1
            tracks[tid] = {"bbox": box, "miss": 0}
            frame_tracks.append(TrackedObject(track_id=tid, bbox=box))

        results.append(frame_tracks)
    return results


def _reverse_template_trace(
    grays: list[np.ndarray],
    start_bbox: tuple[int, int, int, int],
    start_template: np.ndarray,
    *,
    search_margin: float,
    match_threshold: float,
    search_margin_fraction: float = 0.75,
    min_search_margin: float = 6.0,
) -> list[tuple[int, int, int, int] | None]:
    """Walk backward through frames with no usable background model at all
    (warmup/flare, dropped before the flare-settle cutoff) using only
    appearance matching against a single fixed anchor template -- there's no
    diff mask to track against there, and the anchor isn't refreshed frame to
    frame since a flare frame's own crop is a worse reference, not a better
    one. Stops (leaving the rest None) at the first frame that doesn't match,
    rather than keep guessing once the trail goes cold.

    Each step's search window is scaled to the subject's own size (like
    `_run_track_pass`) and biased by the displacement observed on the
    previous step, so the trace follows a plausible path back through the
    flare rather than jumping to a lookalike patch.
    """
    results: list[tuple[int, int, int, int] | None] = []
    bbox = start_bbox
    velocity = (0.0, 0.0)
    for gray in grays:
        margin = min(
            search_margin,
            _size_relative_margin(
                bbox, margin_fraction=search_margin_fraction, min_margin=min_search_margin
            ),
        )
        match = reacquire_by_template(
            gray,
            start_template,
            bbox,
            search_margin=margin,
            match_threshold=match_threshold,
            velocity=velocity,
        )
        if match is None:
            break
        old_center, new_center = _bbox_center(bbox), _bbox_center(match)
        velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
        results.append(match)
        bbox = match
    results.extend([None] * (len(grays) - len(results)))
    return results


def _anchor_exemplar_index(detections: list[FrameDetection]) -> int | None:
    """Index of the frame that shows the subject best, to use as a fixed
    reference for the whole clip. None if no frame is trustworthy enough.

    Only real background-subtraction hits qualify: a box that was itself
    produced by appearance matching is not independent evidence of what the
    subject looks like, so seeding from one would just entrench whatever the
    first match latched onto. Among those, the one closest to the clip's own
    median tracked area wins.

    Picking the *largest* box instead is tempting -- more pixels carry more
    appearance -- but measured on 40 labelled clips it inflated the median
    box area across the clip from 798 to 2184 px, because the sweep carries
    the exemplar's size into every frame it fills, including frames where the
    subject is genuinely smaller. The median-sized exemplar is the one that
    best represents the subject's typical appearance.
    """
    real = [fd for fd in detections if fd.largest is not None and not fd.recovered]
    if not real:
        return None
    areas = sorted(_bbox_area(_contour_bbox(fd.largest)) for fd in real)
    typical = areas[len(areas) // 2]
    return min(
        real, key=lambda fd: abs(_bbox_area(_contour_bbox(fd.largest)) - typical)
    ).index


def _anchor_trace(
    grays: list[np.ndarray],
    anchor_index: int,
    anchor_bbox: tuple[int, int, int, int],
    anchor_template: np.ndarray,
    *,
    search_margin: float,
    match_threshold: float,
    max_streak: int,
    search_margin_fraction: float = 0.75,
    min_search_margin: float = 6.0,
) -> list[tuple[int, int, int, int] | None]:
    """Sweep outward from `anchor_index` in both directions, matching every
    frame against one fixed exemplar of the subject.

    The forward/backward `_run_track_pass` templates are refreshed on each
    real detection, which is what lets them follow a subject that genuinely
    changes appearance -- but it also means a partly-wrong box teaches the
    next match to look for a partly-wrong thing, and the reference can drift
    into background over a run of frames. Matching against a single
    never-updated crop of the clip's best-evidenced frame cannot drift.

    Each direction stops at the first frame that fails to match, or after
    `max_streak` consecutive matches, rather than guessing on past a cold
    trail. The cap matters even though this exemplar can't drift the way a
    refreshed template can: once the real subject has left the search window
    for good (walked out of frame), nothing stops the exemplar from matching
    some unrelated static background patch that merely resembles it -- and
    because that patch never moves, it keeps re-matching itself at high
    confidence indefinitely. Observed on a real clip where the subjects exit
    through the frame edge: uncapped, the sweep locked onto a static patch
    and held it, unmoving, for the remaining 16 frames of a 43-frame clip.
    """
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(grays)
    boxes[anchor_index] = anchor_bbox
    for direction in (1, -1):
        bbox = anchor_bbox
        velocity = (0.0, 0.0)
        index = anchor_index + direction
        steps = 0
        while 0 <= index < len(grays) and steps < max_streak:
            margin = min(
                search_margin,
                _size_relative_margin(
                    bbox, margin_fraction=search_margin_fraction, min_margin=min_search_margin
                ),
            )
            match = reacquire_by_template(
                grays[index],
                anchor_template,
                bbox,
                search_margin=margin,
                match_threshold=match_threshold,
                velocity=velocity,
            )
            if match is None:
                break
            old_center, new_center = _bbox_center(bbox), _bbox_center(match)
            velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
            boxes[index] = match
            bbox = match
            steps += 1
            index += direction
    return boxes


def contour_centroid(contour: np.ndarray) -> tuple[float, float] | None:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def normalized_contour_points(contour: np.ndarray, frame_width: int, frame_height: int) -> list:
    return [(float(x) / frame_width, float(y) / frame_height) for [[x, y]] in contour]


def _whole_frame_contour(frame_width: int, frame_height: int) -> np.ndarray:
    w, h = frame_width - 1, frame_height - 1
    return np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _frame_is_merged(detection: ClipDetection, frame_index: int) -> bool:
    """True if the multi-object tracker sees two or more subjects sharing one
    blob in this frame (`TrackedObject.merged_ids` non-empty) -- that frame's
    largest contour describes a merged group silhouette, not one subject."""
    if frame_index >= len(detection.multi_tracks):
        return False
    return any(t.merged_ids for t in detection.multi_tracks[frame_index])


@dataclass(frozen=True)
class FrameDetection:
    """Everything the detector saw in one frame, before it is reduced to features."""

    index: int
    frame: np.ndarray
    mask: np.ndarray
    all_contours: list[np.ndarray]  # every contour found, before either area gate
    blobs: list[np.ndarray]  # contours passing both the min and max area gates
    largest: np.ndarray | None  # largest contour under max_area, no min gate
    centroid: tuple[float, float] | None
    motion_pixel_fraction: float
    median_grey: float
    is_flare: bool
    recovered: bool = False  # track continued via appearance match, not a real bg-diff blob
    filled_by_reverse: bool = False  # forward pass found nothing here; a backward scan did
    filled_by_anchor: bool = False  # box came from the fixed best-frame exemplar sweep
    # Motion this frame that would have formed a contour if not for `ignore_polygons` --
    # None if there was none, or no ignore region at all. Never fed into tracking or
    # feature scoring, purely so a render can show something (e.g. a swaying branch in
    # front of a known stationary light) was suppressed there rather than nothing.
    suppressed_light_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ClipDetection:
    """Per-frame detector output for a whole clip."""

    frames: list[FrameDetection]
    background: np.ndarray
    frame_width: int
    frame_height: int
    warmup_dropped: int
    total_frames: int
    dropped_frames: list[np.ndarray]  # raw frames before the flare-settle cutoff, undetected
    # bbox per dropped_frames entry, from tracing the first scored frame's appearance
    # backward into the flare/warmup region -- None where the trace didn't reach/match.
    dropped_frame_boxes: list[tuple[int, int, int, int] | None]
    # True where the matching dropped_frame_boxes entry came from a real per-pixel
    # background diff against a photometrically-compensated warmup frame (see
    # `detect_clip`'s `compensate_warmup`), rather than pure appearance matching --
    # the renderer uses this to label which mechanism found the box. Empty unless
    # `compensate_warmup=True`.
    dropped_frame_box_is_photometric: list[bool] = field(default_factory=list)
    # dropped_frames, colour-corrected (brightness + per-channel colour cast) to match
    # the settled background -- display/diagnostic only, never fed into detection.
    # Empty unless `compensate_warmup=True`.
    dropped_frame_compensated: list[np.ndarray] = field(default_factory=list)
    # every persistently-identified subject per frame, from track_multiple_objects --
    # additive diagnostic detail, parallel to `frames`, not used in feature scoring.
    multi_tracks: list[list[TrackedObject]] = field(default_factory=list)
    # fraction of this clip's total blob area that sits on reference-matching
    # ground (see the scenery-motion computation in `detect_clip`) -- 0.0 with
    # no motion or no reference background at all. `has_reference_background`
    # tells the two apart, since "no reference" must never silently read the
    # same as "no scenery motion found".
    scenery_motion_fraction: float = 0.0
    has_reference_background: bool = False


def detect_clip(
    video_path: str,
    *,
    max_area_fraction: float = 0.25,
    min_blob_area_fraction: float = 0.0005,
    threshold: int = 18,
    flare_tolerance: float = 3.0,
    max_flare_fraction: float = 0.4,
    max_track_jump_fraction: float = 0.2,
    max_track_miss_frames: int = 5,
    template_match_threshold: float = 0.55,
    flare_match_relax: float = 0.1,
    track_search_margin_fraction: float = 0.75,
    min_track_search_margin: float = 6.0,
    fragment_close_kernel_size: int = 9,
    max_recovered_streak: int = 12,
    max_size_change_ratio: float = 4.0,
    anchor_refine: bool = True,
    max_anchor_streak: int = 4,
    min_reacquire_area: float = 20.0,
    reference_background: np.ndarray | None = None,
    max_scenery_streak: int = 2,
    scenery_correlation: float = 0.94,
    ignore_polygons: tuple[tuple[tuple[float, float], ...], ...] = (),
    compensate_warmup: bool = False,
    prefer_flashlight_candidate: bool = False,
) -> ClipDetection | None:
    """Run the background-subtraction detector over one clip, keeping per-frame
    detail. Returns None if the clip has no readable frames.

    `ignore_polygons` (normalised [0, 1] points, `CameraZone.ignore`) are
    masked out of every frame's motion diff before contour-finding, so a known
    fixed artifact -- a stationary light left in view, or a lens edge/vignette
    colour-fringing band -- can never itself become a tracked blob or inflate
    `blob_count`, regardless of how much it moves/flickers in IR.

    Consecutive-frame differencing was tried first and failed on the real
    footage: these cameras change IR gain/illuminator state as a global
    exposure step, and that step is a far bigger inter-frame delta than an
    actual animal. It hid a porcupine on cam15/15454 entirely. Rather than
    drop a fixed warmup window -- flare settles anywhere from frame 1 to
    frame 21 depending on the clip, and can recur mid-clip -- the cutoff is
    measured per clip from `flare_settle_index` and everything up to it is
    dropped before the background is modelled. Blobs larger than
    `max_area_fraction` of the frame are additionally rejected as residual
    illumination change rather than a subject.

    The per-frame `largest`/`centroid` is not simply "the biggest blob this
    frame" -- it is a persistent track (see `track_contour`) that prefers
    continuing the previous frame's identity by bounding-box overlap, then by
    nearest centroid within `max_track_jump_fraction` of the frame diagonal.
    This is what keeps a tracked subject from flipping to a second person/
    animal that happens to have a larger blob in a later frame. The track
    tolerates up to `max_track_miss_frames` consecutive frames with no
    matching candidate (e.g. the subject briefly blends into the background)
    before it is dropped and the next frame re-acquires on the largest blob,
    same as the original stateless behaviour.

    Within that miss tolerance, a frame with NO plausible background-diff
    candidate at all is not simply skipped: `reacquire_by_template` searches
    the raw (non-diffed) frame around the last known position for the last
    known appearance (a cropped greyscale patch, refreshed on every real
    detection), so a subject that stops registering against the background
    model (e.g. it stands still long enough to blend in, or the diff briefly
    drops below `threshold`) still gets a `FrameDetection.recovered=True` box
    instead of a gap. This is a template match, not a full-frame search, so it
    can only find the same subject near where it was last seen: the search
    window is capped at `track_search_margin_fraction` of the track's own
    current size (floored at `min_track_search_margin` pixels), not a fixed
    frame-relative radius, and is biased toward the last observed direction
    of travel while still covering a full reversal -- otherwise a small or
    slow-moving subject's box can bounce to an unrelated lookalike patch well
    outside where it could plausibly have moved in one frame.

    That same forward-only pass can still miss the frames before a track ever
    gets its first bg-diff hit (e.g. the subject enters slowly, or is small
    enough that it only starts registering a few frames in) -- there's
    nothing to reacquire against yet at that point. `_run_track_pass` is run a
    second time over these frames in reverse, seeded independently, so a
    track that only "starts" partway through can fill in the earlier frames
    from the back; wherever the forward pass found nothing, the backward
    result (marked `FrameDetection.filled_by_reverse=True`) is used instead.
    That same backward scan is then extended past `considered[0]`, into the
    raw warmup/flare frames dropped before the cutoff, using only appearance
    matching against a single fixed anchor crop (there is no background model
    there to diff against) and a slightly relaxed `template_match_threshold`
    (by `flare_match_relax`) since those frames are noisier/differently lit --
    results land in `ClipDetection.dropped_frame_boxes`, kept separate from
    `frames` since they're still not fed into feature scoring.

    `compensate_warmup=True` (default off) improves on that appearance-only
    guess for the SAME warmup window, without touching `frames`/`background`/
    feature scoring at all: each dropped frame is photometrically matched
    (`src.features.photometric_match`, a per-frame gain/offset fit) onto the
    already-trustworthy settled `background`, which cancels the IR gain step
    well enough that a real per-pixel background diff -- the same
    threshold/morphology/contour pipeline used for every scored frame --
    becomes meaningful there too, instead of only ever appearance-matching a
    single fixed crop.

    The fit itself walks backward from the settled frame, not independently
    per warmup frame: frame `drop-1` (closest to settled) is matched directly
    against `background` -- a small, well-conditioned gain/offset fit -- and
    every earlier frame is then matched against its own already-corrected
    neighbour, one small brightness step at a time, all the way back to frame
    0. A frame at the start of a steep ramp (e.g. near-black) fitting directly
    against the far-away settled background is a much larger, less reliable
    jump than a chain of small steps between adjacent frames that are already
    similar to each other. The detection target is unchanged either way --
    every chained frame is still diffed against the real settled `background`
    for motion, only what each individual fit is computed AGAINST changes.

    That real diff is tracked backward from `considered[0]`'s own established
    box using the same `_run_track_pass` state machine as the rest of the
    clip, so it can find a genuinely moving subject the appearance trace
    would only ever re-confirm by lookalike, and correctly report "nothing
    there" (a frame that really is just flare, not a subject) instead of
    forcing a guess. Falls back to the appearance-only box for any frame this
    real-diff pass can't place (e.g. `considered[0]` itself had no detection
    to seed from) so coverage never regresses. `ClipDetection.dropped_frame_
    box_is_photometric` marks which mechanism produced each box.
    `ClipDetection.dropped_frame_compensated` additionally carries every
    dropped COLOUR frame corrected the same chained way (per-channel,
    `src.features.photometric_match_color`) purely for display -- evening out
    the flare's brightness/colour ramp so a human watching a debug render
    sees roughly what the settled background looks like, not the raw ramp.

    Kept separate from `extract_clip_features` so overlays and diagnostics can
    render exactly what scored a clip rather than a lookalike reimplementation.

    A low-contrast subject against a similarly-coloured background (e.g. a
    brown animal in daylight) often diffs out as several small disconnected
    fragments rather than one solid blob, so `largest`/`centroid` only ever
    covers part of it. After the existing MORPH_OPEN (which removes speckle
    noise), a MORPH_CLOSE with a `fragment_close_kernel_size` kernel bridges
    small gaps between nearby fragments into one contour before anything else
    runs -- set it to 0 to disable and fall back to the raw opened mask.

    That close already absorbs essentially all of the recoverable
    fragmentation. Growing the tracked contour further by absorbing nearby
    motion that a sliding window of neighbouring frames corroborates was
    tried and removed: measured over 60 labelled clips, only 2.2% of motion
    pixels lie within 20px of the tracked box, while 47.3% lie more than 60px
    away. The box holding a mean 63% of frame motion is therefore not a
    fragmentation problem -- the rest is vegetation, other subjects and
    speckle genuinely elsewhere in the scene. The growth bought +1.0% recall
    for -2.8% precision.

    `ClipDetection.multi_tracks` additionally runs `track_multiple_objects`
    over the same per-frame candidates, giving every distinct subject in the
    clip its own persistent id (not just the single `largest` track) -- purely
    additive diagnostic detail for telling separate subjects apart in an
    overlay, including when two of them briefly merge into one blob.

    A track sustained for more than `max_recovered_streak` consecutive frames
    purely by appearance recovery (never reconfirmed by a real background-diff
    hit) is dropped and re-acquired fresh instead of kept indefinitely.
    Appearance matching can't distinguish a genuine subject that's briefly
    blended into the background from a static, high-texture background
    feature (e.g. a wire or vine) that was mistakenly picked up once -- both
    trivially keep re-matching their own unchanging template forever. Without
    this, whichever is anchored first wins permanently, even while a much
    larger, genuinely moving candidate persists elsewhere in the same frames.

    A candidate is also rejected as a continuation if its area balloons or
    collapses by more than `max_size_change_ratio` from the track's own last
    size, both growing and shrinking -- otherwise a real subject's track can
    silently "hand off" onto a much larger or smaller co-located blob (e.g. a
    residual-illumination blob shrinking away while the real, much smaller
    subject happens to sit inside it) and keep going under the same identity,
    at the wrong size, for the rest of the clip. The same ratio also decides
    which early frame is trustworthy enough to seed the backward appearance
    trace into `dropped_frame_boxes`: if `considered[0]`'s own box is itself
    an implausible outlier against the clip's typical tracked size (e.g. that
    same residual-illumination blob, before it's had a chance to collapse
    down to the real subject), the trace seeds instead from the first later
    frame whose size is plausible, and traces backward through the
    intervening frames too, not just the true warmup/flare ones.

    A fresh/unconstrained start (no active track, whether that's the very
    first frame or right after a `max_recovered_streak` force-drop) requires
    the candidate to clear `min_reacquire_area` -- without this floor, that
    "pick whatever's largest" fallback can latch onto a single noise-speck/
    sensor-grain contour a few pixels across just because nothing bigger is
    present that frame (observed on cam13/4101: a real guard track was
    followed, after he left frame, by 16 frames confidently tracking a 9-16px
    speck drifting across the scene). Validated against the smallest known
    genuine subjects in this corpus (a 4px-area dassie frame, a 34px porcupine
    frame) at `min_reacquire_area=20` -- both keep tracking unaffected, while
    cam13/4101's post-exit frames correctly report no detection instead of a
    confident wrong one.

    `reference_background`, when supplied, additionally removes runs of frozen
    appearance-recovered frames that match a background built from OTHER clips
    of the same camera at the same coordinates -- the tracker holding a fence
    rail or a mounting pole rather than a subject (see `_run_track_pass` and
    `src.reference_bg`). It is aligned onto this clip's own median by phase
    correlation first, because cameras drift on their mounts between clips and
    the comparison is per-pixel. Omit it to disable the check.

    `prefer_flashlight_candidate=True` (default off) computes `green_light_
    ratio` for every raw motion candidate in every frame (see `track_contour`)
    and lets it override the largest-area pick for a track's fresh/
    unconstrained start whenever a smaller candidate clears `FLASHLIGHT_
    CANDIDATE_MIN_RATIO`. This exists because "biggest contour wins" has a
    confirmed failure mode distinct from everything else in this function: a
    bigger, static-or-drifting bright blob (a sunlit bush, illuminated
    vegetation outside the fence) can simply outsize the guard's own
    flashlight in the same frame, and once the wrong contour is `largest`
    every colour feature downstream describes the wrong thing for the rest of
    the track (confirmed on cam07/11174 -- a real flashlight sat in its own
    1785px contour while a 3025px bush contour in the same frame won the
    old vote). It does not discard any candidate: every contour `cv2.
    findContours` found is still there and still eligible; this only changes
    which one an untracked frame picks first. Off by default because it is
    unmeasured beyond the one confirmed clip -- see the caller (`scripts.
    backtest`/`extract_clip_features`) for how to sweep it against the full
    labelled corpus before trusting it corpus-wide, same discipline as every
    other `detect_clip`-level change in this file's history.

    Both passes above refresh their appearance template on every real
    detection, which is what lets them follow a genuinely changing subject.
    They can still both come up empty on a frame, so `anchor_refine` adds a
    final pass: pick the single best-evidenced real detection in the clip
    (`_anchor_exemplar_index`) and sweep outward from it in both directions
    matching that one never-updated crop (`_anchor_trace`), filling only the
    frames still left with no box. Measured on 40 labelled clips this cut
    boxless frames from 29 to 12 while leaving box-size jitter and centre-path
    smoothness fractionally better than without it.

    It deliberately does not overwrite boxes that were already recovered by
    appearance matching, even though those come from a template that can
    drift. That was tried, on the theory that a fixed exemplar cannot drift,
    and it was worse on both proxies: size jitter rose from 0.230 to 0.264 and
    centre-path jerk from 13.2 to 14.3 across 343 rewritten frames. A template
    refreshed from a nearby frame tracks a subject through gradual change
    better than one anchored to a distant frame, drift notwithstanding.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        return None

    all_grays = [
        cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in frames
    ]
    all_medians = [float(np.median(g)) for g in all_grays]
    drop = flare_settle_index(
        all_medians, tolerance=flare_tolerance, max_fraction=max_flare_fraction
    )
    considered = frames[drop:]
    grays = all_grays[drop:]
    medians = all_medians[drop:]

    frame_height, frame_width = considered[0].shape[:2]
    max_area = max_area_fraction * frame_height * frame_width
    min_blob_area = min_blob_area_fraction * frame_height * frame_width

    background = np.median(np.stack(grays), axis=0).astype(np.uint8)
    aligned_ref = _aligned_reference(reference_background, background)
    kernel = np.ones((3, 3), np.uint8)
    close_kernel = (
        np.ones((fragment_close_kernel_size, fragment_close_kernel_size), np.uint8)
        if fragment_close_kernel_size > 0
        else None
    )

    flares = flare_frames(medians, tolerance=flare_tolerance)
    max_jump_distance = max_track_jump_fraction * (frame_width**2 + frame_height**2) ** 0.5
    ignore_mask = (
        ignore_region_mask(frame_width, frame_height, ignore_polygons) if ignore_polygons else None
    )

    masks: list[np.ndarray] = []
    per_frame_contours: list[list[np.ndarray]] = []
    per_frame_blobs: list[list[np.ndarray]] = []
    per_frame_candidates: list[list[np.ndarray]] = []
    per_frame_suppressed_boxes: list[tuple[int, int, int, int] | None] = []
    motion_fracs: list[float] = []
    total_motion_area = 0.0
    scenery_motion_area = 0.0
    per_frame_flashlight_scores: list[list[float]] = []
    for frame_index, gray in enumerate(grays):
        diff = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        suppressed_box: tuple[int, int, int, int] | None = None
        if ignore_mask is not None:
            suppressed = mask.copy()
            suppressed[~ignore_mask] = 0
            mask[ignore_mask] = 0
            if np.any(suppressed):
                suppressed_contours, _ = cv2.findContours(
                    suppressed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if suppressed_contours:
                    largest_suppressed = max(suppressed_contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_suppressed) >= min_blob_area:
                        suppressed_box = cv2.boundingRect(largest_suppressed)
        per_frame_suppressed_boxes.append(suppressed_box)
        frame_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks.append(mask)
        per_frame_contours.append(list(frame_contours))
        per_frame_blobs.append(
            [c for c in frame_contours if min_blob_area <= cv2.contourArea(c) <= max_area]
        )
        per_frame_candidates.append(
            [c for c in frame_contours if 0 < cv2.contourArea(c) <= max_area]
        )
        if prefer_flashlight_candidate:
            per_frame_flashlight_scores.append(
                [
                    green_light_ratio(considered[frame_index], c)
                    for c in per_frame_candidates[-1]
                ]
            )
        motion_fracs.append(float(np.count_nonzero(mask)) / mask.size)
        # Scored per blob, not per pixel: a location matching the reference at
        # the SAME coordinates (built from other clips of this camera) is
        # something normally there -- foliage or a fence rail shaking in the
        # wind, not a subject that only ever visits once. See `_run_track_pass`
        # for why the comparison must use a different clip's background, not
        # this clip's own (circular, no separation -- measured and rejected).
        for blob in per_frame_blobs[-1]:
            area = cv2.contourArea(blob)
            total_motion_area += area
            if aligned_ref is not None and _patch_similarity(
                _bbox_crop(gray, _contour_bbox(blob)), _bbox_crop(aligned_ref, _contour_bbox(blob))
            ) >= scenery_correlation:
                scenery_motion_area += area

    track_kwargs = {
        "max_jump_distance": max_jump_distance,
        "max_track_miss_frames": max_track_miss_frames,
        "template_match_threshold": template_match_threshold,
        "search_margin_fraction": track_search_margin_fraction,
        "min_search_margin": min_track_search_margin,
        "max_recovered_streak": max_recovered_streak,
        "max_size_change_ratio": max_size_change_ratio,
        "min_reacquire_area": min_reacquire_area,
        "reference_background": aligned_ref,
        "max_scenery_streak": max_scenery_streak,
        "scenery_correlation": scenery_correlation,
    }
    flashlight_scores_per_frame = (
        per_frame_flashlight_scores if prefer_flashlight_candidate else None
    )
    forward = _run_track_pass(
        grays,
        per_frame_candidates,
        flashlight_scores_per_frame=flashlight_scores_per_frame,
        **track_kwargs,
    )
    # Tell the backward pass which original frames already had a real subject
    # established somewhere earlier in actual time -- its own reverse
    # traversal can't know this on its own (see _run_track_pass docstring).
    forward_established = False
    forward_established_by_index: list[bool] = []
    for contour, _recovered in forward:
        if contour is not None:
            forward_established = True
        forward_established_by_index.append(forward_established)
    backward = list(
        reversed(
            _run_track_pass(
                list(reversed(grays)),
                list(reversed(per_frame_candidates)),
                enforce_min_area=list(reversed(forward_established_by_index)),
                flashlight_scores_per_frame=(
                    list(reversed(flashlight_scores_per_frame))
                    if flashlight_scores_per_frame is not None
                    else None
                ),
                **track_kwargs,
            )
        )
    )
    multi_tracks = track_multiple_objects(
        per_frame_blobs,
        max_jump_distance=max_jump_distance,
        max_track_miss_frames=max_track_miss_frames,
        size_margin_fraction=track_search_margin_fraction,
        min_size_margin=min_track_search_margin,
    )

    detections: list[FrameDetection] = []
    for frame_index, frame in enumerate(considered):
        contour, recovered = forward[frame_index]
        filled_by_reverse = False
        if contour is None:
            back_contour, back_recovered = backward[frame_index]
            if back_contour is not None:
                contour, recovered, filled_by_reverse = back_contour, back_recovered, True
        detections.append(
            FrameDetection(
                index=frame_index,
                frame=frame,
                mask=masks[frame_index],
                all_contours=per_frame_contours[frame_index],
                blobs=per_frame_blobs[frame_index],
                largest=contour,
                centroid=None if contour is None else contour_centroid(contour),
                motion_pixel_fraction=motion_fracs[frame_index],
                median_grey=medians[frame_index],
                recovered=recovered,
                filled_by_reverse=filled_by_reverse,
                is_flare=flares[frame_index],
                suppressed_light_box=per_frame_suppressed_boxes[frame_index],
            )
        )

    # Fill frames that both directional passes left with no box at all, by
    # matching one fixed exemplar of the clip's best frame outward in both
    # directions. Frames that already have a box are never touched -- see the
    # note in the docstring about overwriting appearance-recovered ones.
    if anchor_refine:
        anchor_index = _anchor_exemplar_index(detections)
        if anchor_index is not None:
            ax0, ay0, ax1, ay1 = _contour_bbox(detections[anchor_index].largest)
            anchor_template = grays[anchor_index][ay0:ay1, ax0:ax1]
            if anchor_template.size > 0:
                anchor_boxes = _anchor_trace(
                    grays,
                    anchor_index,
                    (ax0, ay0, ax1, ay1),
                    anchor_template,
                    search_margin=max_jump_distance,
                    match_threshold=template_match_threshold,
                    max_streak=max_anchor_streak,
                    search_margin_fraction=track_search_margin_fraction,
                    min_search_margin=min_track_search_margin,
                )
                for frame_index, box in enumerate(anchor_boxes):
                    if box is None or detections[frame_index].largest is not None:
                        continue
                    anchor_contour = _bbox_to_rect_contour(box)
                    detections[frame_index] = replace(
                        detections[frame_index],
                        largest=anchor_contour,
                        centroid=contour_centroid(anchor_contour),
                        recovered=True,
                        filled_by_anchor=True,
                    )

    dropped_frame_boxes: list[tuple[int, int, int, int] | None] = [None] * drop
    seed_index = 0
    tracked_areas = sorted(
        _bbox_area(_contour_bbox(fd.largest)) for fd in detections if fd.largest is not None
    )
    if tracked_areas:
        typical_area = tracked_areas[len(tracked_areas) // 2]
        for i, fd in enumerate(detections):
            if fd.largest is None:
                continue
            if _size_change_plausible(
                typical_area, _bbox_area(_contour_bbox(fd.largest)), max_size_change_ratio
            ):
                seed_index = i
                break
    if (drop > 0 or seed_index > 0) and detections[seed_index].largest is not None:
        x0, y0, x1, y1 = _contour_bbox(detections[seed_index].largest)
        seed_template = grays[seed_index][y0:y1, x0:x1]
        if seed_template.size > 0:
            traced = _reverse_template_trace(
                list(reversed(all_grays[:drop] + grays[:seed_index])),
                (x0, y0, x1, y1),
                seed_template,
                search_margin=max_jump_distance,
                match_threshold=max(0.0, template_match_threshold - flare_match_relax),
                search_margin_fraction=track_search_margin_fraction,
                min_search_margin=min_track_search_margin,
            )
            traced = list(reversed(traced))
            dropped_frame_boxes = traced[:drop]
            # A seed_index above 0 means considered[0]..considered[seed_index-1]
            # were themselves implausibly-sized outliers (see docstring); the
            # same backward trace covers them too, so replace their detection
            # with the traced (properly-sized) box instead of leaving the
            # original oversized/undersized one in place.
            for offset, box in enumerate(traced[drop:]):
                if box is None:
                    continue
                bbox_contour = _bbox_to_rect_contour(box)
                detections[offset] = replace(
                    detections[offset],
                    largest=bbox_contour,
                    centroid=contour_centroid(bbox_contour),
                    recovered=True,
                    filled_by_reverse=True,
                )

    dropped_frame_box_is_photometric = [False] * drop
    dropped_frame_compensated: list[np.ndarray] = []
    if compensate_warmup and drop > 0:
        # Walk backward from the settled frame, which we trust, rather than
        # fitting each warmup frame independently against it -- frame drop-1 is
        # already close to settled (a small, well-conditioned gain/offset fit),
        # and each earlier frame is fit against its own already-corrected
        # neighbour, one small step at a time, instead of every frame separately
        # trying to jump straight from wherever the ramp caught it to the final
        # settled brightness in one fit. The diff against `background` for
        # motion detection is unchanged -- chaining only changes what each
        # frame's gain/offset FIT is computed against, not the detection target.
        background_bgr = np.median(np.stack(considered), axis=0).astype(np.uint8)
        reference_bgr = background_bgr
        compensated_bgr_reversed = []
        for i in range(drop - 1, -1, -1):
            comp_bgr = photometric_match_color(frames[i], reference_bgr)
            compensated_bgr_reversed.append(comp_bgr)
            reference_bgr = comp_bgr
        dropped_frame_compensated = list(reversed(compensated_bgr_reversed))

        reference_gray = background
        compensated_grays_reversed = []
        for i in range(drop - 1, -1, -1):
            gain, offset = photometric_match(all_grays[i], reference_gray)
            comp_gray = apply_photometric_match(all_grays[i], gain, offset)
            compensated_grays_reversed.append(comp_gray)
            reference_gray = comp_gray
        compensated_grays = list(reversed(compensated_grays_reversed))

        dropped_candidates: list[list[np.ndarray]] = []
        for comp_gray in compensated_grays:
            diff = cv2.absdiff(comp_gray, background)
            _, dmask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            dmask = cv2.morphologyEx(dmask, cv2.MORPH_OPEN, kernel)
            if close_kernel is not None:
                dmask = cv2.morphologyEx(dmask, cv2.MORPH_CLOSE, close_kernel)
            if ignore_mask is not None:
                dmask[ignore_mask] = 0
            dcontours, _ = cv2.findContours(dmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dropped_candidates.append([c for c in dcontours if 0 < cv2.contourArea(c) <= max_area])
        # Seed from considered[0]'s own established box when there is one, so the
        # trace picks up the SAME subject rather than whatever else clears the
        # candidate gate first. When considered[0] itself has no detection (the
        # scored track never establishes at all, e.g. a slow-dwelling subject --
        # see cam06/21520 in repo memory), there is nothing to anchor continuity
        # to; run unseeded instead of giving up, so a real per-pixel diff still
        # gets a chance to find and track a subject purely within the warmup
        # window on its own merits.
        has_seed = detections[0].largest is not None
        seed_grays = [grays[0], *reversed(compensated_grays)] if has_seed else list(
            reversed(compensated_grays)
        )
        seed_candidates = (
            [per_frame_candidates[0], *reversed(dropped_candidates)]
            if has_seed
            else list(reversed(dropped_candidates))
        )
        seed_results = _run_track_pass(seed_grays, seed_candidates, **track_kwargs)
        # With a seed, result[0] is grays[0] itself (already tracked, kept only
        # for continuity) -- drop it. Either way the rest are the dropped frames
        # in reverse-chronological order (drop-1 down to 0).
        traced = seed_results[1:] if has_seed else seed_results
        traced_boxes = list(
            reversed(
                [_contour_bbox(contour) if contour is not None else None for contour, _r in traced]
            )
        )
        for i, box in enumerate(traced_boxes):
            if box is None:
                continue
            dropped_frame_boxes[i] = box
            dropped_frame_box_is_photometric[i] = True


    return ClipDetection(
        frames=detections,
        background=background,
        frame_width=frame_width,
        frame_height=frame_height,
        warmup_dropped=drop,
        total_frames=total_frames,
        dropped_frames=frames[:drop],
        dropped_frame_boxes=dropped_frame_boxes,
        dropped_frame_box_is_photometric=dropped_frame_box_is_photometric,
        dropped_frame_compensated=dropped_frame_compensated,
        multi_tracks=multi_tracks,
        scenery_motion_fraction=(
            scenery_motion_area / total_motion_area if total_motion_area > 0 else 0.0
        ),
        has_reference_background=aligned_ref is not None,
    )


def _warmup_motion_features(
    detection: ClipDetection,
    zone: CameraZone,
    frame_width: int,
    frame_height: int,
    *,
    threshold: int = 18,
) -> dict[str, float]:
    """Track whatever is moving in the DROPPED IR-flare frames and classify
    which side of the fence it was on.

    Each dropped frame is photometrically matched (`src.features.
    photometric_match`, the same chained walk-back-from-settled fit
    `detect_clip`'s `compensate_warmup` uses) onto the settled `background`,
    then diffed against it with the same threshold every scored frame uses.

    Was a per-frame-median-ratio consecutive-frame diff until 2026-09-08. That
    approach divides by each frame's OWN median, which is fine on an ordinary
    frame but explodes on a near-black one: cam07/22289's first 5 dropped
    frames have a whole-frame median of 2.0, an 8-bit value that produces a
    50x gain, amplifying ordinary sensor noise into a false "changed" reading
    across most of the frame (13,406-63,710 of 76,800 pixels flagged per pair)
    -- exactly where a real subject (a guard visible bottom-left, confirmed by
    eye against a brightness-boosted still) should have been the clearest
    signal, not the noisiest. A least-squares gain+offset fit against a fixed,
    stable reference does not have this failure mode.

    Re-verified on cam06/21377 after the rewrite: the tracked box still narrows
    steadily as the guard approaches (8,539px down to 8.5px over 9 dropped
    frames) and every frame still reads `inside`, matching the pre-rewrite
    behaviour this feature was originally validated against.

    KNOWN LIMITATION, not solved by this rewrite: diffing against a fixed
    background also flags a STATIC feature that is simply lit differently
    before the IR gain settles than after -- not sensor noise, a real
    photometric difference, just not a moving subject. cam07/18570's largest
    warmup contour is the same static bright branch that its scored frames
    separately lock onto (`best_contour`'s own largest-area selection has an
    identical failure mode there) -- confirmed by the two matching almost
    exactly (183,91,137,87 recovered scored-frame lock vs. 182,90,138,88 here).
    Telling "lit differently" apart from "moved" needs comparing multiple
    frames against EACH OTHER as well as against the background, which this
    single-frame-vs-background diff does not attempt.

    Only the zone verdict survives. Measured on 424 labelled clips by
    leave-one-out AUC, adding features to the ranker baseline of 0.792:

        warmup_outside_fraction            0.809  (+0.017)  <- kept
        warmup_track_fraction              0.787  (-0.006)
        warmup_zone_classifiable_fraction  0.791  (-0.001)
        warmup_subject_left                0.773  (-0.019)
        all four together                  0.779  (-0.014)

    So "was something moving in the warmup" and "did it continue into the
    scored frames" are both worthless -- an intruder moves and leaves there
    just as readily as a guard. WHERE it moved is what carries information,
    for the same reason `outside_frame_fraction` does: it is real independent
    evidence even when the subject left before scoring started and no
    flashlight was ever visible (cam06/21377 has zero warmup flashlight).

    Not usable as a hard suppression rule: a fully-inside warmup track catches
    97/205 guards but also 4/12 positives, so it stays a ranker input only.
    """
    zeros = {"warmup_outside_fraction": 0.0}
    dropped = detection.dropped_frames
    if len(dropped) < 2:
        return zeros

    background = detection.background
    reference = background
    corrected_reversed = []
    for raw in reversed(dropped):
        grey_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        gain, offset = photometric_match(grey_raw, reference)
        corrected = apply_photometric_match(grey_raw, gain, offset)
        corrected_reversed.append(corrected)
        reference = corrected
    corrected_frames = reversed(corrected_reversed)

    frame_area = float(frame_width * frame_height)
    kernel = np.ones((3, 3), np.uint8)
    track: list[tuple[float, float, int, int, int, int]] = []
    for corrected in corrected_frames:
        diff = cv2.absdiff(corrected, background)
        mask = cv2.morphologyEx(
            (diff > threshold).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        x, y, w, h = cv2.boundingRect(best)
        # A blob spanning most of both axes IS the gain step, not a subject.
        if w > 0.6 * frame_width and h > 0.6 * frame_height:
            continue
        if not 0.01 * frame_area <= area <= 0.35 * frame_area:
            continue
        track.append((x, y, w, h))

    if not track:
        return zeros

    # Same base-of-box convention and same geometry every scored side feature
    # uses, so a warmup verdict is directly comparable to outside_frame_fraction.
    verdicts = [
        classify_zone(((bx + bw / 2.0) / frame_width, (by + bh) / frame_height), zone)
        for bx, by, bw, bh in track
    ]
    classifiable = [v for v in verdicts if v in ("outside", "inside")]
    if not classifiable:
        return zeros

    return {
        "warmup_outside_fraction": sum(1 for v in classifiable if v == "outside")
        / len(classifiable)
    }


def _metric_track_features(
    considered: list[FrameDetection],
    zone: CameraZone,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> dict[str, float]:
    """Metric features from the ground-plane model (see `src.ground_calibration`),
    computed in one pass over genuine bg-diff frames (excludes recovered/
    reverse-filled, matching every other per-frame feature in this module --
    a hallucinated continuation box says nothing about the real subject).

    Only meaningful when this camera has opted in to metric calibration
    (`zone.metric_calibration`); `uncalibrated=1.0` otherwise and every other
    key here is 0.0 rather than misleadingly "clean" or "consistent".

    implausible_height_fraction / off_plane_fraction -- a physics gate: how
    often the blob implies an impossible real-world height, or sits somewhere
    the model says isn't the ground at all. Meant to catch flare/rain/branch
    artifacts before they reach the shape-based rules, not to report a
    trustworthy height (compare `estimated_height_m`, the older per-row-ruler
    estimate, still reported separately).

    height_consistency -- coefficient of variation of the RAW implied height
    (enforce_limits=False) across frames where the base point is on the
    ground plane. A real rigid subject keeps roughly the same real height
    frame to frame; a rain streak, swaying branch or flare does not. LOWER is
    more subject-like -- same convention as `src.features.area_stability`.
    0.0 with fewer than 2 valid samples (no evidence of instability, not
    "perfectly consistent").

    depth_progression -- net ground-plane distance travelled, divided by the
    total distance travelled back and forth. A guard patrolling the fence
    line changes range steadily (ratio near 1); vegetation or a fixed-point
    artifact does not translate in depth at all (ratio near 0, or 0.0 if
    distance never measurably changed). Deliberately independent of the
    guard's flashlight being visible, unlike `green_light_ratio`.

    depth_range_m -- max minus min ground-plane distance across the clip,
    in real metres. Separates a subject traversing the view from one milling
    in place at a single range.

    subject_height_m / subject_width_m / subject_area_m2 / metric_aspect --
    median real-world size over plausible frames only (excludes anything
    already counted in implausible_height_fraction). Scale-invariant
    replacements for `row_normalised_area`/`aspect_ratio`, which conflate a
    near subject with a far one -- a human stays ~1.6-1.9m and ~3:1 upright
    regardless of range, an animal does not. Width comes from the ground
    distance between the bbox's two bottom corners, not a pixel ruler.

    distance_median_m -- median ground-plane distance across the clip.

    speed_mps -- median frame-to-frame ground-plane displacement divided by
    real elapsed time (uses `fps` and the actual frame-index gap between
    genuine detections, since recovered/reverse-filled frames in between are
    excluded). Real walking is ~1.4 m/s, running 3-5 m/s -- unlike
    `normalised_speed` (body-widths per frame), this is comparable across
    subjects of different sizes and ranges.
    """
    zeros = {
        "implausible_height_fraction": 0.0,
        "off_plane_fraction": 0.0,
        "height_consistency": 0.0,
        "depth_progression": 0.0,
        "depth_range_m": 0.0,
        "subject_height_m": 0.0,
        "subject_width_m": 0.0,
        "subject_area_m2": 0.0,
        "metric_aspect": 0.0,
        "distance_median_m": 0.0,
        "speed_mps": 0.0,
    }
    cal = calibrate(zone, frame_width, frame_height)
    if cal is None:
        return {**zeros, "uncalibrated": 1.0}

    genuine = 0
    implausible = 0
    off_plane = 0
    heights: list[float] = []
    distances: list[float] = []
    widths: list[float] = []
    aspects: list[float] = []
    areas: list[float] = []
    ground_tracks: list[tuple[int, np.ndarray]] = []
    for detected in considered:
        if detected.largest is None or detected.recovered or detected.filled_by_reverse:
            continue
        genuine += 1
        x, y, w, h = cv2.boundingRect(detected.largest)
        base = (x + w / 2.0, float(y + h))
        ground = cal.ground_point(base)
        if ground is None:
            off_plane += 1
            continue
        ground_tracks.append((detected.index, ground))
        height = cal.height_m(base, float(y), enforce_limits=False)
        if height is None or height <= 0 or height > MAX_SUBJECT_HEIGHT_M:
            implausible += 1
            continue
        left = cal.ground_point((float(x), float(y + h)))
        right = cal.ground_point((float(x + w), float(y + h)))
        width = None if left is None or right is None else float(np.linalg.norm(right - left))
        # Width is checked as strictly as height: an implausible width means
        # this frame's whole metric reading is nonsense, so it counts as
        # implausible rather than quietly contributing a garbage median.
        if width is None or width <= 0 or width > MAX_SUBJECT_WIDTH_M:
            implausible += 1
            continue
        heights.append(height)
        widths.append(width)
        aspects.append(height / width)
        areas.append(height * width)
        distance = cal.distance_m(base)
        if distance is not None:
            distances.append(distance)

    if genuine == 0:
        return {**zeros, "uncalibrated": 0.0}

    speeds: list[float] = []
    if fps > 0:
        for (idx_a, ga), (idx_b, gb) in zip(ground_tracks, ground_tracks[1:], strict=False):
            dt = (idx_b - idx_a) / fps
            if dt > 0:
                speed = float(np.linalg.norm(gb - ga)) / dt
                # Nothing on this terrain outruns a sprint; a higher reading is
                # a tracker jump between unrelated blobs, not a fast subject.
                if speed <= MAX_SUBJECT_SPEED_MPS:
                    speeds.append(speed)

    return {
        "implausible_height_fraction": implausible / genuine,
        "off_plane_fraction": off_plane / genuine,
        "height_consistency": area_stability(heights),
        "depth_progression": depth_progression(distances),
        "depth_range_m": max(distances) - min(distances) if len(distances) >= 2 else 0.0,
        "subject_height_m": float(np.median(heights)) if heights else 0.0,
        "subject_width_m": float(np.median(widths)) if widths else 0.0,
        "subject_area_m2": float(np.median(areas)) if areas else 0.0,
        "metric_aspect": float(np.median(aspects)) if aspects else 0.0,
        "distance_median_m": float(np.median(distances)) if distances else 0.0,
        "speed_mps": float(np.median(speeds)) if speeds else 0.0,
        "uncalibrated": 0.0,
    }


def extract_clip_features(
    video_path: str,
    zone: CameraZone,
    *,
    reference_row: float | None = None,
    max_area_fraction: float = 0.25,
    min_blob_area_fraction: float = 0.0005,
    threshold: int = 18,
    flare_tolerance: float = 3.0,
    max_flare_fraction: float = 0.4,
    daylight_color_fraction: float = 0.15,
    template_match_threshold: float = 0.55,
    reference_background: np.ndarray | None = None,
    scenery_correlation: float = 0.94,
    daylight_hint: bool | None = None,
    prefer_flashlight_candidate: bool = False,
) -> dict[str, float] | None:
    """Run the detector over one clip and compute features for its largest
    track. Returns None if no motion was detected.

    `daylight_color_fraction` gates `warmup_flashlight_ratio` ONLY. It used to
    gate every green-light feature, on the reasoning that a daytime clip with
    real ambient colour reads the same way to a hue check as a flashlight does.
    That was true of the old, permissive flashlight test (`min_saturation` 60,
    which sat below the entire sunlit-grass saturation distribution) and is no
    longer true of the sharpened one: with `src.features`'s
    FLASHLIGHT_MIN_SATURATION/FLASHLIGHT_MIN_BLOB_AREA, `green_light_ratio`
    reads exactly 0.0000 on all 42 labelled animal+incident clips whether the
    gate is applied or not, while night guards still reach p90 0.224. So the
    scored-frame features no longer need a blunt whole-frame veto, and a
    flashlight in daylight can finally be detected instead of discarded.
    The warmup ratio keeps the gate because it is a whole-frame reading over
    the IR-flare frames, where a dusk clip's colourful foliage genuinely does
    dominate -- removing it there flips real animals (cam08/7360 the rooikat,
    cam10/7631) to guard on `warmup_flashlight_ratio` alone.

    `daylight_hint` is the EXOGENOUS answer to the daylight question -- pass
    `src.features.daylight_hint(timestamp)`. When it is `False`, the image
    statistic above is overruled: the clock says night, so whatever colour is
    in frame is not ambient daylight. This matters because the gate is not
    camera-neutral. Measured 2026-09-08 over all 16,272 genuinely-night clips,
    29.7% trip it anyway, concentrated in the cameras that record colour-cast
    night footage: cam16 95.4%, cam01b 94.8%, cam14 87.0%, cam04 55.4%,
    cam01a 54.8%, cam01 43.3%, cam05 40.6% -- against cam07 3.8%, cam06 5.3%,
    cam02 5.6%. Raw-pixel checks confirm the green is really there and
    detectable (12/12 sampled cam01b and cam16 night clips carry >50
    green-mask pixels, up to 36k, hue 36-40, saturation 255).
    Leave it `None` (the default) to rely on the image statistic alone.

    `reference_background`, when supplied, additionally computes
    `scenery_motion_fraction` (see `detect_clip`) -- the fraction of this
    clip's total blob area that sits on ground the reference says is normally
    there, the candidate signal for wind-shaken vegetation. Omit it (the
    default) to leave both `scenery_motion_fraction` at 0.0 and
    `has_reference_background` at 0.0; a consumer must check the latter
    before treating the former as "no scenery motion found", since a camera
    with no reference at all looks identical to one with a reference that
    simply found nothing.

    `prefer_flashlight_candidate` (see `detect_clip`) is forwarded unchanged.
    Off by default, unmeasured corpus-wide -- pass `True` here (and thread it
    through a caller's own CLI flag) to sweep it before adopting it.
    """
    detection = detect_clip(
        video_path,
        max_area_fraction=max_area_fraction,
        min_blob_area_fraction=min_blob_area_fraction,
        threshold=threshold,
        flare_tolerance=flare_tolerance,
        max_flare_fraction=max_flare_fraction,
        template_match_threshold=template_match_threshold,
        ignore_polygons=zone.ignore,
        reference_background=reference_background,
        scenery_correlation=scenery_correlation,
        prefer_flashlight_candidate=prefer_flashlight_candidate,
    )
    if detection is None:
        return None

    # Real elapsed time between frames for speed_mps -- detect_clip itself
    # only tracks frame INDEX, not wall-clock spacing. Sanitised because ~2% of
    # this corpus reports impossible fps (1005, 16000).
    fps = sane_fps(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))

    frame_width, frame_height = detection.frame_width, detection.frame_height
    considered = detection.frames
    ignore_mask = (
        ignore_region_mask(frame_width, frame_height, zone.ignore) if zone.ignore else None
    )
    # Auto-detected stationary lights (see `detect_stationary_light_mask`) supplement
    # any hand-traced `zone.ignore` polygon -- combined into one exclude_mask so a new
    # fixed light on a camera doesn't need its own polygon before it stops reading as
    # the guard's flashlight. `tracked_region` (every frame's own tracked box, however
    # it was found) is carved out first so a guard who dwells in one spot with the
    # flashlight held steady doesn't get auto-classified as a fixed light and excluded
    # from their own flashlight scoring.
    tracked_region = np.zeros((frame_height, frame_width), dtype=bool)
    for detected in considered:
        if detected.largest is None:
            continue
        x, y, w, h = cv2.boundingRect(detected.largest)
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(frame_width, x + w + pad), min(frame_height, y + h + pad)
        tracked_region[y0:y1, x0:x1] = True
    auto_light_mask = detect_stationary_light_mask(
        [d.frame for d in considered], exclude_region=tracked_region
    )
    if np.any(auto_light_mask):
        ignore_mask = auto_light_mask if ignore_mask is None else (ignore_mask | auto_light_mask)

    centroids: list[tuple[float, float]] = []
    # Genuine bg-diff hits only (excludes recovered/filled_by_reverse frames) --
    # motion statistics computed over inferred/template-dragged boxes measure the
    # tracker's willingness to hallucinate a continuation, not real subject
    # behaviour (see the daylight-gate/provenance retrospective in repo memory).
    genuine_centroids: list[tuple[float, float]] = []
    genuine_blob_areas: list[float] = []
    genuine_detected_indices: list[int] = []
    genuine_frames_detected = 0
    non_genuine_frames = 0
    whole_frame_green_ratios: list[float] = []
    color_fractions: list[float] = []
    best_contour: np.ndarray | None = None
    best_frame: np.ndarray | None = None
    best_area = -1.0
    # Fallback when every detected frame is merged (see below) -- better to
    # describe a merged silhouette than to have no shape features at all.
    best_contour_any: np.ndarray | None = None
    best_frame_any: np.ndarray | None = None
    best_area_any = -1.0
    whole_frame = _whole_frame_contour(frame_width, frame_height)
    motion_pixel_fraction = 0.0
    blob_count = 0
    white_fraction = 0.0
    frames_with_box = 0
    flashlight_bbox_frames = 0
    classified_frames = 0
    outside_frames = 0

    # The guard's flashlight is often visible ONLY in the frames dropped for IR
    # flare: they walk out of shot before the gain settles, so every scored
    # frame afterwards contains just whatever moved next (a vine, a bush, a
    # camera artifact on the last frame). Measured 2026-09-06 on the clips the
    # user reviewed, the warmup green signal ran 76-97x the scored signal on
    # exactly those guard clips. Scoring the dropped frames for the flashlight
    # recovers the guard evidence without letting the flare-corrupted frames
    # anywhere near the motion features.
    warmup_flashlight_ratio = 0.0
    if detection.dropped_frames:
        warmup_colour = [color_saturation_fraction(f) for f in detection.dropped_frames]
        warmup_daylight = (
            sum(warmup_colour) / len(warmup_colour) > daylight_color_fraction
            and daylight_hint is not False
        )
        if not warmup_daylight:
            warmup_flashlight_ratio = max(
                green_light_ratio(f, whole_frame, exclude_mask=ignore_mask)
                for f in detection.dropped_frames
            )

    for detected in considered:
        whole_frame_green_ratios.append(
            green_light_ratio(detected.frame, whole_frame, exclude_mask=ignore_mask)
        )
        color_fractions.append(color_saturation_fraction(detected.frame))
        # Peak-frame readings, not an average -- a storm/wind frame with motion
        # scattered across many small blobs (bushes, branches) reads very
        # differently from a single compact subject even at the same threshold.
        motion_pixel_fraction = max(motion_pixel_fraction, detected.motion_pixel_fraction)
        blob_count = max(blob_count, len(detected.blobs))
        if detected.recovered or detected.filled_by_reverse:
            non_genuine_frames += 1
        contour = detected.largest
        if contour is not None:
            # A bright vegetation/web obstruction against the lens genuinely
            # overexposes the sensor (see blob_white_fraction's docstring) --
            # checked on every frame with a box regardless of provenance,
            # same reasoning as motion_pixel_fraction/blob_count above: a peak
            # reading, since the obstruction only needs to appear once.
            white_fraction = max(white_fraction, blob_white_fraction(detected.frame, contour))
        if contour is None:
            continue
        frames_with_box += 1
        if (
            flashlight_bbox_overlap(
                detected.frame, cv2.boundingRect(contour), exclude_mask=ignore_mask
            )
            > FLASHLIGHT_SUBJECT_THRESHOLD
        ):
            flashlight_bbox_frames += 1
        area = cv2.contourArea(contour)
        # Shape features describe the subject at its clearest, not whichever
        # frame happened to be last -- tracks often end on a fading speck. Any
        # provenance is eligible here: a recovered/reverse-filled box still
        # carries a real, previously-measured silhouette worth describing.
        # A frame where the multi-object tracker sees two+ subjects sharing one
        # blob (e.g. two people merged) describes a group silhouette, not a
        # single subject -- excluded from the "clearest frame" pick unless
        # every detected frame is merged, in which case it's the only option.
        if area > best_area_any:
            best_area_any = area
            best_contour_any = contour
            best_frame_any = detected.frame
        if not _frame_is_merged(detection, detected.index) and area > best_area:
            best_area = area
            best_contour = contour
            best_frame = detected.frame
        if detected.centroid is not None:
            centroids.append(detected.centroid)
            if not detected.recovered and not detected.filled_by_reverse:
                genuine_frames_detected += 1
                genuine_detected_indices.append(detected.index)
                genuine_blob_areas.append(area)
                genuine_centroids.append(detected.centroid)
                # Per-frame side verdict. The single-best-frame
                # `outside_pixel_fraction` below describes the clearest
                # silhouette; this instead asks how much of the TRACK was
                # spent outside, which is what separates a subject that was
                # genuinely out there from one caught outside on a single
                # frame (a beam sweep, or a guard leaning over the line).
                frame_points = normalized_contour_points(contour, frame_width, frame_height)
                if zone_classifiable_fraction(frame_points, zone) > 0.0:
                    classified_frames += 1
                    if outside_pixel_fraction(frame_points, zone) > 0.5:
                        outside_frames += 1

    if best_contour is None or best_frame is None:
        best_contour, best_frame = best_contour_any, best_frame_any

    if best_contour is None or best_frame is None:
        return None

    ref_row = reference_row if reference_row is not None else float(frame_height)
    points = normalized_contour_points(best_contour, frame_width, frame_height)
    track = [(x / frame_width, y / frame_height) for x, y in centroids]
    best_width = float(cv2.boundingRect(best_contour)[2])
    color_fraction = sum(color_fractions) / len(color_fractions) if color_fractions else 0.0

    return {
        "outside_pixel_fraction": outside_pixel_fraction(points, zone),
        "zone_classifiable_fraction": zone_classifiable_fraction(points, zone),
        "outside_frame_fraction": (
            outside_frames / classified_frames if classified_frames else 0.0
        ),
        "aspect_ratio": aspect_ratio(best_contour),
        "solidity": solidity(best_contour),
        "saturation_ratio": saturation_ratio(best_frame, best_contour),
        "color_fraction": color_fraction,
        "green_light_ratio": green_light_ratio(
            best_frame, best_contour, exclude_mask=ignore_mask
        ),
        "green_light_flicker": green_light_flicker(whole_frame_green_ratios),
        # Peak flashlight-hue fraction of the WHOLE frame, the scored-frame
        # counterpart of warmup_flashlight_ratio. green_light_ratio only looks
        # inside the tracked contour, so it reads 0.0 whenever the tracker is
        # following the ground the beam is lighting up rather than the beam
        # itself -- confirmed visually on cam07/11174, where an obvious green
        # flashlight sits on the fence while the tracked box is 40% of the
        # frame away on the illuminated bushes outside it.
        # Diagnostic only, deliberately NOT a classify() rule: measured
        # 2026-09-08 it separates well (guard p90 0.244, max 0.881; every
        # incident under 0.00074) but the worst real ANIMAL clip sits at
        # 0.01119, leaving only 1.8x margin at a useful threshold, and against
        # the current rule set it buys one event. Both numbers would have to
        # improve before it earns a place above the geometry rule.
        "whole_frame_green_ratio": (
            max(whole_frame_green_ratios) if whole_frame_green_ratios else 0.0
        ),
        "warmup_flashlight_ratio": warmup_flashlight_ratio,
        **_warmup_motion_features(
            detection, zone, frame_width, frame_height, threshold=threshold
        ),
        "flashlight_subject_fraction": (
            0.0 if not frames_with_box else flashlight_bbox_frames / frames_with_box
        ),
        "row_normalised_area": row_normalised_area(best_contour, ref_row),
        "edge_density": edge_density(best_frame, best_contour),
        "blob_white_fraction": white_fraction,
        "long_flare_frames": float(detection.warmup_dropped),
        # Whole-clip transition, so it spans the dropped warmup frames AND the
        # scored ones -- the flash itself is often inside the flare window.
        "post_flash_red_shift": post_flash_red_shift(
            list(detection.dropped_frames) + [d.frame for d in considered]
        ),
        "path_length": path_length(genuine_centroids),
        "jitter": jitter(genuine_centroids),
        "persistence": persistence(genuine_frames_detected, len(considered)),
        "motion_pixel_fraction": motion_pixel_fraction,
        "blob_count": float(blob_count),
        # Median counterparts of the two peak readings above. The peak is what a
        # storm needs, but it also fires on a single flare-settle frame at the
        # start of an otherwise quiet clip (cam04/10887 reads [16, 5, 5, 4, 3...]);
        # the median only rises when the scattered motion actually persists.
        "motion_pixel_fraction_median": _median([d.motion_pixel_fraction for d in considered]),
        "blob_count_median": _median([float(len(d.blobs)) for d in considered]),
        "longest_detection_run": longest_detection_run(genuine_detected_indices, len(considered)),
        "area_stability": area_stability(genuine_blob_areas),
        "normalised_speed": normalised_speed(genuine_centroids, best_width),
        "heading_change": heading_change(genuine_centroids),
        "fence_crossed": float(track_crosses_fence(track, zone)),
        "median_fence_distance": median_fence_distance(track, zone),
        "recovered_fraction": (
            non_genuine_frames / len(considered) if considered else 0.0
        ),
        "scenery_motion_fraction": detection.scenery_motion_fraction,
        "has_reference_background": float(detection.has_reference_background),
        **_metric_track_features(considered, zone, frame_width, frame_height, fps),
    }


def iter_labelled_clips_with_files(conn, *, camera_id: str) -> Iterator[dict[str, Any]]:
    """Labelled clips for one camera that have a local file_path to extract from."""
    for row in db.iter_clips(conn, camera_id=camera_id):
        if not row["file_path"]:
            continue
        label_row = db.get_label(conn, row["channel_id"], row["message_id"])
        if label_row is None or label_row["label"] is None:
            continue  # event's class not known yet (may only carry a startup_state)
        yield {
            "channel_id": row["channel_id"],
            "message_id": row["message_id"],
            "camera_id": row["camera_id"],
            "file_path": row["file_path"],
            "label": label_row["label"],
            "timestamp": row["timestamp"],
        }


def run_spike(
    conn,
    cameras: CamerasConfig,
    *,
    camera_id: str,
    operating_window_start: str = "18:00",
    operating_window_end: str = "06:00",
) -> list[dict[str, Any]]:
    camera = cameras.by_id(camera_id)
    if camera is None:
        raise ValueError(f"Unknown camera_id: {camera_id!r}")

    rows: list[dict[str, Any]] = []
    for clip in iter_labelled_clips_with_files(conn, camera_id=camera_id):
        features = extract_clip_features(clip["file_path"], camera.zone_at(clip["timestamp"]))
        if features is None:
            continue
        rows.append(
            {
                "channel_id": clip["channel_id"],
                "message_id": clip["message_id"],
                "camera_id": clip["camera_id"],
                "label": clip["label"],
                "time_of_day": time_of_day(
                    clip["timestamp"],
                    window_start=operating_window_start,
                    window_end=operating_window_end,
                ),
                "is_daylight": is_daylight(clip["timestamp"]),
                **features,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:  # pragma: no cover - requires real labelled footage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="Camera id from config/cameras.yaml")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    cameras_cfg = load_cameras_config("config/cameras.yaml")
    conn = db.connect(app_cfg.db_path)

    rows = run_spike(
        conn,
        cameras_cfg,
        camera_id=args.camera,
        operating_window_start=app_cfg.operating_window_start,
        operating_window_end=app_cfg.operating_window_end,
    )
    conn.close()

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} labelled feature rows to {args.out}")


if __name__ == "__main__":
    main()
