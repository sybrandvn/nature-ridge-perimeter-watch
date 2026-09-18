"""Motion detection, tracking, and versioned feature-cache persistence.

This module owns the image-to-``ClipDetection`` pipeline and its tracking
primitives. Zone-specific feature scoring lives in :mod:`src.scoring`.
Cached values are that extractor's JSON-safe feature mapping, not
``ClipDetection`` (which contains raw NumPy frames, masks, and contours).

Although the database column is named ``motion_fingerprint``, its value here is
an extraction fingerprint: global motion settings plus every per-clip input
that can affect extracted features. This deliberately trades reuse after a zone
edit for correctness until the detector/feature split can cache a smaller,
truly zone-independent payload.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src import db
from src.config import CameraZone
from src.features import (
    FLASHLIGHT_CANDIDATE_MIN_RATIO,
    apply_photometric_match,
    flare_frames,
    flare_settle_index,
    green_light_ratio,
    ignore_region_mask,
    photometric_match,
    photometric_match_color,
)
from src.reference_bg import align

EXTRACTOR_VERSION = "motion-features-v1"
_NO_MOTION_KEY = "__perimeter_watch_no_motion__"
_FLAT_PATCH_STD = 1e-3
_FLAT_PATCH_TOLERANCE = 2.0
_SCENERY_DRIFT_TOLERANCE = 1.0
_UNGATED_COST = 1.0e6


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
    reference_background_primary: bool = False,
    max_scenery_streak: int = 2,
    scenery_correlation: float = 0.94,
    ignore_polygons: tuple[tuple[tuple[float, float], ...], ...] = (),
    compensate_warmup: bool = False,
    prefer_flashlight_candidate: bool = False,
    multi_track_confirm_frames: int = 1,
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

    `reference_background_primary=True` is an EXPERIMENTAL recovery path. It
    uses the aligned cross-clip reference as the scored-frame difference
    target, rather than this clip's median background. This can reveal a
    subject that is already present for most of a short clip and was therefore
    absorbed into its own median. It is off by default and must be measured
    through ``scripts.backtest --reference-background-primary`` before any
    production decision: a reference mismatch can also make stationary scene
    changes look like foreground.

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

    `multi_track_confirm_frames` (default 1, i.e. off) is `track_multiple_
    objects`'s `confirm_frames` for the real `multi_tracks` pass below --
    raising it lets that pass require a new track to be matched several
    consecutive frames before it is reported at all, so a single-frame
    speck of noise (a leaf edge, a compression artifact) never mints its
    own persistent id when tracking EVERY raw `per_frame_candidates`
    contour (any nonzero area up to `max_area_fraction`) instead of only
    min-area `per_frame_blobs` -- the concrete fix for "the multi-object
    tracker cannot see small subjects at all" (docs/detection_improvement_
    review.md section 3, stage 1), the animal population specifically.

    Measured 2026-09-09 at confirm_frames=3 against the full labelled
    corpus and REJECTED at that value: it silently suppresses exactly the
    short-lived flashlight objects `_multi_object_flashlight_features`
    depends on (a torch beam is often visible for only 1-2 frames before
    the tracker's own gating drops or re-splits it), regressing 16 guard
    clips whose `guard_candidate` classification depends on
    `multi_object_max_flashlight_ratio` -- including cam07/11174, the
    exact clip that motivated shipping that feature in the first place
    (its ratio collapsed from 0.429, comfortably over
    `GREEN_LIGHT_RATIO_MIN`, to 0.0). Defaulting to 1 reproduces the
    pre-rewrite immediate-report behaviour exactly (every candidate is
    reported the frame it first appears, same as the old blob-only
    tracker), so `multi_object_max_flashlight_ratio` is unaffected -- see
    `docs/detection_improvement_review.md`'s implementation-status section
    for the full before/after corpus numbers. The candidates-vs-blobs input
    switch (the actual small-subject fix) still applies regardless of this
    value; only the noise-suppression half of the stage-1 tracker rewrite
    is gated behind it, and is left off by default until a real motivating
    case for it is measured.

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
    detection_background = (
        aligned_ref if reference_background_primary and aligned_ref is not None else background
    )
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
        diff = cv2.absdiff(gray, detection_background)
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
        per_frame_candidates,
        max_jump_distance=max_jump_distance,
        max_track_miss_frames=max_track_miss_frames,
        size_margin_fraction=track_search_margin_fraction,
        min_size_margin=min_track_search_margin,
        frames_bgr=considered,
        max_merge_streak=max_recovered_streak,
        confirm_frames=multi_track_confirm_frames,
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


def largest_contour(mask: np.ndarray, *, max_area: float | None = None) -> np.ndarray | None:
    """Largest external contour in a binary motion mask, or ``None`` if empty."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if max_area is not None:
        contours = [contour for contour in contours if cv2.contourArea(contour) <= max_area]
    return max(contours, key=cv2.contourArea) if contours else None


def _contour_bbox(contour: np.ndarray) -> tuple[int, int, int, int]:
    """Return a contour box in corner form ``(x0, y0, x1, y1)``."""
    x, y, width, height = cv2.boundingRect(contour)
    return (x, y, x + width, y + height)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    intersection_width = max(0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def _size_change_plausible(from_area: float, to_area: float, max_ratio: float) -> bool:
    """Whether two tracked-box areas can plausibly describe one subject."""
    if from_area <= 0 or to_area <= 0:
        return True
    ratio = to_area / from_area
    return (1.0 / max_ratio) <= ratio <= max_ratio


def _size_relative_margin(
    bbox: tuple[int, int, int, int], *, margin_fraction: float, min_margin: float
) -> float:
    """A position-search margin scaled to the tracked object's own size."""
    x0, y0, x1, y1 = bbox
    return max(margin_fraction * max(x1 - x0, y1 - y0), min_margin)


def contour_centroid(contour: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def _bbox_to_rect_contour(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Synthesize a rectangular contour from a corner-form bounding box."""
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
    """Find a prior subject template in a local, velocity-biased window."""
    template_height, template_width = template.shape[:2]
    if template_height == 0 or template_width == 0:
        return None
    x0, y0, x1, y1 = last_bbox
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    velocity_x, velocity_y = velocity
    base_half_width = (x1 - x0) / 2.0 + search_margin
    base_half_height = (y1 - y0) / 2.0 + search_margin
    center_x += velocity_x / 2.0
    center_y += velocity_y / 2.0
    half_width = base_half_width + abs(velocity_x) / 2.0
    half_height = base_half_height + abs(velocity_y) / 2.0
    frame_height, frame_width = gray_frame.shape[:2]
    window_x0 = max(int(center_x - half_width), 0)
    window_y0 = max(int(center_y - half_height), 0)
    window_x1 = min(int(center_x + half_width), frame_width)
    window_y1 = min(int(center_y + half_height), frame_height)
    window = gray_frame[window_y0:window_y1, window_x0:window_x1]
    if window.shape[0] < template_height or window.shape[1] < template_width:
        return None
    result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, maximum, _, location = cv2.minMaxLoc(result)
    if maximum < match_threshold:
        return None
    match_x, match_y = location
    return (
        window_x0 + match_x,
        window_y0 + match_y,
        window_x0 + match_x + template_width,
        window_y0 + match_y + template_height,
    )


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
    """Select the contour that plausibly continues a single tracked subject.

    A continuing subject must overlap, or stay within a box-size-scaled search
    margin, and cannot abruptly balloon or collapse in area.  A failed
    continuation is a miss, not a jump to the frame's largest blob: the
    caller's appearance-recovery policy gets the next chance to recover it.
    A fresh/reacquired subject must clear ``min_reacquire_area`` so a forced
    reset cannot silently follow sensor speckle. Optional flashlight scores
    let a smaller verified torch beat an unrelated larger bush, including when
    it appears independently while another track is active.
    """
    if not candidates:
        return None
    if track_bbox is None:
        eligible = [
            index
            for index, contour in enumerate(candidates)
            if cv2.contourArea(contour) >= min_reacquire_area
        ]
        if not eligible:
            return None
        if flashlight_scores is not None:
            flashlight_candidates = [
                index
                for index in eligible
                if flashlight_scores[index] > FLASHLIGHT_CANDIDATE_MIN_RATIO
            ]
            if flashlight_candidates:
                eligible = flashlight_candidates
        return candidates[max(eligible, key=lambda index: cv2.contourArea(candidates[index]))]

    boxes = [_contour_bbox(contour) for contour in candidates]
    track_area = _bbox_area(track_bbox)
    size_ok = [
        _size_change_plausible(track_area, _bbox_area(box), max_size_change_ratio)
        for box in boxes
    ]
    overlaps = [
        _bbox_iou(track_bbox, box) if plausible else 0.0
        for box, plausible in zip(boxes, size_ok, strict=True)
    ]
    best_overlap = max(range(len(candidates)), key=lambda index: overlaps[index])
    if overlaps[best_overlap] > 0:
        return candidates[best_overlap]

    track_center = _bbox_center(track_bbox)
    distances = [
        ((center_x - track_center[0]) ** 2 + (center_y - track_center[1]) ** 2) ** 0.5
        if plausible
        else float("inf")
        for plausible, (center_x, center_y) in zip(
            size_ok, (_bbox_center(box) for box in boxes), strict=True
        )
    ]
    closest = min(range(len(candidates)), key=lambda index: distances[index])
    allowed_distance = min(
        max_jump_distance,
        _size_relative_margin(
            track_bbox, margin_fraction=size_margin_fraction, min_margin=min_size_margin
        ),
    )
    if distances[closest] <= allowed_distance:
        return candidates[closest]

    if flashlight_scores is not None:
        flashlight_candidates = [
            index
            for index, score in enumerate(flashlight_scores)
            if score > FLASHLIGHT_CANDIDATE_MIN_RATIO
            and cv2.contourArea(candidates[index]) >= min_reacquire_area
        ]
        if flashlight_candidates:
            return candidates[
                max(flashlight_candidates, key=lambda index: cv2.contourArea(candidates[index]))
            ]
    return None


def _bbox_crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def _patch_similarity(patch: np.ndarray, other: np.ndarray) -> float:
    """Compare matching patches, including stable handling of flat imagery."""
    if patch.shape != other.shape or patch.size == 0:
        return 0.0
    first, second = patch.astype(np.float32), other.astype(np.float32)
    if float(first.std()) < _FLAT_PATCH_STD or float(second.std()) < _FLAT_PATCH_STD:
        return 1.0 if float(np.abs(first - second).mean()) <= _FLAT_PATCH_TOLERANCE else 0.0
    return float(cv2.matchTemplate(first, second, cv2.TM_CCOEFF_NORMED)[0, 0])


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
    """Apply the single-track state machine in forward or reverse order.

    Background-difference candidates establish/continue a track; nearby
    template matching fills short gaps. Long appearance-only runs are reset,
    and a frozen recovered run matching a cross-clip reference background is
    retrospectively discarded as scenery. ``enforce_min_area`` lets a reverse
    pass retain chronological knowledge of when a track had already existed.
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
        contour = track_contour(
            candidates,
            track_bbox,
            max_jump_distance=max_jump_distance,
            size_margin_fraction=search_margin_fraction,
            min_size_margin=min_search_margin,
            max_size_change_ratio=max_size_change_ratio,
            min_reacquire_area=(min_reacquire_area if ever_tracked or externally_enforced else 0.0),
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
                    track_bbox,
                    margin_fraction=search_margin_fraction,
                    min_margin=min_search_margin,
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


def _bbox_histogram(
    frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]
) -> np.ndarray | None:
    """Return a normalized HSV histogram for a non-empty bounding-box crop."""
    x0, y0, x1, y1 = (int(round(value)) for value in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame_bgr.shape[1], x1), min(frame_bgr.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return histogram


def _histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the bounded Bhattacharyya distance between two histograms."""
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def track_multiple_objects(
    candidates_per_frame: list[list[np.ndarray]],
    *,
    max_jump_distance: float,
    max_track_miss_frames: int,
    size_margin_fraction: float = 0.75,
    min_size_margin: float = 6.0,
    frames_bgr: list[np.ndarray] | None = None,
    appearance_weight: float = 0.25,
    max_merge_streak: int = 30,
    confirm_frames: int = 1,
) -> list[list[TrackedObject]]:
    """Assign stable IDs to all candidate contours across a clip.

    Assignment uses a globally optimal Hungarian solve over velocity-predicted
    boxes, with an optional HSV appearance tie-breaker. Overlapping tracks can
    survive a temporary merged blob, and new tracks may require consecutive
    confirmation before they are emitted.
    """
    next_id = 0
    tracks: dict[int, dict[str, Any]] = {}
    results: list[list[TrackedObject]] = []

    for frame_index, candidates in enumerate(candidates_per_frame):
        boxes = [_contour_bbox(candidate) for candidate in candidates]
        frame_bgr = (
            frames_bgr[frame_index]
            if frames_bgr is not None and frame_index < len(frames_bgr)
            else None
        )
        track_ids = list(tracks)
        predicted: dict[int, tuple[float, float, float, float]] = {}
        for track_id in track_ids:
            track = tracks[track_id]
            velocity_x, velocity_y = track.get("velocity", (0.0, 0.0))
            x0, y0, x1, y1 = track["bbox"]
            predicted[track_id] = (
                x0 + velocity_x,
                y0 + velocity_y,
                x1 + velocity_x,
                y1 + velocity_y,
            )

        primary_for_track: dict[int, int] = {}
        group_for_candidate: dict[int, list[int]] = {}
        if track_ids and boxes:
            cost = np.full((len(track_ids), len(boxes)), _UNGATED_COST, dtype=float)
            for row, track_id in enumerate(track_ids):
                predicted_bbox = predicted[track_id]
                allowed = min(
                    max_jump_distance,
                    _size_relative_margin(
                        predicted_bbox,
                        margin_fraction=size_margin_fraction,
                        min_margin=min_size_margin,
                    ),
                )
                predicted_center = _bbox_center(predicted_bbox)
                track_histogram = tracks[track_id].get("hist")
                for column, box in enumerate(boxes):
                    iou = _bbox_iou(predicted_bbox, box)
                    if iou > 0:
                        geometric_cost = 1.0 - iou
                    else:
                        center_x, center_y = _bbox_center(box)
                        distance = (
                            (center_x - predicted_center[0]) ** 2
                            + (center_y - predicted_center[1]) ** 2
                        ) ** 0.5
                        if distance > allowed:
                            continue
                        geometric_cost = 1.0 + distance / allowed if allowed > 0 else 1.0
                    appearance_cost = 0.0
                    if frame_bgr is not None and track_histogram is not None:
                        candidate_histogram = _bbox_histogram(frame_bgr, box)
                        if candidate_histogram is not None:
                            appearance_cost = _histogram_distance(
                                track_histogram, candidate_histogram
                            )
                    cost[row, column] = geometric_cost + appearance_weight * appearance_cost

            row_indices, column_indices = linear_sum_assignment(cost)
            for row, column in zip(row_indices, column_indices, strict=True):
                if cost[row, column] >= _UNGATED_COST:
                    continue
                track_id = track_ids[row]
                primary_for_track[track_id] = column
                group_for_candidate.setdefault(column, []).append(track_id)

        for track_id in track_ids:
            if track_id in primary_for_track or not boxes:
                continue
            predicted_bbox = predicted[track_id]
            ious = [_bbox_iou(predicted_bbox, box) for box in boxes]
            best_index = max(range(len(boxes)), key=lambda index: ious[index])
            if ious[best_index] > 0 and best_index in group_for_candidate:
                group_for_candidate[best_index].append(track_id)

        frame_tracks: list[TrackedObject] = []
        matched_ids: set[int] = set()
        for candidate_index, track_group in group_for_candidate.items():
            bbox = boxes[candidate_index]
            if len(track_group) == 1:
                track_id = track_group[0]
                old_center = _bbox_center(tracks[track_id]["bbox"])
                new_center = _bbox_center(bbox)
                tracks[track_id]["velocity"] = (
                    new_center[0] - old_center[0],
                    new_center[1] - old_center[1],
                )
                tracks[track_id]["bbox"] = bbox
                tracks[track_id]["miss"] = 0
                tracks[track_id]["merge_streak"] = 0
                tracks[track_id]["pending"] = tracks[track_id].get("pending", 0) + 1
                if frame_bgr is not None:
                    histogram = _bbox_histogram(frame_bgr, bbox)
                    if histogram is not None:
                        tracks[track_id]["hist"] = histogram
                matched_ids.add(track_id)
                if tracks[track_id]["pending"] >= confirm_frames:
                    frame_tracks.append(TrackedObject(track_id=track_id, bbox=bbox))
                continue

            for track_id in track_group:
                tracks[track_id]["merge_streak"] = tracks[track_id].get("merge_streak", 0) + 1
                if tracks[track_id]["merge_streak"] > max_merge_streak:
                    del tracks[track_id]
                    matched_ids.add(track_id)
                    continue
                velocity_x, velocity_y = tracks[track_id].get("velocity", (0.0, 0.0))
                x0, y0, x1, y1 = tracks[track_id]["bbox"]
                tracks[track_id]["bbox"] = (
                    x0 + velocity_x,
                    y0 + velocity_y,
                    x1 + velocity_x,
                    y1 + velocity_y,
                )
                tracks[track_id]["miss"] = 0
                matched_ids.add(track_id)
                merged_ids = tuple(sorted(set(track_group) - {track_id}))
                if tracks[track_id]["pending"] >= confirm_frames:
                    frame_tracks.append(
                        TrackedObject(
                            track_id=track_id,
                            bbox=tracks[track_id]["bbox"],
                            merged_ids=merged_ids,
                        )
                    )

        for track_id in list(tracks):
            if track_id in matched_ids:
                continue
            tracks[track_id]["miss"] += 1
            if tracks[track_id]["miss"] > max_track_miss_frames:
                del tracks[track_id]

        claimed_candidate_indices = set(group_for_candidate)
        for candidate_index, box in enumerate(boxes):
            if candidate_index in claimed_candidate_indices:
                continue
            track_id = next_id
            next_id += 1
            tracks[track_id] = {
                "bbox": box,
                "miss": 0,
                "merge_streak": 0,
                "pending": 1,
            }
            if frame_bgr is not None:
                histogram = _bbox_histogram(frame_bgr, box)
                if histogram is not None:
                    tracks[track_id]["hist"] = histogram
            if confirm_frames <= 1:
                frame_tracks.append(TrackedObject(track_id=track_id, bbox=box))

        results.append(frame_tracks)
    return results


def _aligned_reference(
    reference: np.ndarray | None, background: np.ndarray
) -> np.ndarray | None:
    """Register a compatible cross-clip reference onto this clip's median."""
    if reference is None or reference.shape != background.shape:
        return None
    aligned, _shift = align(reference, background)
    return aligned


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
    """Trace a fixed subject template backward through warmup/flare frames."""
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
    """Select the genuine detection nearest the clip's median tracked area."""
    genuine = [
        detected
        for detected in detections
        if detected.largest is not None and not detected.recovered
    ]
    if not genuine:
        return None
    areas = sorted(_bbox_area(_contour_bbox(detected.largest)) for detected in genuine)
    typical = areas[len(areas) // 2]
    return min(
        genuine,
        key=lambda detected: abs(
            _bbox_area(_contour_bbox(detected.largest)) - typical
        ),
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
    """Sweep a fixed exemplar outward from an established anchor frame."""
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
                    bbox,
                    margin_fraction=search_margin_fraction,
                    min_margin=min_search_margin,
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


@dataclass(frozen=True)
class TrackedObject:
    """One persistently identified object in one frame.

    ``bbox`` is corner-form ``(x0, y0, x1, y1)``. It is intentionally a
    detector-owned, zone-independent DTO: consumers such as feature scoring
    and the debug renderer can share the same observed object without pulling
    tracking implementation back into their module.
    """

    track_id: int
    bbox: tuple[int, int, int, int]
    # Other ids sharing this detector box; empty when this id has its own blob.
    merged_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class FrameDetection:
    """Everything the motion detector observed in one video frame."""

    index: int
    frame: np.ndarray
    mask: np.ndarray
    all_contours: list[np.ndarray]
    blobs: list[np.ndarray]
    largest: np.ndarray | None
    centroid: tuple[float, float] | None
    motion_pixel_fraction: float
    median_grey: float
    is_flare: bool
    recovered: bool = False
    filled_by_reverse: bool = False
    filled_by_anchor: bool = False
    suppressed_light_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ClipDetection:
    """Detector output for one clip, before any zone-specific scoring.

    This is deliberately an in-memory hand-off type, not the cache payload:
    it contains NumPy imagery and contours. Keeping it here establishes the
    detector/feature boundary needed for a later compact raw-track cache.
    """

    frames: list[FrameDetection]
    background: np.ndarray
    frame_width: int
    frame_height: int
    warmup_dropped: int
    total_frames: int
    dropped_frames: list[np.ndarray]
    dropped_frame_boxes: list[tuple[int, int, int, int] | None]
    dropped_frame_box_is_photometric: list[bool] = field(default_factory=list)
    dropped_frame_compensated: list[np.ndarray] = field(default_factory=list)
    multi_tracks: list[list[TrackedObject]] = field(default_factory=list)
    scenery_motion_fraction: float = 0.0
    has_reference_background: bool = False


@dataclass(frozen=True)
class GeometryObservations:
    """Compact, JSON-safe detector evidence needed for fence-side scoring.

    Unlike :class:`ClipDetection`, this deliberately contains no frames,
    masks, or OpenCV contours.  It is the first cacheable slice of detector
    output: enough to replay the fence/depth features after a zone edit, but
    not enough to replay colour, texture, or metric-calibration features.
    Pixel dimensions and multi-object boxes stay in pixel coordinates so a
    replay has exactly the same base-of-box convention as the detector.
    """

    frame_width: int
    frame_height: int
    best_contour_points: tuple[tuple[float, float], ...]
    genuine_contour_points: tuple[tuple[tuple[float, float], ...], ...]
    centroid_track: tuple[tuple[float, float], ...]
    multi_tracks: tuple[tuple[TrackedObject, ...], ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a plain JSON-compatible representation for future storage."""
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "best_contour_points": [list(point) for point in self.best_contour_points],
            "genuine_contour_points": [
                [list(point) for point in contour] for contour in self.genuine_contour_points
            ],
            "centroid_track": [list(point) for point in self.centroid_track],
            "multi_tracks": [
                [
                    {
                        "track_id": obj.track_id,
                        "bbox": list(obj.bbox),
                        "merged_ids": list(obj.merged_ids),
                    }
                    for obj in frame_tracks
                ]
                for frame_tracks in self.multi_tracks
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GeometryObservations:
        """Restore a payload produced by :meth:`to_payload`."""
        return cls(
            frame_width=int(payload["frame_width"]),
            frame_height=int(payload["frame_height"]),
            best_contour_points=tuple(
                (float(x), float(y)) for x, y in payload["best_contour_points"]
            ),
            genuine_contour_points=tuple(
                tuple((float(x), float(y)) for x, y in contour)
                for contour in payload["genuine_contour_points"]
            ),
            centroid_track=tuple((float(x), float(y)) for x, y in payload["centroid_track"]),
            multi_tracks=tuple(
                tuple(
                    TrackedObject(
                        track_id=int(obj["track_id"]),
                        bbox=tuple(int(value) for value in obj["bbox"]),
                        merged_ids=tuple(int(value) for value in obj["merged_ids"]),
                    )
                    for obj in frame_tracks
                )
                for frame_tracks in payload["multi_tracks"]
            ),
        )


@dataclass(frozen=True)
class MetricFrameObservation:
    """A genuine detector box at its original frame index."""

    frame_index: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class MetricObservations:
    """Compact, JSON-safe evidence needed for ground-plane metric scoring.

    Frame imagery and contour detail are unnecessary for the calibration
    features: the frame dimensions, genuine contour boxes, their original
    indices, and the clip frame rate fully determine the existing readings.
    """

    frame_width: int
    frame_height: int
    fps: float
    frames: tuple[MetricFrameObservation, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.fps,
            "frames": [
                {"frame_index": observed.frame_index, "bbox": list(observed.bbox)}
                for observed in self.frames
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MetricObservations:
        return cls(
            frame_width=int(payload["frame_width"]),
            frame_height=int(payload["frame_height"]),
            fps=float(payload["fps"]),
            frames=tuple(
                MetricFrameObservation(
                    frame_index=int(observed["frame_index"]),
                    bbox=tuple(int(value) for value in observed["bbox"]),
                )
                for observed in payload["frames"]
            ),
        )


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_identity(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def extraction_fingerprint(
    *,
    video_path: str | Path,
    motion_fingerprint: str,
    zone: CameraZone,
    reference_background: np.ndarray | None,
    daylight_hint: bool | None,
) -> str:
    """Hash every input that can change the standard feature-extraction path."""
    identity = {
        "video_sha256": _hash_file(video_path),
        "motion": motion_fingerprint,
        "zone": asdict(zone),
        "reference_background": _array_identity(reference_background),
        "daylight_hint": daylight_hint,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:32]


def get_cached_features(
    conn: Any,
    *,
    channel_id: str,
    message_id: int,
    fingerprint: str,
) -> tuple[bool, dict[str, float] | None]:
    """Return ``(hit, features)``; a cached no-motion result is a real hit."""
    payload = db.get_cached_track(
        conn,
        channel_id=channel_id,
        message_id=message_id,
        extractor_version=EXTRACTOR_VERSION,
        motion_fingerprint=fingerprint,
    )
    if payload is None:
        return False, None
    if payload == {_NO_MOTION_KEY: True}:
        return True, None
    return True, {str(key): float(value) for key, value in payload.items()}


def put_cached_features(
    conn: Any,
    *,
    channel_id: str,
    message_id: int,
    fingerprint: str,
    features: Mapping[str, float] | None,
) -> None:
    payload: dict[str, Any] = (
        {_NO_MOTION_KEY: True} if features is None else dict(features)
    )
    db.put_cached_track(
        conn,
        channel_id=channel_id,
        message_id=message_id,
        extractor_version=EXTRACTOR_VERSION,
        motion_fingerprint=fingerprint,
        features=payload,
    )
