"""Zone-aware feature scoring for motion-detector output.

``src.motion`` owns video decoding, detection and tracking. This module reduces
its in-memory ``ClipDetection`` into classifier-ready scalar features, including
compact geometry and metric replay paths. It has no labelled-corpus or CSV
responsibilities; those remain in ``scripts.spike``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.config import CameraZone, MotionThresholds, load_thresholds_config
from src.features import (
    FLASHLIGHT_CANDIDATE_MIN_RATIO,
    FLASHLIGHT_SUBJECT_THRESHOLD,
    apply_photometric_match,
    area_stability,
    aspect_ratio,
    blob_black_white_balance,
    blob_white_fraction,
    color_saturation_fraction,
    depth_progression,
    detect_stationary_light_mask,
    edge_density,
    flashlight_bbox_overlap,
    global_camera_shift_score,
    green_light_flicker,
    green_light_ratio,
    heading_change,
    ignore_region_mask,
    jitter,
    longest_detection_run,
    normalised_speed,
    optical_flow_direction_coherence,
    path_length,
    persistence,
    photometric_match,
    post_flash_red_shift,
    row_normalised_area,
    sane_fps,
    saturation_ratio,
    solidity,
)
from src.ground_calibration import (
    MAX_SUBJECT_HEIGHT_M,
    MAX_SUBJECT_SPEED_MPS,
    MAX_SUBJECT_WIDTH_M,
    calibrate,
)
from src.motion import (
    ClipDetection,
    GeometryObservations,
    MetricFrameObservation,
    MetricObservations,
    TrackedObject,
    _bbox_iou,
    _bbox_to_rect_contour,
    _contour_bbox,
    detect_clip,
)
from src.zones import (
    _fence_x_at_y,
    classify_zone,
    effective_fence,
    median_fence_distance,
    outside_pixel_fraction,
    track_crosses_fence,
    zone_classifiable_fraction,
)


@lru_cache(maxsize=1)
def _default_motion_thresholds() -> MotionThresholds:
    path = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"
    return load_thresholds_config(path).motion_thresholds()


_DEFAULT_MOTION_THRESHOLDS = _default_motion_thresholds()

def normalized_contour_points(contour: np.ndarray, frame_width: int, frame_height: int) -> list:
    return [(float(x) / frame_width, float(y) / frame_height) for [[x, y]] in contour]


def rasterized_zone_fractions(
    points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    frame_width: int,
    frame_height: int,
    zone: CameraZone,
) -> tuple[float, float]:
    """Return (outside, classifiable) fractions over the filled silhouette.

    The legacy ``outside_pixel_fraction`` samples contour vertices, so merely
    changing OpenCV's contour encoding can change its value. This additive
    measurement rasterizes the polygon and weights every occupied pixel
    equally. It remains reporting-only until corpus measurements justify
    replacing a tuned legacy classifier input.
    """
    fence = effective_fence(zone)
    if not points or fence is None or zone.outside is None:
        return 0.0, 0.0
    contour = np.array(
        [
            [
                [
                    int(round(x * frame_width)),
                    int(round(y * frame_height)),
                ]
            ]
            for x, y in points
        ],
        dtype=np.int32,
    )
    contour[:, 0, 0] = np.clip(contour[:, 0, 0], 0, frame_width - 1)
    contour[:, 0, 1] = np.clip(contour[:, 0, 1], 0, frame_height - 1)
    silhouette = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.drawContours(silhouette, [contour], -1, 1, thickness=-1)
    ys, xs = np.nonzero(silhouette)
    total = len(xs)
    if total == 0:
        return 0.0, 0.0

    classifiable = ys.astype(float) / frame_height >= zone.depth_cutoff
    if zone.ignore:
        classifiable &= ~ignore_region_mask(frame_width, frame_height, zone.ignore)[ys, xs]
    if not np.any(classifiable):
        return 0.0, 0.0

    xs_valid = xs[classifiable]
    ys_valid = ys[classifiable]
    fence_x = np.array(
        [_fence_x_at_y(float(y) / frame_height, fence) * frame_width for y in ys_valid]
    )
    outside = xs_valid > fence_x if zone.outside == "right" else xs_valid < fence_x
    return float(np.mean(outside)), float(np.count_nonzero(classifiable) / total)


def _whole_frame_contour(frame_width: int, frame_height: int) -> np.ndarray:
    w, h = frame_width - 1, frame_height - 1
    return np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _frame_is_merged(detection: ClipDetection, frame_index: int) -> bool:
    """True if the multi-object tracker sees two or more subjects sharing one
    blob in this frame (`TrackedObject.merged_ids` non-empty) -- that frame's
    largest contour describes a merged group silhouette, not one subject."""
    if frame_index >= len(detection.multi_tracks) or frame_index >= len(detection.frames):
        return False
    contour = detection.frames[frame_index].largest
    if contour is None:
        return False
    contour_bbox = _contour_bbox(contour)
    overlapping = [
        track
        for track in detection.multi_tracks[frame_index]
        if _bbox_iou(contour_bbox, track.bbox) > 0.0
    ]
    return any(track.merged_ids for track in overlapping)


@dataclass(frozen=True)
class WarmupMotionObject:
    """One visible contour from the consecutive-frame warmup scorer."""

    frame_index: int
    bbox: tuple[int, int, int, int]
    area: float
    disposition: str
    verdict: str | None = None


@dataclass(frozen=True)
class WarmupMotionAnalysis:
    """Shared warmup evidence for classification and the debug renderer."""

    features: dict[str, float]
    corrected_frames: tuple[np.ndarray, ...]
    objects: tuple[WarmupMotionObject, ...]


def warmup_motion_analysis(
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
    zeros = {
        "warmup_outside_fraction": 0.0,
        "warmup_dynamic_frame_fraction": 0.0,
        "warmup_dynamic_outside_fraction": 0.0,
        "warmup_dynamic_classifiable_fraction": 0.0,
        "warmup_dynamic_inside_bottom_left_fraction": 0.0,
    }
    dropped = detection.dropped_frames
    if len(dropped) < 2:
        return WarmupMotionAnalysis(zeros, (), ())

    background = detection.background
    reference = background
    corrected_reversed = []
    for raw in reversed(dropped):
        grey_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        gain, offset = photometric_match(grey_raw, reference)
        corrected = apply_photometric_match(grey_raw, gain, offset)
        corrected_reversed.append(corrected)
        reference = corrected
    corrected_frames = list(reversed(corrected_reversed))

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

    dynamic_verdicts: list[str] = []
    dynamic_inside_bottom_left_hits = 0
    close_kernel = np.ones((9, 9), np.uint8)
    ignore_mask = (
        ignore_region_mask(frame_width, frame_height, zone.ignore) if zone.ignore else None
    )
    dynamic_hits = 0
    observations: list[WarmupMotionObject] = []
    for frame_index, (previous, current) in enumerate(
        zip(corrected_frames, corrected_frames[1:], strict=False), start=1
    ):
        raw_dynamic_mask = cv2.morphologyEx(
            (cv2.absdiff(previous, current) > threshold).astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            kernel,
        )
        raw_dynamic_mask = cv2.morphologyEx(
            raw_dynamic_mask, cv2.MORPH_CLOSE, close_kernel
        )
        dynamic_mask = raw_dynamic_mask.copy()
        if ignore_mask is not None:
            ignored_mask = np.zeros_like(raw_dynamic_mask)
            ignored_mask[ignore_mask] = raw_dynamic_mask[ignore_mask]
            ignored_contours, _ = cv2.findContours(
                ignored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in ignored_contours:
                x, y, width, height = cv2.boundingRect(contour)
                observations.append(
                    WarmupMotionObject(
                        frame_index=frame_index,
                        bbox=(x, y, x + width, y + height),
                        area=float(cv2.contourArea(contour)),
                        disposition="ignored_region",
                    )
                )
            dynamic_mask[ignore_mask] = 0
        contours, _ = cv2.findContours(
            dynamic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if 0.0005 * frame_area <= area <= 0.25 * frame_area:
                candidates.append(contour)
                continue
            x, y, width, height = cv2.boundingRect(contour)
            observations.append(
                WarmupMotionObject(
                    frame_index=frame_index,
                    bbox=(x, y, x + width, y + height),
                    area=area,
                    disposition="ignored_size",
                )
            )
        if not candidates:
            continue
        dynamic_hits += 1
        selected = max(candidates, key=cv2.contourArea)
        for contour in candidates:
            x, y, width, height = cv2.boundingRect(contour)
            base_centre = (
                (x + width / 2.0) / frame_width,
                (y + height) / frame_height,
            )
            observations.append(
                WarmupMotionObject(
                    frame_index=frame_index,
                    bbox=(x, y, x + width, y + height),
                    area=float(cv2.contourArea(contour)),
                    disposition="used" if contour is selected else "not_selected",
                    verdict=classify_zone(base_centre, zone),
                )
            )
        x, y, width, height = cv2.boundingRect(selected)
        base_centre = (
            (x + width / 2.0) / frame_width,
            (y + height) / frame_height,
        )
        verdict = classify_zone(base_centre, zone)
        dynamic_verdicts.append(verdict)
        # Confirmed guard-exit geometry: the person is still on the protected
        # side, low and left in frame, while walking out before IR warmup ends.
        # Count against every dynamic hit (not only classifiable ones), so an
        # uncertain zone verdict cannot inflate this feature.
        if verdict == "inside" and base_centre[0] <= 0.45 and base_centre[1] >= 0.60:
            dynamic_inside_bottom_left_hits += 1

    pair_count = len(corrected_frames) - 1
    dynamic_classifiable = [
        verdict for verdict in dynamic_verdicts if verdict in ("outside", "inside")
    ]
    dynamic_features = {
        "warmup_dynamic_frame_fraction": (
            dynamic_hits / pair_count if pair_count > 0 else 0.0
        ),
        "warmup_dynamic_outside_fraction": (
            sum(1 for verdict in dynamic_classifiable if verdict == "outside")
            / len(dynamic_classifiable)
            if dynamic_classifiable
            else 0.0
        ),
        "warmup_dynamic_classifiable_fraction": (
            len(dynamic_classifiable) / dynamic_hits if dynamic_hits > 0 else 0.0
        ),
        "warmup_dynamic_inside_bottom_left_fraction": (
            dynamic_inside_bottom_left_hits / dynamic_hits if dynamic_hits > 0 else 0.0
        ),
    }

    if not track:
        return WarmupMotionAnalysis(
            {**zeros, **dynamic_features}, tuple(corrected_frames), tuple(observations)
        )

    # Same base-of-box convention and same geometry every scored side feature
    # uses, so a warmup verdict is directly comparable to outside_frame_fraction.
    verdicts = [
        classify_zone(((bx + bw / 2.0) / frame_width, (by + bh) / frame_height), zone)
        for bx, by, bw, bh in track
    ]
    classifiable = [v for v in verdicts if v in ("outside", "inside")]
    if not classifiable:
        return WarmupMotionAnalysis(
            {**zeros, **dynamic_features}, tuple(corrected_frames), tuple(observations)
        )

    return WarmupMotionAnalysis(
        {
            "warmup_outside_fraction": sum(1 for v in classifiable if v == "outside")
            / len(classifiable),
            **dynamic_features,
        },
        tuple(corrected_frames),
        tuple(observations),
    )


def _warmup_motion_features(
    detection: ClipDetection,
    zone: CameraZone,
    frame_width: int,
    frame_height: int,
    *,
    threshold: int = 18,
) -> dict[str, float]:
    """Compatibility wrapper returning the shared warmup analysis features."""
    return warmup_motion_analysis(
        detection, zone, frame_width, frame_height, threshold=threshold
    ).features


def _multi_object_outside_features(
    detection: ClipDetection,
    zone: CameraZone,
    frame_width: int,
    frame_height: int,
    *,
    fallback_outside_fraction: float | None = None,
) -> dict[str, float]:
    """Per-object, whole-clip fence-side reading over EVERY persistently-
    tracked object in the scored frames (`detection.multi_tracks`), not just
    the single largest/tracked contour every other outside-fraction feature
    reads. `multi_tracks` already runs unconditionally inside `detect_clip`
    for diagnostic overlays (see `track_multiple_objects`); this is the first
    feature to actually read it rather than discard it after rendering.

    Motivation, from a real recurring cam07 failure (found 2026-09-09,
    reviewing the fence_bottom-era `outside_pixel_fraction` thresholds with
    the user): `_warmup_motion_features`'s own docstring already documents a
    static bright artifact (a branch) that the SAME best-contour pick locks
    onto in both the warmup and scored frames of cam07/18570. cam07/22393 is
    the same pattern with a spider web: the guard exits frame during warmup,
    never appears in a scored frame at all, and the geometry rule reads a
    coincidentally-outside web as "the subject" because nothing else was
    there to outrank it. A single best-contour reading has no way to tell
    "the one thing found happens to be outside" apart from "the real subject
    is outside" -- per-object identity does, because the web and a guard
    (were they both genuinely present) would be two separate tracks with two
    separate verdicts instead of one blended number.

    Two clip-level readings, deliberately kept separate rather than folded
    into one:
      - `multi_object_outside_fraction_weighted`: every object that moved,
        area-weighted by each object's own total tracked bbox area -- "how
        much of everything that moved was outside."
      - `multi_object_dominant_outside_fraction`: just the single object with
        the largest total tracked area -- a track-wide, multi-candidate-aware
        alternative to `outside_pixel_fraction`'s single-BEST-FRAME reading.
    Plus `multi_object_count`: distinct persistent objects seen, a genuinely
    different signal from `blob_count` (a per-frame count with no identity
    across frames).

    Per-object verdict uses the same base-of-box convention
    `_warmup_motion_features` uses (`TrackedObject` only carries a bbox, not a
    full contour, so this is coarser than `outside_pixel_fraction`'s
    per-point sampling -- consistent with that existing feature's own
    precision, not a new approximation).

    Known simplification, not yet resolved: while two tracks are merged
    (`TrackedObject.merged_ids` non-empty), both track ids currently accrue
    the FULL shared bbox's area independently, rather than splitting it --
    inflates a merged object's apparent size slightly. Left as-is for this
    first cut; revisit if `dominant_id` selection near a merge ever looks
    wrong on real clips.

    `fallback_outside_fraction`: `multi_tracks` only sees genuinely
    background-diffed blobs (`per_frame_blobs`), never a RECOVERED/appearance-
    matched detection the single-tracked-target pipeline also draws on -- so a
    clip whose real subject is mostly recovered (a fast, small, intermittently
    -detected animal, say) can have ZERO real per-object evidence here even
    though `outside_pixel_fraction` correctly read it. Measured 2026-09-09:
    swapping this feature in for `outside_pixel_fraction` with no fallback
    cost exactly the 5 labelled animal clips with `multi_object_count == 0`
    their alert status, all for this reason. When given, both readings default
    to `fallback_outside_fraction` (typically the caller's own
    `outside_pixel_fraction` for the same clip) rather than a bare 0.0 when no
    real per-object evidence exists at all -- 0.0 is a confident "inside"
    claim this function has no basis for making with zero evidence.

    NOT wired into classify() -- reporting/measurement only until proven
    against the labelled corpus, same discipline as `outside_frame_fraction`
    and `warmup_outside_fraction` before it. As of 2026-09-09 a straight
    swap-in for `outside_pixel_fraction` (dominant reading, with this
    fallback) measured guard=55/429 environment=7/159 animal=15/37
    incident=10/10 against the shipped guard=23/429 environment=5/159
    animal=15/37 incident=7/10 -- incident recall genuinely improves (the 3
    gained clips are exactly the known cam06/21519, cam09/21521, cam10/21523
    "blank precursor" clips this repo already tracks by ID), but guard leak
    more than doubles. A frame-count/area "minimum evidence" floor was tried
    as a fix for the guard leak and REJECTED after checking real per-object
    data: the thinnest-evidence dominant object in the whole checked sample
    was the genuinely-valuable cam06/21519 gain (seen in 1 of 4 frames), while
    the worst guard leaks had the MOST substantial evidence (cam08/4293: 12 of
    12 frames, largest area in the sample) -- a floor would filter out the
    exact clip it should keep and leave the exact clips it should suppress.
    Not usable as a classify() gate until the guard leak's actual cause is
    understood (needs visual inspection of what the "dominant" object in a
    leaking guard clip actually is -- a flashlight beam? a second real
    entity? multi-object tracker fragmentation? -- not a hand-tuned number).
    """
    return _multi_object_outside_features_from_tracks(
        detection.multi_tracks,
        zone,
        frame_width,
        frame_height,
        fallback_outside_fraction=fallback_outside_fraction,
    )


def _multi_object_outside_features_from_tracks(
    multi_tracks: tuple[tuple[TrackedObject, ...], ...] | list[list[TrackedObject]],
    zone: CameraZone,
    frame_width: int,
    frame_height: int,
    *,
    fallback_outside_fraction: float | None = None,
) -> dict[str, float]:
    """Fence-side features from compact persistent-track observations."""
    zeros = {
        "multi_object_outside_fraction_weighted": (
            0.0 if fallback_outside_fraction is None else fallback_outside_fraction
        ),
        "multi_object_dominant_outside_fraction": (
            0.0 if fallback_outside_fraction is None else fallback_outside_fraction
        ),
        "multi_object_count": 0.0,
    }
    if not multi_tracks:
        return zeros

    verdicts: dict[int, list[bool]] = {}
    areas: dict[int, float] = {}
    for frame_tracks in multi_tracks:
        for obj in frame_tracks:
            # obj.bbox is corner form (x0, y0, x1, y1) -- see _contour_bbox and
            # track_multiple_objects' own dead-reckoning step, both of which
            # use this convention, and scripts/render_debug.py's renderer,
            # which unpacks it the same way. Do NOT read this as (x, y, w, h):
            # that silently sends the sample point below the frame for any
            # blob in the lower half (classify_zone has no bounds check and
            # side_name extrapolates), and turns `areas` into a function of
            # position rather than size, corrupting `dominant_id`'s "biggest
            # object" selection. Found and fixed 2026-09-09 -- see
            # docs/detection_improvement_review.md section 1.1 for the
            # measured corpus impact (105/141 sampled clips' weighted
            # fraction changed, 49/141 dominant-object flips >0.5).
            x0, y0, x1, y1 = obj.bbox
            point = ((x0 + x1) / 2.0 / frame_width, y1 / frame_height)
            verdict = classify_zone(point, zone)
            if verdict not in ("outside", "inside"):
                continue
            verdicts.setdefault(obj.track_id, []).append(verdict == "outside")
            areas[obj.track_id] = areas.get(obj.track_id, 0.0) + float((x1 - x0) * (y1 - y0))

    if not verdicts:
        return zeros

    object_fractions = {tid: sum(vs) / len(vs) for tid, vs in verdicts.items()}
    total_area = sum(areas[tid] for tid in object_fractions)
    weighted = (
        sum(object_fractions[tid] * areas[tid] for tid in object_fractions) / total_area
        if total_area > 0
        else 0.0
    )
    dominant_id = max(areas, key=lambda tid: areas[tid])

    return {
        "multi_object_outside_fraction_weighted": weighted,
        "multi_object_dominant_outside_fraction": object_fractions[dominant_id],
        "multi_object_count": float(len(object_fractions)),
    }


def geometry_observations_from_detection(
    detection: ClipDetection,
) -> GeometryObservations | None:
    """Reduce a detection to the compact evidence needed for fence replay.

    The point and track-selection rules intentionally mirror
    ``extract_clip_features``: its best contour skips merged multi-object
    frames when possible, while its per-frame side feature uses only genuine
    (not recovered or reverse-filled) contours.  Keeping this reduction free
    of a zone makes the resulting payload reusable for fence, side and depth
    edits.
    """
    best_contour: np.ndarray | None = None
    best_area = -1.0
    best_contour_any: np.ndarray | None = None
    best_area_any = -1.0
    centroids: list[tuple[float, float]] = []
    genuine_contours: list[tuple[tuple[float, float], ...]] = []

    for detected in detection.frames:
        contour = detected.largest
        if contour is None:
            continue
        area = cv2.contourArea(contour)
        if area > best_area_any:
            best_area_any = area
            best_contour_any = contour
        if not _frame_is_merged(detection, detected.index) and area > best_area:
            best_area = area
            best_contour = contour
        if detected.centroid is None:
            continue
        centroids.append(
            (
                detected.centroid[0] / detection.frame_width,
                detected.centroid[1] / detection.frame_height,
            )
        )
        if not detected.recovered and not detected.filled_by_reverse:
            genuine_contours.append(
                tuple(
                    normalized_contour_points(
                        contour, detection.frame_width, detection.frame_height
                    )
                )
            )

    selected = best_contour if best_contour is not None else best_contour_any
    if selected is None:
        return None
    return GeometryObservations(
        frame_width=detection.frame_width,
        frame_height=detection.frame_height,
        best_contour_points=tuple(
            normalized_contour_points(selected, detection.frame_width, detection.frame_height)
        ),
        genuine_contour_points=tuple(genuine_contours),
        centroid_track=tuple(centroids),
        multi_tracks=tuple(tuple(frame_tracks) for frame_tracks in detection.multi_tracks),
    )


def geometry_features_from_observations(
    observations: GeometryObservations, zone: CameraZone
) -> dict[str, float]:
    """Re-score all current fence/depth features without video or imagery.

    This is deliberately narrower than ``features_from_detection``.  It is
    the cacheable replay path for geometry only; colour, texture, warmup and
    metric-calibration features use their own compact replay path; colour,
    texture and warmup features still need the in-memory detection and are
    therefore not silently approximated here.
    """
    points = observations.best_contour_points
    classified = [
        contour
        for contour in observations.genuine_contour_points
        if zone_classifiable_fraction(contour, zone) > 0.0
    ]
    outside_frames = sum(
        outside_pixel_fraction(contour, zone) > 0.5 for contour in classified
    )
    outside_fraction = outside_pixel_fraction(points, zone)
    outside_area_fraction, classifiable_area_fraction = rasterized_zone_fractions(
        points, observations.frame_width, observations.frame_height, zone
    )
    return {
        "outside_pixel_fraction": outside_fraction,
        "zone_classifiable_fraction": zone_classifiable_fraction(points, zone),
        "outside_area_fraction": outside_area_fraction,
        "zone_classifiable_area_fraction": classifiable_area_fraction,
        "outside_frame_fraction": outside_frames / len(classified) if classified else 0.0,
        **_multi_object_outside_features_from_tracks(
            observations.multi_tracks,
            zone,
            observations.frame_width,
            observations.frame_height,
            fallback_outside_fraction=outside_fraction,
        ),
        "fence_crossed": float(track_crosses_fence(observations.centroid_track, zone)),
        "median_fence_distance": median_fence_distance(observations.centroid_track, zone),
    }


def multi_object_flashlight_scores(
    detection: ClipDetection,
    *,
    exclude_mask: np.ndarray | None = None,
) -> dict[int, float]:
    """Return each persistent track's peak rectangular flashlight ratio."""
    scores: dict[int, float] = {}
    for frame_index, frame_tracks in enumerate(detection.multi_tracks):
        if frame_index >= len(detection.frames):
            continue
        frame_bgr = detection.frames[frame_index].frame
        for obj in frame_tracks:
            ratio = green_light_ratio(
                frame_bgr,
                _bbox_to_rect_contour(obj.bbox),
                exclude_mask=exclude_mask,
            )
            scores[obj.track_id] = max(scores.get(obj.track_id, 0.0), ratio)
    return scores


def _multi_object_flashlight_features(
    detection: ClipDetection,
    zone: CameraZone,
    frame_width: int,
    frame_height: int,
    *,
    exclude_mask: np.ndarray | None = None,
    flashlight_min_ratio: float = FLASHLIGHT_CANDIDATE_MIN_RATIO,
) -> dict[str, float]:
    """Scores EVERY persistently-tracked object (`detection.multi_tracks`) for
    flashlight-ness independently, rather than only ever checking whichever
    single contour the best-contour pipeline happened to follow.

    Added 2026-09-09 (docs/detection_improvement_review.md section 3, stage
    2 of the object-linking design): `green_light_ratio`'s own signature
    already accepts an arbitrary contour, so this is a mechanically small
    extension of an existing primitive, not new colour-detection work -- the
    only new part is synthesizing a rectangular contour from each object's
    own bbox (`_bbox_to_rect_contour`, the same approach
    `flashlight_bbox_overlap` already uses for the single tracked box) and
    scoring it per frame the object appears in, taking the PEAK across those
    frames as that object's own flashlight score (same convention as every
    other single-clip flashlight peak feature -- `warmup_flashlight_ratio`,
    `whole_frame_green_ratio`).

    Direct fix for the cam07/11174 family of failure this session's review
    names: a real flashlight beam sitting in its OWN separate track, while a
    much larger unrelated object (a bush) is what the single best-contour
    pipeline follows and therefore all it scores. `green_light_ratio`
    (contour-restricted, single track) reads 0.0 on that clip even though the
    beam is plainly visible one object over. This feature can see it: it
    does not matter which object the single-track pipeline decided to follow.

    `multi_object_dominant_excl_flashlight_outside_fraction` answers a
    different, related question -- given that a flashlight-scoring track
    exists, what does the fence-side reading look like for the largest
    OTHER (non-flashlight) object -- the "ignore the beam, follow the
    person" reading `prefer_flashlight_candidate`'s active-track override
    cannot express (it can only make the SINGLE track follow one thing or
    the other; this reads both independently, which is what the user's own
    framing asked for: "flashlight should always be separately tracked...
    leaves room for tracking the main object"). `_has_evidence` disambiguates
    a real 0.0 (fully inside) from "no non-flashlight object exists at all"
    (e.g. a clip with only one tracked object, and it IS the flashlight) --
    same "don't let an absence read as a confident zero" discipline
    `_multi_object_outside_features`'s own `zone_classifiable_fraction`
    guard and this repo's `uncalibrated`/`has_reference_background` flags
    already use.

    This began reporting-only, then its full-corpus measurement showed no
    protected-class regression and it became the ``multi_object_flashlight``
    guard rule. The dominant non-flashlight outside fields remain reporting
    evidence and do not override that measured rule.
    """
    zeros = {
        "multi_object_flashlight_track_count": 0.0,
        "multi_object_max_flashlight_ratio": 0.0,
        "multi_object_dominant_excl_flashlight_outside_fraction": 0.0,
        "multi_object_dominant_excl_flashlight_has_evidence": 0.0,
    }
    if not detection.multi_tracks:
        return zeros

    flashlight_scores = multi_object_flashlight_scores(
        detection, exclude_mask=exclude_mask
    )
    verdicts: dict[int, list[bool]] = {}
    areas: dict[int, float] = {}
    for frame_tracks in detection.multi_tracks:
        for obj in frame_tracks:
            # obj.bbox is corner form (x0, y0, x1, y1) -- see the note in
            # _multi_object_outside_features above.
            x0, y0, x1, y1 = obj.bbox
            point = ((x0 + x1) / 2.0 / frame_width, y1 / frame_height)
            verdict = classify_zone(point, zone)
            if verdict in ("outside", "inside"):
                verdicts.setdefault(obj.track_id, []).append(verdict == "outside")
            areas[obj.track_id] = areas.get(obj.track_id, 0.0) + float((x1 - x0) * (y1 - y0))

    flashlight_track_ids = {
        tid for tid, score in flashlight_scores.items() if score > flashlight_min_ratio
    }
    non_flashlight_areas = {
        tid: area
        for tid, area in areas.items()
        if tid not in flashlight_track_ids and tid in verdicts
    }
    dominant_excl_fraction = 0.0
    has_evidence = 0.0
    if non_flashlight_areas:
        dominant_id = max(non_flashlight_areas, key=lambda tid: non_flashlight_areas[tid])
        votes = verdicts[dominant_id]
        dominant_excl_fraction = sum(votes) / len(votes)
        has_evidence = 1.0

    return {
        "multi_object_flashlight_track_count": float(len(flashlight_track_ids)),
        "multi_object_max_flashlight_ratio": (
            max(flashlight_scores.values()) if flashlight_scores else 0.0
        ),
        "multi_object_dominant_excl_flashlight_outside_fraction": dominant_excl_fraction,
        "multi_object_dominant_excl_flashlight_has_evidence": has_evidence,
    }


PERSON_HEIGHT_MIN_M = 0.9
PERSON_HEIGHT_MAX_M = 2.2
ARTIFACT_WHITE_FRACTION_MIN = 0.4  # same bar classify()'s blob_white_fraction_min uses
# A track must be seen this many frames before it counts toward any type below --
# see this function's own docstring for why (measured, not assumed).
MIN_TRACK_FRAMES_FOR_TYPE = 3


def _multi_object_type_features(
    detection: ClipDetection,
    zone: CameraZone,
    frame_width: int,
    frame_height: int,
) -> dict[str, float]:
    """Scores every persistently-tracked object (`detection.multi_tracks`) for
    a coarse per-track TYPE -- person, animal, or artifact -- rather than the
    one whole-clip scalar (`subject_height_m`, `blob_white_fraction`, ...)
    every other feature in this module reports for whichever single contour
    `track_contour`'s best-contour pipeline happened to follow.

    Added 2026-09-09 (docs/detection_improvement_review.md section 3, stage
    2 of the object-linking design). Person/animal reuse `GroundCalibration.
    height_m` exactly the way `metric_features_from_observations` does for the
    single tracked subject -- the review's own note that this is "a
    mechanically small extension of an existing primitive, not new detection
    work" applies here just as it did to the per-object flashlight score
    (`_multi_object_flashlight_features`) shipped earlier the same day.
    Artifact reuses `blob_white_fraction` (a lens obstruction/overexposure
    reading) the same way.

    `PERSON_HEIGHT_MIN_M`/`_MAX_M` (0.9-2.2 m) mirror `classify()`'s
    `neighbour_subject_height_min`/`_max` -- duplicated here deliberately
    rather than imported, the same separation every other feature in this
    module keeps: spike.py computes raw signals, classify.py (via
    `config/thresholds.yaml`) owns the actual decision thresholds. A track's
    OWN median height across its genuine frames decides its type; a track
    with no calibrated reading at all (off the ground plane every frame)
    counts toward neither.

    Vegetation is NOT scored here despite being in the review's own table --
    the cheapest per-track evidence for it (matching the per-camera
    reference background at the same coordinates) needs the actual aligned
    reference image, not just the whole-clip `scenery_motion_fraction`
    scalar `ClipDetection` already carries; threading that image out of
    `detect_clip` is a bigger, more invasive change than the two signals
    below, and the review lists three DIFFERENT candidate vegetation
    signals (background match, returns-to-place, incoherent optical flow) --
    picking one deserves its own measurement pass, not a guess bundled into
    this commit.

    `multi_object_type_has_evidence` disambiguates "this camera has no
    metric calibration at all, so person/animal both read 0" from a real
    "no track was person- or animal-height" -- same discipline as every
    other `_has_evidence` flag in this module. It only qualifies the
    person/animal counts: `multi_object_artifact_track_count` has no such
    ambiguity, since `blob_white_fraction` is computable from any frame
    regardless of calibration.

    `MIN_TRACK_FRAMES_FOR_TYPE` (3): a track must be seen at least this many
    frames before it counts toward any type. Measured 2026-09-09, WITHOUT
    this filter, against the full labelled corpus (after switching
    `track_multiple_objects`'s real call site to `per_frame_candidates` with
    `confirm_frames=1` -- see `detect_clip`'s own docstring): every raw
    candidate mints its own track id with no continuity requirement at all,
    so a single frame of wind-shaken foliage or an insect counts exactly the
    same as a real, sustained subject. The result was not a mild false-
    positive rate but a near-universal one -- `multi_object_animal_track_
    count > 0` fired on 138/158 environment clips and 342/425 guard clips
    (vs 26/32 real animal clips), and its median COUNT was actually HIGHER
    for environment (53.5) than for animal (2.5) or incident (13.0) clips,
    because a single windy/insect-heavy clip mints dozens of one-frame
    tracks. Requiring 3 consecutive frames of evidence is the same
    discipline `track_multiple_objects`'s own (off-by-default)
    `confirm_frames` parameter uses, applied locally to this consumer
    instead of globally to the tracker -- global confirm_frames was tried
    and rejected (see `detect_clip`'s docstring) because it suppresses
    short-lived flashlight objects the `guard_candidate` rule depends on;
    a real animal or person track, unlike a torch beam, is expected to
    persist for several frames, so the same fix does not cost anything here.

    3 frames helps (environment's `animal_track_count > 0` rate dropped from
    138/158 to 132/158, median count 53.5 -> 26) but does NOT make
    `multi_object_person_track_count`/`multi_object_animal_track_count`
    separable, and a follow-up check found raising the bar further makes it
    WORSE, not better: at `MIN_TRACK_FRAMES_FOR_TYPE=12`, restricted to
    cam10 (6 animal / 83 environment labelled clips, its best-populated
    camera for both), real animal clips dropped to 0/6 with any qualifying
    track at all, while environment clips still hit 39/83 (mean count 7.86,
    max 68). A real animal's own track is apparently LESS likely to sustain
    12 consecutive frames than wind-shaken vegetation is -- an animal
    crosses the frame or is occluded, while a swaying branch oscillates
    around one fixed area indefinitely. **Do not retry raising this
    threshold as a fix for animal/person separability -- measured backwards
    twice now.** `multi_object_artifact_track_count` looks more promising in
    the same corpus pass (environment 103/158=65% vs guard 75/425=18% vs
    animal 2/32=6%) but is likely highly correlated with the pre-existing
    single-track `blob_white_fraction` (already `classify()`-wired via the
    blinding-foreground gate) -- that overlap needs checking before treating
    it as new information, not assumed.

    Reporting-only: not read by classify() without its own corpus-wide
    measurement pass first, same discipline as every other detect_clip-level
    feature in this file's history.
    """
    zeros = {
        "multi_object_person_track_count": 0.0,
        "multi_object_animal_track_count": 0.0,
        "multi_object_artifact_track_count": 0.0,
        "multi_object_type_has_evidence": 0.0,
    }
    if not detection.multi_tracks:
        return zeros

    cal = calibrate(zone, frame_width, frame_height)
    heights: dict[int, list[float]] = {}
    white_fractions: dict[int, float] = {}
    frame_counts: dict[int, int] = {}
    for frame_index, frame_tracks in enumerate(detection.multi_tracks):
        frame_bgr = (
            detection.frames[frame_index].frame if frame_index < len(detection.frames) else None
        )
        for obj in frame_tracks:
            frame_counts[obj.track_id] = frame_counts.get(obj.track_id, 0) + 1
            x0, y0, x1, y1 = obj.bbox
            if cal is not None:
                base = ((x0 + x1) / 2.0, float(y1))
                height = cal.height_m(base, float(y0), enforce_limits=False)
                if height is not None and 0 < height <= MAX_SUBJECT_HEIGHT_M:
                    heights.setdefault(obj.track_id, []).append(height)
            if frame_bgr is not None:
                fraction = blob_white_fraction(frame_bgr, _bbox_to_rect_contour(obj.bbox))
                white_fractions[obj.track_id] = max(
                    white_fractions.get(obj.track_id, 0.0), fraction
                )

    person_count = 0
    animal_count = 0
    for tid, track_heights in heights.items():
        if frame_counts.get(tid, 0) < MIN_TRACK_FRAMES_FOR_TYPE:
            continue
        median_height = float(np.median(track_heights))
        if PERSON_HEIGHT_MIN_M <= median_height <= PERSON_HEIGHT_MAX_M:
            person_count += 1
        elif median_height < PERSON_HEIGHT_MIN_M:
            animal_count += 1

    artifact_count = sum(
        1
        for tid, fraction in white_fractions.items()
        if fraction >= ARTIFACT_WHITE_FRACTION_MIN
        and frame_counts.get(tid, 0) >= MIN_TRACK_FRAMES_FOR_TYPE
    )

    return {
        "multi_object_person_track_count": float(person_count),
        "multi_object_animal_track_count": float(animal_count),
        "multi_object_artifact_track_count": float(artifact_count),
        "multi_object_type_has_evidence": float(cal is not None),
    }


def metric_observations_from_detection(
    detection: ClipDetection, *, fps: float
) -> MetricObservations:
    """Reduce genuine single-track boxes to zone-independent metric evidence."""
    frames = []
    for detected in detection.frames:
        if (
            detected.largest is None
            or detected.recovered
            or detected.filled_by_reverse
        ):
            continue
        x, y, width, height = cv2.boundingRect(detected.largest)
        frames.append(
            MetricFrameObservation(
                frame_index=detected.index,
                bbox=(x, y, x + width, y + height),
            )
        )
    return MetricObservations(
        frame_width=detection.frame_width,
        frame_height=detection.frame_height,
        fps=fps,
        frames=tuple(frames),
    )


def metric_features_from_observations(
    observations: MetricObservations, zone: CameraZone
) -> dict[str, float]:
    """Re-score ground-plane metric features without video or contour imagery.

    The calculation is identical to the former in-extractor pass. Only genuine
    background-difference boxes are present, and original frame indices retain
    the real time gaps used by the speed calculation.
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
    calibration = calibrate(
        zone, observations.frame_width, observations.frame_height
    )
    if calibration is None:
        return {**zeros, "uncalibrated": 1.0}

    genuine = len(observations.frames)
    implausible = 0
    off_plane = 0
    heights: list[float] = []
    distances: list[float] = []
    widths: list[float] = []
    aspects: list[float] = []
    areas: list[float] = []
    ground_tracks: list[tuple[int, np.ndarray]] = []
    for observed in observations.frames:
        x0, y0, x1, y1 = observed.bbox
        base = ((x0 + x1) / 2.0, float(y1))
        ground = calibration.ground_point(base)
        if ground is None:
            off_plane += 1
            continue
        ground_tracks.append((observed.frame_index, ground))
        height = calibration.height_m(base, float(y0), enforce_limits=False)
        if height is None or height <= 0 or height > MAX_SUBJECT_HEIGHT_M:
            implausible += 1
            continue
        left = calibration.ground_point((float(x0), float(y1)))
        right = calibration.ground_point((float(x1), float(y1)))
        width = None if left is None or right is None else float(np.linalg.norm(right - left))
        if width is None or width <= 0 or width > MAX_SUBJECT_WIDTH_M:
            implausible += 1
            continue
        heights.append(height)
        widths.append(width)
        aspects.append(height / width)
        areas.append(height * width)
        distance = calibration.distance_m(base)
        if distance is not None:
            distances.append(distance)

    if genuine == 0:
        return {**zeros, "uncalibrated": 0.0}

    speeds: list[float] = []
    if observations.fps > 0:
        for (index_a, ground_a), (index_b, ground_b) in zip(
            ground_tracks, ground_tracks[1:], strict=False
        ):
            elapsed = (index_b - index_a) / observations.fps
            if elapsed <= 0:
                continue
            speed = float(np.linalg.norm(ground_b - ground_a)) / elapsed
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


@dataclass(frozen=True)
class _DetectionFeatureSummary:
    """Zone-filtered, per-frame evidence consumed by feature assembly."""

    best_contour: np.ndarray
    best_frame: np.ndarray
    genuine_centroids: tuple[tuple[float, float], ...]
    genuine_blob_areas: tuple[float, ...]
    genuine_detected_indices: tuple[int, ...]
    genuine_frames_detected: int
    non_genuine_frames: int
    reverse_filled_frames: int
    flow_coherence_values: tuple[float, ...]
    whole_frame_green_ratios: tuple[float, ...]
    color_fraction: float
    motion_pixel_fraction: float
    motion_pixel_fraction_median: float
    blob_count: int
    blob_count_median: float
    white_fraction: float
    black_white_balance: float
    rectangular_black_white_balance: float
    frames_with_box: int
    flashlight_bbox_frames: int


def feature_exclude_mask(
    detection: ClipDetection, zone: CameraZone
) -> np.ndarray | None:
    """Combine configured ignore regions with automatically stable lights.

    This is public so diagnostic renderers can use the exact exclusion mask
    used by feature extraction instead of maintaining a visual approximation.
    """
    frame_width, frame_height = detection.frame_width, detection.frame_height
    exclude_mask = (
        ignore_region_mask(frame_width, frame_height, zone.ignore) if zone.ignore else None
    )
    # Carve every tracked box out of automatic light detection so a guard who
    # dwells with a steady torch is not classified as a fixed camera artifact.
    tracked_region = np.zeros((frame_height, frame_width), dtype=bool)
    for detected in detection.frames:
        if detected.largest is None:
            continue
        x, y, width, height = cv2.boundingRect(detected.largest)
        padding = 4
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1 = min(frame_width, x + width + padding)
        y1 = min(frame_height, y + height + padding)
        tracked_region[y0:y1, x0:x1] = True
    automatic = detect_stationary_light_mask(
        [detected.frame for detected in detection.frames],
        exclude_region=tracked_region,
    )
    if np.any(automatic):
        return automatic if exclude_mask is None else (exclude_mask | automatic)
    return exclude_mask


def warmup_flashlight_diagnostics(
    detection: ClipDetection,
    *,
    exclude_mask: np.ndarray | None,
    daylight_color_fraction: float,
    daylight_hint: bool | None,
) -> tuple[bool, tuple[float, ...]]:
    """Return the warmup daylight gate and each raw frame's light ratio.

    Keeping this calculation shared with the debug renderer ensures that its
    warmup outlines, gate explanation and displayed ratios describe the exact
    frames and thresholds used by ``warmup_flashlight_ratio``.
    """
    if not detection.dropped_frames:
        return False, ()
    warmup_colour = [
        color_saturation_fraction(frame) for frame in detection.dropped_frames
    ]
    daylight_gated = (
        sum(warmup_colour) / len(warmup_colour) > daylight_color_fraction
        and daylight_hint is not False
    )
    whole_frame = _whole_frame_contour(
        detection.frame_width, detection.frame_height
    )
    ratios = tuple(
        green_light_ratio(frame, whole_frame, exclude_mask=exclude_mask)
        for frame in detection.dropped_frames
    )
    return daylight_gated, ratios


def _warmup_flashlight_feature(
    detection: ClipDetection,
    *,
    exclude_mask: np.ndarray | None,
    daylight_color_fraction: float,
    daylight_hint: bool | None,
) -> float:
    """Measure flashlight colour without requiring a settled-frame track."""
    daylight_gated, ratios = warmup_flashlight_diagnostics(
        detection,
        exclude_mask=exclude_mask,
        daylight_color_fraction=daylight_color_fraction,
        daylight_hint=daylight_hint,
    )
    if daylight_gated or not ratios:
        return 0.0
    return max(ratios)


def _summarize_detection_features(
    detection: ClipDetection,
    *,
    exclude_mask: np.ndarray | None,
) -> _DetectionFeatureSummary | None:
    """Reduce frame imagery and contours to the scalars used during scoring."""
    genuine_centroids: list[tuple[float, float]] = []
    genuine_blob_areas: list[float] = []
    genuine_detected_indices: list[int] = []
    genuine_frames_detected = 0
    non_genuine_frames = 0
    reverse_filled_frames = 0
    flow_coherence_values: list[float] = []
    previous_genuine_index: int | None = None
    previous_genuine_gray: np.ndarray | None = None
    previous_genuine_contour: np.ndarray | None = None
    whole_frame_green_ratios: list[float] = []
    color_fractions: list[float] = []
    best_contour: np.ndarray | None = None
    best_frame: np.ndarray | None = None
    best_area = -1.0
    best_contour_any: np.ndarray | None = None
    best_frame_any: np.ndarray | None = None
    best_area_any = -1.0
    whole_frame = _whole_frame_contour(detection.frame_width, detection.frame_height)
    motion_pixel_fraction = 0.0
    blob_count = 0
    white_fraction = 0.0
    black_white_balance = 0.0
    rectangular_black_white_balance = 0.0
    frames_with_box = 0
    flashlight_bbox_frames = 0

    for detected in detection.frames:
        for candidate in detected.all_contours:
            area = cv2.contourArea(candidate)
            if area <= 0:
                continue
            _x, _y, width, height = cv2.boundingRect(candidate)
            box_area = width * height
            if box_area <= 0 or area / box_area < 0.5:
                continue
            rectangular_black_white_balance = max(
                rectangular_black_white_balance,
                blob_black_white_balance(detected.frame, candidate),
            )
        whole_frame_green_ratios.append(
            green_light_ratio(detected.frame, whole_frame, exclude_mask=exclude_mask)
        )
        color_fractions.append(color_saturation_fraction(detected.frame))
        motion_pixel_fraction = max(
            motion_pixel_fraction, detected.motion_pixel_fraction
        )
        blob_count = max(blob_count, len(detected.blobs))
        if detected.recovered or detected.filled_by_reverse:
            non_genuine_frames += 1
        if detected.filled_by_reverse:
            reverse_filled_frames += 1
        contour = detected.largest
        if contour is None:
            continue
        white_fraction = max(
            white_fraction, blob_white_fraction(detected.frame, contour)
        )
        frames_with_box += 1
        if (
            flashlight_bbox_overlap(
                detected.frame,
                cv2.boundingRect(contour),
                exclude_mask=exclude_mask,
            )
            > FLASHLIGHT_SUBJECT_THRESHOLD
        ):
            flashlight_bbox_frames += 1
        area = cv2.contourArea(contour)
        if area > best_area_any:
            best_area_any = area
            best_contour_any = contour
            best_frame_any = detected.frame
        if not _frame_is_merged(detection, detected.index) and area > best_area:
            best_area = area
            best_contour = contour
            best_frame = detected.frame
        if (
            detected.centroid is None
            or detected.recovered
            or detected.filled_by_reverse
        ):
            continue
        black_white_balance = max(
            black_white_balance,
            blob_black_white_balance(detected.frame, contour),
        )
        genuine_frames_detected += 1
        genuine_detected_indices.append(detected.index)
        genuine_blob_areas.append(area)
        genuine_centroids.append(detected.centroid)
        current_gray = cv2.cvtColor(detected.frame, cv2.COLOR_BGR2GRAY)
        if previous_genuine_index == detected.index - 1:
            coherence = optical_flow_direction_coherence(
                previous_genuine_gray, current_gray, previous_genuine_contour
            )
            if coherence is not None:
                flow_coherence_values.append(coherence)
        previous_genuine_index = detected.index
        previous_genuine_gray = current_gray
        previous_genuine_contour = contour

    if best_contour is None or best_frame is None:
        best_contour, best_frame = best_contour_any, best_frame_any
    if best_contour is None or best_frame is None:
        return None

    return _DetectionFeatureSummary(
        best_contour=best_contour,
        best_frame=best_frame,
        genuine_centroids=tuple(genuine_centroids),
        genuine_blob_areas=tuple(genuine_blob_areas),
        genuine_detected_indices=tuple(genuine_detected_indices),
        genuine_frames_detected=genuine_frames_detected,
        non_genuine_frames=non_genuine_frames,
        reverse_filled_frames=reverse_filled_frames,
        flow_coherence_values=tuple(flow_coherence_values),
        whole_frame_green_ratios=tuple(whole_frame_green_ratios),
        color_fraction=(
            sum(color_fractions) / len(color_fractions) if color_fractions else 0.0
        ),
        motion_pixel_fraction=motion_pixel_fraction,
        motion_pixel_fraction_median=_median(
            [detected.motion_pixel_fraction for detected in detection.frames]
        ),
        blob_count=blob_count,
        blob_count_median=_median(
            [float(len(detected.blobs)) for detected in detection.frames]
        ),
        white_fraction=white_fraction,
        black_white_balance=black_white_balance,
        rectangular_black_white_balance=rectangular_black_white_balance,
        frames_with_box=frames_with_box,
        flashlight_bbox_frames=flashlight_bbox_frames,
    )


def _appearance_features(
    summary: _DetectionFeatureSummary,
    *,
    exclude_mask: np.ndarray | None,
    reference_row: float,
) -> dict[str, float]:
    """Shape, colour, light and texture readings for the selected subject."""
    contour, frame = summary.best_contour, summary.best_frame
    return {
        "aspect_ratio": aspect_ratio(contour),
        "solidity": solidity(contour),
        "saturation_ratio": saturation_ratio(frame, contour),
        "color_fraction": summary.color_fraction,
        "green_light_ratio": green_light_ratio(
            frame, contour, exclude_mask=exclude_mask
        ),
        "green_light_flicker": green_light_flicker(
            summary.whole_frame_green_ratios
        ),
        # Diagnostic whole-frame counterpart to the subject-only green ratio.
        "whole_frame_green_ratio": (
            max(summary.whole_frame_green_ratios)
            if summary.whole_frame_green_ratios
            else 0.0
        ),
        "flashlight_subject_fraction": (
            0.0
            if not summary.frames_with_box
            else summary.flashlight_bbox_frames / summary.frames_with_box
        ),
        "row_normalised_area": row_normalised_area(contour, reference_row),
        "edge_density": edge_density(frame, contour),
        "blob_frame_fraction": (
            cv2.contourArea(contour) / float(frame.shape[0] * frame.shape[1])
        ),
        "blob_white_fraction": summary.white_fraction,
    }


def _temporal_features(
    detection: ClipDetection, summary: _DetectionFeatureSummary
) -> dict[str, float]:
    """Motion continuity, trajectory and whole-clip transition readings."""
    frame_count = len(detection.frames)
    best_width = float(cv2.boundingRect(summary.best_contour)[2])
    return {
        "post_flash_red_shift": post_flash_red_shift(
            list(detection.dropped_frames)
            + [detected.frame for detected in detection.frames]
        ),
        "path_length": path_length(summary.genuine_centroids),
        "jitter": jitter(summary.genuine_centroids),
        "persistence": persistence(summary.genuine_frames_detected, frame_count),
        "motion_pixel_fraction": summary.motion_pixel_fraction,
        "blob_count": float(summary.blob_count),
        "motion_pixel_fraction_median": summary.motion_pixel_fraction_median,
        "blob_count_median": summary.blob_count_median,
        "longest_detection_run": longest_detection_run(
            summary.genuine_detected_indices, frame_count
        ),
        "area_stability": area_stability(summary.genuine_blob_areas),
        "normalised_speed": normalised_speed(
            summary.genuine_centroids, best_width
        ),
        "heading_change": heading_change(summary.genuine_centroids),
        "flow_direction_coherence": (
            _median(summary.flow_coherence_values)
            if summary.flow_coherence_values
            else 0.0
        ),
        "flow_direction_coherence_has_evidence": float(
            bool(summary.flow_coherence_values)
        ),
        "recovered_fraction": (
            summary.non_genuine_frames / frame_count if frame_count else 0.0
        ),
        "reverse_filled_fraction": (
            summary.reverse_filled_frames / frame_count if frame_count else 0.0
        ),
        "terminal_reverse_seed": float(
            summary.reverse_filled_frames > 0
            and len(summary.genuine_detected_indices) == 1
            and summary.genuine_detected_indices[0] == frame_count - 1
        ),
        "blob_black_white_balance": summary.black_white_balance,
        "rectangular_black_white_balance": summary.rectangular_black_white_balance,
        "global_camera_shift_score": global_camera_shift_score(
            list(detection.dropped_frames)
            + [detected.frame for detected in detection.frames]
        ),
        "scenery_motion_fraction": detection.scenery_motion_fraction,
        "has_reference_background": float(detection.has_reference_background),
    }


def _warmup_features(
    detection: ClipDetection,
    zone: CameraZone,
    *,
    threshold: int,
    exclude_mask: np.ndarray | None,
    daylight_color_fraction: float,
    daylight_hint: bool | None,
    motion_analysis: WarmupMotionAnalysis | None = None,
) -> dict[str, float]:
    """Features sourced from, or explicitly describing, the warmup window."""
    motion = motion_analysis or warmup_motion_analysis(
        detection,
        zone,
        detection.frame_width,
        detection.frame_height,
        threshold=threshold,
    )
    return {
        "warmup_flashlight_ratio": _warmup_flashlight_feature(
            detection,
            exclude_mask=exclude_mask,
            daylight_color_fraction=daylight_color_fraction,
            daylight_hint=daylight_hint,
        ),
        **motion.features,
        "long_flare_frames": float(detection.warmup_dropped),
    }


def _multi_object_features(
    detection: ClipDetection,
    zone: CameraZone,
    *,
    exclude_mask: np.ndarray | None,
) -> dict[str, float]:
    """Combine per-object flashlight and coarse type feature groups."""
    return {
        **_multi_object_flashlight_features(
            detection,
            zone,
            detection.frame_width,
            detection.frame_height,
            exclude_mask=exclude_mask,
        ),
        **_multi_object_type_features(
            detection, zone, detection.frame_width, detection.frame_height
        ),
    }


def extract_clip_features(
    video_path: str,
    zone: CameraZone,
    *,
    reference_row: float | None = None,
    max_area_fraction: float = _DEFAULT_MOTION_THRESHOLDS.max_area_fraction,
    min_blob_area_fraction: float = _DEFAULT_MOTION_THRESHOLDS.min_blob_area_fraction,
    threshold: int = _DEFAULT_MOTION_THRESHOLDS.threshold,
    flare_tolerance: float = _DEFAULT_MOTION_THRESHOLDS.flare_tolerance,
    max_flare_fraction: float = _DEFAULT_MOTION_THRESHOLDS.max_flare_fraction,
    daylight_color_fraction: float = 0.15,
    max_track_jump_fraction: float = _DEFAULT_MOTION_THRESHOLDS.max_track_jump_fraction,
    max_track_miss_frames: int = _DEFAULT_MOTION_THRESHOLDS.max_track_miss_frames,
    template_match_threshold: float = _DEFAULT_MOTION_THRESHOLDS.template_match_threshold,
    flare_match_relax: float = _DEFAULT_MOTION_THRESHOLDS.flare_match_relax,
    track_search_margin_fraction: float = _DEFAULT_MOTION_THRESHOLDS.track_search_margin_fraction,
    min_track_search_margin: float = _DEFAULT_MOTION_THRESHOLDS.min_track_search_margin,
    fragment_close_kernel_size: int = _DEFAULT_MOTION_THRESHOLDS.fragment_close_kernel_size,
    max_recovered_streak: int = _DEFAULT_MOTION_THRESHOLDS.max_recovered_streak,
    max_size_change_ratio: float = _DEFAULT_MOTION_THRESHOLDS.max_size_change_ratio,
    anchor_refine: bool = _DEFAULT_MOTION_THRESHOLDS.anchor_refine,
    max_anchor_streak: int = _DEFAULT_MOTION_THRESHOLDS.max_anchor_streak,
    min_reacquire_area: float = _DEFAULT_MOTION_THRESHOLDS.min_reacquire_area,
    reference_background: np.ndarray | None = None,
    reference_background_primary: bool = False,
    max_scenery_streak: int = _DEFAULT_MOTION_THRESHOLDS.max_scenery_streak,
    scenery_correlation: float = _DEFAULT_MOTION_THRESHOLDS.scenery_correlation,
    daylight_hint: bool | None = None,
    compensate_warmup: bool = _DEFAULT_MOTION_THRESHOLDS.compensate_warmup,
    prefer_flashlight_candidate: bool = _DEFAULT_MOTION_THRESHOLDS.prefer_flashlight_candidate,
    multi_track_confirm_frames: int = _DEFAULT_MOTION_THRESHOLDS.multi_track_confirm_frames,
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

    `reference_background_primary` is an experimental detector mode. When a
    compatible reference is available it uses that aligned cross-clip image as
    the scored-frame difference target, which can recover a subject absorbed
    into this clip's median background. It is reporting-only through
    ``scripts.backtest --reference-background-primary`` and defaults off.

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
        max_track_jump_fraction=max_track_jump_fraction,
        max_track_miss_frames=max_track_miss_frames,
        template_match_threshold=template_match_threshold,
        flare_match_relax=flare_match_relax,
        track_search_margin_fraction=track_search_margin_fraction,
        min_track_search_margin=min_track_search_margin,
        fragment_close_kernel_size=fragment_close_kernel_size,
        max_recovered_streak=max_recovered_streak,
        max_size_change_ratio=max_size_change_ratio,
        anchor_refine=anchor_refine,
        max_anchor_streak=max_anchor_streak,
        min_reacquire_area=min_reacquire_area,
        ignore_polygons=zone.ignore,
        reference_background=reference_background,
        reference_background_primary=reference_background_primary,
        max_scenery_streak=max_scenery_streak,
        scenery_correlation=scenery_correlation,
        compensate_warmup=compensate_warmup,
        prefer_flashlight_candidate=prefer_flashlight_candidate,
        multi_track_confirm_frames=multi_track_confirm_frames,
    )
    if detection is None:
        return None
    # Preserve real frame gaps for metric speed while rejecting the impossible
    # FPS metadata reported by roughly 2% of the corpus (for example 1005/16000).
    fps = sane_fps(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
    return features_from_detection(
        detection,
        zone,
        fps=fps,
        reference_row=reference_row,
        threshold=threshold,
        daylight_color_fraction=daylight_color_fraction,
        daylight_hint=daylight_hint,
    )


def features_from_detection(
    detection: ClipDetection,
    zone: CameraZone,
    *,
    fps: float,
    reference_row: float | None = None,
    threshold: int = _DEFAULT_MOTION_THRESHOLDS.threshold,
    daylight_color_fraction: float = 0.15,
    daylight_hint: bool | None = None,
    warmup_motion: WarmupMotionAnalysis | None = None,
) -> dict[str, float] | None:
    """Re-score an already-observed clip for a zone without rerunning detection.

    This is the public detector/feature seam. ``ClipDetection`` remains an
    in-memory object today, so it is not itself a cache payload; its frames and
    contours are still needed by colour, texture and per-object type features.
    Geometry and single-track metric calibration already reduce through their
    compact observation DTOs inside this scorer.

    The detector currently receives ``zone.ignore`` as an input, so changing
    an ignore polygon still requires detection. Fence, side, depth and metric
    calibration changes can be re-scored from the same detection object.
    """
    ignore_mask = feature_exclude_mask(detection, zone)
    summary = _summarize_detection_features(
        detection,
        exclude_mask=ignore_mask,
    )
    warmup_features = _warmup_features(
        detection,
        zone,
        threshold=threshold,
        exclude_mask=ignore_mask,
        daylight_color_fraction=daylight_color_fraction,
        daylight_hint=daylight_hint,
        motion_analysis=warmup_motion,
    )
    if summary is None:
        if (
            warmup_features["warmup_dynamic_frame_fraction"] <= 0.0
            and warmup_features["warmup_flashlight_ratio"] <= 0.0
        ):
            return None
        return {
            "scored_motion_present": 0.0,
            "green_light_ratio": 0.0,
            "green_light_flicker": 0.0,
            "outside_pixel_fraction": 0.0,
            "outside_area_fraction": 0.0,
            "zone_classifiable_area_fraction": 0.0,
            "median_fence_distance": 0.0,
            "color_fraction": 0.0,
            "blob_count": 0.0,
            "global_camera_shift_score": global_camera_shift_score(
                list(detection.dropped_frames)
                + [detected.frame for detected in detection.frames]
            ),
            "rectangular_black_white_balance": 0.0,
            "scenery_motion_fraction": detection.scenery_motion_fraction,
            "has_reference_background": float(detection.has_reference_background),
            **warmup_features,
        }
    ref_row = (
        reference_row if reference_row is not None else float(detection.frame_height)
    )
    geometry_observations = geometry_observations_from_detection(detection)
    # The compact reduction and frame summary share the same best-contour rule.
    if geometry_observations is None:
        return None
    metric_observations = metric_observations_from_detection(detection, fps=fps)

    return {
        "scored_motion_present": 1.0,
        **geometry_features_from_observations(geometry_observations, zone),
        **_appearance_features(
            summary,
            exclude_mask=ignore_mask,
            reference_row=ref_row,
        ),
        **warmup_features,
        **_multi_object_features(detection, zone, exclude_mask=ignore_mask),
        **_temporal_features(detection, summary),
        **metric_features_from_observations(metric_observations, zone),
    }
