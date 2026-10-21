"""Offline per-object scenery measurements; never used by the live classifier.

Reference resemblance is evidence of scenery, not a vegetation class. Objects
that cannot be measured remain unknown. Only observed motion candidates count
toward trajectories: recovered/template boxes never establish movement.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.config import CameraZone
from src.motion import ClipDetection, _bbox_iou, _bbox_to_rect_contour, track_multiple_objects
from src.scoring import warmup_motion_analysis


@dataclass(frozen=True)
class ProbeSettings:
    reference_similarity: float = 0.94
    novel_similarity_max: float = 0.70
    min_reference_observations: int = 3
    reference_vote_fraction: float = 0.8
    min_patch_pixels: int = 16
    min_patch_std: float = 3.0
    min_alignment_response: float = 0.2
    max_alignment_fraction: float = 0.1
    min_trajectory_observations: int = 5
    min_step_pixels: float = 1.5
    max_stationary_range_pixels: float = 2.0


DEFAULT_SETTINGS = ProbeSettings()


def patch_similarity(
    current: np.ndarray,
    reference: np.ndarray,
    bbox: tuple[int, int, int, int],
    valid: np.ndarray,
    settings: ProbeSettings = DEFAULT_SETTINGS,
) -> float | None:
    """Brightness-invariant resemblance at the same, independently aligned place.

    Small, flat, out-of-frame and registration-border patches are unmeasured.
    A constant patch must not look like confident scenery merely because two
    black rectangles happen to agree.
    """
    if current.shape != reference.shape or valid.shape != reference.shape:
        return None
    x0, y0, x1, y1 = bbox
    height, width = current.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None
    if (x1 - x0) * (y1 - y0) < settings.min_patch_pixels:
        return None
    if not np.all(valid[y0:y1, x0:x1]):
        return None
    first = current[y0:y1, x0:x1].astype(np.float32)
    second = reference[y0:y1, x0:x1].astype(np.float32)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    if min(float(first.std()), float(second.std())) < settings.min_patch_std:
        return None
    value = float(cv2.matchTemplate(first, second, cv2.TM_CCOEFF_NORMED)[0, 0])
    return value if np.isfinite(value) else None


def align_reference(
    reference: np.ndarray | None, background: np.ndarray,
    settings: ProbeSettings = DEFAULT_SETTINGS,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Register a cross-event reference while retaining unsupported image borders."""
    if reference is None or reference.shape != background.shape:
        return None, None, {"usable": False, "reason": "missing_or_incompatible_reference"}
    if min(float(reference.std()), float(background.std())) < settings.min_patch_std:
        return None, None, {"usable": False, "reason": "flat_reference_or_background"}
    (dx, dy), response = cv2.phaseCorrelate(
        reference.astype(np.float32), background.astype(np.float32)
    )
    shift = float(np.hypot(dx, dy))
    good = (np.isfinite([dx, dy, response]).all()
            and response >= settings.min_alignment_response
            and shift <= settings.max_alignment_fraction * np.hypot(*background.shape))
    details = {"usable": bool(good), "shift_pixels": shift, "response": float(response)}
    if not good:
        return None, None, {**details, "reason": "uncertain_registration"}
    height, width = background.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(reference, matrix, (width, height))
    valid = cv2.warpAffine(
        np.ones_like(reference, dtype=np.uint8), matrix, (width, height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    return aligned, valid, details


def trajectory_evidence(
    observations: Sequence[dict[str, Any]], settings: ProbeSettings = DEFAULT_SETTINGS,
) -> dict[str, Any]:
    """Measure bounded returns only across consecutive, directly observed steps."""
    observations = [o for o in observations if not o.get("merged", False)]
    unknown = {"has_evidence": False, "reversals": None, "net_path_ratio": None,
               "stationary": None, "bounded_return": None}
    if len(observations) < settings.min_trajectory_observations:
        return unknown
    indices = np.array([o["index"] for o in observations])
    if np.any(np.diff(indices) != 1):
        return unknown
    centers = np.array([((o["bbox"][0] + o["bbox"][2]) / 2,
                         (o["bbox"][1] + o["bbox"][3]) / 2) for o in observations])
    steps = np.diff(centers, axis=0)
    distances = np.linalg.norm(steps, axis=1)
    path = float(distances.sum())
    span = float(np.linalg.norm(np.ptp(centers, axis=0)))
    if span <= settings.max_stationary_range_pixels:
        return {"has_evidence": True, "stationary": True, "bounded_return": False,
                "reversals": 0, "net_path_ratio": None, "range_pixels": span}
    moving = distances >= settings.min_step_pixels
    if int(moving.sum()) < 3:
        return unknown
    _u, _s, axes = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
    projected = steps @ axes[0]
    signs = np.sign(projected[moving])
    reversals = int(np.count_nonzero(signs[:-1] * signs[1:] < 0))
    ratio = float(np.linalg.norm(centers[-1] - centers[0]) / path) if path else None
    return {"has_evidence": True, "stationary": False,
            "bounded_return": bool(reversals >= 2 and ratio is not None and ratio <= 0.25),
            "reversals": reversals, "net_path_ratio": ratio, "range_pixels": span}


def summarize_object(
    observations: Sequence[dict[str, Any]], settings: ProbeSettings = DEFAULT_SETTINGS,
) -> dict[str, Any]:
    """Describe a track; novel appearance is not a person/animal identification."""
    similarities = [o["similarity"] for o in observations if o["similarity"] is not None]
    matched = sum(v >= settings.reference_similarity for v in similarities)
    novel = sum(v < settings.novel_similarity_max for v in similarities)
    reference_confirmed = (
        len(similarities) >= settings.min_reference_observations
        and matched / len(observations) >= settings.reference_vote_fraction
    )
    trajectory = trajectory_evidence(observations, settings)
    return {"observations": list(observations), "reported_frames": len(observations),
            "observed_frames": sum(not o.get("merged", False) for o in observations),
            "merged_frames": sum(o.get("merged", False) for o in observations),
            "warmup_frames": sum(o["phase"] == "warmup" for o in observations),
            "scored_frames": sum(o["phase"] == "scored" for o in observations),
            "selected_direct_frames": sum(o["selected_direct"] for o in observations),
            "measured_reference_frames": len(similarities), "matched_reference_frames": matched,
            "novel_frames": novel, "reference_confirmed": reference_confirmed,
            "novel_appearance": bool(novel >= 2),
            "unknown_reference_frames": len(observations) - len(similarities),
            "trajectory": trajectory,
            "bounded_reference_track": bool(reference_confirmed and trajectory["has_evidence"]
                and (trajectory["stationary"] or trajectory["bounded_return"]))}


def probe_detection(
    detection: ClipDetection, zone: CameraZone, reference: np.ndarray | None,
    settings: ProbeSettings = DEFAULT_SETTINGS, *, threshold: int = 18,
) -> dict[str, Any]:
    """Track all observed objects through startup and scoring, measuring each.

    Multi-object tracking is recomputed across the warmup boundary from real
    contours. Production templates and reverse-filled boxes are not inputs.
    Shared/merged boxes are retained as unknown reference observations.
    """
    aligned, valid, registration = align_reference(reference, detection.background, settings)
    warmup = warmup_motion_analysis(
        detection, zone, detection.frame_width, detection.frame_height, threshold=threshold,
    )
    drop = len(detection.dropped_frames)
    warmup_gray = list(warmup.corrected_frames)
    if len(warmup_gray) != drop:
        warmup_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in detection.dropped_frames]
    candidates: list[list[np.ndarray]] = [[] for _ in range(drop)]
    for obj in warmup.objects:
        if obj.disposition in ("used", "not_selected"):
            candidates[obj.frame_index].append(_bbox_to_rect_contour(obj.bbox))
    gray = [cv2.GaussianBlur(f, (5, 5), 0) for f in warmup_gray]
    main_boxes = [None] * drop
    for detected in detection.frames:
        gray.append(cv2.GaussianBlur(cv2.cvtColor(detected.frame, cv2.COLOR_BGR2GRAY), (5, 5), 0))
        candidates.append([c for c in detected.all_contours
                           if 0 < cv2.contourArea(c) <= 0.25 * detection.frame_width
                           * detection.frame_height])
        direct = (detected.largest is not None and detected.centroid is not None
                  and not detected.recovered and not detected.filled_by_reverse)
        if direct:
            x, y, width, height = cv2.boundingRect(detected.largest)
            main_boxes.append((x, y, x + width, y + height))
        else:
            main_boxes.append(None)
    tracks = track_multiple_objects(
        candidates,
        max_jump_distance=0.2 * np.hypot(detection.frame_width, detection.frame_height),
        max_track_miss_frames=5,
    )
    by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, objects in enumerate(tracks):
        for obj in objects:
            similarity = None
            if aligned is not None and not obj.merged_ids:
                similarity = patch_similarity(gray[index], aligned, obj.bbox, valid, settings)
            by_id[obj.track_id].append({
                "index": index, "phase": "warmup" if index < drop else "scored",
                "bbox": obj.bbox, "similarity": similarity,
                "merged": bool(obj.merged_ids),
                "selected_direct": main_boxes[index] is not None
                and _bbox_iou(obj.bbox, main_boxes[index]) >= 0.5,
            })
    objects = [{"track_id": tid, **summarize_object(obs, settings)}
               for tid, obs in sorted(by_id.items())]
    selected = [obj for obj in objects if obj["selected_direct_frames"]]
    selected_ids = {obj["track_id"] for obj in selected}
    others = [obj for obj in objects if obj["track_id"] not in selected_ids]
    accounted = {o["index"] for obj in selected for o in obj["observations"]
                 if o["selected_direct"]}
    direct_count = sum(box is not None for box in main_boxes)
    coverage = direct_count > 0 and len(accounted) == direct_count
    main_reference = coverage and all(obj["reference_confirmed"] for obj in selected)
    main_bounded = coverage and all(obj["bounded_reference_track"] for obj in selected)
    novel_others = [obj for obj in others if obj["novel_appearance"]]
    unknown_others = [obj for obj in others
                      if not obj["reference_confirmed"] and not obj["novel_appearance"]]
    return {
        "registration": registration, "objects": objects, "object_count": len(objects),
        "selected_direct_observations": direct_count,
        "selected_direct_observations_accounted": len(accounted),
        "startup_unmeasured": bool(drop == 1),
        "oversized_warmup_objects": sum(obj.disposition == "ignored_size"
            and obj.area > 0.25 * detection.frame_width * detection.frame_height
            for obj in warmup.objects),
        "reference_confirmed_tracks": sum(obj["reference_confirmed"] for obj in objects),
        "bounded_reference_tracks": sum(obj["bounded_reference_track"] for obj in objects),
        "novel_appearance_tracks": sum(obj["novel_appearance"] for obj in objects),
        "warmup_novel_appearance_tracks": sum(obj["novel_appearance"]
            and obj["warmup_frames"] > 0 for obj in objects),
        "cross_boundary_tracks": sum(obj["warmup_frames"] > 0 and obj["scored_frames"] > 0
                                     for obj in objects),
        "unknown_reference_tracks": sum(not obj["reference_confirmed"]
            and not obj["novel_appearance"] for obj in objects),
        "unknown_reference_observations": sum(obj["unknown_reference_frames"] for obj in objects),
        "novel_observations": sum(obj["novel_frames"] for obj in objects),
        "main_reference_confirmed": main_reference, "main_bounded_reference": main_bounded,
        "other_novel_appearance_tracks": len(novel_others),
        "other_unknown_tracks": len(unknown_others),
        "mixed_scene": bool(any(obj["reference_confirmed"] for obj in objects)
                            and any(obj["novel_appearance"] for obj in objects)),
    }
