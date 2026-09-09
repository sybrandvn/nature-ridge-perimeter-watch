"""Rank the full corpus by animal/incident-likeness for review triage.

Fits a small hand-rolled logistic regression (no sklearn; matches the rest of
this repo) on the labelled ground truth -- positive = animal/incident,
negative = every other real class -- over the 19 numeric features in
`scripts.spike.FEATURE_COLUMNS`, scores every downloaded+detected clip in the
corpus, and writes a per-camera stratified review queue: the top N unlabelled
clips per camera by score, plus a matching `.message_ids` file for
`scripts/label.py --message-ids-file`.

Always uses whatever detector `scripts.spike.extract_clip_features` currently
wraps -- rerun this whenever the detector changes, since every feature value
depends on it. See docs/gate2_separability_finding.md for the model's
validated leave-one-out AUC and the label-coverage caveat: this ranks well
within the labelled pool's population, not as a certified corpus-wide
classifier (top-100 by an earlier fit contained zero known positives, because
only ~1.5% of the corpus is labelled and coverage is uneven per camera).

`--maintenance-out` additionally writes the "camera needs cleaning" channel,
which is two files, not one: the per-clip queue itself, and a sibling
`.windows.csv` listing the spans where one camera's own blinding-flag rate is
sustainedly elevated above its own baseline (see `obstruction_windows`). The
windows file is the actionable one -- 30 spans across three years of history
against 1,019 individual flagged clips.

Run:
    uv run python scripts/rank_candidates.py --workers 8 \\
        --out data/reports/candidates_2026-08-30.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts.label import _event_key  # noqa: E402
from scripts.spike import FEATURE_COLUMNS, extract_clip_features  # noqa: E402
from src import db  # noqa: E402
from src.classify import is_blinding_foreground  # noqa: E402
from src.config import CamerasConfig, load_app_config, load_cameras_config  # noqa: E402
from src.features import daylight_hint, is_daylight, sane_fps  # noqa: E402
from src.reference_bg import (  # noqa: E402
    era_of,
    load_manifest,
    load_reference_image,
    reference_for,
)

ExtractFn = Callable[..., "dict[str, float] | None"]

_NON_FEATURE_COLUMNS = (
    "channel_id",
    "message_id",
    "camera_id",
    "label",
    "time_of_day",
    "is_daylight",
    "has_reference_background",
)
NUMERIC_FEATURES = tuple(c for c in FEATURE_COLUMNS if c not in _NON_FEATURE_COLUMNS)
POSITIVE_LABELS = ("animal", "incident")
NEGATIVE_LABELS = ("guard", "environment", "resident", "unknown")

# Cameras where a human reviewer confirmed (watching the actual footage, not
# an inferred threshold) that every is_blinding_foreground hit is a guard
# shining their flashlight directly into the lens, or a permanent structure
# the camera is mounted on -- so these clips don't belong in a "send someone
# to clean this" report. Neither is something anyone can clean.
# cam04, 2026-09-07: user reviewed the full maintenance queue and confirmed
# every cam04 entry was the flashlight pattern.
# cam01a, 2026-09-08: user confirmed this is the pole-mounted camera, it reads
# its own pole as very bright, and the guard flashes it directly. It is the
# worst camera on site for overexposure (33% of clips) and the blinding flag
# (37%), and `post_flash_red_shift` does not catch it -- 39 of its 43 flagged
# clips survive that filter.
NO_MAINTENANCE_CAMERAS = frozenset({"cam04", "cam01a"})

# Only guard clips exceed this on the labelled corpus (guard max 0.581 vs
# environment 0.014, animal 0.038, incident 0.001) -- see
# src.features.post_flash_red_shift.
FLASHLIGHT_RED_SHIFT = 0.10

# Every numeric feature, not a hand-picked subset: the queue is a REVIEW
# artifact, and the whole point of reviewing a candidate is being able to see
# why it scored -- persistence/longest_detection_run to tell a sustained
# subject from a one-frame streak, the metric columns to tell a small animal
# from a person, recovered_fraction to spot a hallucinated track. Re-running
# the 9-minute scoring pass just to add a column is not acceptable.
REPORT_COLUMNS = (
    "score",
    "camera_id",
    "channel_id",
    "message_id",
    "timestamp",
    *NUMERIC_FEATURES,
)


def iter_clips_with_files(conn: Any, *, camera_id: str | None = None) -> Iterator[dict[str, Any]]:
    for row in db.iter_clips(conn, camera_id=camera_id):
        if not row["file_path"]:
            continue
        label_row = db.get_label(conn, row["channel_id"], row["message_id"])
        yield {
            "channel_id": row["channel_id"],
            "message_id": row["message_id"],
            "camera_id": row["camera_id"],
            "timestamp": row["timestamp"],
            "file_path": row["file_path"],
            "caption": row["caption"],
            "label": label_row["label"] if label_row is not None else None,
            "startup_state": label_row["startup_state"] if label_row is not None else None,
        }


def reference_background_for(entries, root, camera, timestamp):
    """This camera's reference background for the clip's era and lighting, or
    None when nothing covers it. Same resolution as scripts/render_debug.py and
    scripts/backtest.py, so every tool scores the same signal the same way."""
    if not entries or timestamp is None:
        return None
    entry = reference_for(
        entries,
        camera.id,
        timestamp,
        era=era_of(camera, timestamp),
        daylight=is_daylight(timestamp),
    )
    return None if entry is None else load_reference_image(Path(root), entry)


def collect_features(
    conn: Any,
    cameras: CamerasConfig,
    *,
    camera_id: str | None = None,
    reference_entries: Any = None,
    reference_root: str = "data/reference_bg",
    extract_fn: ExtractFn = extract_clip_features,
) -> list[dict[str, Any]]:
    """Single-process feature collection -- the testable, injectable core. The
    CLI uses a multiprocessing variant of this same logic for full-corpus runs
    (see `_collect_features_parallel`), since ~16.9k clips take ~30 minutes
    single-core."""
    unknown_cameras: set[str] = set()
    rows: list[dict[str, Any]] = []
    for clip in iter_clips_with_files(conn, camera_id=camera_id):
        camera = cameras.by_id(clip["camera_id"])
        if camera is None:
            unknown_cameras.add(clip["camera_id"])
            continue
        extra: dict[str, Any] = {}
        reference = reference_background_for(
            reference_entries, reference_root, camera, clip["timestamp"]
        )
        if reference is not None:
            extra["reference_background"] = reference
        hint = daylight_hint(clip["timestamp"])
        if hint is not None:
            extra["daylight_hint"] = hint
        features = extract_fn(clip["file_path"], camera.zone_at(clip["timestamp"]), **extra)
        row = {
            "channel_id": clip["channel_id"],
            "message_id": clip["message_id"],
            "camera_id": clip["camera_id"],
            "timestamp": clip["timestamp"],
            "caption": clip.get("caption"),
            "file_path": clip["file_path"],
            "label": clip["label"],
            "startup_state": clip.get("startup_state"),
            "detected": features is not None,
        }
        for feat in NUMERIC_FEATURES:
            row[feat] = 0.0 if features is None else float(features.get(feat) or 0.0)
        rows.append(row)
    for camera_id_ in sorted(unknown_cameras):
        print(f"skipped: no config/cameras.yaml entry for {camera_id_!r}", file=sys.stderr)
    return rows


_WORKER_CAMERAS: CamerasConfig | None = None
_WORKER_REFERENCE_ENTRIES: Any = None
_WORKER_REFERENCE_ROOT: str = "data/reference_bg"


def _init_worker(cameras_path: str, reference_root: str = "data/reference_bg") -> None:
    global _WORKER_CAMERAS, _WORKER_REFERENCE_ENTRIES, _WORKER_REFERENCE_ROOT
    _WORKER_CAMERAS = load_cameras_config(cameras_path)
    _WORKER_REFERENCE_ROOT = reference_root
    # Loaded per worker rather than pickled across the fork: the manifest is
    # small, the reference IMAGES it points at are not.
    _WORKER_REFERENCE_ENTRIES = load_manifest(reference_root) if reference_root else []


def _extract_worker(clip: dict[str, Any]) -> dict[str, Any]:
    row = {
        "channel_id": clip["channel_id"],
        "message_id": clip["message_id"],
        "camera_id": clip["camera_id"],
        "timestamp": clip["timestamp"],
        "caption": clip.get("caption"),
        "file_path": clip["file_path"],
        "label": clip["label"],
        "startup_state": clip.get("startup_state"),
    }
    camera = _WORKER_CAMERAS.by_id(clip["camera_id"]) if _WORKER_CAMERAS else None
    features = None
    if camera is not None:
        try:
            extra: dict[str, Any] = {}
            reference = reference_background_for(
                _WORKER_REFERENCE_ENTRIES, _WORKER_REFERENCE_ROOT, camera, clip["timestamp"]
            )
            if reference is not None:
                extra["reference_background"] = reference
            hint = daylight_hint(clip["timestamp"])
            if hint is not None:
                extra["daylight_hint"] = hint
            features = extract_clip_features(
                clip["file_path"], camera.zone_at(clip["timestamp"]), **extra
            )
        except Exception:
            features = None
    row["detected"] = features is not None
    for feat in NUMERIC_FEATURES:
        row[feat] = 0.0 if features is None else float(features.get(feat) or 0.0)
    return row


def _collect_features_parallel(
    conn: Any, cameras_path: str, *, workers: int, reference_root: str = "data/reference_bg"
) -> list[dict[str, Any]]:
    clips = list(iter_clips_with_files(conn))
    print(f"clips with a file: {len(clips)}")
    rows: list[dict[str, Any]] = []
    with Pool(
        workers, initializer=_init_worker, initargs=(cameras_path, reference_root)
    ) as pool:
        for i, row in enumerate(pool.imap_unordered(_extract_worker, clips, chunksize=40)):
            rows.append(row)
            if i % 2000 == 0:
                print(i, flush=True)
    return rows


def fit_logistic(
    x: np.ndarray, y: np.ndarray, *, l2: float = 1.0, lr: float = 0.5, iterations: int = 3000
) -> np.ndarray:
    """Batch gradient descent logistic regression, L2-regularised, with samples
    weighted inversely to their class frequency so the rare positive class
    (typically ~5% of labelled rows) isn't swamped by the negative majority."""
    n, d = x.shape
    x_aug = np.hstack([np.ones((n, 1)), x])
    weights = np.zeros(d + 1)
    n_pos = max(float(y.sum()), 1.0)
    n_neg = max(float(n - y.sum()), 1.0)
    sample_weight = np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(x_aug @ weights)))
        grad = x_aug.T @ (sample_weight * (p - y)) / n
        grad[1:] += (l2 / n) * weights[1:]
        weights -= lr * grad
    return weights


def predict_proba(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([np.ones((x.shape[0], 1)), x])
    return 1.0 / (1.0 + np.exp(-(x_aug @ weights)))


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC: fraction of positive/negative pairs the
    scores order correctly, ties counted as half a win."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    wins = sum(np.sum(p > scores_neg) + 0.5 * np.sum(p == scores_neg) for p in scores_pos)
    return float(wins) / (len(scores_pos) * len(scores_neg))


def leave_one_out_auc(x: np.ndarray, y: np.ndarray, **fit_kwargs: Any) -> float:
    scores = np.zeros(len(y))
    mask = np.ones(len(y), dtype=bool)
    for i in range(len(y)):
        mask[i] = False
        weights = fit_logistic(x[mask], y[mask], **fit_kwargs)
        scores[i] = predict_proba(x[i : i + 1], weights)[0]
        mask[i] = True
    pos = y == 1
    return auc(scores[pos], scores[~pos])


def clip_duration_seconds(file_path: str) -> float:
    cap = cv2.VideoCapture(file_path)
    try:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = sane_fps(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    return frame_count / fps if frame_count > 0 else 0.0


def prefer_longest_per_event(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse rows sharing an embedded alert timestamp (the same physical
    trigger, per `scripts.label._event_key`) to just the longest clip.

    A review queue should never spend a slot on an event's short startup-only
    clip when its longer, more representative sibling exists -- reviewing the
    short one on its own tells you nothing. Same rule and same helper as
    `scripts.render_debug._prefer_longest_per_event`; rows with no parseable
    event key are always kept.
    """
    best_by_key: dict[str, dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for row in rows:
        key = _event_key(row["camera_id"], row.get("caption"))
        if key is None:
            result.append(row)
            continue
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = row
            result.append(row)
        elif clip_duration_seconds(row.get("file_path") or "") > clip_duration_seconds(
            existing.get("file_path") or ""
        ):
            result[result.index(existing)] = row
            best_by_key[key] = row
    return result


def stratified_top_n(
    rows: list[dict[str, Any]],
    *,
    camera_key: str = "camera_id",
    score_key: str = "score",
    top_n: int,
) -> list[dict[str, Any]]:
    """Top `top_n` rows per distinct `camera_key` value by `score_key`, so one
    high-clip-count camera can't crowd out every other camera's review queue."""
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_camera[row[camera_key]].append(row)
    queue: list[dict[str, Any]] = []
    for camera_id in sorted(by_camera):
        cam_rows = sorted(by_camera[camera_id], key=lambda r: -r[score_key])
        queue.extend(cam_rows[:top_n])
    queue.sort(key=lambda r: -r[score_key])
    return queue


def rank_and_write(
    rows: list[dict[str, Any]], *, top_per_camera: int, out_path: str
) -> list[dict[str, Any]]:
    """Fit on the labelled+detected rows, score every detected row, and write
    the per-camera stratified unlabelled queue. Returns the written queue."""
    detected = [r for r in rows if r.get("detected")]
    labelled = [r for r in detected if r["label"] in POSITIVE_LABELS + NEGATIVE_LABELS]
    if not labelled:
        raise ValueError("no labelled+detected rows to fit against")

    x_labelled = np.array([[r[f] for f in NUMERIC_FEATURES] for r in labelled])
    y_labelled = np.array([1.0 if r["label"] in POSITIVE_LABELS else 0.0 for r in labelled])
    mean = x_labelled.mean(axis=0)
    std = x_labelled.std(axis=0)
    std[std == 0] = 1.0
    x_labelled_std = (x_labelled - mean) / std

    weights = fit_logistic(x_labelled_std, y_labelled)
    loo = leave_one_out_auc(x_labelled_std, y_labelled)
    n_pos = int(y_labelled.sum())
    print(f"labelled rows: {len(labelled)} ({n_pos} positive) -- leave-one-out AUC: {loo:.3f}")

    x_all = np.array([[r[f] for f in NUMERIC_FEATURES] for r in detected])
    x_all_std = (x_all - mean) / std
    scores = predict_proba(x_all_std, weights)
    for row, score in zip(detected, scores, strict=True):
        row["score"] = round(float(score), 3)

    # blank/duplicate rows have nothing worth reviewing on their own -- blank
    # is empty content, duplicate is a literal frame-for-frame prefix of a
    # longer sibling clip. Excluding them here (not just via
    # prefer_longest_per_event) matters because that helper only collapses a
    # pair when BOTH rows are in this same unlabelled/detected pool; a
    # duplicate whose longer sibling has no file, wasn't detected, or is
    # already labelled would otherwise stay in as a solo, uninformative
    # candidate.
    unlabelled = [
        r
        for r in detected
        if not r["label"] and r.get("startup_state") not in ("blank", "duplicate")
    ]

    # A blinding-foreground clip (bright obstruction dominating the tracked
    # blob, or an IR ramp that never settled -- see
    # src.classify.is_blinding_foreground) is a maintenance issue, not an
    # incident/animal lead: measured zero leak into either class, so excluding
    # it here never costs a real sighting. It still gets its own review queue
    # via write_maintenance_candidates below.
    unlabelled = [r for r in unlabelled if not is_blinding_foreground(r)]

    # A sub-1s clip whose embedded alert timestamp is shared by another
    # file-having clip anywhere in the corpus is a redundant review slot --
    # checked against the labelled corpus first: every animal/incident/resident
    # clip under 1s (4 of 33) already has such a sibling, so this never drops
    # the only representation of a real event. Sharing the timestamp is
    # sufficient on its own, without a literal frame match: some of these short
    # clips are themselves corrupt/burst recordings (a real declared sub-second
    # duration, not a metadata bug -- see cam07/18985) that would never pass
    # _frames_prefix_match, but the event is still fully covered by the sibling.
    event_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        key = _event_key(row["camera_id"], row.get("caption"))
        if key is not None:
            event_counts[key] += 1

    def _is_redundant_short_clip(row: dict[str, Any]) -> bool:
        key = _event_key(row["camera_id"], row.get("caption"))
        if key is None or event_counts[key] < 2:
            return False
        return clip_duration_seconds(row.get("file_path") or "") < 1.0

    unlabelled = [r for r in unlabelled if not _is_redundant_short_clip(r)]
    unlabelled = prefer_longest_per_event(unlabelled)
    queue = stratified_top_n(unlabelled, top_n=top_per_camera)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in queue:
            writer.writerow({col: row.get(col, "") for col in REPORT_COLUMNS})

    ids_path = str(Path(out_path).with_suffix(".message_ids"))
    with open(ids_path, "w") as f:
        for row in queue:
            f.write(f"{row['message_id']}\n")

    print(f"wrote {len(queue)} candidates to {out_path} and {ids_path}")
    return queue


def write_maintenance_candidates(
    rows: list[dict[str, Any]], *, top_per_camera: int, out_path: str
) -> list[dict[str, Any]]:
    """Every detected clip flagged `is_blinding_foreground` (see
    src.classify), any label state -- this is a "camera needs cleaning"
    report, not an incident lead, so an already-labelled guard/environment
    clip still belongs here. Ranked by whichever of the two triggering
    features reads more extreme, per-camera stratified same as the main
    queue.

    Excludes flashlight-into-lens clips via `post_flash_red_shift` (see
    src.features): a guard's flashlight flash makes the whole frame settle red
    afterwards, a physical obstruction never does. Measured 2026-09-07 on the
    already-flagged population -- 27/85 guard clips exceed 0.10 but 0/24
    environment, 0/2 resident and 0/2 unknown do, so this removes flashlight
    clips without touching a single real obstruction.

    `NO_MAINTENANCE_CAMERAS` stays as a second, human-confirmed override: the
    red-shift threshold catches most but not all of cam04 (some of its
    confirmed flashlight clips read below 0.10), and the user verified that
    camera's queue by eye.
    """
    detected = [
        r
        for r in rows
        if r.get("detected")
        and is_blinding_foreground(r)
        and r.get("post_flash_red_shift", 0.0) < FLASHLIGHT_RED_SHIFT
        and r["camera_id"] not in NO_MAINTENANCE_CAMERAS
    ]
    detected = prefer_longest_per_event(detected)
    for row in detected:
        row["maintenance_score"] = max(
            row.get("blob_white_fraction", 0.0), row.get("long_flare_frames", 0.0) / 50.0
        )
    queue = stratified_top_n(detected, score_key="maintenance_score", top_n=top_per_camera)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    columns = (*REPORT_COLUMNS[:1], "maintenance_score", *REPORT_COLUMNS[1:])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in queue:
            writer.writerow({col: row.get(col, "") for col in columns})

    ids_path = str(Path(out_path).with_suffix(".message_ids"))
    with open(ids_path, "w") as f:
        for row in queue:
            f.write(f"{row['message_id']}\n")

    print(f"wrote {len(queue)} maintenance candidates to {out_path} and {ids_path}")
    return queue


# Chosen against the corpus-wide behaviour of the flag itself (measured
# 2026-09-07): every camera's per-clip rate is 4-23% as a BASELINE, so a
# window only means something if it is both well above that camera's own
# normal AND high in absolute terms. `min_clips` keeps a quiet camera's
# handful of clips from producing a 100% "window" out of two hits.
OBSTRUCTION_WINDOW_DAYS = 14
OBSTRUCTION_MIN_CLIPS = 12
OBSTRUCTION_MIN_RATE = 0.25
OBSTRUCTION_MIN_RATIO = 2.0


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def obstruction_windows(
    rows: list[dict[str, Any]],
    *,
    window_days: int = OBSTRUCTION_WINDOW_DAYS,
    min_clips: int = OBSTRUCTION_MIN_CLIPS,
    min_rate: float = OBSTRUCTION_MIN_RATE,
    min_ratio: float = OBSTRUCTION_MIN_RATIO,
) -> list[dict[str, Any]]:
    """Spans where ONE camera's own blinding-flag rate is sustainedly elevated
    above its own baseline -- the actionable form of "go clean this camera".

    The per-clip flag is not that signal, and measuring it said so plainly: on
    the 16,554-clip corpus the flag fires on 8.9% of clips but shows almost no
    temporal clustering (P(next clip from the same camera also flagged | this
    one flagged) is 0-19%, and the longest consecutive run anywhere is 5).
    A spider web on a lens does not behave like that -- it would flag every
    clip until someone wipes it. So the per-clip flag is measuring transient
    bright events, and a maintenance queue built directly on it (1,019 clips
    corpus-wide) is not a work order.

    The rate over a window IS that signal. cam07 runs a 6.4% baseline and hits
    43.1% across 60 clips over 2026-02-17..03-03 -- 25 flagged clips spread
    over 12 separate nights, blob_white_fraction 0.41-0.93, and
    post_flash_red_shift at or below 0.027 on every one of them, so it is not
    the guard's flashlight. Several of those same clips were also leaking into
    the alert channel as incident_candidate before the 2026-09-07 blinded-lens
    gate in src.classify.classify, which is corroboration rather than
    coincidence: one physical obstruction, showing up in both channels.

    Overlapping qualifying windows are merged, so a month-long obstruction
    reports as one span rather than one row per clip in it.

    `NO_MAINTENANCE_CAMERAS` is honoured here too, for consistency with
    `write_maintenance_candidates` -- though note cam04 does produce two
    strongly elevated windows (47.4% and 43.9% against a 9.1% baseline), so
    that human override is worth re-checking against this windowed view
    rather than the per-clip queue it was originally made on.
    """
    by_camera: dict[str, list[tuple[datetime, bool, Any]]] = {}
    for row in rows:
        ts = _parse_ts(row.get("timestamp"))
        if ts is None or not row.get("detected"):
            continue
        if row["camera_id"] in NO_MAINTENANCE_CAMERAS:
            continue
        flagged = is_blinding_foreground(row) and (
            row.get("post_flash_red_shift", 0.0) < FLASHLIGHT_RED_SHIFT
        )
        by_camera.setdefault(row["camera_id"], []).append((ts, flagged, row["message_id"]))

    span = timedelta(days=window_days)
    results: list[dict[str, Any]] = []
    for camera_id, series in sorted(by_camera.items()):
        series.sort(key=lambda item: item[0])
        baseline = sum(1 for _, f, _ in series if f) / len(series)
        if baseline <= 0:
            continue
        merged: list[dict[str, Any]] = []
        start = 0
        for end, (ts, _, _) in enumerate(series):
            while series[start][0] < ts - span:
                start += 1
            chunk = series[start : end + 1]
            if len(chunk) < min_clips:
                continue
            flagged = [item for item in chunk if item[1]]
            rate = len(flagged) / len(chunk)
            if rate < min_rate or rate < baseline * min_ratio:
                continue
            window = {
                "camera_id": camera_id,
                "window_start": chunk[0][0].isoformat(),
                "window_end": ts.isoformat(),
                "clips": len(chunk),
                "flagged": len(flagged),
                "rate": round(rate, 4),
                "baseline_rate": round(baseline, 4),
                "message_ids": " ".join(str(m) for _, _, m in flagged),
            }
            if merged and chunk[0][0].isoformat() <= merged[-1]["window_end"]:
                previous = merged[-1]
                previous["window_end"] = window["window_end"]
                previous["clips"] = max(previous["clips"], window["clips"])
                previous["flagged"] = max(previous["flagged"], window["flagged"])
                previous["rate"] = max(previous["rate"], window["rate"])
                previous["message_ids"] = " ".join(
                    dict.fromkeys(previous["message_ids"].split() + window["message_ids"].split())
                )
            else:
                merged.append(window)
        results.extend(merged)
    results.sort(key=lambda w: (-w["rate"], w["camera_id"]))
    return results


def write_obstruction_windows(rows: list[dict[str, Any]], *, out_path: str) -> list[dict[str, Any]]:
    windows = obstruction_windows(rows)
    columns = (
        "camera_id",
        "window_start",
        "window_end",
        "clips",
        "flagged",
        "rate",
        "baseline_rate",
        "message_ids",
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(windows)
    print(f"wrote {len(windows)} obstruction windows to {out_path}")
    return windows


def main() -> None:  # pragma: no cover - requires the real corpus/db
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=None, help="restrict to one camera id")
    parser.add_argument("--top-per-camera", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1, help="1 runs single-process")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--maintenance-out",
        default=None,
        help="also write a blinding-foreground ('camera needs cleaning') review queue here",
    )
    parser.add_argument(
        "--reference-bg",
        default="data/reference_bg",
        help="per-camera reference background directory (build with scripts/build_reference_bg.py)",
    )
    parser.add_argument(
        "--no-reference-bg",
        action="store_true",
        help="disable the reference-background scenery veto, for before/after comparison",
    )
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    cameras_cfg = load_cameras_config("config/cameras.yaml")
    conn = db.connect(app_cfg.db_path)

    reference_root = "" if args.no_reference_bg else args.reference_bg
    reference_entries = [] if args.no_reference_bg else load_manifest(args.reference_bg)
    if args.workers > 1 and args.camera is None:
        rows = _collect_features_parallel(
            conn, "config/cameras.yaml", workers=args.workers, reference_root=reference_root
        )
    else:
        rows = collect_features(
            conn,
            cameras_cfg,
            camera_id=args.camera,
            reference_entries=reference_entries,
            reference_root=args.reference_bg,
        )
    conn.close()

    print(f"detected: {sum(1 for r in rows if r.get('detected'))}/{len(rows)}")
    rank_and_write(rows, top_per_camera=args.top_per_camera, out_path=args.out)
    if args.maintenance_out:
        write_maintenance_candidates(
            rows, top_per_camera=args.top_per_camera, out_path=args.maintenance_out
        )
        write_obstruction_windows(
            rows, out_path=str(Path(args.maintenance_out).with_suffix(".windows.csv"))
        )


if __name__ == "__main__":
    main()
