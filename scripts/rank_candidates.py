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
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts.label import _event_key  # noqa: E402
from scripts.spike import FEATURE_COLUMNS, extract_clip_features  # noqa: E402
from src import db  # noqa: E402
from src.config import CamerasConfig, load_app_config, load_cameras_config  # noqa: E402
from src.features import is_daylight  # noqa: E402
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

REPORT_COLUMNS = (
    "score",
    "camera_id",
    "channel_id",
    "message_id",
    "timestamp",
    "outside_pixel_fraction",
    "aspect_ratio",
    "blob_count",
    "fence_crossed",
    "area_stability",
    "normalised_speed",
    "saturation_ratio",
    "green_light_ratio",
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
        features = extract_fn(clip["file_path"], camera.zone_at(clip["timestamp"]), **extra)
        row = {
            "channel_id": clip["channel_id"],
            "message_id": clip["message_id"],
            "camera_id": clip["camera_id"],
            "timestamp": clip["timestamp"],
            "caption": clip.get("caption"),
            "file_path": clip["file_path"],
            "label": clip["label"],
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
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return frame_count / fps if frame_count > 0 and fps > 0 else 0.0


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

    unlabelled = [r for r in detected if not r["label"]]
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


def main() -> None:  # pragma: no cover - requires the real corpus/db
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=None, help="restrict to one camera id")
    parser.add_argument("--top-per-camera", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1, help="1 runs single-process")
    parser.add_argument("--out", required=True)
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


if __name__ == "__main__":
    main()
