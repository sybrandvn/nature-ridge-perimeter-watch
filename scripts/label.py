"""Interactive ground-truth labeling CLI.

Defaults to clips that already have a downloaded video (`file_path` set), since most
backfilled rows don't and walking them produces nothing to watch. Pass --include-no-file
to fall back to the old metadata-only behaviour (label off caption/camera/timestamp alone).

Many triggers send a very short (<2s) near-blank clip immediately, followed a few minutes
later by the real (longer) clip -- the camera waking up, nothing visible. --short-clip-seconds
flags these; after --confirm-short-count of them get the same label in a row, you're asked
once whether to bulk-apply that label to the rest without reviewing each one.

Some triggers also send two alert messages for the same physical event -- e.g. an
(Initial*) caption and a (Stopped*)/follow-up caption sharing one embedded camera
timestamp. Once you label one, its still-unlabeled sibling shows a suggested label
you can accept with Enter, or override by typing another letter/word as usual.

Omitting --camera walks every camera's remaining clips shuffled and interleaved
round-robin across cameras, so a session gets a spread instead of exhausting one
camera's queue before moving to the next. Priority clips (known rare events) still
come first, unshuffled.

Run:
    uv run python scripts/label.py [--camera CAM_ID] [--limit N] [--include-no-file]
        [--short-clip-seconds SECS] [--confirm-short-count N]
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402
from src.config import load_app_config  # noqa: E402
from src.db import VALID_LABELS  # noqa: E402

PromptFn = Callable[[Mapping], "tuple[str, str | None] | None"]  # None => quit
BulkConfirmFn = Callable[[str, int], bool]
DurationFn = Callable[[str], "float | None"]

# Known rare-class clips from README.md section 4 (crawl incident, probe, animal
# sightings), so a labeling session surfaces them early instead of behind months of
# ordinary timestamp-ordered clips. Does not pre-fill labels -- the user still labels
# every clip themselves.
PRIORITY_MESSAGE_IDS: dict[str, frozenset[int]] = {
    "cam06": frozenset({21519, 21520}),  # crawl incident, 2026-07-21
    "cam08": frozenset({4054, 4055, 7360}),  # probe (4054/4055) + animal (7360)
    "cam05": frozenset({18269}),  # animal, 2026-01-06
}

# Captions like "(Initial*) ... @ 14-03-24 20:44:22" and "(Stopped*) ... @ 14-03-24
# 20:44:22" are two separate alert messages for the same physical trigger, sharing
# this embedded camera timestamp -- not independent events.
_EVENT_TS_RE = re.compile(r"@\s*(\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _prioritize(rows: list[Mapping]) -> tuple[list[Mapping], list[Mapping]]:
    """Split into (priority, rest), each preserving timestamp order."""
    priority, rest = [], []
    for row in rows:
        known_ids = PRIORITY_MESSAGE_IDS.get(row["camera_id"], frozenset())
        (priority if row["message_id"] in known_ids else rest).append(row)
    return priority, rest


def _spread_by_camera(rows: list[Mapping], rng: random.Random) -> list[Mapping]:
    """Shuffle within each camera's clips, then round-robin across cameras, so a
    --camera-less session doesn't spend dozens of clips in a row on one camera before
    moving to the next. A no-op when rows only span one camera (i.e. --camera was set)."""
    by_camera: dict[str, list[Mapping]] = {}
    for row in rows:
        by_camera.setdefault(row["camera_id"], []).append(row)
    if len(by_camera) <= 1:
        return rows
    for group in by_camera.values():
        rng.shuffle(group)
    camera_order = list(by_camera)
    rng.shuffle(camera_order)
    result: list[Mapping] = []
    while any(by_camera[cam] for cam in camera_order):
        for cam in camera_order:
            if by_camera[cam]:
                result.append(by_camera[cam].pop(0))
    return result


def _event_key(camera_id: str, caption: str | None) -> str | None:
    """Group key for alert messages sharing one embedded camera timestamp (same
    physical trigger), or None if the caption doesn't carry one."""
    if not caption:
        return None
    match = _EVENT_TS_RE.search(caption)
    return f"{camera_id}|{match.group(1)}" if match else None


def _event_label_map(conn) -> dict[str, str]:
    """event_key -> label for clips already labeled (this or a prior session), so a
    still-unlabeled Initial/Stopped sibling can suggest the same label."""
    rows = conn.execute(
        """
        SELECT clips.camera_id, clips.caption, labels.label
        FROM clips JOIN labels
            ON clips.channel_id = labels.channel_id AND clips.message_id = labels.message_id
        """
    )
    mapping: dict[str, str] = {}
    for row in rows:
        key = _event_key(row["camera_id"], row["caption"])
        if key is not None:
            mapping[key] = row["label"]
    return mapping


# One line per src.db.VALID_LABELS entry, per docs/plan.md's Ground truth labels section.
LABEL_EXAMPLES: dict[str, str] = {
    "guard": "guard on patrol, flashlight visible, usually near/interior side",
    "animal": "an animal crossing -- not a person",
    "incident": "a person: crawling, probing, or climbing, usually far/exterior side",
    "environment": "IR-attracted insects, rain streaks, wind-blown vegetation, shadow artifacts",
    "unknown": "can't tell / too ambiguous to call confidently",
    "startup": "short blank clip, camera waking up -- not ambiguous, just empty",
}


def _resolve_label(raw: str) -> str | None:
    """Accept a full label name or its single-letter shortcut (g/a/i/e/u); else None."""
    raw = raw.strip().lower()
    if raw in VALID_LABELS:
        return raw
    return next((label for label in VALID_LABELS if raw == label[0]), None)


def _clip_duration_seconds(path: str) -> float | None:
    """Video duration in seconds, or None if the file is missing/unreadable."""
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    if not fps or not frame_count:
        return None
    return frame_count / fps


def _default_bulk_confirm(label: str, remaining: int) -> bool:  # pragma: no cover - interactive I/O
    raw = input(
        f"{remaining} more short clip(s) look like the same pattern -- label them all "
        f"'{label}' without reviewing each one? [y/N] "
    ).strip().lower()
    return raw in ("y", "yes")


def default_prompt(clip: Mapping) -> tuple[str, str | None] | None:
    print(
        f"\n[{clip['channel_id']}#{clip['message_id']}] "
        f"camera={clip['camera_id']} time={clip['timestamp']}"
    )
    if clip["caption"]:
        print(f"  caption: {clip['caption']}")
    if clip["file_path"]:
        print(f"  file: {clip['file_path']}")
        duration = _clip_duration_seconds(clip["file_path"])
        if duration is not None:
            note = "  <- often 'startup': camera waking up, usually blank" if duration < 2.0 else ""
            print(f"  duration: {duration:.1f}s{note}")
    else:
        print("  (no local file yet -- metadata-only label)")

    print("  labels:")
    for label in VALID_LABELS:
        print(f"    [{label[0]}] {label:<11} {LABEL_EXAMPLES[label]}")

    suggested = clip.get("_suggested_label")
    if suggested is not None:
        print(f"  suggested: {suggested} (same event as a labeled clip -- Enter to accept)")

    while True:
        raw = input("label (letter or full word, q to quit): ").strip().lower()
        if raw == "q":
            return None
        if raw == "" and suggested is not None:
            notes = input("notes (optional): ").strip() or None
            return suggested, notes
        resolved = _resolve_label(raw)
        if resolved is not None:
            notes = input("notes (optional): ").strip() or None
            return resolved, notes
        print(f"Not a valid label: {raw!r}")


def run_labeling_session(
    conn,
    *,
    prompt_fn: PromptFn,
    limit: int | None = None,
    camera_id: str | None = None,
    with_file_only: bool = False,
    short_clip_seconds: float | None = None,
    confirm_short_count: int = 3,
    bulk_confirm_fn: BulkConfirmFn | None = None,
    duration_fn: DurationFn = _clip_duration_seconds,
    rng: random.Random | None = None,
) -> int:
    labeled = 0
    rows = db.iter_unlabeled_clips(conn, camera_id=camera_id, with_file_only=with_file_only)
    priority, rest = _prioritize([dict(row) for row in rows])
    rest = _spread_by_camera(rest, rng or random.Random())
    rows = priority + rest

    # Priority clips are known real events -- never eligible for the short-clip bulk
    # shortcut below, even if one happens to be short (e.g. cam06's Initial alert 21519).
    priority_ids = {
        (row["channel_id"], row["message_id"])
        for row in rows
        if row["message_id"] in PRIORITY_MESSAGE_IDS.get(row["camera_id"], frozenset())
    }

    def is_short(row: Mapping) -> bool:
        if short_clip_seconds is None or not row["file_path"]:
            return False
        if (row["channel_id"], row["message_id"]) in priority_ids:
            return False
        duration = duration_fn(row["file_path"])
        return duration is not None and duration < short_clip_seconds

    short_flags = {(row["channel_id"], row["message_id"]): is_short(row) for row in rows}
    remaining_short = sum(short_flags.values())
    confirm = bulk_confirm_fn or _default_bulk_confirm
    last_short_label: str | None = None
    short_streak = 0
    bulk_label: str | None = None
    event_labels = _event_label_map(conn)

    for row in rows:
        if limit is not None and labeled >= limit:
            break
        key = (row["channel_id"], row["message_id"])
        short = short_flags[key]
        event_key = _event_key(row["camera_id"], row["caption"])

        if short and bulk_label is not None:
            db.upsert_label(
                conn,
                channel_id=row["channel_id"],
                message_id=row["message_id"],
                label=bulk_label,
                notes="bulk: matches short blank-clip pattern",
            )
            labeled += 1
            remaining_short -= 1
            if event_key is not None:
                event_labels[event_key] = bulk_label
            continue

        row["_suggested_label"] = event_labels.get(event_key) if event_key is not None else None
        result = prompt_fn(row)
        if result is None:
            break
        label, notes = result
        db.upsert_label(
            conn,
            channel_id=row["channel_id"],
            message_id=row["message_id"],
            label=label,
            notes=notes,
        )
        labeled += 1
        if event_key is not None:
            event_labels[event_key] = label

        if short:
            remaining_short -= 1
            short_streak = short_streak + 1 if label == last_short_label else 1
            last_short_label = label
            if (
                bulk_label is None
                and short_streak == confirm_short_count
                and remaining_short > 0
                and confirm(label, remaining_short)
            ):
                bulk_label = label
    return labeled


def main() -> None:  # pragma: no cover - interactive I/O
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=None, help="Restrict to one camera id")
    parser.add_argument("--limit", type=int, default=None, help="Max clips to label this run")
    parser.add_argument(
        "--include-no-file",
        action="store_true",
        help="Also walk clips with no downloaded video (metadata-only labeling)",
    )
    parser.add_argument(
        "--short-clip-seconds",
        type=float,
        default=2.0,
        help="Flag clips shorter than this as likely blank pre-alert clips (0 disables)",
    )
    parser.add_argument(
        "--confirm-short-count",
        type=int,
        default=3,
        help="Short clips to confirm individually before offering to bulk-label the rest",
    )
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    conn = db.connect(app_cfg.db_path)
    labeled = run_labeling_session(
        conn,
        prompt_fn=default_prompt,
        limit=args.limit,
        camera_id=args.camera,
        with_file_only=not args.include_no_file,
        short_clip_seconds=args.short_clip_seconds or None,
        confirm_short_count=args.confirm_short_count,
    )
    conn.close()
    print(f"\nLabeled {labeled} clip(s).")


if __name__ == "__main__":
    main()
