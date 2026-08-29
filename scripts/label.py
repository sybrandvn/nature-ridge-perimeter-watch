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

The earlier half of such a pair is often not an independent blank clip at all --
it's a literal frame-for-frame prefix of the later clip (same recording, sent early
as a preview). Before an interactive session starts, any unlabeled earlier-half clip
whose frames match the paired clip's start (mean pixel difference near zero across
every frame) is auto-labeled `startup` and never prompted for. Known rare-event clips
(PRIORITY_MESSAGE_IDS) are never auto-labeled this way even if they happen to match.
Pass --no-startup-overlap-check to disable and review these by hand instead.

`startup` only says a clip is a redundant early duplicate -- it says nothing about
whether the clip's own content would be usable if that's all a real-time system had
(no future clip to compare against yet). `startup_clear` and `startup_blank` capture
that: some short/early clips are actually clear enough to make out the subject on
their own, some are genuinely blank. Use --relabel-label startup to re-walk clips
already labeled `startup` (or any other label) and re-triage each by hand into
whichever label actually fits, without touching the rest of the unlabeled queue.

Omitting --camera walks every camera's remaining clips shuffled and interleaved
round-robin across cameras, so a session gets a spread instead of exhausting one
camera's queue before moving to the next. Priority clips (known rare events) still
come first, unshuffled.

Pass --message-ids-file to review a curated shortlist instead (e.g. candidates from
scripts/backtest.py) -- one message_id per line, reviewed in the order given rather
than timestamp/round-robin order. Already-labeled ids in the file are skipped.

Run:
    uv run python scripts/label.py [--camera CAM_ID] [--limit N] [--include-no-file]
        [--short-clip-seconds SECS] [--confirm-short-count N] [--no-startup-overlap-check]
        [--relabel-label LABEL] [--message-ids-file PATH]
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402
from src.config import load_app_config  # noqa: E402
from src.db import VALID_LABELS  # noqa: E402

PromptFn = Callable[[Mapping], "tuple[str, str | None] | None"]  # None => quit
BulkConfirmFn = Callable[[str, int], bool]
DurationFn = Callable[[str], "float | None"]
FrameMatchFn = Callable[[str, str], bool]

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


def _event_clips_map(conn) -> dict[str, list[Mapping]]:
    """event_key -> every clip sharing that embedded-timestamp event (any label
    state), ordered earliest-first by message timestamp. The earliest clip in a pair
    is the one that can be an auto-labeled `startup` prefix duplicate; unlike
    `_event_label_map` this doesn't require either clip to be labeled yet."""
    mapping: dict[str, list[Mapping]] = {}
    for row in conn.execute("SELECT * FROM clips ORDER BY timestamp"):
        key = _event_key(row["camera_id"], row["caption"])
        if key is not None:
            mapping.setdefault(key, []).append(row)
    return mapping


def _read_frames(path: str, limit: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    try:
        while len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _frame_mad(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        a = cv2.resize(a, (b.shape[1], b.shape[0]))
    return float(np.mean(np.abs(a.astype(int) - b.astype(int))))


def _frames_prefix_match(
    short_path: str, long_path: str, *, mad_threshold: float = 4.0, max_frames: int = 10
) -> bool:
    """True if every frame of `short_path` matches (low pixel difference) the
    same-index frame of `long_path` -- i.e. the shorter clip is a literal
    recording-start duplicate of the longer one, not independent footage.

    Confirmed empirically 2026-08-29 on cam06/cam08/cam15 Initial/Stopped pairs:
    mean absolute pixel difference stays under 3 across every frame of the shorter
    clip when it's a genuine duplicate start, versus jumping past 15 by frame 2-3
    on a pair that just happens to share an embedded timestamp but isn't (one
    cam01b pair had corrupt fps metadata and diverged immediately)."""
    short_frames = _read_frames(short_path, max_frames)
    if not short_frames:
        return False
    long_frames = _read_frames(long_path, len(short_frames))
    if len(long_frames) < len(short_frames):
        return False
    return all(_frame_mad(s, long_frames[i]) < mad_threshold for i, s in enumerate(short_frames))


def _apply_startup_prefix_duplicates(
    conn,
    *,
    camera_id: str | None = None,
    frame_match_fn: FrameMatchFn = _frames_prefix_match,
) -> int:
    """Auto-label the earlier half of an Initial/Stopped pair as `startup` when its
    frames are a literal duplicate of the paired clip's start, so it never needs a
    human to watch it. Skips clips already labeled (handled by iter_unlabeled_clips),
    clips with no downloaded file, and known rare-event clips (PRIORITY_MESSAGE_IDS)
    even if they'd otherwise match -- a known real event is never auto-labeled."""
    event_clips = _event_clips_map(conn)
    applied = 0
    for row in db.iter_unlabeled_clips(conn, camera_id=camera_id, with_file_only=True):
        if row["message_id"] in PRIORITY_MESSAGE_IDS.get(row["camera_id"], frozenset()):
            continue
        key = _event_key(row["camera_id"], row["caption"])
        if key is None:
            continue
        siblings = event_clips.get(key, [])
        if len(siblings) < 2:
            continue
        earliest, partner = siblings[0], siblings[1]
        if (earliest["channel_id"], earliest["message_id"]) != (
            row["channel_id"],
            row["message_id"],
        ):
            continue  # only the earlier half of the pair is a startup candidate
        if not partner["file_path"]:
            continue
        if not frame_match_fn(row["file_path"], partner["file_path"]):
            continue
        db.upsert_label(
            conn,
            channel_id=row["channel_id"],
            message_id=row["message_id"],
            label="startup",
            notes=f"auto: frame-identical prefix of {partner['camera_id']}#{partner['message_id']}",
        )
        applied += 1
    return applied


# One line per src.db.VALID_LABELS entry, per docs/plan.md's Ground truth labels section.
LABEL_EXAMPLES: dict[str, str] = {
    "guard": "guard on patrol, flashlight visible, usually near/interior side",
    "animal": "an animal crossing -- not a person",
    "incident": "a person: crawling, probing, or climbing, usually far/exterior side",
    "environment": "IR-attracted insects, rain streaks, wind-blown vegetation, shadow artifacts",
    "unknown": "can't tell / too ambiguous to call confidently",
    "startup": "exact frame-duplicate of a later clip's start (auto-detected, rarely hand-picked)",
    "startup_clear": "short/early pre-alert clip, but clear enough to make out the subject",
    "startup_blank": "short/early pre-alert clip, genuinely nothing visible",
}

# Distinct single-letter shortcut per label -- can't derive from label[0] since startup,
# startup_clear, and startup_blank all start with 's'.
LABEL_SHORTCUTS: dict[str, str] = {
    "guard": "g",
    "animal": "a",
    "incident": "i",
    "environment": "e",
    "unknown": "u",
    "startup": "s",
    "startup_clear": "c",
    "startup_blank": "b",
}


def _resolve_label(raw: str) -> str | None:
    """Accept a full label name or its LABEL_SHORTCUTS letter; else None."""
    raw = raw.strip().lower()
    if raw in VALID_LABELS:
        return raw
    return next((label for label, letter in LABEL_SHORTCUTS.items() if raw == letter), None)


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
            note = (
                "  <- often startup_clear/startup_blank: camera waking up"
                if duration < 2.0
                else ""
            )
            print(f"  duration: {duration:.1f}s{note}")
    else:
        print("  (no local file yet -- metadata-only label)")

    print("  labels:")
    for label in VALID_LABELS:
        print(f"    [{LABEL_SHORTCUTS[label]}] {label:<14} {LABEL_EXAMPLES[label]}")

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
    detect_prefix_duplicates: bool = False,
    frame_match_fn: FrameMatchFn = _frames_prefix_match,
    relabel_label: str | None = None,
    message_ids: list[int] | None = None,
    rng: random.Random | None = None,
) -> int:
    """`relabel_label`, if set, re-walks clips that ALREADY carry that label (e.g.
    `startup`) instead of unlabeled clips -- for re-triaging a prior label into a
    finer-grained one (e.g. splitting `startup` into `startup_clear`/`startup_blank`).
    Disables `detect_prefix_duplicates` (auto-labeling only applies to unlabeled clips).

    `message_ids`, if set, restricts the queue to exactly those clips (any camera),
    in the order given -- e.g. a curated screening shortlist -- instead of the usual
    timestamp/round-robin ordering. Already-labeled clips in the list are skipped.
    Disables `detect_prefix_duplicates` too (the full-corpus auto-scan is irrelevant
    to a small curated list, and pointlessly widens the window for a concurrent
    writer -- e.g. scripts/download_clips.py -- to collide on the db).
    """
    labeled = 0
    auto_labeled = 0
    if detect_prefix_duplicates and relabel_label is None and message_ids is None:
        auto_labeled = _apply_startup_prefix_duplicates(
            conn, camera_id=camera_id, frame_match_fn=frame_match_fn
        )
    if message_ids is not None:
        by_id = {row["message_id"]: dict(row) for row in db.iter_clips(conn)}
        rows = []
        for mid in message_ids:
            row = by_id.get(mid)
            if row is None:
                continue
            if with_file_only and not row["file_path"]:
                continue
            if db.get_label(conn, row["channel_id"], row["message_id"]) is not None:
                continue
            rows.append(row)
    else:
        if relabel_label is not None:
            rows = db.iter_clips_with_label(
                conn, relabel_label, camera_id=camera_id, with_file_only=with_file_only
            )
        else:
            rows = db.iter_unlabeled_clips(
                conn, camera_id=camera_id, with_file_only=with_file_only
            )
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
    return labeled + auto_labeled


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
    parser.add_argument(
        "--no-startup-overlap-check",
        action="store_true",
        help=(
            "Disable auto-labeling the earlier half of an Initial/Stopped pair as "
            "'startup' when its frames are a literal duplicate of the paired clip's start"
        ),
    )
    parser.add_argument(
        "--relabel-label",
        default=None,
        choices=VALID_LABELS,
        help=(
            "Re-walk clips that already carry this label instead of unlabeled clips, "
            "e.g. --relabel-label startup to split prior 'startup' rows into "
            "startup_clear/startup_blank"
        ),
    )
    parser.add_argument(
        "--message-ids-file",
        default=None,
        help=(
            "Review exactly the message_ids listed in this file (one per line, "
            "# comments/blank lines ignored), in the order given, instead of the usual "
            "unlabeled queue -- e.g. a scripts/backtest.py screening shortlist"
        ),
    )
    args = parser.parse_args()
    if args.message_ids_file and args.relabel_label:
        parser.error("--message-ids-file and --relabel-label are mutually exclusive")

    message_ids = None
    if args.message_ids_file:
        text = Path(args.message_ids_file).read_text()
        message_ids = [
            int(line.split("#", 1)[0].strip())
            for line in text.splitlines()
            if line.split("#", 1)[0].strip()
        ]

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
        detect_prefix_duplicates=not args.no_startup_overlap_check,
        relabel_label=args.relabel_label,
        message_ids=message_ids,
    )
    conn.close()
    print(f"\nLabeled {labeled} clip(s).")


if __name__ == "__main__":
    main()
