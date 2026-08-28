"""Interactive ground-truth labeling CLI.

Defaults to clips that already have a downloaded video (`file_path` set), since most
backfilled rows don't and walking them produces nothing to watch. Pass --include-no-file
to fall back to the old metadata-only behaviour (label off caption/camera/timestamp alone).

Many triggers send a very short (<2s) near-blank clip immediately, followed a few minutes
later by the real (longer) clip -- the camera waking up, nothing visible. --short-clip-seconds
flags these; after --confirm-short-count of them get the same label in a row, you're asked
once whether to bulk-apply that label to the rest without reviewing each one.

Run:
    uv run python scripts/label.py [--camera CAM_ID] [--limit N] [--include-no-file]
        [--short-clip-seconds SECS] [--confirm-short-count N]
"""

from __future__ import annotations

import argparse
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


def _prioritize(rows: list[Mapping]) -> list[Mapping]:
    """Move PRIORITY_MESSAGE_IDS clips to the front, preserving timestamp order elsewhere."""
    priority, rest = [], []
    for row in rows:
        known_ids = PRIORITY_MESSAGE_IDS.get(row["camera_id"], frozenset())
        (priority if row["message_id"] in known_ids else rest).append(row)
    return priority + rest


# One line per src.db.VALID_LABELS entry, per docs/plan.md's Ground truth labels section.
LABEL_EXAMPLES: dict[str, str] = {
    "guard": "guard on patrol, flashlight visible, usually near/interior side",
    "animal": "an animal crossing -- not a person",
    "incident": "a person: crawling, probing, or climbing, usually far/exterior side",
    "environment": "IR-attracted insects, rain streaks, wind-blown vegetation, shadow artifacts",
    "unknown": "can't tell / too ambiguous to call confidently",
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
            note = "  <- often blank/pre-alert, camera waking up" if duration < 2.0 else ""
            print(f"  duration: {duration:.1f}s{note}")
    else:
        print("  (no local file yet -- metadata-only label)")

    print("  labels:")
    for label in VALID_LABELS:
        print(f"    [{label[0]}] {label:<11} {LABEL_EXAMPLES[label]}")

    while True:
        raw = input("label (letter or full word, q to quit): ").strip().lower()
        if raw == "q":
            return None
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
) -> int:
    labeled = 0
    rows = db.iter_unlabeled_clips(conn, camera_id=camera_id, with_file_only=with_file_only)
    rows = _prioritize([dict(row) for row in rows])

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

    for row in rows:
        if limit is not None and labeled >= limit:
            break
        key = (row["channel_id"], row["message_id"])
        short = short_flags[key]

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
            continue

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
