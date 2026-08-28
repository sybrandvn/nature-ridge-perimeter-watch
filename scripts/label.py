"""Interactive ground-truth labeling CLI.

Defaults to clips that already have a downloaded video (`file_path` set), since most
backfilled rows don't and walking them produces nothing to watch. Pass --include-no-file
to fall back to the old metadata-only behaviour (label off caption/camera/timestamp alone).

Run:
    uv run python scripts/label.py [--camera CAM_ID] [--limit N] [--include-no-file]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402
from src.config import load_app_config  # noqa: E402
from src.db import VALID_LABELS  # noqa: E402

PromptFn = Callable[[Mapping], "tuple[str, str | None] | None"]  # None => quit

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


def _hyperlink(path_str: str) -> str:
    """OSC 8 terminal hyperlink so `path_str` is clickable in terminals that render it
    (VS Code, iTerm2, GNOME Terminal, Windows Terminal); plain text otherwise."""
    uri = Path(path_str).resolve().as_uri()
    return f"\033]8;;{uri}\033\\{path_str}\033]8;;\033\\"


def _resolve_label(raw: str) -> str | None:
    """Accept a full label name or its single-letter shortcut (g/a/i/e/u); else None."""
    raw = raw.strip().lower()
    if raw in VALID_LABELS:
        return raw
    return next((label for label in VALID_LABELS if raw == label[0]), None)


def default_prompt(clip: Mapping) -> tuple[str, str | None] | None:
    print(
        f"\n[{clip['channel_id']}#{clip['message_id']}] "
        f"camera={clip['camera_id']} time={clip['timestamp']}"
    )
    if clip["caption"]:
        print(f"  caption: {clip['caption']}")
    if clip["file_path"]:
        print(f"  file: {_hyperlink(clip['file_path'])}")
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
) -> int:
    labeled = 0
    rows = db.iter_unlabeled_clips(conn, camera_id=camera_id, with_file_only=with_file_only)
    for row in _prioritize([dict(row) for row in rows]):
        if limit is not None and labeled >= limit:
            break
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
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    conn = db.connect(app_cfg.db_path)
    labeled = run_labeling_session(
        conn,
        prompt_fn=default_prompt,
        limit=args.limit,
        camera_id=args.camera,
        with_file_only=not args.include_no_file,
    )
    conn.close()
    print(f"\nLabeled {labeled} clip(s).")


if __name__ == "__main__":
    main()
