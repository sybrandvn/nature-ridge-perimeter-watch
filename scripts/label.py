"""Interactive ground-truth labeling CLI.

Metadata-only for now: until scripts/meta_backfill.py's future video-download
step populates `file_path`, this labels off caption/camera/timestamp context
alone. Real visual labeling becomes practical once clips have local files.

Run:
    uv run python scripts/label.py [--camera CAM_ID] [--limit N]
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


def default_prompt(clip: Mapping) -> tuple[str, str | None] | None:
    print(
        f"\n[{clip['channel_id']}#{clip['message_id']}] "
        f"camera={clip['camera_id']} time={clip['timestamp']}"
    )
    if clip["caption"]:
        print(f"  caption: {clip['caption']}")
    if clip["file_path"]:
        print(f"  file: {clip['file_path']}")
    else:
        print("  (no local file yet -- metadata-only label)")

    while True:
        raw = input(f"label [{'/'.join(VALID_LABELS)}] (q to quit): ").strip().lower()
        if raw == "q":
            return None
        if raw in VALID_LABELS:
            notes = input("notes (optional): ").strip() or None
            return raw, notes
        print(f"Not a valid label: {raw!r}")


def run_labeling_session(
    conn,
    *,
    prompt_fn: PromptFn,
    limit: int | None = None,
    camera_id: str | None = None,
) -> int:
    labeled = 0
    for row in db.iter_unlabeled_clips(conn, camera_id=camera_id):
        if limit is not None and labeled >= limit:
            break
        result = prompt_fn(dict(row))
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
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    conn = db.connect(app_cfg.db_path)
    labeled = run_labeling_session(
        conn, prompt_fn=default_prompt, limit=args.limit, camera_id=args.camera
    )
    conn.close()
    print(f"\nLabeled {labeled} clip(s).")


if __name__ == "__main__":
    main()
