"""Phase 0b: infer fence camera order from real backfilled clip timestamps.

Reads (camera_id, timestamp) pairs from the `clips` table populated by
scripts/meta_backfill.py, runs src.sequence.infer_camera_order, and prints a
human-readable hypothesis for the fence order. This is a hypothesis to confirm
against known gates/landmarks (see docs/plan.md, Phase 0b) -- it does not write
to config/cameras.yaml automatically.

Run:
    uv run python scripts/infer_camera_order.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402
from src.config import load_app_config  # noqa: E402
from src.sequence import ClipEvent, infer_camera_order, split_half_stability  # noqa: E402


def load_events(conn, *, exclude_camera_ids: frozenset[str]) -> list[ClipEvent]:
    events = []
    for row in db.iter_clips(conn):
        if row["camera_id"] in exclude_camera_ids:
            continue
        timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
        events.append(ClipEvent(camera_id=row["camera_id"], timestamp=timestamp))
    return events


def format_report(result, stability: float | None) -> str:
    lines = [
        "Inferred fence order (low confidence at the ends until confirmed against a gate):",
        "  " + " -> ".join(result.order),
        "",
        f"Reversal cost (lower is better; ~1 per patrol night is expected): {result.reversal_cost}",
        f"Cross-check agreement (spectral vs. greedy-chain): {result.cross_check_agreement:.2f}",
        f"Entry point candidates: {', '.join(result.entry_point_candidates)}",
    ]
    if result.unplaced:
        lines.append(f"Unplaced cameras (too little data to place): {', '.join(result.unplaced)}")
    if result.colocated_groups:
        lines.append("Co-located camera groups (treated as one node during inference):")
        for rep, members in result.colocated_groups.items():
            lines.append(f"  {rep}: {', '.join(members)}")
    if stability is not None:
        lines.append(f"Split-half stability: {stability:.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-unknown",
        action="store_true",
        default=True,
        help="Exclude clips whose camera_id could not be resolved (default: on).",
    )
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    conn = db.connect(app_cfg.db_path)
    exclude = frozenset({"unknown"}) if args.exclude_unknown else frozenset()
    events = load_events(conn, exclude_camera_ids=exclude)
    conn.close()

    if not events:
        print("No clip metadata found. Run scripts/meta_backfill.py first.")
        return

    result = infer_camera_order(events)

    stability: float | None
    try:
        stability = split_half_stability(events)
    except Exception:
        stability = None

    print(format_report(result, stability))


if __name__ == "__main__":
    main()
