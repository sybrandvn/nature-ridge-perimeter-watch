"""Resolve sibling clips into a security verdict and a communication state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.classify import classify_event

URGENT_CATEGORIES = frozenset({"animal_candidate", "incident_candidate"})
PRELIMINARY_CATEGORIES = frozenset({"incident_candidate"})


@dataclass(frozen=True)
class EventResolution:
    category: str
    reason: str
    resolution_state: str
    representative: Mapping[str, Any]
    notification_representative: Mapping[str, Any]
    initial_category: str | None
    complete_category: str | None


def _category(rows: Sequence[Mapping[str, Any]]) -> str | None:
    return None if not rows else classify_event(row["category"] for row in rows)


def resolve_event(clips: Sequence[Mapping[str, Any]]) -> EventResolution:
    """Keep the conservative verdict while describing sibling agreement honestly.

    A completed guard clip is positive benign evidence and can produce a
    ``likely_resolved`` communication. Blank/environment completions are only
    ``conflicting`` because they cannot prove what left the frame after startup.
    """
    if not clips:
        raise ValueError("cannot resolve an event without clips")

    category = classify_event(row["category"] for row in clips)
    matches = [row for row in clips if row["category"] == category]
    representative = max(
        matches,
        key=lambda row: (row["phase"] == "complete", row["message_id"]),
    )
    initial = [row for row in clips if row["phase"] == "initial"]
    complete = [row for row in clips if row["phase"] == "complete"]
    standalone = [row for row in clips if row["phase"] == "standalone"]
    initial_category = _category(initial)
    complete_category = _category(complete)

    notification_representative = representative
    if initial_category in URGENT_CATEGORIES:
        if not complete:
            resolution_state = "unconfirmed"
        else:
            notification_representative = max(complete, key=lambda row: row["message_id"])
            if complete_category in URGENT_CATEGORIES:
                resolution_state = "confirmed"
            elif complete_category == "guard_candidate":
                resolution_state = "likely_resolved"
            else:
                resolution_state = "conflicting"
    elif complete or standalone:
        resolution_state = "confirmed"
    else:
        resolution_state = "unconfirmed"

    return EventResolution(
        category=category,
        reason=str(representative["reason"]),
        resolution_state=resolution_state,
        representative=representative,
        notification_representative=notification_representative,
        initial_category=initial_category,
        complete_category=complete_category,
    )
