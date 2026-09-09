"""Cross-camera corroboration for environment/storm candidate discovery.

A single clip's `blob_count > 10` (`src.classify.classify`'s
`environment_candidate` rule) already identifies scattered/storm-like motion
reasonably well in isolation (AUC ~0.86-0.93 environment-vs-guard, depending on
label volume -- see repo memory). This module adds a second, independent axis:
whether a NEIGHBOURING camera (physically adjacent along the fence, per
`config/cameras.yaml`'s `order`) also reads `environment_candidate` within a
short time window. Real wind/storm activity tends to affect more than one
camera's field of view at once; a lone camera's own noise does not.

Validated 2026-09-04 on this repo's real corpus: at the default
`window_minutes=15, neighbor_distance=2`, two independent review batches found
100% precision (11/11 then correctly `environment`). Widening the search
(e.g. `window_minutes=30, neighbor_distance=3`) finds more real events but also
picks up a different false-positive mode: a guard walking a stretch of
adjacent cameras triggers them in sequence within the window, which looks
identical in shape to a real multi-camera storm. Always review the output
visually (`scripts/label.py --message-ids-file`) before trusting it -- this
module only proposes candidates, it never labels anything.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta


def _parse(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


@dataclass(frozen=True)
class ClipSignal:
    """One clip's identity, timing and whether it independently reads as a
    storm/environment candidate -- the minimal input this module needs."""

    camera_id: str
    message_id: int
    timestamp: str
    is_candidate: bool


def neighbor_cameras(camera_id: str, order: dict[str, int], *, distance: int) -> list[str]:
    """Other camera ids within `distance` fence positions of `camera_id`.

    Physical adjacency stands in for "could plausibly share a weather event",
    since this repo has no other geographic model of the property. A camera
    missing from `order` (not yet configured) has no neighbours.
    """
    own_order = order.get(camera_id)
    if own_order is None:
        return []
    return [
        other_id
        for other_id, other_order in order.items()
        if other_id != camera_id and abs(other_order - own_order) <= distance
    ]


def find_corroborated_events(
    clips: list[ClipSignal],
    order: dict[str, int],
    *,
    window_minutes: float,
    neighbor_distance: int,
) -> list[list[ClipSignal]]:
    """Group candidate clips into multi-camera events.

    Two candidate clips are linked if they are from order-adjacent cameras and
    fall within `window_minutes` of each other; linked clips are merged
    transitively (connected components), so a chain of overlapping pairs
    across 3+ cameras collapses into one event rather than several
    overlapping ones. Non-candidate clips (`is_candidate=False`) are ignored
    entirely -- they can never anchor or join an event, only a camera's own
    `environment_candidate` clips can.

    Returned events are sorted by their earliest member's timestamp; each
    event's own members are sorted chronologically. A clip with no
    corroborating neighbour within the window does not appear in any event.
    """
    candidates = [c for c in clips if c.is_candidate]
    by_camera: dict[str, list[ClipSignal]] = {}
    for clip in candidates:
        by_camera.setdefault(clip.camera_id, []).append(clip)
    for camera_clips in by_camera.values():
        camera_clips.sort(key=lambda c: _parse(c.timestamp))

    def key_of(clip: ClipSignal) -> tuple[str, int]:
        return (clip.camera_id, clip.message_id)

    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(key: tuple[str, int]) -> tuple[str, int]:
        while parent.get(key, key) != key:
            parent[key] = parent.get(parent[key], parent[key])
            key = parent[key]
        return key

    def union(a: tuple[str, int], b: tuple[str, int]) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    row_by_key = {key_of(clip): clip for clip in candidates}
    involved: set[tuple[str, int]] = set()

    for clip in candidates:
        ts = _parse(clip.timestamp)
        lo = ts - timedelta(minutes=window_minutes)
        hi = ts + timedelta(minutes=window_minutes)
        for neighbor_id in neighbor_cameras(clip.camera_id, order, distance=neighbor_distance):
            neighbor_clips = by_camera.get(neighbor_id)
            if not neighbor_clips:
                continue
            times = [_parse(c.timestamp) for c in neighbor_clips]
            i = bisect_left(times, lo)
            while i < len(neighbor_clips) and times[i] <= hi:
                other = neighbor_clips[i]
                key_a, key_b = key_of(clip), key_of(other)
                parent.setdefault(key_a, key_a)
                parent.setdefault(key_b, key_b)
                involved.add(key_a)
                involved.add(key_b)
                union(key_a, key_b)
                i += 1

    groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for key in involved:
        groups.setdefault(find(key), []).append(key)

    events = [
        sorted((row_by_key[key] for key in keys), key=lambda c: _parse(c.timestamp))
        for keys in groups.values()
    ]
    events.sort(key=lambda members: _parse(members[0].timestamp))
    return events
