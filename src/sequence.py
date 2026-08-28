"""Camera-order inference from message metadata, and (later) patrol/probe analysis.

No video is needed: patrol passes are bursts of clips across several cameras in a
short window, and the correct fence order is the one under which those passes read
as long monotonic walks rather than zigzags. This module implements that as an
optimisation over (camera_id, timestamp) pairs only.

The output of infer_camera_order is a *hypothesis* to be confirmed by a human
against known gates/landmarks (see docs/plan.md, Phase 0b) — nothing here writes
to config/cameras.yaml directly.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.errors import SequenceError


@dataclass(frozen=True)
class ClipEvent:
    camera_id: str
    timestamp: float  # unix epoch seconds


@dataclass(frozen=True)
class Pass:
    events: tuple[ClipEvent, ...]  # time-ordered


@dataclass(frozen=True)
class TransitionStats:
    count: int
    median_dt: float
    iqr_dt: float


@dataclass(frozen=True)
class OrderResult:
    order: tuple[str, ...]
    unplaced: tuple[str, ...]
    colocated_groups: dict[str, tuple[str, ...]]
    reversal_cost: int
    cross_check_agreement: float
    entry_point_candidates: tuple[str, ...]
    transit_confidence: dict[tuple[str, str], TransitionStats | None]


# --------------------------------------------------------------------------
# Pass segmentation
# --------------------------------------------------------------------------


def segment_passes(
    events: Sequence[ClipEvent], *, max_gap_seconds: float, min_cameras: int
) -> list[Pass]:
    """Group events into bursts (gaps under max_gap_seconds), keeping only bursts
    that touch at least min_cameras distinct cameras — the structural signature
    of a patrol pass, independent of any CV classification."""
    ordered = sorted(events, key=lambda e: e.timestamp)
    groups: list[list[ClipEvent]] = []
    current: list[ClipEvent] = []
    for event in ordered:
        if current and (event.timestamp - current[-1].timestamp) > max_gap_seconds:
            groups.append(current)
            current = []
        current.append(event)
    if current:
        groups.append(current)

    return [
        Pass(events=tuple(group))
        for group in groups
        if len({e.camera_id for e in group}) >= min_cameras
    ]


# --------------------------------------------------------------------------
# Transition matrix
# --------------------------------------------------------------------------


def build_transition_matrix(passes: Sequence[Pass]) -> dict[frozenset[str], TransitionStats]:
    """Aggregate consecutive-in-pass camera transitions into symmetric pair stats."""
    dts_by_pair: dict[frozenset[str], list[float]] = {}
    for p in passes:
        for a, b in zip(p.events, p.events[1:], strict=False):
            if a.camera_id == b.camera_id:
                continue
            key = frozenset({a.camera_id, b.camera_id})
            dts_by_pair.setdefault(key, []).append(abs(b.timestamp - a.timestamp))

    matrix: dict[frozenset[str], TransitionStats] = {}
    for key, dts in dts_by_pair.items():
        arr = np.array(dts)
        matrix[key] = TransitionStats(
            count=len(dts),
            median_dt=float(np.median(arr)),
            iqr_dt=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        )
    return matrix


# --------------------------------------------------------------------------
# Co-located camera detection
# --------------------------------------------------------------------------


def detect_colocated(
    matrix: dict[frozenset[str], TransitionStats], *, max_median_dt: float, min_count: int
) -> dict[str, str]:
    """Union pairs that consistently trigger near-simultaneously into one group,
    represented by the alphabetically smallest camera id in the group."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for pair, stats in matrix.items():
        if stats.count >= min_count and stats.median_dt <= max_median_dt:
            a, b = tuple(pair)
            union(a, b)

    all_ids = {cid for pair in matrix for cid in pair}
    return {cid: find(cid) for cid in all_ids}


def collapse_colocated(passes: Sequence[Pass], colocated_map: dict[str, str]) -> list[Pass]:
    return [
        Pass(
            events=tuple(
                ClipEvent(
                    camera_id=colocated_map.get(e.camera_id, e.camera_id),
                    timestamp=e.timestamp,
                )
                for e in p.events
            )
        )
        for p in passes
    ]


def _group_by_representative(colocated_map: dict[str, str]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for original, rep in colocated_map.items():
        groups.setdefault(rep, []).append(original)
    return {rep: tuple(sorted(members)) for rep, members in groups.items() if len(members) > 1}


# --------------------------------------------------------------------------
# Ordering: spectral seriation + reversal-minimising refinement
# --------------------------------------------------------------------------


def spectral_order(
    camera_ids: Sequence[str], matrix: dict[frozenset[str], TransitionStats]
) -> tuple[list[str], list[str]]:
    """Linear spectral seriation via the Fiedler vector of the transition graph.

    Cameras with no observed transitions to any other camera are reported as
    unplaced rather than forced into the chain.
    """
    ids = sorted(set(camera_ids))
    n = len(ids)
    if n <= 1:
        return ids, []

    idx = {cid: i for i, cid in enumerate(ids)}
    weights = np.zeros((n, n))
    for pair, stats in matrix.items():
        a, b = tuple(pair)
        if a in idx and b in idx:
            weights[idx[a], idx[b]] = stats.count
            weights[idx[b], idx[a]] = stats.count

    degree = weights.sum(axis=1)
    placed_ids = [cid for cid in ids if degree[idx[cid]] > 0]
    unplaced_ids = [cid for cid in ids if degree[idx[cid]] == 0]

    if len(placed_ids) <= 1:
        return placed_ids, unplaced_ids

    sub_idx = {cid: i for i, cid in enumerate(placed_ids)}
    m = len(placed_ids)
    sub_weights = np.zeros((m, m))
    for pair, stats in matrix.items():
        a, b = tuple(pair)
        if a in sub_idx and b in sub_idx:
            sub_weights[sub_idx[a], sub_idx[b]] = stats.count
            sub_weights[sub_idx[b], sub_idx[a]] = stats.count

    laplacian = np.diag(sub_weights.sum(axis=1)) - sub_weights
    _, eigenvectors = np.linalg.eigh(laplacian)
    fiedler = eigenvectors[:, 1]
    order = [placed_ids[i] for i in np.argsort(fiedler)]
    return order, unplaced_ids


def total_reversals(order: Sequence[str], passes: Sequence[Pass]) -> int:
    """Cost function: count of direction flips across all passes under `order`.

    A clean out-and-back patrol under the correct order contributes exactly one
    reversal (the turn point); a wrong order fragments passes into many.
    """
    position = {cam: i for i, cam in enumerate(order)}
    total = 0
    for p in passes:
        positions = [position[e.camera_id] for e in p.events if e.camera_id in position]
        collapsed: list[int] = []
        for pos in positions:
            if not collapsed or collapsed[-1] != pos:
                collapsed.append(pos)
        if len(collapsed) < 3:
            continue
        diffs = [b - a for a, b in zip(collapsed, collapsed[1:], strict=False)]
        signs = [1 if d > 0 else -1 for d in diffs]
        total += sum(1 for s1, s2 in zip(signs, signs[1:], strict=False) if s1 != s2)
    return total


def refine_order_2opt(
    order: Sequence[str], passes: Sequence[Pass], *, max_iterations: int = 2000
) -> list[str]:
    """Local search minimising total_reversals, combining segment-reversal (2-opt)
    and single-element relocation (or-opt) moves so the search can escape local
    optima that either move type alone gets stuck in."""
    current = list(order)
    if len(current) < 3:
        return current

    best_cost = total_reversals(current, passes)
    iterations = 0
    improved = True
    while improved and iterations < max_iterations:
        improved = False
        n = len(current)

        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = current[:i] + current[i : j + 1][::-1] + current[j + 1 :]
                iterations += 1
                cost = total_reversals(candidate, passes)
                if cost < best_cost:
                    current, best_cost = candidate, cost
                    improved = True
                if iterations >= max_iterations:
                    return current

        for i in range(n):
            elem = current[i]
            rest = current[:i] + current[i + 1 :]
            for pos in range(len(rest) + 1):
                candidate = rest[:pos] + [elem] + rest[pos:]
                iterations += 1
                cost = total_reversals(candidate, passes)
                if cost < best_cost:
                    current, best_cost = candidate, cost
                    improved = True
                if iterations >= max_iterations:
                    return current
    return current


def greedy_chain_order(
    camera_ids: Sequence[str], matrix: dict[frozenset[str], TransitionStats], *, start_id: str
) -> list[str]:
    """Independent seriation method used only to cross-check spectral_order."""
    remaining = set(camera_ids) - {start_id}
    order = [start_id]
    while remaining:
        last = order[-1]
        best = max(
            remaining,
            key=lambda cid: (
                matrix[frozenset({last, cid})].count if frozenset({last, cid}) in matrix else 0,
                cid,
            ),
        )
        order.append(best)
        remaining.remove(best)
    return order


def _adjacent_pairs(order: Sequence[str]) -> set[frozenset[str]]:
    return {frozenset({a, b}) for a, b in zip(order, order[1:], strict=False)}


def order_agreement(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    """Adjacency-set overlap; invariant to overall reversal since pairs are unordered."""
    pairs_a, pairs_b = _adjacent_pairs(order_a), _adjacent_pairs(order_b)
    if not pairs_a and not pairs_b:
        return 1.0
    return len(pairs_a & pairs_b) / len(pairs_a | pairs_b)


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def entry_point_frequencies(passes: Sequence[Pass]) -> Counter[str]:
    """How often each camera is the first or last clip of a pass — candidate gates."""
    counter: Counter[str] = Counter()
    for p in passes:
        if not p.events:
            continue
        counter[p.events[0].camera_id] += 1
        counter[p.events[-1].camera_id] += 1
    return counter


def transit_time_confidence(
    order: Sequence[str], matrix: dict[frozenset[str], TransitionStats]
) -> dict[tuple[str, str], TransitionStats | None]:
    """Transition stats for each adjacent pair in the final order; None means no
    direct observations support that adjacency — a low-confidence placement."""
    return {
        (a, b): matrix.get(frozenset({a, b})) for a, b in zip(order, order[1:], strict=False)
    }


def split_half_stability(
    events: Sequence[ClipEvent],
    *,
    max_gap_seconds: float = 180.0,
    min_cameras_per_pass: int = 3,
    colocate_max_dt: float = 2.0,
    colocate_min_count: int = 3,
) -> float:
    """Infer the order independently on the first and second half of history
    (split by pass count, not wall time) and compare via order_agreement."""
    passes = segment_passes(
        events, max_gap_seconds=max_gap_seconds, min_cameras=min_cameras_per_pass
    )
    if len(passes) < 2:
        raise SequenceError("Need >= 2 passes to assess split-half stability")

    midpoint = len(passes) // 2
    first_half_events = [e for p in passes[:midpoint] for e in p.events]
    second_half_events = [e for p in passes[midpoint:] for e in p.events]

    result_a = infer_camera_order(
        first_half_events,
        max_gap_seconds=max_gap_seconds,
        min_cameras_per_pass=min_cameras_per_pass,
        colocate_max_dt=colocate_max_dt,
        colocate_min_count=colocate_min_count,
    )
    result_b = infer_camera_order(
        second_half_events,
        max_gap_seconds=max_gap_seconds,
        min_cameras_per_pass=min_cameras_per_pass,
        colocate_max_dt=colocate_max_dt,
        colocate_min_count=colocate_min_count,
    )
    return order_agreement(result_a.order, result_b.order)


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------


def infer_camera_order(
    events: Sequence[ClipEvent],
    *,
    max_gap_seconds: float = 180.0,
    min_cameras_per_pass: int = 3,
    colocate_max_dt: float = 2.0,
    colocate_min_count: int = 3,
) -> OrderResult:
    passes = segment_passes(
        events, max_gap_seconds=max_gap_seconds, min_cameras=min_cameras_per_pass
    )
    if not passes:
        raise SequenceError(
            f"No patrol-like passes found (need bursts across >= {min_cameras_per_pass} cameras "
            f"within {max_gap_seconds}s)"
        )

    matrix = build_transition_matrix(passes)
    colocated_map = detect_colocated(
        matrix, max_median_dt=colocate_max_dt, min_count=colocate_min_count
    )
    passes = collapse_colocated(passes, colocated_map)
    matrix = build_transition_matrix(passes)

    all_camera_ids = sorted({e.camera_id for p in passes for e in p.events})
    order, unplaced = spectral_order(all_camera_ids, matrix)
    order = refine_order_2opt(order, passes)

    if len(order) >= 2:
        alt_seed = greedy_chain_order(order, matrix, start_id=order[0])
        alt_order = refine_order_2opt(alt_seed, passes)
        agreement = order_agreement(order, alt_order)
    else:
        agreement = 1.0

    entry_freq = entry_point_frequencies(passes)
    top_entries = tuple(cam for cam, _ in entry_freq.most_common(2))

    return OrderResult(
        order=tuple(order),
        unplaced=tuple(unplaced),
        colocated_groups=_group_by_representative(colocated_map),
        reversal_cost=total_reversals(order, passes),
        cross_check_agreement=agreement,
        entry_point_candidates=top_entries,
        transit_confidence=transit_time_confidence(order, matrix),
    )
