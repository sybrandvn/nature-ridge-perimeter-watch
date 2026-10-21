import random

import pytest

from src.errors import SequenceError
from src.sequence import (
    ClipEvent,
    Pass,
    build_transition_matrix,
    detect_colocated,
    entry_point_frequencies,
    greedy_chain_order,
    infer_camera_order,
    order_agreement,
    refine_order_2opt,
    segment_passes,
    spectral_order,
    split_half_stability,
    total_reversals,
)

FENCE_ORDER = ["camA", "camB", "camC", "camD", "camE"]


def simulate_patrol_nights(
    order: list[str],
    *,
    nights: int,
    step_seconds: float = 30.0,
    jitter_seconds: float = 4.0,
    night_gap_seconds: float = 8 * 3600,
    seed: int = 0,
) -> list[ClipEvent]:
    """Guard walks out along `order` then back, once per night."""
    rng = random.Random(seed)
    events: list[ClipEvent] = []
    t = 0.0
    for _ in range(nights):
        walk = order + list(reversed(order[:-1]))
        for camera_id in walk:
            t += step_seconds + rng.uniform(-jitter_seconds, jitter_seconds)
            events.append(ClipEvent(camera_id=camera_id, timestamp=t))
        t += night_gap_seconds
    return events


# --------------------------------------------------------------------------
# segment_passes
# --------------------------------------------------------------------------


def test_segment_passes_splits_on_gap():
    events = [
        ClipEvent("camA", 0),
        ClipEvent("camB", 10),
        ClipEvent("camC", 20),
        ClipEvent("camA", 1000),
        ClipEvent("camB", 1010),
        ClipEvent("camC", 1020),
    ]
    passes = segment_passes(events, max_gap_seconds=60, min_cameras=3)
    assert len(passes) == 2
    assert len(passes[0].events) == 3
    assert len(passes[1].events) == 3


def test_segment_passes_drops_single_camera_bursts():
    events = [ClipEvent("camA", 0), ClipEvent("camA", 5), ClipEvent("camA", 10)]
    passes = segment_passes(events, max_gap_seconds=60, min_cameras=3)
    assert passes == []


def test_segment_passes_requires_min_distinct_cameras():
    events = [ClipEvent("camA", 0), ClipEvent("camB", 5)]
    passes = segment_passes(events, max_gap_seconds=60, min_cameras=3)
    assert passes == []


# --------------------------------------------------------------------------
# build_transition_matrix / detect_colocated
# --------------------------------------------------------------------------


def test_build_transition_matrix_counts_and_median():
    passes = [
        Pass(events=(ClipEvent("A", 0), ClipEvent("B", 10), ClipEvent("A", 20))),
        Pass(events=(ClipEvent("A", 100), ClipEvent("B", 112))),
    ]
    matrix = build_transition_matrix(passes)
    key = frozenset({"A", "B"})
    assert matrix[key].count == 3
    assert matrix[key].median_dt == pytest.approx(10.0)


def test_build_transition_matrix_ignores_self_transitions():
    passes = [Pass(events=(ClipEvent("A", 0), ClipEvent("A", 5), ClipEvent("B", 10)))]
    matrix = build_transition_matrix(passes)
    assert frozenset({"A"}) not in matrix
    assert frozenset({"A", "B"}) in matrix


def test_detect_colocated_groups_frequent_near_zero_dt_pair():
    passes = [
        Pass(
            events=(
                ClipEvent("gate1a", t),
                ClipEvent("gate1b", t + 0.2),
                ClipEvent("camC", t + 30),
            )
        )
        for t in (0, 100, 200, 300)
    ]
    matrix = build_transition_matrix(passes)
    grouping = detect_colocated(matrix, max_median_dt=1.0, min_count=3)
    assert grouping["gate1a"] == grouping["gate1b"]


def test_detect_colocated_leaves_distant_pairs_ungrouped():
    passes = [Pass(events=(ClipEvent("A", 0), ClipEvent("B", 30))) for _ in range(5)]
    matrix = build_transition_matrix(passes)
    grouping = detect_colocated(matrix, max_median_dt=1.0, min_count=3)
    assert grouping["A"] != grouping["B"]


# --------------------------------------------------------------------------
# total_reversals / order_agreement
# --------------------------------------------------------------------------


def test_total_reversals_out_and_back_is_one_per_pass():
    passes = [
        Pass(events=tuple(ClipEvent(c, i) for i, c in enumerate(["A", "B", "C", "B", "A"])))
        for _ in range(3)
    ]
    assert total_reversals(["A", "B", "C"], passes) == 3


def test_total_reversals_wrong_order_is_higher():
    passes = [Pass(events=tuple(ClipEvent(c, i) for i, c in enumerate(["A", "B", "C", "B", "A"])))]
    correct = total_reversals(["A", "B", "C"], passes)
    wrong = total_reversals(["B", "A", "C"], passes)
    assert wrong > correct


def test_order_agreement_identical_and_reversed_orders():
    assert order_agreement(["A", "B", "C"], ["A", "B", "C"]) == 1.0
    assert order_agreement(["A", "B", "C"], ["C", "B", "A"]) == 1.0


def test_order_agreement_disjoint_orders_is_low():
    score = order_agreement(["A", "B", "C", "D"], ["A", "C", "B", "D"])
    assert score < 1.0


# --------------------------------------------------------------------------
# spectral_order / refine_order_2opt / greedy_chain_order
# --------------------------------------------------------------------------


def test_spectral_order_recovers_simple_path():
    passes = simulate_patrol_nights(FENCE_ORDER, nights=10, jitter_seconds=0)
    passes_grouped = segment_passes(passes, max_gap_seconds=180, min_cameras=3)
    matrix = build_transition_matrix(passes_grouped)
    order, unplaced = spectral_order(FENCE_ORDER, matrix)
    assert unplaced == []
    assert order_agreement(order, FENCE_ORDER) == 1.0


def test_spectral_order_reports_disconnected_camera_as_unplaced():
    passes = [Pass(events=(ClipEvent("A", 0), ClipEvent("B", 10), ClipEvent("C", 20)))]
    matrix = build_transition_matrix(passes)
    order, unplaced = spectral_order(["A", "B", "C", "Z"], matrix)
    assert "Z" in unplaced
    assert "Z" not in order


def test_greedy_chain_order_prefers_stronger_edges():
    matrix = build_transition_matrix(
        [
            Pass(events=(ClipEvent("A", 0), ClipEvent("B", 10))),
            Pass(events=(ClipEvent("A", 100), ClipEvent("B", 110))),
            Pass(events=(ClipEvent("A", 200), ClipEvent("C", 210))),
        ]
    )
    order = greedy_chain_order(["A", "B", "C"], matrix, start_id="A")
    assert order[1] == "B"  # A-B has 2 observations vs A-C's 1


def test_refine_order_2opt_fixes_a_mildly_perturbed_order():
    events = simulate_patrol_nights(FENCE_ORDER, nights=15, jitter_seconds=2, seed=1)
    passes = segment_passes(events, max_gap_seconds=180, min_cameras=3)
    perturbed = ["camA", "camC", "camB", "camD", "camE"]  # one adjacent transposition
    refined = refine_order_2opt(perturbed, passes)
    assert order_agreement(refined, FENCE_ORDER) == 1.0


def test_refine_order_2opt_recovers_from_full_shuffle():
    events = simulate_patrol_nights(FENCE_ORDER, nights=15, jitter_seconds=2, seed=1)
    passes = segment_passes(events, max_gap_seconds=180, min_cameras=3)
    shuffled = ["camC", "camA", "camE", "camB", "camD"]
    refined = refine_order_2opt(shuffled, passes)
    assert order_agreement(refined, FENCE_ORDER) == 1.0


# --------------------------------------------------------------------------
# entry_point_frequencies
# --------------------------------------------------------------------------


def test_entry_point_frequencies_identifies_endpoints():
    # simulate_patrol_nights walks out-and-back to the SAME starting camera each
    # night (a single-gate round trip), so camA dominates as both start and end.
    events = simulate_patrol_nights(FENCE_ORDER, nights=20, jitter_seconds=2, seed=2)
    passes = segment_passes(events, max_gap_seconds=180, min_cameras=3)
    freq = entry_point_frequencies(passes)
    assert freq.most_common(1)[0][0] == "camA"


def test_entry_point_frequencies_identifies_both_gates_when_alternating():
    # Some nights enter at camA and exit at camE (one-way); others the reverse —
    # modelling two active perimeter gates rather than a single round-trip point.
    events: list[ClipEvent] = []
    t = 0.0
    for night in range(20):
        walk = FENCE_ORDER if night % 2 == 0 else list(reversed(FENCE_ORDER))
        for camera_id in walk:
            t += 30.0
            events.append(ClipEvent(camera_id, t))
        t += 8 * 3600
    passes = segment_passes(events, max_gap_seconds=180, min_cameras=3)
    freq = entry_point_frequencies(passes)
    assert {cam for cam, _ in freq.most_common(2)} == {"camA", "camE"}


# --------------------------------------------------------------------------
# infer_camera_order (end-to-end, gate 1)
# --------------------------------------------------------------------------


def test_infer_camera_order_recovers_ground_truth_with_noise():
    events = simulate_patrol_nights(FENCE_ORDER, nights=25, jitter_seconds=3, seed=3)

    # Inject noise: isolated single-camera "animal sighting" clips that must not
    # be mistaken for patrol structure (segment_passes' min_cameras filter handles this).
    rng = random.Random(42)
    noise = [
        ClipEvent(rng.choice(FENCE_ORDER), rng.uniform(0, events[-1].timestamp)) for _ in range(30)
    ]

    result = infer_camera_order(events + noise)

    assert order_agreement(result.order, FENCE_ORDER) == 1.0
    assert set(result.order) == set(FENCE_ORDER)
    assert result.unplaced == ()
    assert result.cross_check_agreement >= 0.8
    assert set(result.entry_point_candidates) == {"camA", "camE"}
    # One reversal (the turn point) per night is the expected minimum.
    assert result.reversal_cost <= 25 + 5


def test_infer_camera_order_handles_missing_camera():
    partial_order = [c for c in FENCE_ORDER if c != "camC"]
    events = simulate_patrol_nights(partial_order, nights=20, jitter_seconds=2, seed=4)

    result = infer_camera_order(events)

    assert "camC" not in result.order
    assert "camC" not in result.unplaced  # never observed at all, not merely disconnected
    assert order_agreement(result.order, partial_order) == 1.0


def test_infer_camera_order_collapses_colocated_gate_cameras():
    # camA - gate1a/gate1b (co-located, ~0.3s apart) - camD - camE, walked out and back.
    order_with_gate_pair = ["camA", "gate1a", "gate1b", "camD", "camE"]
    step_seconds = {
        ("camA", "gate1a"): 30.0,
        ("gate1a", "gate1b"): 0.3,
        ("gate1b", "camD"): 30.0,
        ("camD", "camE"): 30.0,
    }

    def build_night(start_t: float) -> list[ClipEvent]:
        walk = order_with_gate_pair + list(reversed(order_with_gate_pair[:-1]))
        events = []
        t = start_t
        for a, b in zip(walk, walk[1:], strict=False):
            key = (a, b) if (a, b) in step_seconds else (b, a)
            t += step_seconds.get(key, 30.0)
            events.append(ClipEvent(b, t))
        return events

    events = [ClipEvent(order_with_gate_pair[0], 0.0)]
    t = 0.0
    for _ in range(20):
        events += build_night(t)
        t = events[-1].timestamp + 8 * 3600

    result = infer_camera_order(events, colocate_max_dt=2.0, colocate_min_count=3)
    grouped_ids = {cid for members in result.colocated_groups.values() for cid in members}
    assert {"gate1a", "gate1b"} <= grouped_ids

    strict_result = infer_camera_order(events, colocate_max_dt=0.05, colocate_min_count=3)
    assert strict_result.colocated_groups == {}


def test_infer_camera_order_raises_when_no_passes_found():
    events = [ClipEvent("camA", 0), ClipEvent("camA", 5)]
    with pytest.raises(SequenceError):
        infer_camera_order(events)


def test_split_half_stability_high_for_consistent_patrols():
    events = simulate_patrol_nights(FENCE_ORDER, nights=30, jitter_seconds=3, seed=6)
    score = split_half_stability(events)
    assert score >= 0.8


def test_split_half_stability_raises_with_too_few_passes():
    events = simulate_patrol_nights(FENCE_ORDER, nights=1, seed=7)
    with pytest.raises(SequenceError):
        split_half_stability(events)
