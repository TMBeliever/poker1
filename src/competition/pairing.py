from __future__ import annotations
from random import Random
from typing import Any, Sequence


def random_groups(ids: Sequence[str], seats: int = 6, rng: Random | None = None) -> list[list[str]]:
    """Randomly partition a list of agent IDs into tables of size `seats`."""
    ids_list = list(ids)
    r = rng if rng is not None else Random()
    r.shuffle(ids_list)
    return [ids_list[i : i + seats] for i in range(0, len(ids_list), seats)]


def swiss_groups(
    ids: Sequence[str],
    scores: dict[str, Any] | None = None,
    seats: int = 6,
    standings: Sequence[Any] | None = None,
) -> list[list[str]]:
    """Partition agents into Swiss tables of size `seats` based on rankings.

    If pre-sorted `standings` are supplied, preserves their full multi-key tiebreak
    order (cumulative BB/100 -> R4-R10 net BB -> hands -> id).
    Otherwise, sorts by `scores` dictionary.
    """
    if standings is not None:
        ordered = [s.agent_id if hasattr(s, "agent_id") else s for s in standings]
    elif scores is not None:
        # scores can map aid -> float, or aid -> tuple of tiebreak criteria
        ordered = sorted(
            list(ids),
            key=lambda x: (
                -scores[x] if isinstance(scores.get(x), (int, float))
                else ([-v for v in scores[x]] if isinstance(scores.get(x), (list, tuple)) else 0),
                x,
            ),
        )
    else:
        ordered = list(ids)

    return [ordered[i : i + seats] for i in range(0, len(ordered), seats)]


def verify_pairing_integrity(
    groups: list[list[str]], expected_ids: Sequence[str], seats: int = 6
) -> bool:
    """Verify that groups constitute a valid, complete, and disjoint partition."""
    expected_set = set(expected_ids)
    all_assigned: list[str] = []
    for g in groups:
        if len(g) != seats:
            return False
        all_assigned.extend(g)
    return len(all_assigned) == len(expected_set) and set(all_assigned) == expected_set
