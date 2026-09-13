"""Officially calibrated context ladder data for 120-player tournament format.

FIELD_SIZE = 120
QUALIFY_RANK = 12 (Top 10% advance from Preliminary to Semifinals)
GOLDEN_LINE_BB100 = 28.0 (Empirical qualification line for 12th place)
"""
from __future__ import annotations

FIELD_SIZE: int = 120
QUALIFY_RANK: int = 12

RANK_LADDER: list[tuple[float, int]] = [
    (-80.0, 120),
    (-40.0, 105),
    (-15.0, 85),
    (0.0, 60),
    (12.0, 32),
    (20.0, 20),
    (25.0, 13),
    (28.0, 12),
    (45.0, 6),
    (80.0, 1),
]

GOLDEN_LINE_BB100: float = 28.0
