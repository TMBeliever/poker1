"""State definitions for tournament engine."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class Stage(Enum):
    """Strict tournament stages."""
    PRELIMINARY = auto()
    SEMIFINAL = auto()
    FINAL = auto()
    FINISHED = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class TournamentState:
    """Tournament context representation provided to agents or observation layer.
    
    Contains all tournament-level situational indicators specified in ApexPoker requirements.
    """
    stage: Stage
    hand_no: int
    hands_remaining: int
    player_id: int
    rank: int
    cutoff_rank: int
    rank_margin_bb: float
    cumulative_net_bb: float
    rebuy_count: int
    table_rank: int
    table_id: int
    stack_bb: float
    effective_stack_bb: float
    table_strength: float


@dataclass
class PlayerRecord:
    """Persistent player tournament profile and accounting record."""
    player_id: int
    stack_bb: float = 50.0
    rebuy_count: int = 0
    hands_played: int = 0
    stage_hands_played: int = 0
    cumulative_net_bb: float = 0.0
    stage_net_bb: float = 0.0
    is_active: bool = True
    current_table_id: int = -1
    seat_id: int = -1

    def reset_for_stage(self, starting_stack_bb: float = 50.0) -> None:
        """Reset player chips and stage-specific counters at stage boundary."""
        self.stack_bb = starting_stack_bb
        self.stage_net_bb = 0.0
        self.stage_hands_played = 0
        self.rebuy_count = 0  # Stage rebuy counter resets for independent stage evaluation
