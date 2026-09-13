"""Tournament Reward Engine for ApexPoker multi-stage tournaments."""

from dataclasses import dataclass
import math
from typing import Optional
from src.tournament.state import Stage


@dataclass
class TournamentRewardBreakdown:
    """Individual reward components tracked for policy optimization and analysis."""
    hand_reward: float
    stage_reward: float
    final_reward: float
    net_bb: float
    rank: int
    rank_margin: float
    total_reward: float

    def to_dict(self) -> dict:
        return {
            "hand_reward": self.hand_reward,
            "stage_reward": self.stage_reward,
            "final_reward": self.final_reward,
            "net_bb": self.net_bb,
            "rank": self.rank,
            "rank_margin": self.rank_margin,
            "total_reward": self.total_reward,
        }


class TournamentRewardCalculator:
    """Computes stage-specific tournament rewards with clear component separation.
    
    Philosophy:
    - Preliminary: Net BB / BB100 + P(Top12 qualification)
    - Semifinal: Net BB + P(Top3 per table qualification)
    - Final: Champion probability (Rank 1)
    
    Preserves and independently records:
    - hand_reward
    - stage_reward
    - final_reward
    - net_bb
    - rank
    - rank_margin
    """

    def __init__(
        self,
        hand_scale: float = 0.01,
        top12_bonus: float = 1.0,
        top3_bonus: float = 2.0,
        champion_bonus: float = 5.0,
        margin_weight: float = 0.1,
    ):
        self.hand_scale = hand_scale
        self.top12_bonus = top12_bonus
        self.top3_bonus = top3_bonus
        self.champion_bonus = champion_bonus
        self.margin_weight = margin_weight

    def calculate(
        self,
        stage: Stage,
        hand_delta_bb: float,
        current_rank: int,
        cutoff_rank: int,
        rank_margin_bb: float,
        cumulative_net_bb: float,
        hands_remaining: int,
        is_stage_terminal: bool = False,
    ) -> TournamentRewardBreakdown:
        """Compute the breakdown of tournament rewards."""
        # 1. Immediate hand reward (based on chip profit/loss in BB)
        hand_reward = hand_delta_bb * self.hand_scale

        stage_reward = 0.0
        final_reward = 0.0

        # Continuous shaping on rank margin when near the end of stage
        urgency = 1.0 / (1.0 + max(0, hands_remaining) / 20.0)
        margin_shaping = math.tanh(rank_margin_bb / 50.0) * self.margin_weight * urgency

        if stage == Stage.PRELIMINARY:
            # Preliminary terminal reward: Qualification for Top 12
            if is_stage_terminal:
                if current_rank <= cutoff_rank:
                    stage_reward = self.top12_bonus
                else:
                    stage_reward = -0.5 * self.top12_bonus
            else:
                stage_reward = margin_shaping

        elif stage == Stage.SEMIFINAL:
            # Semifinal terminal reward: Qualification for Final (Top 3 on table)
            if is_stage_terminal:
                if current_rank <= cutoff_rank:
                    stage_reward = self.top3_bonus
                else:
                    stage_reward = -0.5 * self.top3_bonus
            else:
                stage_reward = margin_shaping

        elif stage == Stage.FINAL:
            # Final terminal reward: Crowned Champion
            if is_stage_terminal:
                if current_rank == 1:
                    final_reward = self.champion_bonus
                else:
                    final_reward = self.champion_bonus / (1.0 + current_rank)
            else:
                final_reward = margin_shaping

        total_reward = hand_reward + stage_reward + final_reward

        return TournamentRewardBreakdown(
            hand_reward=hand_reward,
            stage_reward=stage_reward,
            final_reward=final_reward,
            net_bb=cumulative_net_bb,
            rank=current_rank,
            rank_margin=rank_margin_bb,
            total_reward=total_reward,
        )
