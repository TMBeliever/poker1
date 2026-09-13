"""Dynamic Opponent Modeling and Profiling with GRU and In-game Statistics."""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pokers as pkrs
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.opponent_modeling.opponent_model import ActionHistoryEncoder
from src.tournament.state import Stage


@dataclass
class OpponentStats:
    """Empirical poker statistics tracked dynamically for an opponent."""
    hands_tracked: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    total_actions: int = 0
    aggressive_actions: int = 0  # bets and raises
    passive_actions: int = 0      # checks and calls
    fold_actions: int = 0
    three_bet_opportunities: int = 0
    three_bets: int = 0

    @property
    def vpip(self) -> float:
        """Voluntarily Put Money in Pot (fraction)."""
        return self.vpip_hands / max(1, self.hands_tracked)

    @property
    def pfr(self) -> float:
        """Preflop Raise (fraction)."""
        return self.pfr_hands / max(1, self.hands_tracked)

    @property
    def aggression_frequency(self) -> float:
        """Aggression Frequency (AFq)."""
        denom = self.aggressive_actions + self.passive_actions + self.fold_actions
        return self.aggressive_actions / max(1, denom)

    @property
    def three_bet_rate(self) -> float:
        """3-bet percentage."""
        return self.three_bets / max(1, self.three_bet_opportunities)

    @property
    def fold_rate(self) -> float:
        """General fold tendency."""
        denom = self.aggressive_actions + self.passive_actions + self.fold_actions
        return self.fold_actions / max(1, denom)

    def to_vector(self, stack_pressure: float = 1.0, stage_val: float = 0.0) -> np.ndarray:
        """Convert stats to a normalized summary vector of length 7."""
        return np.array(
            [
                self.vpip,
                self.pfr,
                self.aggression_frequency,
                self.three_bet_rate,
                self.fold_rate,
                math.tanh(stack_pressure),
                stage_val,
            ],
            dtype=np.float32,
        )


class DynamicOpponentEncoder(nn.Module):
    """Dynamic opponent embedding integrating GRU sequence encoding with statistical features."""

    def __init__(
        self,
        gru_output_dim: int = 64,
        stats_dim: int = 7,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.stats_dim = stats_dim
        self.embedding_dim = embedding_dim

        # Reusable sequence encoder
        self.gru_encoder = ActionHistoryEncoder(
            action_dim=4,
            state_dim=20,
            hidden_dim=128,
            output_dim=gru_output_dim,
        )

        # Statistical feature projector
        self.stats_projector = nn.Sequential(
            nn.Linear(stats_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )

        # Joint embedding fusion
        self.fusion = nn.Sequential(
            nn.Linear(gru_output_dim + 32, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        action_seq: torch.Tensor,
        state_ctx: torch.Tensor,
        stats_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Produce dense, continuous opponent embeddings."""
        gru_emb = self.gru_encoder(action_seq, state_ctx)
        stats_emb = self.stats_projector(stats_vec)
        combined = torch.cat([gru_emb, stats_emb], dim=-1)
        return self.fusion(combined)


class OpponentTracker:
    """Manages online opponent histories and statistics across tournament hands."""

    def __init__(self, max_history_actions: int = 20):
        self.max_history = max_history_actions
        self.stats: Dict[int, OpponentStats] = {}
        self.action_buffers: Dict[int, deque] = {}

    def get_or_create_stats(self, player_id: int) -> OpponentStats:
        if player_id not in self.stats:
            self.stats[player_id] = OpponentStats()
            self.action_buffers[player_id] = deque(maxlen=self.max_history)
        return self.stats[player_id]

    def record_hand_start(self, player_id: int) -> None:
        stats = self.get_or_create_stats(player_id)
        stats.hands_tracked += 1

    def record_action(
        self,
        player_id: int,
        action: pkrs.Action,
        is_preflop: bool = False,
        is_facing_raise: bool = False,
    ) -> None:
        """Update empirical tendencies based on observed actions."""
        stats = self.get_or_create_stats(player_id)
        stats.total_actions += 1

        if action.action == pkrs.ActionEnum.Fold:
            stats.fold_actions += 1
        elif action.action in (pkrs.ActionEnum.Check, pkrs.ActionEnum.Call):
            stats.passive_actions += 1
            if is_preflop and action.action == pkrs.ActionEnum.Call:
                stats.vpip_hands += 1
        elif action.action == pkrs.ActionEnum.Raise:
            stats.aggressive_actions += 1
            if is_preflop:
                stats.vpip_hands += 1
                stats.pfr_hands += 1
            if is_facing_raise:
                stats.three_bets += 1

        if is_facing_raise:
            stats.three_bet_opportunities += 1

    def get_summary_vector(self, player_id: int, stack_pressure: float = 1.0, stage: Stage = Stage.PRELIMINARY) -> np.ndarray:
        stats = self.get_or_create_stats(player_id)
        stage_val = float(list(Stage).index(stage)) / 3.0
        return stats.to_vector(stack_pressure=stack_pressure, stage_val=stage_val)
