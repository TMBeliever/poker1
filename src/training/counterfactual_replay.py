"""Counterfactual and High-Leverage Experience Replay for Tournament Deep CFR."""

from dataclasses import dataclass
import numpy as np
import torch
from typing import Any, List, Optional, Tuple

from src.tournament.state import Stage, TournamentState


@dataclass
class HighLeverageExperience:
    """Experience tuple annotated with high-leverage tournament situation attributes."""
    poker_vec: np.ndarray
    tourn_vec: Optional[np.ndarray]
    action_type: int
    bet_size: float
    regret: float
    # Tournament situational indicators
    pot_size_bb: float = 0.0
    stack_bb: float = 100.0
    is_bubble: bool = False
    is_late_stage: bool = False
    rank_swing_bb: float = 0.0


class HighLeverageReplayBuffer:
    """Prioritized replay buffer prioritizing high-leverage and counterfactual decisions.
    
    Priority modifiers:
    1. High approximate regret
    2. Bubble states (near cutoff rank with few hands remaining)
    3. Large pots
    4. Short stack states
    5. Late tournament stages (Semifinal / Final)
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[HighLeverageExperience] = []
        self.priorities: List[float] = []
        self.position = 0

    def compute_leverage_multiplier(self, exp: HighLeverageExperience) -> float:
        """Compute situational priority multiplier."""
        multiplier = 1.0

        # Bubble multiplier (close to cutoff with high stakes)
        if exp.is_bubble:
            multiplier += 1.5

        # Large pot multiplier
        if exp.pot_size_bb > 50.0:
            multiplier += min(2.0, exp.pot_size_bb / 50.0)

        # Short stack multiplier
        if exp.stack_bb < 25.0:
            multiplier += 1.2

        # Late stage multiplier
        if exp.is_late_stage:
            multiplier += 1.0

        return multiplier

    def add(self, exp: HighLeverageExperience) -> None:
        """Add a high-leverage experience with situational priority."""
        base_priority = max(abs(exp.regret), 0.01)
        leverage_mult = self.compute_leverage_multiplier(exp)
        priority = (base_priority * leverage_mult) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = exp
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[HighLeverageExperience], List[int], np.ndarray]:
        """Sample batch according to leverage priorities with importance sampling correction."""
        n = len(self.buffer)
        if n == 0:
            return [], [], np.array([])

        if n <= batch_size:
            return list(self.buffer), list(range(n)), np.ones(n, dtype=np.float32)

        total_p = sum(self.priorities)
        probs = [p / total_p for p in self.priorities]

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        weights = np.array([(n * probs[idx]) ** (-beta) for idx in indices], dtype=np.float32)
        weights /= weights.max()

        return samples, list(indices), weights
