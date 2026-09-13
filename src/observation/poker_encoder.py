"""Poker state feature encoder."""

import numpy as np
import torch
import pokers as pkrs
from src.core.model import encode_state


class PokerEncoder:
    """Encodes 6-Max NLHE poker state into a normalized feature vector.
    
    Feature schema (156 dimensions total):
    - hole cards: 52-dim one-hot
    - board cards: 52-dim one-hot
    - street: 5-dim one-hot (Preflop, Flop, Turn, River, Showdown)
    - pot: 1-dim normalized by initial stake
    - button position: 6-dim one-hot
    - current player: 6-dim one-hot
    - player states: 24-dim (6 players x [active, bet_chips, pot_chips, stake])
    - min bet: 1-dim normalized
    - legal actions: 4-dim (Fold, Check, Call, Raise)
    - action history / previous action: 5-dim (4 types + normalized amount)
    """

    FEATURE_DIM = 156

    def __init__(self):
        pass

    def encode(self, state: pkrs.State, player_id: int = 0) -> np.ndarray:
        """Encode pokers state into a 1D numpy float array."""
        return encode_state(state, player_id=player_id)

    def encode_tensor(self, state: pkrs.State, player_id: int = 0, device: str = "cpu") -> torch.Tensor:
        """Encode pokers state directly into a PyTorch tensor."""
        vec = self.encode(state, player_id=player_id)
        return torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
