"""Benchmark and League Poker Player Archetypes.

Supports:
1. Nit (extremely tight, folds often, raises monsters only)
2. TAG (Tight-Aggressive, selective hands, high aggression when entering pot)
3. LAG (Loose-Aggressive, plays wide range with aggressive betting)
4. Calling Station (loose-passive, calls down often, rarely folds or raises)
5. Maniac (hyper-aggressive, bets and raises frequently)
6. Overfolder (folds immediately to any bet or raise)
7. Overbluffer (fires large bluffs when checked to)
8. Adaptive Reg (adjusts aggression based on tournament context and stack depth)
"""

import random
from typing import Optional
import pokers as pkrs
from src.utils.actions import preset_raise_action, sanitize_action


class BaseArchetypeAgent:
    """Base class for style archetype agents."""

    def __init__(self, player_id: int, name: str = "Archetype"):
        self.player_id = player_id
        self.name = f"{name}_{player_id}"
        self.rng = random.Random(player_id * 1000 + 7)

    def choose_action(self, state: pkrs.State, **kwargs) -> pkrs.Action:
        raise NotImplementedError


class NitAgent(BaseArchetypeAgent):
    """Nit: Extremely tight, folds ~85% of hands preflop, only calls/raises premium hands."""

    def __init__(self, player_id: int):
        super().__init__(player_id, name="Nit")

    def choose_action(self, state: pkrs.State, **kwargs) -> pkrs.Action:
        if pkrs.ActionEnum.Check in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))
        
        # 80% fold rate to any bet
        if pkrs.ActionEnum.Fold in state.legal_actions and self.rng.random() < 0.80:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))

        if pkrs.ActionEnum.Raise in state.legal_actions and self.rng.random() < 0.5:
            return sanitize_action(state, preset_raise_action(state, "pot"))

        if pkrs.ActionEnum.Call in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))

        return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))


class TAGAgent(BaseArchetypeAgent):
    """TAG: Tight-Aggressive, selective (~22% VPIP) and highly aggressive."""

    def __init__(self, player_id: int):
        super().__init__(player_id, name="TAG")

    def choose_action(self, state: pkrs.State, **kwargs) -> pkrs.Action:
        if pkrs.ActionEnum.Check in state.legal_actions:
            if pkrs.ActionEnum.Raise in state.legal_actions and self.rng.random() < 0.45:
                return sanitize_action(state, preset_raise_action(state, "half_pot"))
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))

        # Facing bet: 55% fold, 35% raise, 10% call
        roll = self.rng.random()
        if roll < 0.55 and pkrs.ActionEnum.Fold in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))
        elif roll < 0.90 and pkrs.ActionEnum.Raise in state.legal_actions:
            return sanitize_action(state, preset_raise_action(state, "pot"))
        elif pkrs.ActionEnum.Call in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))
        
        return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))


class LAGAgent(BaseArchetypeAgent):
    """LAG: Loose-Aggressive, enters ~38% of pots with high raise frequency."""

    def __init__(self, player_id: int):
        super().__init__(player_id, name="LAG")

    def choose_action(self, state: pkrs.State, **kwargs) -> pkrs.Action:
        if pkrs.ActionEnum.Check in state.legal_actions:
            if pkrs.ActionEnum.Raise in state.legal_actions and self.rng.random() < 0.65:
                return sanitize_action(state, preset_raise_action(state, "pot"))
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))

        roll = self.rng.random()
        if roll < 0.35 and pkrs.ActionEnum.Fold in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))
        elif roll < 0.75 and pkrs.ActionEnum.Raise in state.legal_actions:
            return sanitize_action(state, preset_raise_action(state, "half_pot"))
        elif pkrs.ActionEnum.Call in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))

        return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))


class CallingStationAgent(BaseArchetypeAgent):
    """Calling Station: Rarely folds, rarely raises, calls almost everything."""

    def __init__(self, player_id: int):
        super().__init__(player_id, name="CallingStation")

    def choose_action(self, state: pkrs.State, **kwargs) -> pkrs.Action:
        if pkrs.ActionEnum.Check in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))
        if pkrs.ActionEnum.Call in state.legal_actions:
            # 90% call, 10% fold
            if self.rng.random() < 0.90:
                return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))
        if pkrs.ActionEnum.Fold in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))
        return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))


class ManiacAgent(BaseArchetypeAgent):
    """Maniac: Hyper-aggressive, raises and all-ins constantly."""

    def __init__(self, player_id: int):
        super().__init__(player_id, name="Maniac")

    def choose_action(self, state: pkrs.State, **kwargs) -> pkrs.Action:
        if pkrs.ActionEnum.Raise in state.legal_actions:
            # 85% raise
            if self.rng.random() < 0.85:
                sizing = "all_in" if self.rng.random() < 0.3 else "pot"
                return sanitize_action(state, preset_raise_action(state, sizing))
        if pkrs.ActionEnum.Check in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))
        if pkrs.ActionEnum.Call in state.legal_actions and self.rng.random() < 0.7:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))
        return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))


class OverfolderAgent(BaseArchetypeAgent):
    """Overfolder: Folds whenever facing any bet or raise."""

    def __init__(self, player_id: int):
        super().__init__(player_id, name="Overfolder")

    def choose_action(self, state: pkrs.State, **kwargs) -> pkrs.Action:
        if pkrs.ActionEnum.Check in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))
        if pkrs.ActionEnum.Fold in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))
        return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))


class OverblufferAgent(BaseArchetypeAgent):
    """Overbluffer: Checks when facing a bet, but always bets maximum when checked to."""

    def __init__(self, player_id: int):
        super().__init__(player_id, name="Overbluffer")

    def choose_action(self, state: pkrs.State, **kwargs) -> pkrs.Action:
        if pkrs.ActionEnum.Check in state.legal_actions and pkrs.ActionEnum.Raise in state.legal_actions:
            # Opportunity to bluff when pot is open
            return sanitize_action(state, preset_raise_action(state, "pot"))
        if pkrs.ActionEnum.Check in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))
        if pkrs.ActionEnum.Fold in state.legal_actions and self.rng.random() < 0.7:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))
        return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))


class AdaptiveRegAgent(BaseArchetypeAgent):
    """Adaptive Reg: Adjusts aggression dynamically based on table position and tournament stage."""

    def __init__(self, player_id: int):
        super().__init__(player_id, name="AdaptiveReg")

    def choose_action(self, state: pkrs.State, tournament_context=None, **kwargs) -> pkrs.Action:
        # Default solid mixed strategy
        if pkrs.ActionEnum.Check in state.legal_actions:
            if pkrs.ActionEnum.Raise in state.legal_actions and self.rng.random() < 0.4:
                return sanitize_action(state, preset_raise_action(state, "half_pot"))
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))

        # Context-aware adjustments
        fold_threshold = 0.50
        if tournament_context is not None and tournament_context.rank_margin_bb > 20.0:
            # Ahead of cutoff: play more conservative
            fold_threshold = 0.65

        roll = self.rng.random()
        if roll < fold_threshold and pkrs.ActionEnum.Fold in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))
        elif pkrs.ActionEnum.Raise in state.legal_actions and self.rng.random() < 0.4:
            return sanitize_action(state, preset_raise_action(state, "pot"))
        elif pkrs.ActionEnum.Call in state.legal_actions:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))

        return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))
