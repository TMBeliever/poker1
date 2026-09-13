"""Profiled Poker Agent driven by empirical online Bayesian user profiles.

Translates empirical player parameters (VPIP, PFR, 3-bet, AF, C-bet, sizing preferences)
into an active player agent for realistic digital twin tournament simulations.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import pokers as pkrs
from src.agents.archetypes import BaseArchetypeAgent
from src.utils.actions import build_raise_action, preset_raise_action, sanitize_action


class ProfiledAgent(BaseArchetypeAgent):
    """An agent that plays according to real-world behavioral statistics extracted from online matches."""

    def __init__(
        self,
        player_id: int,
        profile: Dict[str, Any],
        sb: float = 1.0,
        bb: float = 2.0,
        stake: float = 200.0,
        seed: Optional[int] = None,
    ):
        raw_name = profile.get("name") or f"Agent_{player_id}"
        super().__init__(player_id, name=raw_name)
        self.profile = profile
        self.sb = sb
        self.bb = bb
        self.stake = stake
        self.rng = random.Random(seed if seed is not None else player_id * 1000 + 42)

        # Core rates
        self.vpip = float(profile.get("vpip", 0.25))
        self.pfr = float(profile.get("pfr", 0.18))
        self.threebet = float(profile.get("threebet", 0.08))
        self.af = float(profile.get("af", 2.0))
        self.cbet_flop = float(profile.get("cbet_flop", 0.55))
        self.fold_to_cbet = float(profile.get("fold_to_cbet", 0.45))
        self.archetype = profile.get("archetype", "Balanced")

        # Preferred sizing
        self.open_size_bb = float(profile.get("open_size_bb", 2.5))
        self.cbet_size = float(profile.get("cbet_size", 0.50))
        self.value_bet_size = float(profile.get("value_bet_size", 0.65))
        self.raise_size = float(profile.get("raise_size", 0.70))

        # Format display name
        short_arch = self.archetype.split("(")[0].strip()
        self.name = f"{raw_name} [{short_arch}]"

    def choose_action(self, state: pkrs.State, **kwargs) -> pkrs.Action:
        """Select action strictly matching the player's Bayesian profile."""
        legal = state.legal_actions

        # 1. Preflop Decision
        if state.stage == pkrs.Stage.Preflop:
            return self._preflop_action(state)

        # 2. Postflop Decision (Flop, Turn, River)
        return self._postflop_action(state)

    def _preflop_action(self, state: pkrs.State) -> pkrs.Action:
        legal = state.legal_actions
        curr = state.current_player
        p_state = state.players_state[curr]
        current_bet = self.stake - p_state.stake
        max_table_bet = max(self.stake - p.stake for p in state.players_state)
        facing_raise = max_table_bet > current_bet and max_table_bet > self.bb

        roll = self.rng.random()

        # Facing a preflop raise (3-bet / call / fold decision)
        if facing_raise:
            if roll < self.threebet and pkrs.ActionEnum.Raise in legal:
                # 3-bet with preferred raise sizing
                raise_amount = max(max_table_bet * 2.8, self.bb * self.open_size_bb * 3.0)
                return build_raise_action(state, raise_amount)
            
            call_threshold = self.threebet + max(0.05, self.vpip * 0.4)
            if roll < call_threshold and pkrs.ActionEnum.Call in legal:
                return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))
            
            if pkrs.ActionEnum.Fold in legal:
                return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))

        # Unopened or limped pot (Open raise / limp / check decision)
        if roll < self.pfr and pkrs.ActionEnum.Raise in legal:
            open_amount = self.bb * self.open_size_bb
            return build_raise_action(state, open_amount)

        if roll < self.vpip and pkrs.ActionEnum.Call in legal:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))

        if pkrs.ActionEnum.Check in legal:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))

        if pkrs.ActionEnum.Fold in legal:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))

        return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))

    def _postflop_action(self, state: pkrs.State) -> pkrs.Action:
        legal = state.legal_actions
        curr = state.current_player
        p_state = state.players_state[curr]
        current_bet = self.stake - p_state.stake
        max_table_bet = max(self.stake - p.stake for p in state.players_state)
        facing_bet = max_table_bet > current_bet

        roll = self.rng.random()
        # Aggression probability derived from Aggression Factor (AF = Aggressive / Passive)
        # P(aggress) = AF / (AF + 1.0)
        p_aggress = min(0.75, max(0.15, self.af / (self.af + 1.0)))

        if facing_bet:
            # Facing bet: Raise / Call / Fold
            if roll < (p_aggress * 0.25) and pkrs.ActionEnum.Raise in legal:
                raise_amount = max_table_bet * (1.0 + self.raise_size)
                return build_raise_action(state, raise_amount)

            # Fold vs call based on fold_to_cbet profile
            if roll < self.fold_to_cbet and pkrs.ActionEnum.Fold in legal:
                return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))

            if pkrs.ActionEnum.Call in legal:
                return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Call))

            if pkrs.ActionEnum.Fold in legal:
                return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))

            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))

        # Checked to: Bet / Check
        if roll < p_aggress and pkrs.ActionEnum.Raise in legal:
            # Sized bet based on pot fraction
            pot = state.pot
            frac = self.cbet_size if state.stage == pkrs.Stage.Flop else self.value_bet_size
            bet_amt = max(self.bb, pot * frac)
            return build_raise_action(state, bet_amt)

        if pkrs.ActionEnum.Check in legal:
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check))

        return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold))
