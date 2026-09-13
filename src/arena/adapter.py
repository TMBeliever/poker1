"""Cross-project competitor adapter linking agentpoker_final 2 models with ApexPoker."""

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pokers as pkrs

from src.agents.archetypes import (
    AdaptiveRegAgent,
    BaseArchetypeAgent,
    CallingStationAgent,
    LAGAgent,
    ManiacAgent,
    NitAgent,
    OverblufferAgent,
    OverfolderAgent,
    TAGAgent,
)
from src.tournament.state import TournamentState
from src.tournament.tournament_deep_cfr import TournamentDeepCFRAgent
from src.utils.actions import sanitize_action


def _apply_dataclass_compat():
    """Polyfill slots=True and kw_only=True for Python < 3.10 to import agentpoker_final 2."""
    if sys.version_info < (3, 10):
        _orig_dataclass = dataclasses.dataclass

        def _dataclass_compat(*args, **kwargs):
            kwargs.pop("slots", None)
            kwargs.pop("kw_only", None)
            return _orig_dataclass(*args, **kwargs)

        dataclasses.dataclass = _dataclass_compat


_apply_dataclass_compat()


def pkrs_card_to_str(c: Any) -> str:
    """Convert pokers Card instance to standard 2-char string like 'Ah', 'Ks', '2c'."""
    suits = ["c", "d", "h", "s"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    try:
        r_idx = int(c.rank)
        s_idx = int(c.suit)
        return f"{ranks[r_idx]}{suits[s_idx]}"
    except Exception:
        return "2c"


def resolve_agentpoker_path(path_str: str) -> Path:
    """Resolve model path against local dir or ../agentpoker_final 2."""
    p = Path(path_str)
    if p.exists():
        return p

    repo_root = Path(__file__).resolve().parent.parent.parent
    sibling_root = repo_root.parent / "agentpoker_final 2"

    candidates = [
        sibling_root / path_str,
        sibling_root / "models" / path_str,
        sibling_root / "models" / f"{path_str}.json",
        repo_root / path_str,
    ]
    for c in candidates:
        if c.exists():
            return c
    return p


class AgentPokerFinalAgent(BaseArchetypeAgent):
    """Wraps an agentpoker_final 2 StrategyAgent (champion.json / candidate_base.json)
    to execute seamlessly within ApexPoker's PokerTableEnv and TournamentEnv.
    """

    def __init__(
        self,
        player_id: int,
        model_path: str,
        name: str = "AgentPoker_Champion",
        sb: float = 1.0,
        bb: float = 2.0,
        stake: float = 200.0,
    ):
        super().__init__(player_id, name=name)
        self.sb = sb
        self.bb = bb
        self.stake = stake
        self.model_path = str(resolve_agentpoker_path(model_path))

        # Dynamically import StrategyAgent from agentpoker_final 2
        repo_root = Path(__file__).resolve().parent.parent.parent
        sibling_root = repo_root.parent / "agentpoker_final 2"
        for candidate in [sibling_root, Path(self.model_path).resolve().parent.parent]:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        try:
            from agentpoker.strategy import StrategyAgent

            self.strategy = StrategyAgent.load(self.model_path)
            print(f"[{name}] Successfully loaded agentpoker_final 2 model from: {self.model_path}")
        except Exception as e:
            print(f"[{name}] Warning: Failed to load StrategyAgent ({e}), falling back to TAG.")
            self.strategy = None

    def choose_action(
        self,
        state: pkrs.State,
        tournament_context: Optional[TournamentState] = None,
        strict: bool = False,
        **kwargs,
    ) -> pkrs.Action:
        if self.strategy is None:
            # Fallback to standard TAG check/fold
            if pkrs.ActionEnum.Check in state.legal_actions:
                return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check), strict=strict)
            return sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Fold), strict=strict)

        curr = state.current_player
        allowed = []
        if pkrs.ActionEnum.Fold in state.legal_actions:
            allowed.append({"type": "fold"})
        if pkrs.ActionEnum.Check in state.legal_actions:
            allowed.append({"type": "check"})
        if pkrs.ActionEnum.Call in state.legal_actions:
            allowed.append({"type": "call", "amount": self.bb})
        if pkrs.ActionEnum.Raise in state.legal_actions:
            allowed.append({"type": "raise", "minAmount": self.bb * 2, "maxAmount": self.stake})

        t_ctx = {}
        if tournament_context is not None:
            t_ctx = {
                "stage": str(tournament_context.stage).lower(),
                "rank": tournament_context.rank,
                "target_rank": tournament_context.cutoff_rank,
                "hands_remaining": tournament_context.hands_remaining,
                "buffer_bb100": tournament_context.rank_margin_bb,
                "bb100": tournament_context.cumulative_net_bb,
                "rebuy_count": tournament_context.rebuy_count,
                "table_strength": tournament_context.table_strength,
                "table_rank": tournament_context.table_rank,
            }

        obs = {
            "agentId": f"player_{curr}",
            "table": {
                "initialStake": self.stake,
                "bigBlind": self.bb,
                "smallBlind": self.sb,
                "dealerSeat": state.button,
                "hand": {
                    "pot": state.pot,
                    "communityCards": [pkrs_card_to_str(c) for c in state.public_cards],
                    "actions": [],
                },
                "players": [
                    {
                        "agentId": f"player_{i}",
                        "seat": i,
                        "stack": state.players_state[i].stake,
                        "handState": {
                            "holeCards": [pkrs_card_to_str(c) for c in state.players_state[i].hand]
                            if i == curr
                            else [],
                            "currentBet": state.players_state[i].bet_chips,
                        },
                    }
                    for i in range(len(state.players_state))
                ],
            },
            "actionRequest": {"allowedActions": allowed},
            "tournamentContext": t_ctx,
        }

        dec = self.strategy.choose(obs)
        atype = dec.get("type", "fold")

        if atype == "fold" and pkrs.ActionEnum.Fold in state.legal_actions:
            raw = pkrs.Action(pkrs.ActionEnum.Fold)
        elif atype == "check" and pkrs.ActionEnum.Check in state.legal_actions:
            raw = pkrs.Action(pkrs.ActionEnum.Check)
        elif atype == "call" and pkrs.ActionEnum.Call in state.legal_actions:
            raw = pkrs.Action(pkrs.ActionEnum.Call)
        elif atype in ("raise", "bet", "allIn") and pkrs.ActionEnum.Raise in state.legal_actions:
            amt = float(dec.get("amount", self.bb * 2.5))
            raw = pkrs.Action(pkrs.ActionEnum.Raise, amt)
        else:
            raw = pkrs.Action(
                pkrs.ActionEnum.Check if pkrs.ActionEnum.Check in state.legal_actions else pkrs.ActionEnum.Fold
            )

        return sanitize_action(state, raw, strict=strict)


ARCHETYPE_MAP = {
    "TAG": TAGAgent,
    "NIT": NitAgent,
    "LAG": LAGAgent,
    "MANIAC": ManiacAgent,
    "STATION": CallingStationAgent,
    "CALLING_STATION": CallingStationAgent,
    "CALLINGSTATION": CallingStationAgent,
    "OVERFOLDER": OverfolderAgent,
    "OVERBLUFFER": OverblufferAgent,
    "ADAPTIVE": AdaptiveRegAgent,
    "ADAPTIVE_REG": AdaptiveRegAgent,
    "ADAPTIVEREG": AdaptiveRegAgent,
}


def build_competitor(
    name: str,
    spec: str,
    player_id: int,
    sb: float = 1.0,
    bb: float = 2.0,
    stake: float = 200.0,
) -> Tuple[Any, str]:
    """Factory function instantiating any competitor:
    1. Deep CFR checkpoint (.pt)
    2. agentpoker_final 2 model (.json / champion)
    3. Canonical archetype bot (TAG, LAG, Nit, Maniac, Station, etc.)
    """
    spec_upper = spec.strip().upper()
    spec_clean = spec_upper.replace("AGENT", "")
    if spec_upper in ARCHETYPE_MAP:
        cls_ = ARCHETYPE_MAP[spec_upper]
        return cls_(player_id), name
    if spec_clean in ARCHETYPE_MAP:
        cls_ = ARCHETYPE_MAP[spec_clean]
        return cls_(player_id), name

    # Check if real user profile from online competition
    if spec.startswith("profile:") or spec.startswith("user:"):
        prof_key = spec.split(":", 1)[1].strip()
        from src.agents.profiled_agent import ProfiledAgent
        from src.opponent_modeling.profile_manager import OpponentProfileManager
        pm = OpponentProfileManager("data/profiles/opponent_profiles.json")
        p_data = pm.get_profile(prof_key)
        agent = ProfiledAgent(player_id=player_id, profile=p_data, sb=sb, bb=bb, stake=stake)
        disp_name = f"{p_data.get('name', name)} [{p_data.get('archetype', 'S10')}]"
        return agent, disp_name

    # Check if agentpoker_final 2 json model
    resolved_json = resolve_agentpoker_path(spec)
    if resolved_json.suffix == ".json" and resolved_json.exists():
        agent = AgentPokerFinalAgent(
            player_id=player_id,
            model_path=str(resolved_json),
            name=name,
            sb=sb,
            bb=bb,
            stake=stake,
        )
        return agent, name

    # Check if Deep CFR PyTorch checkpoint (.pt)
    if spec.endswith(".pt") or os.path.exists(spec):
        from src.tournament.tournament_deep_cfr import AGENT_TYPE_TOURNAMENT
        from src.utils.agents import CheckpointAgent
        from src.utils.checkpoints import load_checkpoint
        try:
            ckpt = load_checkpoint(spec, map_location="cpu")
            if ckpt.get("agent_type") == AGENT_TYPE_TOURNAMENT or "curriculum" in str(spec):
                agent = TournamentDeepCFRAgent(player_id=player_id, num_players=6)
                agent.load_checkpoint(spec)
                return agent, name
            else:
                agent = CheckpointAgent(player_id=player_id, model_path=spec, sanitize_actions=True)
                return agent, name
        except Exception:
            agent = TournamentDeepCFRAgent(player_id=player_id, num_players=6)
            if os.path.exists(spec):
                agent.load_checkpoint(spec)
            return agent, name

    # Default fallback
    print(f"Warning: Unknown competitor spec '{spec}', defaulting to TAG.")
    return TAGAgent(player_id), name
