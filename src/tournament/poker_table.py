"""Isolated 6-Max Poker Table Environment for tournament play."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import pokers as pkrs

from src.utils.actions import sanitize_action
from src.utils.logging import apply_action_with_logging
from src.utils.evaluation import choose_agent_action
from src.tournament.state import PlayerRecord


class PokerTableEnv:
    """Runs a single 6-Max NLHE hand at a tournament table.
    
    Ensures complete isolation of poker rules from tournament meta-rules.
    """

    def __init__(
        self,
        num_players: int = 6,
        sb: float = 1.0,
        bb: float = 2.0,
        stake: float = 200.0,
        strict: bool = False,
    ):
        self.num_players = num_players
        self.sb = sb
        self.bb = bb
        self.stake = stake
        self.strict = strict

    def play_hand(
        self,
        agents: Sequence[Any],
        button: int,
        seed: int,
        context_provider: Optional[Any] = None,
    ) -> Tuple[List[float], pkrs.State]:
        """Play a single 6-Max hand to completion with given agents.
        
        Args:
            agents: List of 6 agents (one per seat 0..5)
            button: Seat index of the button (0..5)
            seed: Deterministic hand seed
            context_provider: Optional callable(seat_id) returning TournamentState
            
        Returns:
            Tuple of (deltas_bb, final_state)
            deltas_bb: List of net BB won/lost by each seat (sum == 0.0)
        """
        if len(agents) != self.num_players:
            raise ValueError(f"Expected {self.num_players} agents, got {len(agents)}")

        state = pkrs.State.from_seed(
            n_players=self.num_players,
            button=button % self.num_players,
            sb=self.sb,
            bb=self.bb,
            stake=self.stake,
            seed=seed,
        )

        while not state.final_state:
            current_player = state.current_player
            agent = agents[current_player]
            
            # Retrieve optional tournament context for this player
            tournament_context = None
            if context_provider is not None:
                tournament_context = context_provider(current_player)

            # Choose and sanitize action
            raw_action = choose_agent_action(
                agent,
                state,
                tournament_context=tournament_context,
                strict=self.strict,
            )
            action = sanitize_action(
                state,
                raw_action,
                strict=self.strict,
            )

            # Apply action in poker engine
            new_state, log_file, status = apply_action_with_logging(
                state,
                action,
                strict=self.strict,
                error_prefix="PokerTableEnv error",
            )
            if new_state is None:
                raise RuntimeError(f"Invalid game state during hand: {status}, log={log_file}")
            state = new_state

        # Calculate deltas in BB
        deltas_bb = [p.reward / self.bb for p in state.players_state]
        return deltas_bb, state
