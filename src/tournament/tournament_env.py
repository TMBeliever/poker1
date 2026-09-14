"""Tournament Environment orchestrating the 120-player ApexPoker tournament."""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.tournament.state import Stage, TournamentState, PlayerRecord
from src.tournament.rebuy import RebuyManager
from src.tournament.ranking import RankingEngine
from src.tournament.swiss import SwissPairing
from src.tournament.advancement import AdvancementManager
from src.tournament.poker_table import PokerTableEnv


class TournamentEnv:
    """Complete 120-player Multi-Stage Tournament Environment.
    
    Structure:
    1. 120 Players -> 20 tables of 6-Max
    2. PRELIMINARY: 200 hands (10 rounds of 20 hands)
       - R1-R3: Random Seeding
       - R4-R10: Swiss Pairing
       - Top 12 qualify for Semifinal
    3. SEMIFINAL: 12 players -> 2 tables of 6
       - Snake Seeding: A = [1,4,5,8,9,12], B = [2,3,6,7,10,11]
       - 100 BB reset per table
       - 20 hands per table
       - Top 3 from each table advance to Final (6 players)
    4. FINAL: 6 players -> 1 table
       - 100 BB reset
       - 30 hands
       - Rank 1 in net BB is crowned Champion
    5. FINISHED
    """

    def __init__(
        self,
        num_players: int = 120,
        table_size: int = 6,
        starting_stack_bb: float = 100.0,
        sb: float = 500.0,
        bb: float = 1000.0,
        seed: Optional[int] = 42,
        strict: bool = False,
        prelim_rounds: int = 10,
        hands_per_prelim_round: int = 20,
        semifinal_hands: int = 20,
        final_hands: int = 30,
    ):
        self.total_players = num_players
        self.table_size = table_size
        self.starting_stack_bb = starting_stack_bb
        self.sb = sb
        self.bb = bb
        self.seed = seed
        self.strict = strict
        self.prelim_rounds = prelim_rounds
        self.hands_per_prelim_round = hands_per_prelim_round
        self.semifinal_hands = semifinal_hands
        self.final_hands = final_hands
        self.rng = random.Random(seed)

        # Internal components
        self.poker_env = PokerTableEnv(
            num_players=table_size,
            sb=sb,
            bb=bb,
            stake=starting_stack_bb * bb,
            strict=strict,
        )
        self.rebuy_manager = RebuyManager(starting_stack_bb=starting_stack_bb)
        self.ranking_engine = RankingEngine()
        self.swiss_pairing = SwissPairing(table_size=table_size, ranking_engine=self.ranking_engine)
        self.advancement_manager = AdvancementManager(ranking_engine=self.ranking_engine)

        # State tracking
        self.stage = Stage.PRELIMINARY
        self.current_round = 0
        self.current_hand = 0
        self.players: Dict[int, PlayerRecord] = {}
        self.active_players: List[PlayerRecord] = []
        self.current_tables: List[List[PlayerRecord]] = []
        
        # Standings archives
        self.prelim_standings: List[PlayerRecord] = []
        self.semifinal_standings: Dict[str, List[PlayerRecord]] = {}
        self.final_standings: List[PlayerRecord] = []
        self.champion: Optional[PlayerRecord] = None

        self.reset(seed=self.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the tournament environment to the beginning."""
        if seed is not None:
            self.seed = seed
        
        if self.seed is not None:
            self.rng = random.Random(self.seed)
            random.seed(self.seed)
            try:
                import numpy as np
                np.random.seed(self.seed)
            except ImportError:
                pass
            try:
                import torch
                torch.manual_seed(self.seed)
            except ImportError:
                pass

        self.stage = Stage.PRELIMINARY
        self.current_round = 0
        self.current_hand = 0
        self.prelim_standings = []
        self.semifinal_standings = {}
        self.final_standings = []
        self.champion = None

        # Create all 120 player records
        self.players = {
            i: PlayerRecord(player_id=i, stack_bb=self.starting_stack_bb)
            for i in range(self.total_players)
        }
        self.active_players = list(self.players.values())
        self.current_tables = []

    def get_tournament_state(self, player_id: int) -> TournamentState:
        """Construct the rich TournamentState observation for a specific player."""
        player = self.players[player_id]
        
        # Hands remaining in current stage
        if self.stage == Stage.PRELIMINARY:
            total_stage_hands = self.prelim_rounds * self.hands_per_prelim_round
            hands_remaining = max(0, total_stage_hands - self.current_hand)
        elif self.stage == Stage.SEMIFINAL:
            total_stage_hands = self.semifinal_hands
            hands_remaining = max(0, total_stage_hands - self.current_hand)
        elif self.stage == Stage.FINAL:
            total_stage_hands = self.final_hands
            hands_remaining = max(0, total_stage_hands - self.current_hand)
        else:
            hands_remaining = 0

        # Stage standings
        standings = self.ranking_engine.compute_standings_and_margins(
            self.active_players,
            stage=self.stage,
            use_stage_net=True,
        )
        p_info = standings.get(player_id, {"rank": 1, "cutoff_rank": 1, "rank_margin_bb": 0.0})

        # Table statistics
        table_rank = 1
        effective_stack = player.stack_bb
        table_strength = 0.0

        if player.current_table_id >= 0 and player.current_table_id < len(self.current_tables):
            table = self.current_tables[player.current_table_id]
            table_ranks = self.ranking_engine.compute_table_rank(table, use_stage_net=True)
            table_rank = table_ranks.get(player_id, 1)

            opponents = [p for p in table if p.player_id != player_id]
            if opponents:
                effective_stack = min(player.stack_bb, max(p.stack_bb for p in opponents))
                table_strength = sum(p.cumulative_net_bb for p in opponents) / len(opponents)

        return TournamentState(
            stage=self.stage,
            hand_no=self.current_hand,
            hands_remaining=hands_remaining,
            player_id=player_id,
            rank=p_info["rank"],
            cutoff_rank=p_info["cutoff_rank"],
            rank_margin_bb=p_info["rank_margin_bb"],
            cumulative_net_bb=player.cumulative_net_bb,
            rebuy_count=player.rebuy_count,
            table_rank=table_rank,
            table_id=player.current_table_id,
            stack_bb=player.stack_bb,
            effective_stack_bb=effective_stack,
            table_strength=table_strength,
        )

    def _play_table_hand(
        self,
        table: List[PlayerRecord],
        button: int,
        hand_seed: int,
        agent_map: Dict[int, Any],
    ) -> None:
        """Execute a single hand at a table and process rebuys and chip updates."""
        agents = [agent_map[p.player_id] for p in table]
        deltas_bb, _ = self.poker_env.play_hand(
            agents=agents,
            button=button,
            seed=hand_seed,
            context_provider=lambda seat: self.get_tournament_state(table[seat].player_id),
        )

        # Apply chip accounting and auto-rebuy for each seat
        for seat_id, delta in enumerate(deltas_bb):
            player = table[seat_id]
            self.rebuy_manager.apply_hand_result(player, delta)

    def run_preliminary(self, agent_map: Dict[int, Any]) -> List[PlayerRecord]:
        """Execute preliminary stage across 20 tables.
        
        R1-R3: Random pairing
        R4-R10: Swiss pairing
        """
        self.stage = Stage.PRELIMINARY
        self.current_hand = 0

        for round_no in range(1, self.prelim_rounds + 1):
            self.current_round = round_no
            # Table pairing for this round
            self.current_tables = self.swiss_pairing.pair_round(
                self.active_players,
                round_no=round_no,
                rng=self.rng,
            )

            # Play hands for every table in this round
            for hand_idx in range(self.hands_per_prelim_round):
                self.current_hand += 1
                button = hand_idx % self.table_size
                for table_idx, table in enumerate(self.current_tables):
                    hand_seed = self.rng.randint(0, 10_000_000)
                    self._play_table_hand(table, button, hand_seed, agent_map)

        self.prelim_standings = self.ranking_engine.rank_players(self.active_players, use_stage_net=True)
        return self.prelim_standings

    def run_semifinal(self, agent_map: Dict[int, Any]) -> Tuple[List[PlayerRecord], List[PlayerRecord]]:
        """Execute the Semifinal stage across 2 tables (Table A & B)."""
        self.stage = Stage.SEMIFINAL
        self.current_hand = 0

        # Advance top 12 with snake seeding and 100 BB reset
        table_a, table_b = self.advancement_manager.advance_preliminary_to_semifinal(
            self.prelim_standings,
            rng=self.rng,
        )
        self.current_tables = [table_a, table_b]
        self.active_players = table_a + table_b

        # Play hands per table
        for hand_idx in range(self.semifinal_hands):
            self.current_hand += 1
            button = hand_idx % self.table_size
            for table_idx, table in enumerate(self.current_tables):
                hand_seed = self.rng.randint(0, 10_000_000)
                self._play_table_hand(table, button, hand_seed, agent_map)

        standings_a = self.ranking_engine.rank_players(table_a, use_stage_net=True)
        standings_b = self.ranking_engine.rank_players(table_b, use_stage_net=True)
        self.semifinal_standings = {"Table_A": standings_a, "Table_B": standings_b}
        return standings_a, standings_b

    def run_final(self, agent_map: Dict[int, Any]) -> Tuple[PlayerRecord, List[PlayerRecord]]:
        """Execute the Final stage across 1 table of 6 players."""
        self.stage = Stage.FINAL
        self.current_hand = 0

        # Advance top 3 from A and top 3 from B with 100 BB reset
        final_table = self.advancement_manager.advance_semifinal_to_final(
            self.semifinal_standings["Table_A"],
            self.semifinal_standings["Table_B"],
            rng=self.rng,
        )
        self.current_tables = [final_table]
        self.active_players = final_table

        # Play final hands
        for hand_idx in range(self.final_hands):
            self.current_hand += 1
            button = hand_idx % self.table_size
            hand_seed = self.rng.randint(0, 10_000_000)
            self._play_table_hand(final_table, button, hand_seed, agent_map)

        champion, final_standings = self.advancement_manager.determine_champion(final_table)
        self.champion = champion
        self.final_standings = final_standings
        self.stage = Stage.FINISHED
        return champion, final_standings

    def run_full_tournament(self, agent_map: Dict[int, Any]) -> Dict[str, Any]:
        """Execute the complete 120-player tournament end to end."""
        self.reset(seed=self.seed)
        
        prelim = self.run_preliminary(agent_map)
        semi_a, semi_b = self.run_semifinal(agent_map)
        champ, final_standings = self.run_final(agent_map)

        return {
            "champion": champ,
            "final_standings": final_standings,
            "semifinal_table_a": semi_a,
            "semifinal_table_b": semi_b,
            "preliminary_top12": prelim[:12],
            "total_prelim_players": len(prelim),
        }
