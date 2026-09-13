"""Swiss pairing and table assignment for tournament preliminary rounds."""

import random
from typing import List
from src.tournament.state import PlayerRecord
from src.tournament.ranking import RankingEngine


class SwissPairing:
    """Deterministic table pairing engine for 120-player preliminary stage.
    
    Rules:
    - R1-R3: Random table pairing across all active players.
    - R4-R10: Swiss pairing by cumulative standings (top 6 at Table 0, next 6 at Table 1, etc.).
    - Every table contains exactly table_size (6) players.
    """

    def __init__(self, table_size: int = 6, ranking_engine: RankingEngine = None):
        self.table_size = table_size
        self.ranking_engine = ranking_engine or RankingEngine()

    def pair_round(
        self,
        players: List[PlayerRecord],
        round_no: int,
        rng: random.Random,
    ) -> List[List[PlayerRecord]]:
        """Pair players into tables for the specified preliminary round (1-indexed).
        
        Args:
            players: List of active player records (must be divisible by table_size)
            round_no: Current preliminary round number (1 to 10)
            rng: Random instance for deterministic reproducibility
            
        Returns:
            List of tables, where each table is a List[PlayerRecord] of length table_size.
        """
        num_players = len(players)
        if num_players % self.table_size != 0:
            raise ValueError(
                f"Player count ({num_players}) must be a multiple of table size ({self.table_size})"
            )

        num_tables = num_players // self.table_size

        if round_no in (1, 2, 3):
            # R1-R3: Random pairing
            shuffled_players = list(players)
            rng.shuffle(shuffled_players)
            tables = [
                shuffled_players[i * self.table_size : (i + 1) * self.table_size]
                for i in range(num_tables)
            ]
        else:
            # R4-R10: Swiss pairing based on cumulative net BB standings
            ranked_players = self.ranking_engine.rank_players(players, use_stage_net=True)
            tables = []
            for i in range(num_tables):
                table_members = ranked_players[i * self.table_size : (i + 1) * self.table_size]
                # Shuffle seat positions within each table deterministically to avoid seat bias
                shuffled_seats = list(table_members)
                rng.shuffle(shuffled_seats)
                tables.append(shuffled_seats)

        # Update player table and seat metadata
        for table_id, table in enumerate(tables):
            for seat_id, p in enumerate(table):
                p.current_table_id = table_id
                p.seat_id = seat_id

        return tables
