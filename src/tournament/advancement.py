"""Advancement and stage transition manager for ApexPoker tournaments."""

import random
from typing import Dict, List, Tuple
from src.tournament.state import PlayerRecord, Stage
from src.tournament.ranking import RankingEngine


class AdvancementManager:
    """Manages player qualification, snake seeding, and stage transitions."""

    def __init__(self, ranking_engine: RankingEngine = None, starting_stack_bb: float = 50.0):
        self.ranking_engine = ranking_engine or RankingEngine()
        self.starting_stack_bb = starting_stack_bb

    def advance_preliminary_to_semifinal(
        self,
        all_prelim_players: List[PlayerRecord],
        rng: random.Random = None,
    ) -> Tuple[List[PlayerRecord], List[PlayerRecord]]:
        """Advance Top 12 preliminary players into Semifinal Table A and Table B using snake seeding.
        
        Seeding rule (1-indexed preliminary ranks):
        Table A: 1, 4, 5, 8, 9, 12
        Table B: 2, 3, 6, 7, 10, 11
        
        Both tables are independently reset to 100 BB.
        
        Returns:
            Tuple of (table_a_players, table_b_players)
        """
        ranked_players = self.ranking_engine.rank_players(all_prelim_players, use_stage_net=True)
        if len(ranked_players) < 12:
            raise ValueError(f"Need at least 12 players for semifinal, got {len(ranked_players)}")

        top12 = ranked_players[:12]
        
        # 1-indexed ranks to 0-indexed indices:
        # A: 1, 4, 5, 8, 9, 12 -> indices [0, 3, 4, 7, 8, 11]
        # B: 2, 3, 6, 7, 10, 11 -> indices [1, 2, 5, 6, 9, 10]
        table_a_indices = [0, 3, 4, 7, 8, 11]
        table_b_indices = [1, 2, 5, 6, 9, 10]

        table_a = [top12[i] for i in table_a_indices]
        table_b = [top12[i] for i in table_b_indices]

        # Reset each advancing player for semifinal (Rule 11: stage reset)
        for p in table_a + table_b:
            p.reset_for_stage(starting_stack_bb=self.starting_stack_bb)

        # Set table and seat IDs
        for seat_id, p in enumerate(table_a):
            p.current_table_id = 0  # Table A
            p.seat_id = seat_id

        for seat_id, p in enumerate(table_b):
            p.current_table_id = 1  # Table B
            p.seat_id = seat_id

        return table_a, table_b

    def advance_semifinal_to_final(
        self,
        table_a: List[PlayerRecord],
        table_b: List[PlayerRecord],
        rng: random.Random = None,
    ) -> List[PlayerRecord]:
        """Advance Top 3 from Table A and Top 3 from Table B to 6-player Final.
        
        Rule 12: Top 3 each advance (6 players total).
        Rule 13: Final resets to 100 BB.
        """
        ranked_a = self.ranking_engine.rank_players(table_a, use_stage_net=True)
        ranked_b = self.ranking_engine.rank_players(table_b, use_stage_net=True)

        top3_a = ranked_a[:3]
        top3_b = ranked_b[:3]

        final_table = top3_a + top3_b
        
        # Reset stack for final (Rule 13: stage reset)
        for p in final_table:
            p.reset_for_stage(starting_stack_bb=self.starting_stack_bb)

        # Assign final table and seats
        if rng is not None:
            rng.shuffle(final_table)

        for seat_id, p in enumerate(final_table):
            p.current_table_id = 0
            p.seat_id = seat_id

        return final_table

    def determine_champion(
        self,
        final_table: List[PlayerRecord],
    ) -> Tuple[PlayerRecord, List[PlayerRecord]]:
        """Determine final standings and champion based on Final table net profit.
        
        Rule 15: Final net BB rank 1 is Champion.
        """
        ranked_final = self.ranking_engine.rank_players(final_table, use_stage_net=True)
        champion = ranked_final[0]
        return champion, ranked_final
