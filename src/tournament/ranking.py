"""Ranking engine and tie-break management for ApexPoker tournaments."""

from typing import Dict, List, Optional
from src.tournament.state import PlayerRecord, Stage


class RankingEngine:
    """Deterministic ranking engine adhering to tournament tie-break rules.
    
    Tie-break hierarchy:
    1. Net BB (descending: higher net profit is better)
    2. Rebuy count (ascending: fewer rebuys is better)
    3. Stage hands played / completion rate (descending: more hands is better)
    4. Deterministic Player ID (ascending: lower ID is better for full determinism)
    """

    @staticmethod
    def _sort_key(player: PlayerRecord, use_stage_net: bool = True):
        net_bb = player.stage_net_bb if use_stage_net else player.cumulative_net_bb
        hands = player.stage_hands_played if use_stage_net else player.hands_played
        return (
            -round(net_bb, 4),           # Descending net BB
            player.rebuy_count,          # Ascending rebuy count (fewer is better)
            -hands,                      # Descending completion
            player.player_id,            # Ascending player_id
        )

    def rank_players(
        self,
        players: List[PlayerRecord],
        use_stage_net: bool = True,
    ) -> List[PlayerRecord]:
        """Return player records sorted according to deterministic tournament standings."""
        return sorted(players, key=lambda p: self._sort_key(p, use_stage_net=use_stage_net))

    def get_stage_cutoff(self, stage: Stage) -> int:
        """Return the qualification cutoff rank for a given stage."""
        if stage == Stage.PRELIMINARY:
            return 12
        elif stage == Stage.SEMIFINAL:
            return 3  # Top 3 per table in semifinal
        elif stage == Stage.FINAL:
            return 1  # Rank 1 is Champion
        return 1

    def compute_standings_and_margins(
        self,
        players: List[PlayerRecord],
        stage: Stage,
        use_stage_net: bool = True,
    ) -> Dict[int, Dict[str, float]]:
        """Compute rankings and margins for all active players in a stage.
        
        Returns:
            Dict mapping player_id -> {
                'rank': int,
                'cutoff_rank': int,
                'rank_margin_bb': float,
            }
        """
        sorted_players = self.rank_players(players, use_stage_net=use_stage_net)
        cutoff_rank = self.get_stage_cutoff(stage)
        
        # Determine cutoff player reference net BB
        cutoff_idx = min(cutoff_rank - 1, len(sorted_players) - 1)
        cutoff_ref_net = (
            sorted_players[cutoff_idx].stage_net_bb
            if use_stage_net
            else sorted_players[cutoff_idx].cumulative_net_bb
        )

        results = {}
        for rank_1idx, player in enumerate(sorted_players, start=1):
            player_net = player.stage_net_bb if use_stage_net else player.cumulative_net_bb
            margin = player_net - cutoff_ref_net
            results[player.player_id] = {
                "rank": rank_1idx,
                "cutoff_rank": cutoff_rank,
                "rank_margin_bb": margin,
            }
        return results

    def compute_table_rank(
        self,
        table_players: List[PlayerRecord],
        use_stage_net: bool = True,
    ) -> Dict[int, int]:
        """Compute rank within a single table (1-indexed)."""
        sorted_table = self.rank_players(table_players, use_stage_net=use_stage_net)
        return {p.player_id: rank for rank, p in enumerate(sorted_table, start=1)}
