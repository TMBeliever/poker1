"""Rebuy management and chip accounting for ApexPoker tournaments."""

from src.tournament.state import PlayerRecord


class RebuyManager:
    """Deterministic Rebuy and accounting manager.
    
    Rules:
    1. Each player starts with 100 BB.
    2. Bust does not permanently eliminate players.
    3. Auto-rebuy immediately tops up player stack to 100 BB when stack <= 0 or < 1 BB.
    4. Net BB strictly tracks exact chip gains and losses:
       stage_net_bb += delta_bb
       cumulative_net_bb += delta_bb
       Zero-sum across all players is strictly invariant: sum(net_bb) == 0.0.
    """

    def __init__(self, starting_stack_bb: float = 50.0, min_stack_bb: float = 1.0):
        self.starting_stack_bb = starting_stack_bb
        self.min_stack_bb = min_stack_bb

    def compute_net_bb(self, player: PlayerRecord) -> float:
        """Compute net BB from current stage net."""
        return player.stage_net_bb

    def apply_hand_result(self, player: PlayerRecord, delta_bb: float) -> bool:
        """Apply chip changes from a completed hand and trigger auto-rebuy if needed.
        
        Args:
            player: Player record to update
            delta_bb: Net change in chips from the hand (in BB)
            
        Returns:
            True if auto-rebuy was triggered, False otherwise.
        """
        player.hands_played += 1
        player.stage_hands_played += 1
        player.stage_net_bb += delta_bb
        player.cumulative_net_bb += delta_bb
        player.stack_bb += delta_bb

        # Check if stack is busted or below playable threshold (e.g. all-in loss)
        rebought = False
        if player.stack_bb < self.min_stack_bb or delta_bb <= -(self.starting_stack_bb - 0.1):
            rebought = True
            player.rebuy_count += 1
            # Replenish stack back to 100 BB for next hand
            player.stack_bb = self.starting_stack_bb

        return rebought
