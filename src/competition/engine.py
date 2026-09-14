from __future__ import annotations
from dataclasses import dataclass, field
from random import Random
from typing import Any
from .cards import Card, deck, full_hand_rank

@dataclass
class Action:
    seat: int
    street: str
    type: str
    amount: int = 0
    is_forced: bool = False
    cause: str | None = None

@dataclass
class Player:
    agent_id: str
    seat: int
    stack: int
    hole: list[Card] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False
    committed_total: int = 0
    committed_round: int = 0

@dataclass
class HandResult:
    payouts: dict[str, int]
    returned_bets: dict[str, int]
    actions: list[Action]
    board: list[Card]
    final_stacks: dict[str, int]

class NLHEngine:
    """Deterministic local NLHE research engine.

    The engine intentionally mirrors the amount semantics used by Agent Poker:
    bet/raise amounts are the extra chips committed by the current action.
    """
    def __init__(self, small_blind: int = 500, big_blind: int = 1000, seed: int = 7):
        self.sb = small_blind
        self.bb = big_blind
        self.rng = Random(seed)
        # Monotonic per-engine hand counter, exposed to policies as obs["hand_id"] so that
        # stateful observers can tell one hand from the next.
        self.hand_seq = 0

    def play_hand(self, agents, stacks: dict[str, int], dealer_index: int, policies,
                  context_provider=None) -> tuple[HandResult, int]:
        ids = list(agents)
        n = len(ids)
        if not 2 <= n <= 10:
            raise ValueError("NLHE requires 2..10 players")
        players = [Player(a, i, int(stacks[a])) for i, a in enumerate(ids)]
        cards = deck(); self.rng.shuffle(cards)
        for p in players:
            p.hole = [cards.pop(), cards.pop()]

        self.hand_seq += 1
        board: list[Card] = []
        actions: list[Action] = []
        # Betting events are appended here as they happen and the *same list object* is
        # handed to the policy on every decision, so a stateful observer can consume only
        # what is new instead of rescanning the whole history (previously O(n^2) per hand).
        recent_actions: list[dict[str, Any]] = []
        seat_map = {p.seat: p.agent_id for p in players}

        def record(seat: int, street: str, typ: str, amount: int,
                   is_forced: bool = False, cause: str | None = None) -> None:
            actions.append(Action(seat, street, typ, amount, is_forced, cause))
            recent_actions.append({"agentId": seat_map.get(seat), "type": typ,
                                   "amount": amount, "street": street})

        pot = 0

        if n == 2:
            sb_seat = dealer_index
            bb_seat = (dealer_index + 1) % n
            current = dealer_index
        else:
            sb_seat = (dealer_index + 1) % n
            bb_seat = (dealer_index + 2) % n
            current = (bb_seat + 1) % n

        for seat, amt, name in ((sb_seat, self.sb, "smallBlind"), (bb_seat, self.bb, "bigBlind")):
            p = players[seat]
            put = min(p.stack, amt)
            p.stack -= put
            p.committed_round += put
            p.committed_total += put
            pot += put
            if p.stack == 0:
                p.all_in = True
            record(seat, "preflop", name, put, is_forced=True)

        current_bet = max(p.committed_round for p in players)
        last_raise = self.bb
        streets = [("preflop", 0), ("flop", 3), ("turn", 1), ("river", 1)]

        for si, (street, board_count) in enumerate(streets):
            if si > 0:
                cards.pop()  # burn
                board.extend(cards.pop() for _ in range(board_count))
                for p in players:
                    p.committed_round = 0
                current_bet = 0
                last_raise = self.bb
                current = (dealer_index + 1) % n

            if len([p for p in players if not p.folded]) <= 1:
                break
            if all(p.folded or p.all_in for p in players):
                break

            current = self._next_live(players, current)
            acted_since_raise: set[int] = set()
            action_count = 0
            while True:
                action_count += 1
                if action_count > 400:
                    # Fail closed in the local research engine rather than looping forever.
                    live = [p for p in players if not p.folded and not p.all_in]
                    for p in live:
                        if p.committed_round < current_bet:
                            p.folded = True
                            record(p.seat, street, "fold", 0, cause="engine_guard")
                    break
                live = [p for p in players if not p.folded and not p.all_in]
                if not live:
                    break
                if len(live) == 1 and live[0].committed_round == current_bet:
                    break
                p = players[current]
                if p.folded or p.all_in:
                    current = self._next_live(players, (current + 1) % n)
                    continue

                to_call = max(0, current_bet - p.committed_round)
                # Legality mirrors the real Agent Poker action space: "check" only exists
                # when there is nothing to call, and an opening wager is a "bet" rather
                # than a "raise". Getting this wrong silently turns folds into free
                # checks and bets into checks, so it is asserted by tests.
                if to_call > 0:
                    legal: dict[str, Any] = {"fold": None, "call": to_call}
                    open_type = "raise"
                else:
                    legal = {"check": None}
                    open_type = "bet"
                other_live = [q for q in live if q.seat != p.seat]
                if p.stack > 0:
                    if p.stack <= to_call:
                        legal["allIn"] = p.stack
                    elif other_live:
                        min_extra = min(p.stack, to_call + (self.bb if current_bet == 0 else last_raise))
                        if min_extra > to_call:
                            legal[open_type] = (min_extra, p.stack)

                ctx = context_provider(p.agent_id) if context_provider else {}
                obs = self._obs(players, board, p, current_bet, pot, legal, street, ctx,
                                dealer_index=dealer_index, recent_actions=recent_actions,
                                hand_id=self.hand_seq)
                decision = policies[p.agent_id].choose(obs)
                t = decision.get("type")
                if t not in legal:
                    # Defensive fallback. Fold is the safe default when facing a bet --
                    # never silently match a bet on behalf of a misbehaving policy.
                    t = "check" if "check" in legal else "fold"

                put = 0
                if t == "fold":
                    p.folded = True
                elif t == "check":
                    pass
                elif t == "call":
                    put = min(p.stack, to_call)
                elif t == "allIn":
                    put = p.stack
                elif t in ("bet", "raise"):
                    lo, hi = legal[t]
                    put = max(lo, min(hi, int(decision.get("amount", lo))))

                if put:
                    p.stack -= put
                    p.committed_round += put
                    p.committed_total += put
                    pot += put
                if p.stack == 0 and not p.folded:
                    p.all_in = True

                if t in ("bet", "raise"):
                    new_bet = p.committed_round
                    delta = new_bet - current_bet
                    if current_bet == 0 or delta >= last_raise:
                        last_raise = max(self.bb, delta)
                    current_bet = new_bet
                    acted_since_raise = {p.seat}
                else:
                    acted_since_raise.add(p.seat)

                record(p.seat, street, t, put)

                remaining = [q for q in players if not q.folded]
                if len(remaining) <= 1:
                    break
                active = [q for q in remaining if not q.all_in]
                if not active:
                    break
                if all(q.seat in acted_since_raise and q.committed_round == current_bet for q in active):
                    break
                current = self._next_live(players, (current + 1) % n)

            if len([p for p in players if not p.folded]) <= 1:
                break
            if all(p.folded or p.all_in for p in players):
                break

        if len([p for p in players if not p.folded]) > 1:
            while len(board) < 5:
                cards.pop()
                board.append(cards.pop())
        payouts, returned = self._settle(players, board)
        final_stacks = {
            p.agent_id: p.stack + payouts.get(p.agent_id, 0) + returned.get(p.agent_id, 0)
            for p in players
        }
        return HandResult(payouts, returned, actions, board, final_stacks), (dealer_index + 1) % n

    def _obs(self, players, board, me, current_bet, pot, legal, street, context=None,
             dealer_index=None, recent_actions=None, hand_id=None):
        # `recent_actions` is the live, incrementally-built list owned by play_hand. It is
        # passed by reference on purpose: rebuilding it per decision was the single
        # hottest path in the simulator. Observers must consume it with a cursor rather
        # than assume it is a private snapshot.
        return {
            "hand_id": hand_id,
            "context": context or {},
            "hero": [str(c) for c in me.hole],
            "board": [str(c) for c in board],
            "street": street,
            "pot": pot,
            "current_bet": current_bet,
            "stack": me.stack,
            "position": me.seat,
            "dealer_seat": dealer_index,
            "big_blind": self.bb,
            "agentId": me.agent_id,
            "recent_actions": recent_actions if recent_actions is not None else [],
            "players": [
                {
                    "agentId": p.agent_id,
                    "stack": p.stack,
                    "folded": p.folded,
                    "allIn": p.all_in,
                    "currentBet": p.committed_round,
                }
                for p in players
            ],
            "legal": legal,
        }

    @staticmethod
    def _next_live(players, start):
        n = len(players)
        for i in range(n):
            p = players[(start + i) % n]
            if not p.folded and not p.all_in:
                return p.seat
        return start

    def _settle(self, players, board):
        # First return a uniquely unmatched final contribution (uncalled excess).
        returned = {p.agent_id: 0 for p in players}
        totals = sorted({p.committed_total for p in players if p.committed_total > 0}, reverse=True)
        if totals:
            top = totals[0]
            max_players = [p for p in players if p.committed_total == top]
            second = totals[1] if len(totals) > 1 else 0
            if len(max_players) == 1 and top > second:
                p = max_players[0]
                excess = top - second
                p.committed_total -= excess
                # The unmatched excess leaves the pot. It is *credited* via `returned`
                # only -- play_hand adds `returned` on top of `p.stack`, so also mutating
                # p.stack here double-counts it and mints chips out of nothing.
                returned[p.agent_id] += excess

        levels = sorted({p.committed_total for p in players if p.committed_total > 0})
        prev = 0
        payouts = {p.agent_id: 0 for p in players}
        for level in levels:
            contributors = [p for p in players if p.committed_total >= level]
            amount = (level - prev) * len(contributors)
            prev = level
            if amount <= 0:
                continue
            eligible = [p for p in contributors if not p.folded]
            if not eligible:
                continue
            if len(eligible) == 1:
                payouts[eligible[0].agent_id] += amount
                continue
            ranks = {p.agent_id: full_hand_rank(p.hole + board) for p in eligible}
            best = max(ranks.values())
            winners = [p for p in eligible if ranks[p.agent_id] == best]
            share, remainder = divmod(amount, len(winners))
            for p in winners:
                payouts[p.agent_id] += share
            if remainder:
                payouts[winners[0].agent_id] += remainder
        return payouts, returned
