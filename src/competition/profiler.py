from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json, re, time
from typing import Any

@dataclass
class StatMetric:
    """Rigorous poker statistical metric container tracking exact opportunities, raw rates,
    Bayesian confidence, population priors, and smoothed shrinkage rates."""
    name: str
    count: int = 0
    opportunities: int = 0
    population_prior: float = 0.0
    prior_weight: float = 8.0

    def __post_init__(self) -> None:
        if self.opportunities < self.count:
            self.opportunities = self.count
        if self.opportunities < 0:
            self.opportunities = 0
        if self.count < 0:
            self.count = 0

    @property
    def has_sample(self) -> bool:
        """True only when at least one true opportunity has been observed."""
        return self.opportunities > 0

    @property
    def rate(self) -> float:
        if self.opportunities <= 0:
            return round(self.population_prior, 4)
        return round(min(1.0, max(0.0, self.count / self.opportunities)), 4)

    @property
    def confidence(self) -> float:
        if self.opportunities <= 0:
            return 0.0
        return round(min(1.0, max(0.0, self.opportunities / (self.opportunities + self.prior_weight))), 4)

    @property
    def smoothed_rate(self) -> float:
        denom = float(self.opportunities) + self.prior_weight
        if denom <= 0:
            return round(self.population_prior, 4)
        return round(min(1.0, max(0.0, (self.count + self.prior_weight * self.population_prior) / denom)), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "opportunities": self.opportunities,
            "has_sample": self.has_sample,
            "rate": self.rate,
            "confidence": self.confidence,
            "population_prior": self.population_prior,
            "smoothed_rate": self.smoothed_rate,
        }

class OpponentProfiler:
    """Extracts, aggregates, and Bayesian-smooths opponent profiles from hand history data
    using true poker opportunity-based denominators."""

    def __init__(self):
        self.stats: dict[str, dict[str, Any]] = {}
        self.names: dict[str, str] = {}

    @staticmethod
    def _new_stats() -> dict[str, Any]:
        return {
            "hands": 0,
            # Opportunity counters and event counts
            "preflop_opps": 0,
            "vpip_count": 0,
            "pfr_opps": 0,
            "pfr_count": 0,
            "threebet_opps": 0,
            "threebet_count": 0,
            "fold_to_threebet_opps": 0,
            "fold_to_threebet_count": 0,
            "cbet_opps": 0,
            "cbet_count": 0,
            "fold_to_cbet_opps": 0,
            "fold_to_cbet_count": 0,
            "turn_barrel_opps": 0,
            "turn_barrel_count": 0,
            "fold_to_turn_opps": 0,
            "fold_to_turn_count": 0,
            "river_bet_opps": 0,
            "river_bet_count": 0,
            "fold_to_river_opps": 0,
            "fold_to_river_count": 0,
            "wtsd_opps": 0,
            "wtsd_count": 0,
            "wsd_opps": 0,
            "wsd_count": 0,
            # Legacy raw action counts for full backward compatibility
            "vpip": 0,
            "pfr": 0,
            "raises": 0,
            "bets": 0,
            "calls": 0,
            "folds": 0,
            "checks": 0,
            "allins": 0,
            "wtsd": 0,
            "net_pnl": 0,
            "bb_sum": 0.0,
            "street_actions": {
                "preflop": {"bets": 0, "raises": 0, "calls": 0, "folds": 0, "checks": 0, "allins": 0},
                "flop": {"bets": 0, "raises": 0, "calls": 0, "folds": 0, "checks": 0, "allins": 0},
                "turn": {"bets": 0, "raises": 0, "calls": 0, "folds": 0, "checks": 0, "allins": 0},
                "river": {"bets": 0, "raises": 0, "calls": 0, "folds": 0, "checks": 0, "allins": 0},
            },
            # Bet-sizing samples
            "open_bb_samples": [],
            "cbet_frac_samples": [],
            "bet_frac_samples": [],
            "raise_frac_samples": [],
        }

    def seed_from_profiles(self, profiles: dict[str, Any]) -> int:
        """Load a previously exported profile file back into raw counts and opportunity accumulators."""
        seeded = 0
        for aid, p in (profiles or {}).items():
            if not isinstance(p, dict):
                continue
            st = self.stats.setdefault(aid, self._new_stats())
            hands = int(p.get("hands", 0) or 0)
            st["hands"] += hands

            st["preflop_opps"] += int(p.get("preflop_opps") if p.get("preflop_opps") is not None else p.get("vpip_opps", hands) or 0)
            st["vpip_count"] += int(p.get("vpip_count", p.get("vpip", 0)) or 0)
            st["pfr_opps"] += int(p.get("pfr_opps") if p.get("pfr_opps") is not None else hands)
            st["pfr_count"] += int(p.get("pfr_count", p.get("pfr", 0)) or 0)
            st["threebet_opps"] += int(p.get("threebet_opps", 0) or 0)
            st["threebet_count"] += int(p.get("threebet_count", 0) or 0)
            st["fold_to_threebet_opps"] += int(p.get("fold_to_threebet_opps", 0) or 0)
            st["fold_to_threebet_count"] += int(p.get("fold_to_threebet_count", 0) or 0)
            st["cbet_opps"] += int(p.get("cbet_opps", 0) or 0)
            st["cbet_count"] += int(p.get("cbet_count", 0) or 0)
            st["fold_to_cbet_opps"] += int(p.get("fold_to_cbet_opps", 0) or 0)
            st["fold_to_cbet_count"] += int(p.get("fold_to_cbet_count", 0) or 0)
            st["turn_barrel_opps"] += int(p.get("turn_barrel_opps", 0) or 0)
            st["turn_barrel_count"] += int(p.get("turn_barrel_count", 0) or 0)
            st["fold_to_turn_opps"] += int(p.get("fold_to_turn_opps", 0) or 0)
            st["fold_to_turn_count"] += int(p.get("fold_to_turn_count", 0) or 0)
            st["river_bet_opps"] += int(p.get("river_bet_opps", 0) or 0)
            st["river_bet_count"] += int(p.get("river_bet_count", 0) or 0)
            st["fold_to_river_opps"] += int(p.get("fold_to_river_opps", 0) or 0)
            st["fold_to_river_count"] += int(p.get("fold_to_river_count", 0) or 0)
            st["wtsd_opps"] += int(p.get("wtsd_opps", 0) or 0)
            st["wtsd_count"] += int(p.get("wtsd_count", p.get("wtsd", 0)) or 0)
            st["wsd_opps"] += int(p.get("wsd_opps", 0) or 0)
            st["wsd_count"] += int(p.get("wsd_count", 0) or 0)

            # Legacy raw action counts
            st["raises"] += int(p.get("raises_count", p.get("raises", 0)) or 0)
            st["bets"] += int(p.get("bets_count", p.get("bets", 0)) or 0)
            st["calls"] += int(p.get("calls_count", p.get("calls", 0)) or 0)
            st["folds"] += int(p.get("folds_count", p.get("folds", 0)) or 0)
            st["checks"] += int(p.get("checks_count", p.get("checks", 0)) or 0)
            st["allins"] += int(p.get("allins_count", p.get("allins", 0)) or 0)
            st["wtsd"] += int(p.get("wtsd_count", p.get("wtsd", 0)) or 0)
            st["vpip"] = st["vpip_count"]
            st["pfr"] = st["pfr_count"]

            net_pnl = int(p.get("net_pnl", 0) or 0)
            st["net_pnl"] += net_pnl

            bb_sum = float(p.get("bb_sum", 0.0) or 0.0)
            if bb_sum <= 0.0 and net_pnl and p.get("bb_100"):
                ratio = float(p["bb_100"]) / 100.0
                if ratio:
                    bb_sum = net_pnl / ratio
            st["bb_sum"] += bb_sum

            total_n = int(p.get("sizing_samples", 0) or 0)
            for sample_key, avg_key, count_key in (
                ("open_bb_samples", "open_size_bb", "open_size_n"),
                ("cbet_frac_samples", "cbet_size", "cbet_n"),
                ("bet_frac_samples", "value_bet_size", "bet_n"),
                ("raise_frac_samples", "raise_size", "raise_n"),
            ):
                avg = p.get(avg_key)
                if avg is None:
                    continue
                n = int(p.get(count_key, 0) or 0) or total_n // 4
                if n > 0:
                    st[sample_key].extend([float(avg)] * n)

            name = p.get("name")
            if name and name != "Unknown":
                self.names[aid] = name
            seeded += 1
        return seeded

    def ingest_hand(self, data: dict[str, Any]) -> None:
        """Ingest a single hand observation, raw event, or API hand dictionary with opportunity tracking."""
        if isinstance(data, dict) and "actions" in data and "players" in data:
            hand = data
            table = data
        else:
            table = data.get("table") if isinstance(data.get("table"), dict) else data
            hand = table.get("hand") if isinstance(table.get("hand"), dict) else data.get("hand", data)

        if not isinstance(hand, dict) or not hand.get("actions"):
            return

        players = table.get("players") or hand.get("players") or []
        participating_agents = set()
        for p in players:
            if isinstance(p, dict):
                aid = p.get("agentId")
                name = p.get("name")
                if aid:
                    participating_agents.add(aid)
                    if name:
                        self.names[aid] = name
            elif isinstance(p, str):
                participating_agents.add(p)

        for p in players:
            if isinstance(p, dict):
                aid = p.get("agentId")
                if aid:
                    st = self.stats.setdefault(aid, self._new_stats())
                    st["net_pnl"] += int(p.get("netChange", 0))

        actions = hand.get("actions") or []

        # Pot and street tracking
        pot = 0.0
        bb_size: float | None = None
        street_commit: dict[str, float] = {}
        cur_street = "preflop"
        street_raises = 0
        preflop_opener: str | None = None
        preflop_aggressor: str | None = None

        flop_first_bet_made = False
        flop_cbet_aggressor: str | None = None
        turn_first_bet_made = False
        turn_barrel_aggressor: str | None = None
        river_first_bet_made = False
        river_bet_aggressor: str | None = None

        active_players = set(participating_agents)
        saw_flop: set[str] = set()
        saw_turn: set[str] = set()
        saw_river: set[str] = set()
        showdown_players: set[str] = set()
        winning_players: set[str] = set()

        p_vpip: set[str] = set()
        p_pfr_opp: set[str] = set()
        p_pfr: set[str] = set()
        p_threebet_opp: set[str] = set()
        p_threebet: set[str] = set()
        p_fold_to_threebet_opp: set[str] = set()
        p_fold_to_threebet: set[str] = set()

        p_cbet_flop_opp: set[str] = set()
        p_cbet_flop: set[str] = set()
        p_fold_to_cbet_opp: set[str] = set()
        p_fold_to_cbet: set[str] = set()

        p_turn_barrel_opp: set[str] = set()
        p_turn_barrel: set[str] = set()
        p_fold_to_turn_opp: set[str] = set()
        p_fold_to_turn: set[str] = set()

        p_river_bet_opp: set[str] = set()
        p_river_bet: set[str] = set()
        p_fold_to_river_opp: set[str] = set()
        p_fold_to_river: set[str] = set()

        for a in actions:
            if not isinstance(a, dict):
                continue
            aid = a.get("agentId")
            if not aid:
                continue
            st = self.stats.setdefault(aid, self._new_stats())
            typ = a.get("type")
            street = a.get("street", "preflop")
            amount = float(a.get("amount") or 0)

            if street != cur_street:
                cur_street = street
                street_commit = {}
                street_raises = 0
                if street == "flop":
                    saw_flop.update(active_players)
                elif street == "turn":
                    saw_flop.update(active_players)
                    saw_turn.update(active_players)
                elif street == "river":
                    saw_flop.update(active_players)
                    saw_turn.update(active_players)
                    saw_river.update(active_players)

            if typ == "bigBlind" and amount > 0:
                bb_size = amount
            if typ in {"smallBlind", "bigBlind"}:
                pot += amount
                continue

            # Bet-sizing extraction
            if typ in {"raise", "bet", "allIn"} and amount > 0:
                if street == "preflop":
                    if street_raises == 0:
                        total = street_commit.get(aid, 0.0) + amount
                        if bb_size:
                            st["open_bb_samples"].append(total / bb_size)
                elif pot > 0:
                    frac = amount / pot
                    if street == "flop" and typ == "bet" and aid == preflop_aggressor:
                        st["cbet_frac_samples"].append(frac)
                    elif typ == "bet":
                        st["bet_frac_samples"].append(frac)
                    else:
                        st["raise_frac_samples"].append(frac)

            street_commit[aid] = street_commit.get(aid, 0.0) + amount
            pot += amount

            # Raw action counts & street action tracking
            s_acts = st["street_actions"].setdefault(street, {"bets": 0, "raises": 0, "calls": 0, "folds": 0, "checks": 0, "allins": 0})
            if typ in {"raise", "bet"}:
                st["raises"] += 1
                st["bets"] += 1
                s_acts["raises"] += 1
                s_acts["bets"] += 1
            elif typ == "allIn":
                st["allins"] += 1
                st["raises"] += 1
                st["bets"] += 1
                s_acts["allins"] += 1
                s_acts["raises"] += 1
                s_acts["bets"] += 1
            elif typ == "call":
                st["calls"] += 1
                s_acts["calls"] += 1
            elif typ == "fold":
                st["folds"] += 1
                s_acts["folds"] += 1
            elif typ == "check":
                st["checks"] += 1
                s_acts["checks"] += 1

            # Opportunity State Machine
            if street == "preflop":
                if typ in {"raise", "bet", "allIn"}:
                    p_vpip.add(aid)
                    if street_raises == 0:
                        # Open raise / 2-bet
                        p_pfr_opp.add(aid)
                        p_pfr.add(aid)
                        preflop_opener = aid
                        preflop_aggressor = aid
                        street_raises = 1
                    elif street_raises == 1:
                        # 3-bet facing open
                        p_threebet_opp.add(aid)
                        p_threebet.add(aid)
                        preflop_aggressor = aid
                        street_raises = 2
                    else:
                        street_raises += 1
                        preflop_aggressor = aid
                elif typ == "call":
                    p_vpip.add(aid)
                    if street_raises == 0:
                        p_pfr_opp.add(aid)
                    elif street_raises == 1:
                        p_threebet_opp.add(aid)
                    elif street_raises == 2 and aid == preflop_opener:
                        p_fold_to_threebet_opp.add(aid)
                elif typ == "fold":
                    active_players.discard(aid)
                    if street_raises == 0:
                        p_pfr_opp.add(aid)
                    elif street_raises == 1:
                        p_threebet_opp.add(aid)
                    elif street_raises == 2 and aid == preflop_opener:
                        p_fold_to_threebet_opp.add(aid)
                        p_fold_to_threebet.add(aid)
                elif typ == "check":
                    if street_raises == 0:
                        p_pfr_opp.add(aid)

            elif street == "flop":
                if typ == "check":
                    if not flop_first_bet_made and aid == preflop_aggressor:
                        p_cbet_flop_opp.add(aid)
                elif typ in {"bet", "raise", "allIn"}:
                    if not flop_first_bet_made:
                        flop_first_bet_made = True
                        if aid == preflop_aggressor:
                            p_cbet_flop_opp.add(aid)
                            p_cbet_flop.add(aid)
                            flop_cbet_aggressor = aid
                    else:
                        if flop_cbet_aggressor and aid != flop_cbet_aggressor:
                            p_fold_to_cbet_opp.add(aid)
                elif typ == "fold":
                    active_players.discard(aid)
                    if flop_cbet_aggressor and aid != flop_cbet_aggressor:
                        p_fold_to_cbet_opp.add(aid)
                        p_fold_to_cbet.add(aid)
                elif typ == "call":
                    if flop_cbet_aggressor and aid != flop_cbet_aggressor:
                        p_fold_to_cbet_opp.add(aid)

            elif street == "turn":
                if typ == "check":
                    if not turn_first_bet_made and aid == flop_cbet_aggressor and flop_cbet_aggressor:
                        p_turn_barrel_opp.add(aid)
                elif typ in {"bet", "raise", "allIn"}:
                    if not turn_first_bet_made:
                        turn_first_bet_made = True
                        if aid == flop_cbet_aggressor and flop_cbet_aggressor:
                            p_turn_barrel_opp.add(aid)
                            p_turn_barrel.add(aid)
                            turn_barrel_aggressor = aid
                    else:
                        if turn_barrel_aggressor and aid != turn_barrel_aggressor:
                            p_fold_to_turn_opp.add(aid)
                elif typ == "fold":
                    active_players.discard(aid)
                    if turn_barrel_aggressor and aid != turn_barrel_aggressor:
                        p_fold_to_turn_opp.add(aid)
                        p_fold_to_turn.add(aid)
                elif typ == "call":
                    if turn_barrel_aggressor and aid != turn_barrel_aggressor:
                        p_fold_to_turn_opp.add(aid)

            elif street == "river":
                if typ in {"bet", "raise", "allIn"}:
                    if not river_first_bet_made:
                        river_first_bet_made = True
                        p_river_bet_opp.add(aid)
                        p_river_bet.add(aid)
                        river_bet_aggressor = aid
                    else:
                        if river_bet_aggressor and aid != river_bet_aggressor:
                            p_fold_to_river_opp.add(aid)
                elif typ == "check":
                    if not river_first_bet_made:
                        p_river_bet_opp.add(aid)
                elif typ == "fold":
                    active_players.discard(aid)
                    if river_bet_aggressor and aid != river_bet_aggressor:
                        p_fold_to_river_opp.add(aid)
                        p_fold_to_river.add(aid)
                elif typ == "call":
                    if river_bet_aggressor and aid != river_bet_aggressor:
                        p_fold_to_river_opp.add(aid)

        # Showdown extraction: only when multiple players reach showdown or explicit showdown data exists
        showdowns = hand.get("showdown") or []
        if isinstance(showdowns, list) and len(showdowns) >= 2:
            for s in showdowns:
                if isinstance(s, dict) and s.get("agentId"):
                    showdown_players.add(s["agentId"])
                    if s.get("amount", 0) > 0:
                        winning_players.add(s["agentId"])
                elif isinstance(s, str):
                    showdown_players.add(s)
        elif isinstance(showdowns, dict) and len(showdowns) >= 2:
            for s_aid in showdowns:
                showdown_players.add(s_aid)

        if not showdown_players and saw_flop and len(active_players) > 1:
            showdown_players.update(active_players)

        # Winning players from payouts/results
        payouts = hand.get("payouts") or (hand.get("result") or {}).get("payouts") or []
        if isinstance(payouts, list):
            for p in payouts:
                if isinstance(p, dict) and p.get("agentId") and p.get("amount", 0) > 0:
                    winning_players.add(p["agentId"])
        elif isinstance(payouts, dict):
            for p_aid, amt in payouts.items():
                if amt > 0:
                    winning_players.add(p_aid)

        for p in players:
            if isinstance(p, dict):
                p_aid = p.get("agentId")
                if p_aid and p.get("netChange", 0) > 0:
                    winning_players.add(p_aid)

        # Big-blind sum accumulation
        if bb_size:
            for aid in participating_agents:
                if aid in self.stats:
                    self.stats[aid]["bb_sum"] += bb_size

        # Accumulate opportunities and counts for all participating agents
        for aid in participating_agents:
            st = self.stats.setdefault(aid, self._new_stats())
            st["hands"] += 1
            st["preflop_opps"] += 1
            if aid in p_vpip:
                st["vpip_count"] += 1
                st["vpip"] += 1
            if aid in p_pfr_opp:
                st["pfr_opps"] += 1
            if aid in p_pfr:
                st["pfr_count"] += 1
                st["pfr"] += 1
            if aid in p_threebet_opp:
                st["threebet_opps"] += 1
            if aid in p_threebet:
                st["threebet_count"] += 1
            if aid in p_fold_to_threebet_opp:
                st["fold_to_threebet_opps"] += 1
            if aid in p_fold_to_threebet:
                st["fold_to_threebet_count"] += 1
            if aid in p_cbet_flop_opp:
                st["cbet_opps"] += 1
            if aid in p_cbet_flop:
                st["cbet_count"] += 1
            if aid in p_fold_to_cbet_opp:
                st["fold_to_cbet_opps"] += 1
            if aid in p_fold_to_cbet:
                st["fold_to_cbet_count"] += 1
            if aid in p_turn_barrel_opp:
                st["turn_barrel_opps"] += 1
            if aid in p_turn_barrel:
                st["turn_barrel_count"] += 1
            if aid in p_fold_to_turn_opp:
                st["fold_to_turn_opps"] += 1
            if aid in p_fold_to_turn:
                st["fold_to_turn_count"] += 1
            if aid in p_river_bet_opp:
                st["river_bet_opps"] += 1
            if aid in p_river_bet:
                st["river_bet_count"] += 1
            if aid in p_fold_to_river_opp:
                st["fold_to_river_opps"] += 1
            if aid in p_fold_to_river:
                st["fold_to_river_count"] += 1
            if aid in saw_flop or aid in showdown_players:
                st["wtsd_opps"] += 1
            if aid in showdown_players:
                st["wtsd_count"] += 1
                st["wtsd"] += 1
                st["wsd_opps"] += 1
                if aid in winning_players:
                    st["wsd_count"] += 1

    def ingest_file(self, path: str | Path) -> int:
        """Parse a JSONL file containing raw events or exported hands."""
        p = Path(path)
        if not p.exists():
            return 0
        count = 0
        text = p.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            # Support both raw event {'data': {...}} and direct hand dict
            payload = item.get("data", item) if isinstance(item, dict) else item
            self.ingest_hand(payload)
            count += 1
        return count

    def pull_competition_hands(self, client: Any, cid: str, max_hands: int | None = None, save_hands_path: str | Path | None = "data/processed/hands.jsonl") -> int:
        """Fetch completed hands from GET /api/competitions/:cid/hands and ingest them."""
        cursor = None
        total_fetched = 0
        save_file = Path(save_hands_path) if save_hands_path else None
        if save_file:
            save_file.parent.mkdir(parents=True, exist_ok=True)
            f_out = save_file.open("w", encoding="utf-8")
        else:
            f_out = None

        try:
            while True:
                limit = 50
                if max_hands:
                    remaining = max_hands - total_fetched
                    if remaining <= 0:
                        break
                    limit = min(50, remaining)

                resp = None
                for attempt in range(4):
                    try:
                        resp = client.hands(cid, limit=limit, cursor=cursor)
                        break
                    except Exception as e:
                        if attempt == 3:
                            raise
                        time.sleep(1.0 * (attempt + 1))
                hands = resp.get("hands") or []
                if not hands:
                    break

                for h in hands:
                    self.ingest_hand(h)
                    if f_out:
                        f_out.write(json.dumps(h, ensure_ascii=False) + "\n")
                    total_fetched += 1

                if total_fetched % 500 == 0:
                    print(f"[Profiler] Pulled {total_fetched} hands from competition {cid}...")

                cursor = resp.get("nextCursor")
                if not cursor:
                    break
        finally:
            if f_out:
                f_out.close()

        return total_fetched

    def export(
        self,
        out_path: str | Path | None = None,
        prior_weight: float = 8.0,
        min_hands: int = 1,
        filter_afk: bool = False,
    ) -> dict[str, Any]:
        """Export computed smoothed profiles, optionally filtering low-quality agents, saving to JSON."""
        profiles = {}
        filtered = {}
        for aid, st in self.stats.items():
            name = self.names.get(aid, "Unknown")
            hands = st["hands"]

            # Quality Filter 1: Minimum sample size
            if min_hands > 1 and hands < min_hands:
                filtered[aid] = {"name": name, "hands": hands, "reason": f"Sample size too small ({hands} < {min_hands} hands)"}
                continue

            # Quality Filter 2: AFK / Zombie bot filter (disconnected bot auto-folding every hand)
            if filter_afk and hands >= 15:
                raw_fold_rate = st["folds"] / max(1, hands)
                if st["vpip"] == 0 and raw_fold_rate >= 0.95:
                    filtered[aid] = {"name": name, "hands": hands, "reason": f"AFK/Zombie bot (hands={hands}, vpip=0, folds={st['folds']})"}
                    continue

            n = max(1, st["hands"])
            w = max(1.0, float(prior_weight))
            v_opp = st.get("preflop_opps") if st.get("preflop_opps") is not None else hands
            pfr_opp = st.get("pfr_opps") if st.get("pfr_opps") is not None else hands

            metrics = {
                "vpip": StatMetric("vpip", st.get("vpip_count", st.get("vpip", 0)), v_opp, 0.25, w),
                "pfr": StatMetric("pfr", st.get("pfr_count", st.get("pfr", 0)), pfr_opp, 0.18, w),
                "threebet": StatMetric("threebet", st.get("threebet_count", 0), st.get("threebet_opps", 0), 0.08, 10.0),
                "fold_to_threebet": StatMetric("fold_to_threebet", st.get("fold_to_threebet_count", 0), st.get("fold_to_threebet_opps", 0), 0.55, 8.0),
                "cbet_flop": StatMetric("cbet_flop", st.get("cbet_count", 0), st.get("cbet_opps", 0), 0.55, 8.0),
                "fold_to_cbet": StatMetric("fold_to_cbet", st.get("fold_to_cbet_count", 0), st.get("fold_to_cbet_opps", 0), 0.45, 8.0),
                "turn_barrel": StatMetric("turn_barrel", st.get("turn_barrel_count", 0), st.get("turn_barrel_opps", 0), 0.45, 8.0),
                "fold_to_turn": StatMetric("fold_to_turn", st.get("fold_to_turn_count", 0), st.get("fold_to_turn_opps", 0), 0.42, 8.0),
                "river_bet": StatMetric("river_bet", st.get("river_bet_count", 0), st.get("river_bet_opps", 0), 0.38, 8.0),
                "fold_to_river": StatMetric("fold_to_river", st.get("fold_to_river_count", 0), st.get("fold_to_river_opps", 0), 0.40, 8.0),
                "wtsd": StatMetric("wtsd", st.get("wtsd_count", st.get("wtsd", 0)), st.get("wtsd_opps", 0), 0.30, 8.0),
                "wsd": StatMetric("wsd", st.get("wsd_count", 0), st.get("wsd_opps", 0), 0.50, 8.0),
            }

            denom = float(st["hands"]) + w
            svpip = metrics["vpip"].smoothed_rate
            spfr = metrics["pfr"].smoothed_rate
            sraise = (st["raises"] + w * 0.16) / denom
            sfold = (st["folds"] + w * 0.52) / denom
            scall = (st["calls"] + w * 0.32) / denom

            af = round((st["raises"] + st["bets"]) / max(1, st["calls"]), 2)
            net_pnl = st.get("net_pnl", 0)
            bb_100 = round(net_pnl / max(1.0, st.get("bb_sum", 0.0)) * 100, 1)

            def _avg(key: str, default: float, lo: float, hi: float) -> float:
                vals = st.get(key) or []
                if not vals:
                    return default
                return round(max(lo, min(hi, sum(vals) / len(vals))), 3)

            # Measured sizings
            open_size_bb = _avg("open_bb_samples", 2.35, 2.0, 6.0)
            cbet_size = _avg("cbet_frac_samples", 0.47, 0.15, 1.5)
            value_bet_size = _avg("bet_frac_samples", 0.69, 0.15, 1.5)
            raise_size = _avg("raise_frac_samples", 0.68, 0.15, 1.5)
            sizing_samples = (len(st.get("open_bb_samples") or []) + len(st.get("cbet_frac_samples") or [])
                              + len(st.get("bet_frac_samples") or []) + len(st.get("raise_frac_samples") or []))

            is_station = (scall >= 0.38 and sfold <= 0.36) or (metrics["fold_to_cbet"].smoothed_rate <= 0.35 and metrics["wtsd"].smoothed_rate >= 0.35)
            is_nit = svpip <= 0.18 and (sfold >= 0.58 or spfr <= 0.14)
            is_maniac = svpip >= 0.42 and (sraise >= 0.22 or metrics["threebet"].smoothed_rate >= 0.15 or af >= 2.5)
            is_passive = (sraise <= 0.10 or spfr <= 0.10) and af < 0.8
            is_pushfold = (st.get("allins", 0) / max(1, st["hands"])) >= 0.12

            if is_maniac:
                archetype = "Maniac (狂徒/高频乱搞)"
                advice = "绝不河牌纯诈唬；顶对/中对放宽抓诈(Bluff-catch)；手握强牌翻前翻后多做Check-raise引诱其推注。"
            elif is_nit:
                archetype = "Nit (极紧/岩石)"
                advice = "后位无脑偷盲捡底池；对其实施高频C-Bet；一旦其在转牌/河牌反常下注或加注，坚决盖掉中强牌。"
            elif is_station:
                archetype = "Calling Station (跟注站)"
                advice = "绝不进行三条街纯诈唬；顶对好踢脚做大价值下注(75%-100%底池)，他们会用弱对/听牌跟到底。"
            elif is_passive:
                archetype = "Passive (被动鱼)"
                advice = "一旦对方Check直接开枪抢池；对方主动下注或加注代表极强成牌，立即弃牌止损。"
            elif is_pushfold:
                archetype = "Push/Fold (短码梭哈客)"
                advice = "收紧跟注All-In范围至TT+、AQs+；在无弃牌率的情况下不要对其施加轻量诈唬。"
            elif svpip > 0.28 and sraise > 0.18:
                archetype = "LAG (松凶强手)"
                advice = "位置劣势避免玩投机牌；后位利用位置优势做3-bet施压，警惕其转牌河牌连开两枪。"
            elif 0.18 <= svpip <= 0.28 and sraise >= 0.12:
                archetype = "TAG (紧凶稳健)"
                advice = "尊重其早位Open和3-Bet范围；多在底池较小或有位置时针对其翻牌Miss的牌面浮动跟注偷底。"
            else:
                archetype = "Balanced (均衡常规)"
                advice = "按照标准GTO价值/诈唬比例对抗，重点观察其入池位置与下注尺度偏好。"

            profiles[aid] = {
                "name": self.names.get(aid, "Unknown"),
                "hands": st["hands"],
                "metrics": {k: m.to_dict() for k, m in metrics.items()},
                # Opportunity counters
                "vpip_opps": metrics["vpip"].opportunities,
                "vpip_count": metrics["vpip"].count,
                "pfr_opps": metrics["pfr"].opportunities,
                "pfr_count": metrics["pfr"].count,
                "threebet_opps": metrics["threebet"].opportunities,
                "threebet_count": metrics["threebet"].count,
                "fold_to_threebet_opps": metrics["fold_to_threebet"].opportunities,
                "fold_to_threebet_count": metrics["fold_to_threebet"].count,
                "cbet_opps": metrics["cbet_flop"].opportunities,
                "cbet_count": metrics["cbet_flop"].count,
                "fold_to_cbet_opps": metrics["fold_to_cbet"].opportunities,
                "fold_to_cbet_count": metrics["fold_to_cbet"].count,
                "turn_barrel_opps": metrics["turn_barrel"].opportunities,
                "turn_barrel_count": metrics["turn_barrel"].count,
                "fold_to_turn_opps": metrics["fold_to_turn"].opportunities,
                "fold_to_turn_count": metrics["fold_to_turn"].count,
                "river_bet_opps": metrics["river_bet"].opportunities,
                "river_bet_count": metrics["river_bet"].count,
                "fold_to_river_opps": metrics["fold_to_river"].opportunities,
                "fold_to_river_count": metrics["fold_to_river"].count,
                "wtsd_opps": metrics["wtsd"].opportunities,
                "wtsd_count": metrics["wtsd"].count,
                "wsd_opps": metrics["wsd"].opportunities,
                "wsd_count": metrics["wsd"].count,
                # Action counts
                "raises_count": st["raises"],
                "calls_count": st["calls"],
                "folds_count": st["folds"],
                "bets_count": st["bets"],
                "checks_count": st["checks"],
                "allins_count": st.get("allins", 0),
                "street_actions": st.get("street_actions", {}),
                "net_pnl": net_pnl,
                "bb_100": bb_100,
                "bb_sum": round(st.get("bb_sum", 0.0), 2),
                "open_size_n": len(st.get("open_bb_samples") or []),
                "cbet_n": len(st.get("cbet_frac_samples") or []),
                "bet_n": len(st.get("bet_frac_samples") or []),
                "raise_n": len(st.get("raise_frac_samples") or []),
                "vpip": round(svpip, 3),
                "pfr": round(spfr, 3),
                "threebet": round(metrics["threebet"].smoothed_rate, 3),
                "cbet_flop": round(metrics["cbet_flop"].smoothed_rate, 3),
                "fold_to_cbet": round(metrics["fold_to_cbet"].smoothed_rate, 3),
                "turn_barrel": round(metrics["turn_barrel"].smoothed_rate, 3),
                "fold_to_turn": round(metrics["fold_to_turn"].smoothed_rate, 3),
                "river_bet": round(metrics["river_bet"].smoothed_rate, 3),
                "fold_to_river": round(metrics["fold_to_river"].smoothed_rate, 3),
                "wtsd": round(metrics["wtsd"].smoothed_rate, 3),
                "wsd": round(metrics["wsd"].smoothed_rate, 3),
                "raise": round(sraise, 3),
                "fold": round(sfold, 3),
                "call": round(scall, 3),
                "af": af,
                "open_size_bb": open_size_bb,
                "cbet_size": cbet_size,
                "value_bet_size": value_bet_size,
                "raise_size": raise_size,
                "sizing_samples": sizing_samples,
                "archetype": archetype,
                "is_station": is_station,
                "is_nit": is_nit,
                "is_maniac": is_maniac,
                "is_passive": is_passive,
                "is_pushfold": is_pushfold,
                "exploit_advice": advice,
            }

        self.last_filtered = filtered
        if filtered:
            print(f"[Profiler] Quality Filter: Filtered out {len(filtered)} low-quality agents ({len(profiles)} retained).")
            for aid, info in filtered.items():
                print(f"  [Filtered] {info['name']:<16} ({info['reason']})")

        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")

        return profiles
