from __future__ import annotations
from dataclasses import dataclass
import math
from random import Random
from .config import TournamentConfig
from .engine import NLHEngine
from .context import build_context, TableStrengthModel
from .pairing import random_groups, swiss_groups
from .scoring import Standing, bb100, rank_standings
from .strategy import StrategyAgent, StrategyParams

@dataclass
class SimAgent:
    agent_id:str
    strategy:StrategyAgent

class LeagueSimulator:
    def __init__(self, agents:list[SimAgent], sb=500, bb=1000, rounds=10, hands_per_round=20, seats=6, min_completion=0.80, seed=7, config: TournamentConfig | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = TournamentConfig(
                field_size=max(12, len(agents)),
                small_blind=sb,
                big_blind=bb,
                starting_stack_bb=100,
                seats_per_table=seats,
                preliminary_rounds=rounds,
                hands_per_round=hands_per_round,
                min_completion_rate=min_completion,
            )
        self.agents = agents
        self.rounds = self.config.preliminary_rounds
        self.hpr = self.config.hands_per_round
        self.seats = self.config.seats_per_table
        self.min_completion = self.config.min_completion_rate
        self.engine = NLHEngine(self.config.small_blind, self.config.big_blind, seed)
        self.rng = Random(seed)
        self.big_blind = self.config.big_blind

    def _state(self):
        return {
            a.agent_id: {
                'stack': 100 * self.big_blind,
                'initial_stack': 100 * self.big_blind,
                'rebuy_count': 0,
                'rebuy_cost_bb': 0.0,
                'gross_winnings_bb': 0.0,
                'gross_losses_bb': 0.0,
                'net_bb': 0.0,
                'hands': 0,
                'last_round_bb100': None,
                'from_r4_net_bb': 0.0,
                'from_r4_hands': 0,
            }
            for a in self.agents
        }

    def _standings(self, st):
        tie_round = {aid: float(v.get('from_r4_net_bb', 0.0) or 0.0) for aid, v in st.items()}
        rows = [Standing(aid, v['hands'], v['net_bb'], bb100(v['net_bb'], v['hands'])) for aid, v in st.items()]
        return rank_standings(rows, tie_round=tie_round)

    def _context(self, aid, st, round_no, hands_remaining):
        return self._context_map(st, round_no, hands_remaining).get(aid, {})

    def _context_map(self, st, round_no, hands_remaining, table_strength_map=None, active_ids=None):
        """Tournament context for every agent, computed once per hand."""
        rows = self._standings(st)
        r12 = next((x.bb100 for x in rows if x.rank == 12), None)
        r13 = next((x.bb100 for x in rows if x.rank == 13), None)
        ts_map = table_strength_map or {}
        target_rows = [x for x in rows if x.agent_id in active_ids] if active_ids is not None else rows
        return {
            x.agent_id: build_context(
                x.rank, x.bb100, r12, r13, hands_remaining, round_no,
                table_strength=ts_map.get(x.agent_id, 0.0),
                stage="preliminary",
                target_rank=self.config.semifinal_qualifiers,
                total_stage_hands=self.config.total_preliminary_hands,
            )
            for x in target_rows
        }

    def _play_group(self, group, st, round_no, hands_this_round):
        dealer = 0
        policies = {a.agent_id: a.strategy for a in self.agents}

        # Table strength metric for Swiss pairing and random rounds (strictly excluding Hero)
        ts_map = TableStrengthModel.compute_table_strength_map(group, st)

        group_set = set(group)
        for hand_i in range(hands_this_round):
            # Check for any busted players before the hand: auto-rebuy 100 BB
            for aid in group:
                if st[aid]['stack'] <= 0:
                    st[aid]['stack'] = 100 * self.big_blind
                    st[aid]['rebuy_count'] += 1
                    st[aid]['rebuy_cost_bb'] += 100.0

            before = {aid: st[aid]['stack'] for aid in group}
            hands_remaining = self.rounds * self.hpr - (round_no - 1) * self.hpr - hand_i
            provider = self._context_map(st, round_no, hands_remaining, table_strength_map=ts_map, active_ids=group_set).get
            res, dealer = self.engine.play_hand(
                group,
                {aid: st[aid]['stack'] for aid in group},
                dealer,
                policies,
                context_provider=provider,
            )
            for aid in group:
                st[aid]['stack'] = res.final_stacks[aid]
                delta = (st[aid]['stack'] - before[aid]) / self.big_blind
                if delta > 0:
                    st[aid]['gross_winnings_bb'] += delta
                elif delta < 0:
                    st[aid]['gross_losses_bb'] += (-delta)
                st[aid]['hands'] += 1
                if st[aid]['stack'] <= 0:
                    st[aid]['stack'] = 100 * self.big_blind
                    st[aid]['rebuy_count'] += 1
                    st[aid]['rebuy_cost_bb'] += 100.0

                # Unified tournament net calculation:
                st[aid]['net_bb'] = (st[aid]['stack'] - 100 * self.big_blind) / self.big_blind - st[aid]['rebuy_cost_bb']

    def run_preliminary(self):
        st = self._state()
        ids = list(st)
        for rnd in range(1, self.rounds + 1):
            standings = self._standings(st)
            groups = random_groups(ids, self.seats, self.rng) if rnd <= 3 else swiss_groups(ids, seats=self.seats, standings=standings)
            for g in groups:
                g = list(g)
                self.rng.shuffle(g)
                self._play_group(g, st, rnd, self.hpr)
            if rnd == 3:
                for aid, v in st.items():
                    v['_r3_net_bb'] = v['net_bb']
                    v['_r3_hands'] = v['hands']
            if rnd >= 4:
                for aid, v in st.items():
                    v['from_r4_net_bb'] = v['net_bb'] - v.get('_r3_net_bb', 0.0)
                    v['from_r4_hands'] = v['hands'] - v.get('_r3_hands', 0)
        return self._standings(st)

    def run_event(self):
        prelim = self.run_preliminary()
        min_hands = int(self.rounds * self.hpr * self.min_completion)
        qualified = [s for s in prelim if s.hands >= min_hands][:self.config.semifinal_qualifiers]
        result = {'preliminary': prelim, 'qualified': qualified, 'semifinal': [], 'final': []}
        if len(qualified) < self.config.semifinal_qualifiers:
            return result
        rank_map = {s.agent_id: s.rank for s in prelim}
        byid = {a.agent_id: a for a in self.agents}
        groups = [[qualified[i - 1].agent_id for i in (1, 4, 5, 8, 9, 12)], [qualified[i - 1].agent_id for i in (2, 3, 6, 7, 10, 11)]]
        finals = []
        for grp in groups:
            grp = list(grp)
            self.rng.shuffle(grp)
            local = {
                aid: {
                    'stack': 100 * self.big_blind,
                    'rebuy_count': 0,
                    'rebuy_cost_bb': 0.0,
                    'gross_winnings_bb': 0.0,
                    'gross_losses_bb': 0.0,
                    'net_bb': 0.0,
                    'hands': 0,
                }
                for aid in grp
            }
            # Semifinal match; rankings inside it use this match only.
            dealer = 0
            policies = {a.agent_id: a.strategy for a in self.agents}
            for h in range(self.config.semifinal_hands):
                for aid in grp:
                    if local[aid]['stack'] <= 0:
                        local[aid]['stack'] = 100 * self.big_blind
                        local[aid]['rebuy_count'] += 1
                        local[aid]['rebuy_cost_bb'] += 100.0
                before = {aid: local[aid]['stack'] for aid in grp}
                _h_sf = h
                _local_sf = local
                def _sf_ctx(aid, _local=_local_sf, _h=_h_sf):
                    rows = sorted(_local.items(), key=lambda x: -(x[1]['net_bb'] / max(1, x[1]['hands']) * 100 if x[1]['hands'] > 0 else float('-inf')))
                    rank = next((i + 1 for i, (a, _) in enumerate(rows) if a == aid), None)
                    b = (_local[aid]['net_bb'] / _local[aid]['hands'] * 100) if _local[aid]['hands'] > 0 else None
                    r3 = (rows[2][1]['net_bb'] / rows[2][1]['hands'] * 100) if len(rows) >= 3 and rows[2][1]['hands'] > 0 else None
                    r4 = (rows[3][1]['net_bb'] / rows[3][1]['hands'] * 100) if len(rows) >= 4 and rows[3][1]['hands'] > 0 else None
                    ts = TableStrengthModel.compute(grp, hero_id=aid, profiles=_local)
                    return build_context(
                        rank=rank, bb100=b, hands_remaining=self.config.semifinal_hands - _h, round_no=11,
                        stage="semifinal", target_rank=3, total_stage_hands=self.config.semifinal_hands,
                        rank3_bb100=r3, rank4_bb100=r4, table_strength=ts,
                    )
                res, dealer = self.engine.play_hand(grp, {aid: local[aid]['stack'] for aid in grp}, dealer, policies, context_provider=_sf_ctx)
                for aid in grp:
                    local[aid]['stack'] = res.final_stacks[aid]
                    delta = (local[aid]['stack'] - before[aid]) / self.big_blind
                    if delta > 0:
                        local[aid]['gross_winnings_bb'] += delta
                    elif delta < 0:
                        local[aid]['gross_losses_bb'] += (-delta)
                    local[aid]['hands'] += 1
                    if local[aid]['stack'] <= 0:
                        local[aid]['stack'] = 100 * self.big_blind
                        local[aid]['rebuy_count'] += 1
                        local[aid]['rebuy_cost_bb'] += 100.0
                    local[aid]['net_bb'] = (local[aid]['stack'] - 100 * self.big_blind) / self.big_blind - local[aid]['rebuy_cost_bb']
            rows = rank_standings([Standing(aid, v['hands'], v['net_bb'], bb100(v['net_bb'], v['hands'])) for aid, v in local.items()])
            rows.sort(key=lambda x: (-(x.bb100 or float('-inf')), rank_map.get(x.agent_id, 999), x.agent_id))
            for i, x in enumerate(rows, 1):
                x.rank = i
            result['semifinal'].append(rows)
            finals.extend(rows[:3])

        # Final uses fresh final match, no carry-over score.
        ids = [s.agent_id for s in finals]
        self.rng.shuffle(ids)
        local = {
            aid: {
                'stack': 100 * self.big_blind,
                'rebuy_count': 0,
                'rebuy_cost_bb': 0.0,
                'gross_winnings_bb': 0.0,
                'gross_losses_bb': 0.0,
                'net_bb': 0.0,
                'hands': 0,
            }
            for aid in ids
        }
        dealer = 0
        policies = {a.agent_id: a.strategy for a in self.agents}
        for h in range(self.config.final_hands):
            for aid in ids:
                if local[aid]['stack'] <= 0:
                    local[aid]['stack'] = 100 * self.big_blind
                    local[aid]['rebuy_count'] += 1
                    local[aid]['rebuy_cost_bb'] += 100.0
            before = {aid: local[aid]['stack'] for aid in ids}
            _h_f = h
            _local_f = local
            def _f_ctx(aid, _local=_local_f, _h=_h_f):
                rows = sorted(_local.items(), key=lambda x: -(x[1]['net_bb'] / max(1, x[1]['hands']) * 100 if x[1]['hands'] > 0 else float('-inf')))
                rank = next((i + 1 for i, (a, _) in enumerate(rows) if a == aid), None)
                b = (_local[aid]['net_bb'] / _local[aid]['hands'] * 100) if _local[aid]['hands'] > 0 else None
                r1 = (rows[0][1]['net_bb'] / rows[0][1]['hands'] * 100) if len(rows) >= 1 and rows[0][1]['hands'] > 0 else None
                r2 = (rows[1][1]['net_bb'] / rows[1][1]['hands'] * 100) if len(rows) >= 2 and rows[1][1]['hands'] > 0 else None
                ts = TableStrengthModel.compute(ids, hero_id=aid, profiles=_local)
                return build_context(
                    rank=rank, bb100=b, hands_remaining=self.config.final_hands - _h, round_no=12,
                    stage="final", target_rank=1, total_stage_hands=self.config.final_hands,
                    leader_bb100=r1, second_bb100=r2, rank1_bb100=r1, rank2_bb100=r2, table_strength=ts,
                )
            res, dealer = self.engine.play_hand(ids, {aid: local[aid]['stack'] for aid in ids}, dealer, policies, context_provider=_f_ctx)
            for aid in ids:
                local[aid]['stack'] = res.final_stacks[aid]
                delta = (local[aid]['stack'] - before[aid]) / self.big_blind
                if delta > 0:
                    local[aid]['gross_winnings_bb'] += delta
                elif delta < 0:
                    local[aid]['gross_losses_bb'] += (-delta)
                local[aid]['hands'] += 1
                if local[aid]['stack'] <= 0:
                    local[aid]['stack'] = 100 * self.big_blind
                    local[aid]['rebuy_count'] += 1
                    local[aid]['rebuy_cost_bb'] += 100.0
                local[aid]['net_bb'] = (local[aid]['stack'] - 100 * self.big_blind) / self.big_blind - local[aid]['rebuy_cost_bb']
        # Final tie-break by preliminary rank.
        fr = [Standing(aid, v['hands'], v['net_bb'], bb100(v['net_bb'], v['hands'])) for aid, v in local.items()]
        fr = sorted(fr, key=lambda x: (-(x.bb100 if x.bb100 is not None else float('-inf')), rank_map.get(x.agent_id, 999), x.agent_id))
        for i, x in enumerate(fr, 1):
            x.rank = i
        result['final'] = fr
        return result

    def evaluate(self, runs=100):
        if not self.agents: return {'top12_rate':0.0,'final_rate':0.0,'champion_rate':0.0}
        focal=self.agents[0].agent_id; top=final_rate=champ=0
        for i in range(runs):
            sim_agents=[SimAgent(a.agent_id,StrategyAgent(a.strategy.params,seed=1000+i*17+j,name=a.agent_id)) for j,a in enumerate(self.agents)]
            sim=LeagueSimulator(sim_agents,seed=self.engine.rng.randrange(10**9),config=self.config)
            r=sim.run_event()
            if focal in [x.agent_id for x in r['qualified']]: top+=1
            if focal in [x.agent_id for x in r['final']]: final_rate+=1
            if r['final'] and r['final'][0].agent_id==focal: champ+=1
        return {'top12_rate':top/max(1,runs),'final_rate':final_rate/max(1,runs),'champion_rate':champ/max(1,runs)}
