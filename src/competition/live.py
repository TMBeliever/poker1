from __future__ import annotations
import time, copy, json, os
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .protocol import AgentPokerClient, APIError, Config
from .config import TournamentConfig, ExecutionMode
from .strategy import StrategyAgent
from .collector import JSONLCollector
from .profiler import OpponentProfiler
from .context import context_from_standings, synthetic_context

class LiveRunner:
    def __init__(
        self,
        client: AgentPokerClient,
        strategy: StrategyAgent | None = None,
        competition_id: str | None = None,
        collector: JSONLCollector | None = None,
        idle_seconds: int = 30,
        queued_seconds: int = 5,
        seated_seconds: int = 2,
        round_hands: int = 20,
        cycle_hands: int = 200,
        auto_profile: bool = True,
        profiles_path: str = "models/opponent_profiles.json",
        report_path: str = "data/live_reports.jsonl",
        strategy_path: str | None = None,
        config: TournamentConfig | None = None,
    ):
        self.config = config or TournamentConfig.official_120()
        self.client = client
        self.strategy_path = strategy_path or "models/champion.json"
        if strategy is not None:
            self.strategy = strategy
        else:
            if not os.path.exists(self.strategy_path):
                raise FileNotFoundError(
                    f"Production model not found at '{self.strategy_path}'. "
                    f"Live play strictly requires an official certified champion model and will not fallback to untrained defaults."
                )
            prof_file = profiles_path if profiles_path and os.path.exists(profiles_path) else None
            self.strategy = StrategyAgent.load(self.strategy_path, profiles=prof_file)

        self._strategy_mtime = os.path.getmtime(self.strategy_path) if os.path.exists(self.strategy_path) else 0.0
        self.cid = competition_id or client.cfg.competition_id
        self.table_id = None
        self.collector = collector
        self.idle = idle_seconds
        self.queued = queued_seconds
        self.seated = seated_seconds
        self.stop_requested = False

        # Round & Cycle tournament tracker
        self.round_hands = max(1, int(round_hands))
        self.cycle_hands = max(self.round_hands, int(cycle_hands))
        self.auto_profile = auto_profile
        self.profiles_path = profiles_path
        self.report_path = report_path
        self.profiler = OpponentProfiler()
        # Seed from the stored profile file so this session's hands accumulate onto
        # existing counts instead of replacing them (see seed_from_profiles).
        try:
            _pf = Path(profiles_path)
            if _pf.exists():
                self.profiler.seed_from_profiles(json.loads(_pf.read_text(encoding="utf-8")))
        except Exception:
            pass

        self.total_hands_played = 0
        self.current_round = 1
        self.current_cycle = 1
        self.round_net_bb = 0.0
        self.cycle_net_bb = 0.0
        self.session_net_bb = 0.0
        self.session_start = datetime.now(timezone.utc)
        self.best_stage_reached = "预选赛"
        self._report_written = False

        self.prev_hand_id: str | None = None
        self.hand_start_stack: int | None = None
        self.last_completed_hand_obj: dict[str, Any] | None = None
        self._standings_rows: list[dict[str, Any]] = []
        # Net BB for every player at hero's own table, accumulated across the current
        # 200-hand cycle only -- this is what "how did we do this run" actually means,
        # since a global standings lookup mixes in whatever happened before this session.
        self._table_cycle_bb: dict[str, float] = {}
        self.last_cycle_table_rank: tuple[int, int] | None = None
        self._hero_agent_id: str | None = None

        if not self.cid:
            raise ValueError('AGENTPOKER_COMPETITION_ID is required for live mode')

    def run(self, max_steps: int | None = None, max_hands: int | None = None):
        if not self.client.cfg.key:
            raise ValueError('AGENTPOKER_KEY is required')
        steps = 0
        obs = None
        print(f"[Live] Starting Agent loop for competition: {self.cid}")
        print(f"[Live] In-Place Round Controller activated: {self.round_hands} hands/round, {self.cycle_hands} hands/cycle")
        try:
            while True:
                if max_steps and steps >= max_steps:
                    return
                if max_hands and self.total_hands_played >= max_hands:
                    print(f"\n[Live] Reached target max_hands ({max_hands}). Safely stopping...")
                    if obs:
                        self._leave(obs)
                    self._final_report("max_hands_reached")
                    return

                steps += 1
                if obs is None:
                    self._join_until_ready()
                    obs = self._observe_competition()

                self._record(obs)
                self._track_hand_progress(obs)

                if obs.get('table'):
                    self.table_id = obs['table']['id']
                elif obs.get('status') != 'seated':
                    self.table_id = None

                if self.stop_requested:
                    self._leave(obs)
                    return

                # Remote table switch trigger (e.g. from Web Dashboard)
                switch_flag = Path("data/.switch_table_flag")
                if switch_flag.exists():
                    try:
                        switch_flag.unlink()
                    except Exception:
                        pass
                    if obs.get('table') or self.table_id:
                        print("\n[Live] ⚡ 收到换桌控制台指令，正在离开当前牌桌重新匹配...", flush=True)
                        self._leave(obs)
                        self.table_id = None
                        obs = self._observe_competition()
                        continue

                status = obs.get('status')
                cstatus = obs.get('competitionStatus')

                if status == 'idle':
                    if cstatus in ('ended', 'cancelled'):
                        print(f"[Live] Competition {cstatus}. Final bankroll: {obs.get('bankroll')}")
                        self._final_report(f"competition_{cstatus}")
                        return
                    time.sleep(self.idle)
                    obs = None
                    continue

                if status == 'queued':
                    self.table_id = None
                    time.sleep(self.queued)
                    obs = self._observe_competition()
                    continue

                if status == 'seated' and obs.get('actionRequest') is None:
                    time.sleep(self.seated)
                    obs = self._observe_current()
                    continue

                if obs.get('actionRequest') is not None:
                    body = self._make_action_body(obs)
                    dec = body.get('decision', {})
                    amt_str = f" amount={dec.get('amount')}" if 'amount' in dec else ""
                    print(f"[Live] Hand #{self.total_hands_played + 1} | Action -> {dec.get('type')}{amt_str}")
                    obs = self._action_with_retry(body, obs)
                    continue

                obs = self._observe_current()
        except KeyboardInterrupt:
            print("\n[Live] User interrupt received. Executing Leave Play...")
            if obs:
                self._leave(obs)
            elif self.table_id:
                try:
                    self.client.leave({'tableId': self.table_id})
                except Exception:
                    pass
            elif self.cid:
                try:
                    self.client.leave({'competitionId': self.cid})
                except Exception:
                    pass
            print("[Live] Safely left competition. Stopped.")
            self._final_report("user_interrupt")

    def _track_hand_progress(self, obs: dict[str, Any]) -> None:
        table = obs.get("table")
        if not table:
            return
        hand = table.get("hand") or {}
        hid = hand.get("id")
        if not hid:
            return

        hero_id = obs.get("agentId")
        if hero_id:
            self._hero_agent_id = hero_id
        players = table.get("players") or []
        hero_p = next((p for p in players if p.get("agentId") == hero_id), None)
        curr_stack = hero_p.get("stack") if hero_p else None
        bb_size = float(table.get("bigBlind") or 200.0)

        # Initial hand tracking
        if self.prev_hand_id is None:
            self.prev_hand_id = hid
            self.hand_start_stack = curr_stack
            self.last_completed_hand_obj = hand
            return

        # Detect new hand transition
        if hid != self.prev_hand_id:
            self.total_hands_played += 1

            # Ingest previous completed hand into profiler
            if self.last_completed_hand_obj:
                try:
                    self.profiler.ingest_hand({"table": table, "hand": self.last_completed_hand_obj})
                except Exception:
                    pass

            # Calculate chip delta
            net_change = None
            if self.last_completed_hand_obj:
                prev_players = self.last_completed_hand_obj.get("players") or []
                prev_hero = next((p for p in prev_players if p.get("agentId") == hero_id), None)
                if prev_hero and "netChange" in prev_hero and prev_hero["netChange"] is not None:
                    net_change = float(prev_hero["netChange"])

            if net_change is None:
                if curr_stack is not None and self.hand_start_stack is not None:
                    net_change = float(curr_stack - self.hand_start_stack)
                else:
                    net_change = 0.0

            delta_bb = net_change / bb_size
            self.round_net_bb += delta_bb
            self.cycle_net_bb += delta_bb
            self.session_net_bb += delta_bb

            # Track every player at this table for the current cycle, so we can rank
            # hero against actual tablemates instead of a global (and stale) standings call.
            if self.last_completed_hand_obj:
                for p in self.last_completed_hand_obj.get("players") or []:
                    aid = p.get("agentId")
                    nc = p.get("netChange")
                    if aid and nc is not None:
                        self._table_cycle_bb[aid] = self._table_cycle_bb.get(aid, 0.0) + float(nc) / bb_size

            # Check for round completion
            hands_in_round = self.total_hands_played % self.round_hands
            if hands_in_round == 0:
                self._on_round_complete(bb_size)

            # Check for cycle completion
            hands_in_cycle = self.total_hands_played % self.cycle_hands
            if hands_in_cycle == 0:
                self._on_cycle_complete()

            # Advance tracking pointers
            self.prev_hand_id = hid
            self.hand_start_stack = curr_stack
            self.last_completed_hand_obj = hand
        else:
            # Same hand, refresh snapshot
            self.last_completed_hand_obj = hand

    def _on_round_complete(self, bb_size: float) -> None:
        round_bb100 = (self.round_net_bb / max(1, self.round_hands)) * 100.0
        hands_in_cycle = self.total_hands_played % self.cycle_hands
        if hands_in_cycle == 0:
            hands_in_cycle = self.cycle_hands
        cycle_bb100 = (self.cycle_net_bb / max(1, hands_in_cycle)) * 100.0

        stage = "探索建仓期 (R1~R3)" if self.current_round <= 3 else ("积分保线期 (R4~R8)" if self.current_round <= 8 else "气泡冲线期 (R9~R10)")
        status_str = "🟢 稳居保线区" if cycle_bb100 >= 20.0 else "🔴 濒死冲线区"

        print("\n" + "=" * 68)
        print(f" 🏆 【第 {self.current_round} 轮预选赛结算】 (周期手牌: {hands_in_cycle}/{self.cycle_hands} 手 | 累计总局: {self.total_hands_played} 手)")
        print(f" • 比赛阶段: {stage} | 状态评估: {status_str}")
        print(f" • 本轮盈亏: {self.round_net_bb:+.1f} BB ({round_bb100:+.1f} BB/100)")
        print(f" • 赛季累计: {self.cycle_net_bb:+.1f} BB ({cycle_bb100:+.1f} BB/100) vs 晋级黄金线 (+20.0 BB/100)")

        if self.auto_profile:
            self._update_and_reload_profiles()
        self._refresh_standings()

        print("=" * 68 + "\n", flush=True)

        self.round_net_bb = 0.0
        self.current_round = (self.current_round % 10) + 1

    def _on_cycle_complete(self) -> None:
        cycle_bb100 = (self.cycle_net_bb / max(1, self.cycle_hands)) * 100.0
        qualified = cycle_bb100 >= 20.0
        qual_str = "🎉 成功锁定 TOP 12 晋级资格！" if qualified else "⚠️ 遗憾未达出线基准线 (+20.0 BB/100)"
        if qualified:
            self.best_stage_reached = "预选晋级 (Top 12)"

        rank, field = self._table_rank()
        if rank is not None:
            self.last_cycle_table_rank = (rank, field)

        print("\n" + "#" * 68)
        print(f" 🌟 【第 {self.current_cycle} 届虚拟锦标赛 200 手全赛季大结账】")
        print(f" • 赛季总战绩: {self.cycle_net_bb:+.1f} BB ({cycle_bb100:+.1f} BB/100)")
        print(f" • 晋级推演: {qual_str}")
        if rank is not None:
            print(f" • 本周期同桌排名: 第 {rank} 名 (共 {field} 人同桌)")
        print(f" • 下一届虚拟锦标赛 (Cycle {self.current_cycle + 1}) 原地平滑开启...")
        print("#" * 68 + "\n", flush=True)

        self.current_cycle += 1
        self.cycle_net_bb = 0.0
        self.current_round = 1
        self._table_cycle_bb = {}

    def _table_rank(self) -> tuple[int | None, int]:
        """Hero's rank by net BB among tablemates seen during the current cycle.

        Deliberately scoped to *this* 200-hand cycle and *this* table, not a global
        standings call -- a full-field rank mixes in every hand played before this
        session started, which answers a different question than "how did this run go".
        """
        rows = sorted(self._table_cycle_bb.items(), key=lambda kv: -kv[1])
        field = len(rows)
        hero_id = self._hero_agent_id
        if not hero_id or field == 0:
            return None, field
        rank = next((i + 1 for i, (aid, _) in enumerate(rows) if aid == hero_id), None)
        return rank, field

    def _final_report(self, reason: str) -> None:
        """Print and persist a session summary so results are comparable across runs.

        Without this, `run()` returning left no trace beyond the per-round prints
        that scroll past during play -- there was no single number to compare this
        session against a previous one, or against the `evaluate` command's
        simulated fitness for the same strategy.
        """
        if self._report_written:
            return
        self._report_written = True

        hands = self.total_hands_played
        session_bb100 = (self.session_net_bb / hands * 100.0) if hands > 0 else 0.0
        duration_s = (datetime.now(timezone.utc) - self.session_start).total_seconds()

        # Prefer the rank captured when the last full 200-hand cycle closed; if the
        # session stopped mid-cycle, fall back to ranking against whatever tablemates
        # were seen so far in the still-open cycle.
        if self.last_cycle_table_rank is not None:
            final_rank, field_size = self.last_cycle_table_rank
        else:
            final_rank, field_size = self._table_rank()

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "competition_id": self.cid,
            "stop_reason": reason,
            "duration_seconds": round(duration_s, 1),
            "hands_played": hands,
            "session_net_bb": round(self.session_net_bb, 2),
            "session_bb100": round(session_bb100, 2),
            "best_stage_reached": self.best_stage_reached,
            "table_rank": final_rank,
            "table_field_size": field_size,
        }

        print("\n" + "*" * 68)
        print(" 📋 【本场对局总结】")
        print(f" • 停止原因: {reason} | 用时: {duration_s/60:.1f} 分钟")
        print(f" • 总手数: {hands} 手 | 总盈亏: {self.session_net_bb:+.1f} BB ({session_bb100:+.1f} BB/100)")
        print(f" • 最高阶段: {self.best_stage_reached}")
        if final_rank is not None:
            print(f" • 本周期同桌排名: 第 {final_rank} 名 (共 {field_size} 人同桌)")

        history = self._append_report(report)
        if len(history) > 1:
            prev = history[-2]
            prev_bb100 = prev.get("session_bb100")
            if prev_bb100 is not None:
                delta = session_bb100 - prev_bb100
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                print(f" • 对比上一场 ({prev.get('timestamp','?')[:10]}): {prev_bb100:+.1f} BB/100 {arrow} 本场 {delta:+.1f} BB/100 变化")
        print(f" • 已记录到 {self.report_path} (历史场次: {len(history)})")
        print("*" * 68 + "\n", flush=True)

    def _append_report(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        p = Path(self.report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    history.append(json.loads(line))
                except Exception:
                    continue
        history.append(report)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
        return history

    def _update_and_reload_profiles(self) -> None:
        """Fold the hands seen so far into the profile file and reload the strategy.

        The profiler was seeded from this file at startup, so its export already holds
        stored history *plus* this session -- writing it back merges the two rather
        than overwriting the file with a session-only snapshot.
        """
        try:
            profiles = self.profiler.export(out_path=self.profiles_path, prior_weight=8.0, min_hands=1, filter_afk=False)
            loaded = self.strategy.load_opponent_profiles(profiles)
            print(f" • [画像热更] 画像库已累计更新至 {len(profiles)} 位对手 (策略已加载 {loaded} 位)")
        except Exception as e:
            print(f" • [画像热更] 提示: 增量更新画像跳过 ({e})")

    def _check_and_reload_strategy(self) -> None:
        """Hot-reload strategy parameters if the strategy file on disk has been updated."""
        if not self.strategy_path or not os.path.exists(self.strategy_path):
            return
        try:
            mtime = os.path.getmtime(self.strategy_path)
            if mtime > self._strategy_mtime:
                prof_file = self.profiles_path if self.profiles_path and os.path.exists(self.profiles_path) else None
                fresh_agent = StrategyAgent.load(self.strategy_path, profiles=prof_file)
                old_equity = getattr(self.strategy.params, 'equity_samples', 0)
                if old_equity:
                    fresh_agent.params.equity_samples = old_equity
                self.strategy.params = fresh_agent.params
                self._strategy_mtime = mtime
                print(f"\n[Live] ⚡ 策略热重载成功！已在比赛中平滑切换至新模型: {self.strategy_path} (VPIP={self.strategy.params.vpip:.1%}, 偷盲={self.strategy.params.steal_frequency:.1%})", flush=True)
        except Exception as e:
            pass

    def _join_until_ready(self):
        while True:
            try:
                r = self.client.join(self.cid)
            except APIError as e:
                if e.code == 'registration_required':
                    raise
                if e.code == 'competition_not_active':
                    self.cid = self._rediscover()
                    continue
                if e.code == 'insufficient_bankroll':
                    raise
                raise
            if r.get('status') in ('queued', 'seated'):
                return
            time.sleep(self.idle)

    def _rediscover(self):
        data = self.client.discover('active')
        cs = data.get('competitions') or []
        if not cs:
            raise RuntimeError('no active competitions')
        if self.cid and any(c.get('id') == self.cid for c in cs):
            return self.cid
        return cs[0]['id']

    def _observe_competition(self):
        try:
            return self.client.observe(self.cid)
        except APIError as e:
            if e.code == 'stale_table':
                return self.client.observe(self.cid)
            if e.code == 'participation_not_found':
                self._join_until_ready()
                return self.client.observe(self.cid)
            raise

    def _observe_current(self):
        if not self.table_id:
            return self._observe_competition()
        try:
            return self.client.observe(self.cid, self.table_id)
        except APIError as e:
            if e.code == 'stale_table':
                self.table_id = None
                return self.client.observe(self.cid)
            raise

    def _tournament_context(self, hero_id: str | None = None) -> dict[str, Any]:
        """Context for the current decision, via the shared builder.

        Prefers real standings when the API exposes a usable BB/100 ranking, and
        otherwise infers a rank through the same calibrated ladder the simulator's
        ranks came from. The previous five-bucket ladder with hardcoded 20.0 / 15.0
        golden lines did not match the distribution the policy was trained on.
        """
        hands_in_cycle = self.total_hands_played % self.cycle_hands
        rem = max(0, self.cycle_hands - hands_in_cycle)
        cycle_bb100 = (self.cycle_net_bb / max(1, hands_in_cycle)) * 100.0 if hands_in_cycle > 0 else 0.0

        ctx = None
        if self._standings_rows:
            ctx = context_from_standings(self._standings_rows, hero_id,
                                         rem, self.current_round, self.current_cycle)
        if ctx is None:
            ctx = synthetic_context(cycle_bb100, rem, self.current_round, self.current_cycle)
        return ctx

    _get_tournament_context = _tournament_context

    def _refresh_standings(self) -> None:
        """Best-effort standings refresh; failures leave the synthetic ladder in place."""
        try:
            data = self.client.standings(self.cid)
        except Exception:
            self._standings_rows = []
            return
        rows = data.get("standings") or data.get("rows") or data.get("competitors") or []
        self._standings_rows = rows if isinstance(rows, list) else []

    @staticmethod
    def _fallback_action(legal: dict[str, Any]) -> dict[str, Any]:
        """Safest guaranteed legal action in priority order: check -> call(0) -> fold -> call -> allIn."""
        if 'check' in legal:
            return {'type': 'check'}
        if 'call' in legal:
            call_spec = legal['call']
            c_amt = int(call_spec.get('amount', 0)) if isinstance(call_spec, dict) else 0
            if c_amt == 0:
                return {'type': 'call'}
        if 'fold' in legal:
            return {'type': 'fold'}
        if 'call' in legal:
            return {'type': 'call'}
        if 'allIn' in legal:
            return {'type': 'allIn'}
        for k in ('raise', 'bet'):
            if k in legal:
                spec = legal[k]
                lo = int(spec.get('minAmount', 1)) if isinstance(spec, dict) else 1
                return {'type': k, 'amount': lo}
        raise RuntimeError("No legal actions available in actionRequest")

    def _make_action_body(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._check_and_reload_strategy()
        obs["tournamentContext"] = self._tournament_context(obs.get("agentId"))

        req = copy.deepcopy(obs['actionRequest'])
        allowed = req.get('allowedActions') or []
        legal = {a['type']: a for a in allowed if isinstance(a, dict) and a.get('type')}

        try:
            decision = self.strategy.choose(obs)
        except Exception as e:
            print(f"[Live] Warning: strategy error: {e}, falling back to safe action")
            decision = self._fallback_action(legal)

        if not decision or decision.get('type') not in legal:
            print(f"[Live] Warning: illegal action {decision} generated, falling back to safe action")
            decision = self._fallback_action(legal)

        if decision['type'] in ('bet', 'raise'):
            spec = legal[decision['type']]
            lo = int(spec.get('minAmount', 1)) if isinstance(spec, dict) else 1
            hi = int(spec.get('maxAmount', lo)) if isinstance(spec, dict) else lo
            amt = int(decision.get('amount', lo))
            if not (lo <= amt <= hi) or amt <= 0:
                decision['amount'] = max(lo, min(hi, amt)) if lo <= hi else lo
        elif 'amount' in decision:
            decision.pop('amount', None)
        return {
            'competitionId': obs['competitionId'],
            'tableId': obs['table']['id'],
            'actionRequestId': req['id'],
            'decision': decision,
        }

    def _action_with_retry(self, body, original_obs, max_attempts: int = 4):
        frozen = json.loads(json.dumps(body, ensure_ascii=False))
        attempts = 0
        backoff = 0.5
        while True:
            attempts += 1
            try:
                return self.client.action(frozen)
            except requests.RequestException as net_err:
                if attempts >= max_attempts:
                    print(f"[Live] Action network error after {attempts} attempts: {net_err}, refreshing table state")
                    return self._observe_current()
                time.sleep(backoff)
                backoff = min(8.0, backoff * 2.0)
                continue
            except APIError as e:
                if e.code in ('temporarily_unavailable', 'service_unavailable', 'gateway_timeout') or e.status in (502, 503, 504):
                    if attempts >= max_attempts:
                        return self._observe_current()
                    time.sleep(e.retry_after or backoff)
                    backoff = min(8.0, backoff * 2.0)
                    continue
                if e.code in ('stale_action_request', 'idempotency_conflict', 'action_timeout', 'action_expired'):
                    return self._observe_current()
                if e.code == 'stale_table':
                    self.table_id = None
                    return self.client.observe(self.cid)
                if e.code == 'invalid_decision':
                    fresh = self._observe_current()
                    if fresh.get('actionRequest'):
                        return self.client.action(self._make_action_body(fresh))
                    return fresh
                if e.code in ('authentication_required', 'insufficient_bankroll'):
                    raise
                if attempts >= max_attempts:
                    raise
                time.sleep(backoff)
                backoff = min(8.0, backoff * 2.0)

    def _leave(self, obs):
        body = {'tableId': obs['table']['id']} if obs.get('table') else {'competitionId': self.cid}
        while True:
            try:
                r = self.client.leave(body)
                if r.get('status') == 'accepted':
                    return
            except APIError as e:
                if e.code == 'stale_table' and 'tableId' in body:
                    body = {'competitionId': self.cid}
                    continue
                if e.code == 'temporarily_unavailable':
                    time.sleep(e.retry_after or 1)
                    continue
                raise

    def _record(self, obs):
        if self.collector:
            self.collector.write({'kind': 'observation', 'data': obs})
