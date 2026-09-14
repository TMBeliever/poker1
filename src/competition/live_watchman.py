"""Live Watchman Loop adhering strictly to skill.md Sections 3, 4, 5, 6.

Manages:
1. Join with idempotent retry on idle.
2. Queued wait and table discovery.
3. Seated table polling.
4. Action request handling and submission with exact-body retry.
5. Graceful Leave Play on interrupt or target hand completion.
"""

from __future__ import annotations
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .protocol import AgentPokerClient, APIError, Config
from .cfr_bridge import CFRCompetitionBridge
from .soul import SoulManager
from .context import context_from_standings, synthetic_context


class CompetitionWatchman:
    """Live game loop runner implementing the skill.md watchman lifecycle."""

    def __init__(
        self,
        client: AgentPokerClient,
        competition_id: str,
        bridge: Optional[CFRCompetitionBridge] = None,
        model_path: str = "models/base/base_checkpoint_iter_2000.pt",
        idle_seconds: int = 30,
        queued_seconds: int = 5,
        seated_seconds: int = 2,
        round_hands: int = 20,
        cycle_hands: int = 200,
        report_path: str = "data/live_reports.jsonl",
    ):
        self.client = client
        self.cid = competition_id
        self.soul = SoulManager()
        self.bridge = bridge or CFRCompetitionBridge(model_path=model_path, soul_manager=self.soul)
        self.idle = idle_seconds
        self.queued = queued_seconds
        self.seated = seated_seconds
        self.round_hands = round_hands
        self.cycle_hands = cycle_hands
        self.report_path = report_path

        self.table_id: Optional[str] = None
        self.stop_requested = False
        self.total_hands = 0
        self.session_net_bb = 0.0
        self.round_net_bb = 0.0
        self.cycle_net_bb = 0.0
        self.session_start = datetime.now(timezone.utc)
        self.prev_hand_id: Optional[str] = None
        self.hand_start_stack: Optional[int] = None
        self._hero_agent_id: Optional[str] = None
        self._report_written = False

    def run(self, max_steps: Optional[int] = None, max_hands: Optional[int] = None) -> None:
        """Execute the primary watchman loop."""
        print("=" * 70)
        print(f" [ApexPoker Live] Entering Competition Watchman")
        print(f" Competition ID:  {self.cid}")
        print(f" Model Engine:    {self.bridge.model_path}")
        print(f" Round/Cycle:     {self.round_hands} hands/round, {self.cycle_hands} hands/cycle")
        print("=" * 70)

        steps = 0
        obs: Optional[Dict[str, Any]] = None

        try:
            while True:
                if max_steps and steps >= max_steps:
                    print(f"[Live] Reached max steps ({max_steps}). Exiting.")
                    if obs:
                        self._leave(obs)
                    return

                if max_hands and self.total_hands >= max_hands:
                    print(f"\n[Live] Reached target hands ({max_hands}). Safely stopping...")
                    if obs:
                        self._leave(obs)
                    self._final_report("max_hands_reached")
                    return

                steps += 1
                if obs is None:
                    self._join_until_ready()
                    obs = self._observe_competition()

                self._track_hand_progress(obs)

                if obs.get("table"):
                    self.table_id = obs["table"]["id"]
                elif obs.get("status") != "seated":
                    self.table_id = None

                if self.stop_requested:
                    self._leave(obs)
                    return

                status = obs.get("status")
                cstatus = obs.get("competitionStatus")

                # State 1: Idle
                if status == "idle":
                    if cstatus in ("ended", "cancelled"):
                        print(f"[Live] Competition {cstatus}. Final bankroll: {obs.get('bankroll')}")
                        self._final_report(f"competition_{cstatus}")
                        return
                    print(f"[Live] Status is 'idle' (waiting for queue/table opening). Sleeping {self.idle}s...")
                    time.sleep(self.idle)
                    obs = None
                    continue

                # State 2: Queued
                if status == "queued":
                    self.table_id = None
                    print(f"[Live] Status is 'queued' (waiting for seating). Sleeping {self.queued}s...")
                    time.sleep(self.queued)
                    obs = self._observe_competition()
                    continue

                # State 3: Seated but no action request
                if status == "seated" and obs.get("actionRequest") is None:
                    time.sleep(self.seated)
                    obs = self._observe_current()
                    continue

                # State 4: Action Request active!
                if obs.get("actionRequest") is not None:
                    body = self._make_action_body(obs)
                    dec = body.get("decision", {})
                    amt_str = f" amount={dec.get('amount')}" if "amount" in dec else ""
                    chat_str = f" | Chat: \"{body.get('chat')}\"" if body.get("chat") else ""
                    print(f"[Live] Hand #{self.total_hands + 1} | Decision -> {dec.get('type')}{amt_str}{chat_str}")
                    obs = self._action_with_retry(body, obs)
                    continue

                # Fallback: observe
                obs = self._observe_current()

        except KeyboardInterrupt:
            print("\n[Live] User interrupt received. Executing safe Leave Play...")
            if obs:
                self._leave(obs)
            elif self.table_id:
                try:
                    self.client.leave({"tableId": self.table_id})
                except Exception:
                    pass
            elif self.cid:
                try:
                    self.client.leave({"competitionId": self.cid})
                except Exception:
                    pass
            print("[Live] Safely left competition. Stopped.")
            self._final_report("user_interrupt")

    def _join_until_ready(self) -> None:
        """Call POST /api/competitions/join idempotently until queued or seated."""
        while True:
            try:
                res = self.client.join(self.cid)
                status = res.get("status")
                print(f"[Live] Joined competition '{self.cid}' -> status: {status}")
                if status in ("queued", "seated"):
                    return
                print(f"[Live] Join returned '{status}'. Polling again in {self.idle}s...")
                time.sleep(self.idle)
            except APIError as e:
                if e.code == "registration_required" or e.status == 409:
                    print(f"[Live] Error: Registration required for competition {self.cid}. Please register on the web first.")
                    raise
                elif e.code == "insufficient_bankroll":
                    print(f"[Live] Error: Insufficient bankroll to join competition {self.cid}.")
                    raise
                print(f"[Live] Join warning ({e.code}): {e.message}. Retrying in {self.idle}s...")
                time.sleep(self.idle)
            except Exception as net_e:
                print(f"[Live] Network warning during Join: {net_e}. Retrying in 10s...")
                time.sleep(10)

    def _observe_competition(self) -> Dict[str, Any]:
        """Competition-only observe (no tableId) to discover table or refresh queue."""
        try:
            return self.client.observe(self.cid)
        except Exception as e:
            print(f"[Live] Observe competition warning: {e}")
            time.sleep(self.queued)
            return {"status": "queued"}

    def _observe_current(self) -> Dict[str, Any]:
        """Table observe if table_id is known, else competition-only observe."""
        if self.table_id:
            try:
                return self.client.observe(self.cid, self.table_id)
            except APIError as e:
                if e.code == "stale_table" or e.status == 409:
                    print(f"[Live] Table {self.table_id} is stale, switching to competition-only observe.")
                    self.table_id = None
                    return self._observe_competition()
                raise
        return self._observe_competition()

    def _get_tournament_context(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Construct tournament context from live competition state or synthetic ladder."""
        hands_in_cycle = self.total_hands % self.cycle_hands
        rem = max(0, self.cycle_hands - hands_in_cycle)
        cycle_bb100 = (self.cycle_net_bb / max(1, hands_in_cycle)) * 100.0 if hands_in_cycle > 0 else 0.0
        current_round = (self.total_hands // max(1, self.round_hands)) + 1
        current_cycle = (self.total_hands // max(1, self.cycle_hands)) + 1

        rows = []
        try:
            standings_res = self.client.standings(self.cid)
            rows = standings_res.get("standings") or standings_res.get("rows") or []
        except Exception:
            rows = []

        hero_id = obs.get("agentId")
        if rows:
            ctx = context_from_standings(rows, hero_id, rem, current_round, current_cycle)
            if ctx:
                return ctx if isinstance(ctx, dict) else ctx.to_dict()

        synth = synthetic_context(cycle_bb100, rem, current_round, current_cycle)
        return synth if isinstance(synth, dict) else synth.to_dict()

    def _make_action_body(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate action body using the CFRCompetitionBridge."""
        if "tournamentContext" not in obs or not obs["tournamentContext"]:
            obs["tournamentContext"] = self._get_tournament_context(obs)

        req = obs.get("actionRequest") or {}
        res = self.bridge.choose_action(obs)
        decision = res.get("decision", {"type": "fold"})
        chat = res.get("chat")

        body: Dict[str, Any] = {
            "competitionId": obs["competitionId"],
            "tableId": obs["table"]["id"],
            "actionRequestId": req["id"],
            "decision": decision,
        }
        if chat:
            body["chat"] = chat[:140]
        return body

    def _action_with_retry(self, body: Dict[str, Any], original_obs: Dict[str, Any], max_attempts: int = 4) -> Dict[str, Any]:
        """Submit action with exact body preservation on network retry per skill.md."""
        frozen_body = copy.deepcopy(body)
        attempts = 0
        backoff = 0.5
        while True:
            attempts += 1
            try:
                return self.client.action(frozen_body)
            except requests.RequestException as net_err:
                if attempts >= max_attempts:
                    print(f"[Live] Network error after {attempts} attempts ({net_err}). Refreshing table state...")
                    return self._observe_current()
                time.sleep(backoff)
                backoff = min(8.0, backoff * 2.0)
                continue
            except APIError as e:
                if e.code in ("stale_action_request", "idempotency_conflict") or e.status == 409:
                    print(f"[Live] Action request stale ({e.code}), dropping action and re-observing.")
                    return self._observe_current()
                elif e.code == "invalid_decision":
                    print(f"[Live] Invalid decision rejected ({e.message}), falling back to safe action.")
                    frozen_body["decision"] = {"type": "fold"}
                    try:
                        return self.client.action(frozen_body)
                    except Exception:
                        return self._observe_current()
                elif e.code in ("temporarily_unavailable", "service_unavailable") or e.status in (502, 503, 504):
                    retry_wait = e.retry_after or backoff
                    print(f"[Live] Server temporarily unavailable, retrying in {retry_wait}s...")
                    time.sleep(retry_wait)
                    continue
                raise

    def _track_hand_progress(self, obs: Dict[str, Any]) -> None:
        """Track completed hands and net BB accumulation."""
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
        bb_size = float(table.get("bigBlind") or 1000.0)

        if self.prev_hand_id is None:
            self.prev_hand_id = hid
            self.hand_start_stack = curr_stack
            return

        if hid != self.prev_hand_id:
            self.total_hands += 1
            delta_bb = 0.0
            if curr_stack is not None and self.hand_start_stack is not None:
                delta_chips = curr_stack - self.hand_start_stack
                delta_bb = float(delta_chips) / bb_size
            self.round_net_bb += delta_bb
            self.cycle_net_bb += delta_bb
            self.session_net_bb += delta_bb

            print(f"[Live] Hand #{self.total_hands} completed | Hand Delta: {delta_bb:+.1f} BB | Session Net: {self.session_net_bb:+.1f} BB")

            if self.total_hands % self.round_hands == 0:
                round_no = self.total_hands // self.round_hands
                print(f"[Live] 🏁 Round #{round_no} Finished! Round Net: {self.round_net_bb:+.1f} BB")
                self.round_net_bb = 0.0

            self.prev_hand_id = hid
            self.hand_start_stack = curr_stack

    def _leave(self, obs: Dict[str, Any]) -> None:
        """Execute Leave Play per skill.md Section 5 until accepted."""
        body = {"tableId": self.table_id} if self.table_id else {"competitionId": self.cid}
        print(f"[Live] Submitting Leave Play ({body})...")
        try:
            res = self.client.leave(body)
            if res.get("status") == "accepted":
                print("[Live] Leave accepted by server.")
        except Exception as e:
            print(f"[Live] Leave response/warning: {e}")

    def _final_report(self, reason: str) -> None:
        """Write session summary report to JSONL."""
        if self._report_written:
            return
        self._report_written = True

        duration_s = (datetime.now(timezone.utc) - self.session_start).total_seconds()
        hands = self.total_hands
        bb100 = (self.session_net_bb / hands * 100.0) if hands > 0 else 0.0

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "competition_id": self.cid,
            "stop_reason": reason,
            "duration_seconds": round(duration_s, 1),
            "hands_played": hands,
            "session_net_bb": round(self.session_net_bb, 2),
            "session_bb100": round(bb100, 2),
        }

        print("\n" + "=" * 70)
        print(" 📋 【APEXPOKER 比赛值守报告】")
        print(f" • 停止原因: {reason} | 用时: {duration_s/60:.1f} 分钟")
        print(f" • 完成手数: {hands} 手 | 总净收益: {self.session_net_bb:+.1f} BB ({bb100:+.1f} BB/100)")
        print("=" * 70)

        try:
            p = Path(self.report_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(report, ensure_ascii=False) + "\n")
            print(f" • 报告已追加记录至: {self.report_path}\n")
        except Exception:
            pass
