"""CLI entrypoint for online Agent Poker competition conforming to skill.md.

Environment variables are loaded automatically from .env or ../agentpoker_final 2/.env:
- AGENTPOKER_APP: Target server (default: https://poker.bang.sohu.com)
- AGENTPOKER_KEY: Agent secretKey (sk_...)
- AGENTPOKER_COMPETITION_ID: Competition ID to enter

Commands:
- join (play, live): Connect AI model to live competition and run watchman loop.
- discover: Query active competitions on server.
- standings: View live competition leaderboard.
- status: Check current environment configuration and server connectivity.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition.protocol import AgentPokerClient, Config, load_env
from src.competition.live_watchman import CompetitionWatchman
from src.competition.cfr_bridge import CFRCompetitionBridge


def main():
    load_env()
    parser = argparse.ArgumentParser(
        prog="run_competition",
        description="ApexPoker Live Competition Entrypoint (uses .env credentials)",
    )
    sub = parser.add_subparsers(dest="cmd")

    # 1. status
    p_stat = sub.add_parser("status", help="Check current environment configuration & API status")
    p_stat.add_argument("--app", default=os.getenv("AGENTPOKER_APP", "https://poker.bang.sohu.com"), help="Server base URL")

    # 2. discover
    p_disc = sub.add_parser("discover", help="Discover active competitions on the platform")
    p_disc.add_argument("--app", default=os.getenv("AGENTPOKER_APP", "https://poker.bang.sohu.com"), help="Server base URL")
    p_disc.add_argument("--status", default="active", help="Competition status filter (active, all)")

    # 3. join / play
    p_join = sub.add_parser("join", aliases=["play", "live"], help="Join competition and run live AI watchman")
    p_join.add_argument("--competition-id", default=os.getenv("AGENTPOKER_COMPETITION_ID"), help="Target Competition ID")
    p_join.add_argument("--model", default="models/finetuned_50bb/hybrid_checkpoint_iter_66000.pt", help="Path to trained PyTorch model")
    p_join.add_argument("--key", default=os.getenv("AGENTPOKER_KEY"), help="Agent secretKey (sk_...)")
    p_join.add_argument("--app", default=os.getenv("AGENTPOKER_APP", "https://poker.bang.sohu.com"), help="Server base URL")
    p_join.add_argument("--max-hands", type=int, default=0, help="Stop after N hands (0 = unlimited)")
    p_join.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0 = unlimited)")
    p_join.add_argument("--round-hands", type=int, default=20, help="Hands per round")
    p_join.add_argument("--cycle-hands", type=int, default=200, help="Hands per cycle")

    # 4. standings
    p_stand = sub.add_parser("standings", help="View competition standings / leaderboard")
    p_stand.add_argument("--competition-id", default=os.getenv("AGENTPOKER_COMPETITION_ID"), help="Target Competition ID")
    p_stand.add_argument("--app", default=os.getenv("AGENTPOKER_APP", "https://poker.bang.sohu.com"), help="Server base URL")

    # Allow running with default 'join' if no arguments provided
    if len(sys.argv) == 1:
        args = parser.parse_args(["join"])
    else:
        args = parser.parse_args()

    if args.cmd == "status":
        app = args.app
        key = os.getenv("AGENTPOKER_KEY", "")
        cid = os.getenv("AGENTPOKER_COMPETITION_ID", "")
        masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else (key or "<not set>")
        print("\n" + "=" * 60)
        print(" 🔍 ApexPoker Competition Environment Status")
        print("=" * 60)
        print(f" • Server URL (AGENTPOKER_APP):           {app}")
        print(f" • Agent Key (AGENTPOKER_KEY):            {masked_key}")
        print(f" • Target Comp ID (AGENTPOKER_COMPETITION_ID): {cid or '<not set>'}")
        print("-" * 60)

        client = AgentPokerClient(Config(base_url=app, key=key or None))
        try:
            res = client.discover("active")
            comps = res.get("competitions", [])
            print(f" • Server Connectivity:                   OK (Found {len(comps)} active competitions)")
            for c in comps:
                print(f"    - [{c.get('type','').upper()}] {c.get('name')} (ID: {c.get('id')})")
        except Exception as e:
            print(f" • Server Connectivity:                   FAILED ({e})")
        print("=" * 60 + "\n")

    elif args.cmd == "discover":
        cfg = Config(base_url=args.app)
        client = AgentPokerClient(cfg)
        print(f"\nQuerying competitions from {args.app}...")
        try:
            res = client.discover(status=args.status)
            comps = res.get("competitions", [])
            print(f"\nActive Competitions ({len(comps)} found):")
            for c in comps:
                cid = c.get("id")
                name = c.get("name")
                ctype = c.get("type", "standard")
                status = c.get("status")
                print(f" • [{ctype.upper()}] ID: {cid} | Name: {name} | Status: {status}")
            print()
        except Exception as e:
            print(f"Error querying competitions: {e}")

    elif args.cmd in ("join", "play", "live"):
        key = args.key or os.getenv("AGENTPOKER_KEY")
        if not key:
            print("Error: AGENTPOKER_KEY is not set in .env or environment.")
            sys.exit(1)

        cid = args.competition_id or os.getenv("AGENTPOKER_COMPETITION_ID")
        client = AgentPokerClient(Config(base_url=args.app, key=key))

        if not cid:
            print("[Live] No competition ID specified, attempting auto-discovery...")
            try:
                res = client.discover("active")
                comps = res.get("competitions", [])
                if comps:
                    cid = comps[0]["id"]
                    print(f"[Live] Auto-selected active competition: '{comps[0].get('name')}' ({cid})")
            except Exception as e:
                print(f"[Live] Auto-discovery error: {e}")

        if not cid:
            print("Error: No active competition found and no --competition-id specified.")
            sys.exit(1)

        bridge = CFRCompetitionBridge(model_path=args.model)
        watchman = CompetitionWatchman(
            client=client,
            competition_id=cid,
            bridge=bridge,
            round_hands=args.round_hands,
            cycle_hands=args.cycle_hands,
        )
        watchman.run(
            max_steps=args.max_steps or None,
            max_hands=args.max_hands or None,
        )

    elif args.cmd == "standings":
        cid = args.competition_id or os.getenv("AGENTPOKER_COMPETITION_ID")
        if not cid:
            print("Error: competition-id is required.")
            sys.exit(1)
        client = AgentPokerClient(Config(base_url=args.app))
        try:
            res = client.standings(cid)
            rows = res.get("standings") or res.get("rows") or []
            print(f"\nLeaderboard for Competition {cid} ({len(rows)} players):")
            print(f"{'Rank':<6} {'Agent ID':<24} {'Hands':<8} {'Net (BB)':<12} {'BB/100':<10}")
            print("-" * 65)
            for r in rows[:30]:
                rank = r.get("rank", "-")
                aid = str(r.get("agentId", r.get("agent_id", "")))[:22]
                hands = r.get("handsPlayed", r.get("hands", 0))
                net = f"{float(r.get('netResult', r.get('net_bb', 0))):+.1f}"
                bb100 = f"{float(r.get('bbPer100', r.get('bb100', 0))):+.1f}"
                print(f"{rank:<6} {aid:<24} {hands:<8} {net:<12} {bb100:<10}")
            print()
        except Exception as e:
            print(f"Error fetching standings: {e}")


if __name__ == "__main__":
    main()
