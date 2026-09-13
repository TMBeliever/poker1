from __future__ import annotations
import os, sys, json, math, random, datetime, shutil
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from .strategy import StrategyAgent, StrategyParams
from .tournament import LeagueSimulator, SimAgent
from .training import ARCHETYPES, ArenaEvaluator, profile_to_params, _load_params_safe
from .ecosystem import build_ecosystem_pool


@dataclass
class CompetitorCandidate:
    cid: str
    name: str
    category: str  # "model", "archetype", "profile"
    params: StrategyParams
    description: str = ""
    source: str = ""


@dataclass
class CompetitorStats:
    cid: str
    name: str
    category: str
    champ_count: int = 0
    final_count: int = 0
    top12_count: int = 0
    ranks: list[float] = field(default_factory=list)
    bbs: list[float] = field(default_factory=list)
    total_hands: int = 0
    total_net_bb: float = 0.0

    @property
    def avg_rank(self) -> float:
        return sum(self.ranks) / max(1, len(self.ranks))

    @property
    def avg_bb100(self) -> float:
        return (self.total_net_bb / max(1, self.total_hands)) * 100.0 if self.total_hands > 0 else 0.0

    def score(self, runs: int, pool_size: int) -> float:
        runs = max(1, runs)
        champ_rate = self.champ_count / runs
        final_rate = self.final_count / runs
        top12_rate = self.top12_count / runs
        norm_rank = 1.0 - min(self.avg_rank - 1.0, pool_size - 1.0) / max(1.0, pool_size - 1.0)
        bb_bonus = 1.0 / (1.0 + math.exp(-max(-200.0, min(200.0, self.avg_bb100)) / 40.0))
        return (champ_rate * 50.0 + final_rate * 25.0 + top12_rate * 15.0 + norm_rank * 5.0 + bb_bonus * 5.0)


@dataclass
class CertificationCriterion:
    name: str
    required: str
    actual: str
    passed: bool
    detail: str = ""


@dataclass
class ChampionCertificationResult:
    certified: bool
    candidate_id: str
    candidate_name: str
    criteria: list[CertificationCriterion]
    summary: str
    recommendation: str  # "PROMOTE_TO_CHAMPION", "RETAIN_EXISTING", "REJECT"
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "criteria": [asdict(c) for c in self.criteria],
            "summary": self.summary,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
        }


def discover_candidates(
    models_dir: str | Path = "models",
    profiles_path: str | Path = "models/opponent_profiles.json",
    max_profiles: int = 8,
) -> list[CompetitorCandidate]:
    """Scan the environment for all available agents that can enter the battle arena."""
    candidates: list[CompetitorCandidate] = []
    models_path = Path(models_dir)

    # 1. Models from models_dir
    model_files: list[Path] = []
    if models_path.exists():
        # Root level models
        for p in sorted(models_path.glob("*.json")):
            if p.name not in ("opponent_profiles.json", "calibration_table.json"):
                model_files.append(p)
        # Checkpoint archives (scan all archive* subdirectories)
        for sub_dir in sorted(models_path.glob("archive*")):
            if sub_dir.is_dir():
                for p in sorted(sub_dir.glob("*.json"), reverse=True):
                    model_files.append(p)

    # Priority sorting: champion.json first, then other models, then archives
    def model_sort_key(p: Path):
        name = p.name
        if name == "champion.json":
            return (0, name)
        if "champion" in name:
            return (1, name)
        return (2, str(p))

    model_files.sort(key=model_sort_key)

    for p in model_files:
        try:
            rel = str(p.relative_to(models_path.parent)) if p.is_relative_to(models_path.parent) else str(p)
        except Exception:
            rel = str(p)
        try:
            agent = StrategyAgent.load(p)
            vpip_str = f"VPIP {agent.params.vpip:.1%}"
            pfr_str = f"PFR {agent.params.open_frequency * 0.7:.1%}"
            desc = f"演化模型 ({vpip_str}, {pfr_str})"
            if p.name == "champion.json":
                desc += " [官方生产冠军模型 (Production Champion)]"
            elif "candidate" in str(p) or "champion" in p.name:
                desc += " [候选冠军模型]"
            elif "archive" in str(p):
                desc += f" [历史归档代数 {p.stem}]"

            clean_rel = str(p.relative_to(models_path)).replace("/", "_").replace(".json", "")
            cid = f"model:{clean_rel}"
            candidates.append(CompetitorCandidate(
                cid=cid,
                name=rel,
                category="model",
                params=agent.params,
                description=desc,
                source=str(p),
            ))
        except Exception:
            continue

    # 2. Archetypes
    archetype_desc = {
        "tight": ("TAG稳健激进型 (官方推荐基线, 紧凶)", "tight"),
        "balanced": ("Balanced默认均衡型 (标准平衡打法)", "balanced"),
        "lag": ("LAG松凶压迫型 (宽入池高进攻, 偷盲激进)", "lag"),
        "nit": ("Nit超紧坚果型 (极端保守, 只打超强牌)", "nit"),
        "station": ("Calling Station被动跟注站 (极少加注, 极难打跑)", "station"),
        "maniac": ("Maniac狂热进攻型 (超高入池与推入频率)", "maniac"),
    }
    for key, params in ARCHETYPES.items():
        desc, name = archetype_desc.get(key, (f"{key} 原型", key))
        candidates.append(CompetitorCandidate(
            cid=f"archetype:{key}",
            name=f"bot_{key}",
            category="archetype",
            params=params,
            description=f"内置原型: {desc}",
            source="builtin",
        ))

    # 3. Real opponent profiles
    prof_p = Path(profiles_path)
    if prof_p.exists():
        try:
            raw = json.loads(prof_p.read_text(encoding="utf-8"))
            valid = [
                (aid, info) for aid, info in raw.items()
                if isinstance(info, dict) and info.get("hands", 0) >= 30
            ]
            valid.sort(key=lambda x: x[1].get("hands", 0), reverse=True)
            for aid, info in valid[:max_profiles]:
                h = info.get("hands", 0)
                vpip = float(info.get("vpip", 0.25))
                pfr = float(info.get("raise", info.get("pfr", 0.15)))
                style = "松凶" if vpip > 0.32 else ("超紧" if vpip < 0.18 else "常客")
                short_id = aid[:14]
                candidates.append(CompetitorCandidate(
                    cid=f"profile:{aid}",
                    name=f"real_{short_id}",
                    category="profile",
                    params=profile_to_params(info),
                    description=f"真实画像 ({style}, {h}手, VPIP {vpip:.0%}, PFR {pfr:.0%})",
                    source=f"profile:{aid}",
                ))
        except Exception:
            pass

    return candidates


def build_opponent_pool(
    mode: str,
    count: int,
    profiles_path: str | Path | None = "models/opponent_profiles.json",
    min_hands: int = 15,
) -> list[StrategyParams]:
    """Build background opponent field of size `count` using ecosystem generator."""
    eff_mode = mode or "pyramid"
    return build_ecosystem_pool(
        count=count,
        mode=eff_mode,
        profiles_path=profiles_path,
        min_hands=min_hands,
    )


def _jitter_params(p: StrategyParams, rng: random.Random, scale: float = 0.02) -> StrategyParams:
    """Slight jitter to prevent clone stagnation across tournament runs."""
    d = asdict(p)
    d.pop("equity_samples", None)
    for k, v in d.items():
        if isinstance(v, float) and 0.0 < v < 1.0:
            delta = rng.gauss(0, scale * max(0.05, v))
            d[k] = max(0.01, min(0.99, v + delta))
    return StrategyParams(**d)


def _battle_tournament_worker(payload: tuple) -> dict[str, Any]:
    """Multiprocessing worker to simulate 1 complete tournament run with all competitors."""
    (
        run_idx,
        seed,
        competitor_payload,  # list of (cid, name, params_dict)
        opponent_payload,    # list of params_dict
        field_size,
        equity_samples,
    ) = payload

    rng = random.Random(seed)

    # 1. Build competitor agents
    comp_ids = [item[0] for item in competitor_payload]
    agents: list[SimAgent] = []
    for cid, cname, p_dict in competitor_payload:
        p = _load_params_safe(p_dict)
        p = replace(p, equity_samples=equity_samples)
        agents.append(SimAgent(cid, StrategyAgent(p, seed=seed + 101, name=cname)))

    # 2. Build opponent agents with slight jitter
    needed_opponents = max(0, field_size - len(agents))
    for i in range(needed_opponents):
        base_p = _load_params_safe(opponent_payload[i % len(opponent_payload)])
        jittered_p = _jitter_params(base_p, rng)
        jittered_p = replace(jittered_p, equity_samples=equity_samples)
        opp_id = f"opp_{i:02d}"
        agents.append(SimAgent(opp_id, StrategyAgent(jittered_p, seed=seed + 2000 + i * 31, name=opp_id)))

    # Ensure total agents is at least 12 and multiple of 6
    while len(agents) < 12 or len(agents) % 6 != 0:
        base_p = _load_params_safe(opponent_payload[len(agents) % len(opponent_payload)])
        opp_id = f"opp_{len(agents):02d}"
        agents.append(SimAgent(opp_id, StrategyAgent(base_p, seed=seed + 3000 + len(agents), name=opp_id)))

    # Random initial seating order
    rng.shuffle(agents)

    sim = LeagueSimulator(agents, sb=100, bb=200, rounds=10, hands_per_round=20, seats=6, seed=seed)
    res = sim.run_event()

    prelim_list = res.get("preliminary", [])
    prelim_map = {s.agent_id: s for s in prelim_list}
    qualified_ids = [s.agent_id for s in res.get("qualified", [])]
    semifinal_groups = res.get("semifinal", [])
    final_list = res.get("final", [])
    final_ids = [s.agent_id for s in final_list]

    # Map semifinal ranks (7..12)
    semi_standings: dict[str, Any] = {}
    for grp in semifinal_groups:
        for s in grp:
            semi_standings[s.agent_id] = s

    # Semifinal non-advancers ranked 7..12
    semi_non_advancers = [aid for aid in qualified_ids if aid not in final_ids]
    semi_non_advancers.sort(
        key=lambda aid: (
            -(semi_standings[aid].bb100 if aid in semi_standings and semi_standings[aid].bb100 is not None else float("-inf")),
            prelim_map.get(aid).rank if aid in prelim_map else 999
        )
    )
    semi_rank_map = {aid: 7 + idx for idx, aid in enumerate(semi_non_advancers)}

    # Compute stats for competitors
    comp_results: dict[str, Any] = {}
    for cid in comp_ids:
        p_standing = prelim_map.get(cid)
        prelim_rank = p_standing.rank if p_standing else len(agents)
        total_hands = p_standing.hands if p_standing else 0
        total_net_bb = p_standing.net_bb if p_standing else 0.0

        if cid in final_ids:
            f_idx = next(i for i, s in enumerate(final_list) if s.agent_id == cid)
            final_rank = f_idx + 1
            f_standing = final_list[f_idx]
            total_hands += f_standing.hands
            total_net_bb += f_standing.net_bb
            if cid in semi_standings:
                total_hands += semi_standings[cid].hands
                total_net_bb += semi_standings[cid].net_bb
        elif cid in semi_rank_map:
            final_rank = semi_rank_map[cid]
            if cid in semi_standings:
                total_hands += semi_standings[cid].hands
                total_net_bb += semi_standings[cid].net_bb
        else:
            final_rank = prelim_rank

        comp_results[cid] = {
            "rank": final_rank,
            "prelim_rank": prelim_rank,
            "is_champ": (final_rank == 1),
            "is_final": (final_rank <= 6),
            "is_top12": (final_rank <= 12),
            "total_hands": total_hands,
            "total_net_bb": total_net_bb,
            "bb100": (total_net_bb / total_hands * 100.0) if total_hands > 0 else 0.0,
        }

    # Head-to-Head pairwise wins in this tournament
    h2h_pairs: list[tuple[str, str, int]] = []  # (cid_A, cid_B, outcome: 1 if A beats B, -1 if B beats A, 0 tie)
    for i in range(len(comp_ids)):
        for j in range(i + 1, len(comp_ids)):
            ca, cb = comp_ids[i], comp_ids[j]
            ra, rb = comp_results[ca]["rank"], comp_results[cb]["rank"]
            if ra < rb:
                h2h_pairs.append((ca, cb, 1))
            elif rb < ra:
                h2h_pairs.append((ca, cb, -1))
            else:
                h2h_pairs.append((ca, cb, 0))

    champ_id = final_list[0].agent_id if final_list else None
    return {
        "run_idx": run_idx,
        "champ_id": champ_id,
        "finalists": final_ids,
        "comp_results": comp_results,
        "h2h_pairs": h2h_pairs,
    }


class ArenaBattle:
    """Manages multi-model tournament arena battles with comprehensive metrics."""

    def __init__(
        self,
        competitors: list[CompetitorCandidate],
        opponent_mode: str = "pyramid",
        profiles_path: str | Path = "models/opponent_profiles.json",
        field_size: int = 120,
        equity_samples: int = 0,
        workers: int = 0,
        seed: int = 42,
        official: bool = False,
    ):
        if official and field_size != 120:
            raise ValueError(f"Official benchmark strictly requires field_size == 120, got {field_size}")
        self.official = official

        if len(competitors) < 2:
            raise ValueError("ArenaBattle requires at least 2 competitor agents.")

        # Ensure all competitors have strictly unique CIDs
        seen_ids = set()
        sanitized_competitors = []
        for idx, c in enumerate(competitors):
            unique_cid = c.cid
            if unique_cid in seen_ids:
                unique_cid = f"{c.cid}_{idx}"
            seen_ids.add(unique_cid)
            sanitized_competitors.append(replace(c, cid=unique_cid))
        self.competitors = sanitized_competitors

        self.opponent_mode = opponent_mode
        self.profiles_path = profiles_path
        # Field size must be at least max(12, competitors) and multiple of 6
        min_field = max(12, len(self.competitors))
        if min_field % 6 != 0:
            min_field += 6 - (min_field % 6)
        self.field_size = max(min_field, field_size)
        if self.field_size % 6 != 0:
            self.field_size += 6 - (self.field_size % 6)

        if self.official and self.field_size != 120:
            raise ValueError(f"Official benchmark strictly requires field_size == 120, got {self.field_size}")

        self.equity_samples = equity_samples
        self.workers = workers or min(os.cpu_count() or 4, 8)
        self.seed = seed

    def run(self, runs: int = 20, verbose: bool = True) -> dict[str, Any]:
        runs = max(1, runs)
        workers = min(self.workers, runs)

        # Pre-build background opponent pool
        needed_opp = self.field_size - len(self.competitors)
        opp_params = build_opponent_pool(self.opponent_mode, max(self.field_size, needed_opp), self.profiles_path)
        opp_payload = [asdict(p) for p in opp_params]
        comp_payload = [(c.cid, c.name, asdict(c.params)) for c in self.competitors]

        tasks = [
            (r, self.seed + r * 7919, comp_payload, opp_payload, self.field_size, self.equity_samples)
            for r in range(runs)
        ]

        stats_map = {
            c.cid: CompetitorStats(cid=c.cid, name=c.name, category=c.category)
            for c in self.competitors
        }
        # h2h_matrix[A][B] = {"wins": 0, "losses": 0, "ties": 0}
        h2h_matrix: dict[str, dict[str, dict[str, int]]] = {
            c.cid: {other.cid: {"wins": 0, "losses": 0, "ties": 0} for other in self.competitors if other.cid != c.cid}
            for c in self.competitors
        }

        if verbose:
            print(f"\n[擂台启动] 正在并行模拟 {runs} 场全流程锦标赛 (多核并发: {workers} 进程)...", flush=True)

        completed = 0
        if workers > 1 and runs > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_battle_tournament_worker, t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    res = fut.result()
                    completed += 1
                    self._record_run(res, stats_map, h2h_matrix)
                    if verbose:
                        self._print_run_progress(completed, runs, res)
        else:
            for t in tasks:
                res = _battle_tournament_worker(t)
                completed += 1
                self._record_run(res, stats_map, h2h_matrix)
                if verbose:
                    self._print_run_progress(completed, runs, res)

        return self._build_report(runs, stats_map, h2h_matrix)

    def _record_run(self, res: dict[str, Any], stats_map: dict[str, CompetitorStats], h2h: dict):
        comp_res = res["comp_results"]
        for cid, info in comp_res.items():
            if cid in stats_map:
                st = stats_map[cid]
                if info["is_champ"]:
                    st.champ_count += 1
                if info["is_final"]:
                    st.final_count += 1
                if info["is_top12"]:
                    st.top12_count += 1
                st.ranks.append(float(info["rank"]))
                st.bbs.append(float(info["bb100"]))
                st.total_hands += info["total_hands"]
                st.total_net_bb += info["total_net_bb"]

        for ca, cb, outcome in res["h2h_pairs"]:
            if ca == cb:
                continue
            if ca not in h2h:
                h2h[ca] = {}
            if cb not in h2h:
                h2h[cb] = {}
            if cb not in h2h[ca]:
                h2h[ca][cb] = {"wins": 0, "losses": 0, "ties": 0}
            if ca not in h2h[cb]:
                h2h[cb][ca] = {"wins": 0, "losses": 0, "ties": 0}

            if outcome == 1:
                h2h[ca][cb]["wins"] += 1
                h2h[cb][ca]["losses"] += 1
            elif outcome == -1:
                h2h[ca][cb]["losses"] += 1
                h2h[cb][ca]["wins"] += 1
            else:
                h2h[ca][cb]["ties"] += 1
                h2h[cb][ca]["ties"] += 1

    def _print_run_progress(self, completed: int, total: int, res: dict[str, Any]):
        cid_to_name = {c.cid: c.name for c in self.competitors}
        champ_name = cid_to_name.get(res["champ_id"], res["champ_id"] or "陪练选手")
        finalists_desc = []
        for aid in res["finalists"]:
            if aid in cid_to_name:
                r = res["comp_results"][aid]["rank"]
                finalists_desc.append(f"{cid_to_name[aid]} (第{r}名)")
        f_str = ", ".join(finalists_desc) if finalists_desc else "无擂主入决赛"
        prefix = "🏆" if res["champ_id"] in cid_to_name else "⚔️"
        print(
            f"[对局进度 {completed:02d}/{total:02d}] {prefix} 本场冠军: {champ_name:<16} | 擂主决赛桌: [{f_str}]",
            flush=True,
        )

    def _build_report(self, runs: int, stats_map: dict[str, CompetitorStats], h2h: dict) -> dict[str, Any]:
        leaderboard = []
        for c in self.competitors:
            st = stats_map[c.cid]
            champ_r = st.champ_count / runs
            final_r = st.final_count / runs
            top12_r = st.top12_count / runs
            score = st.score(runs, self.field_size)

            champ_se = math.sqrt(champ_r * (1.0 - champ_r) / runs) if runs > 0 else 0.0
            final_se = math.sqrt(final_r * (1.0 - final_r) / runs) if runs > 0 else 0.0
            top12_se = math.sqrt(top12_r * (1.0 - top12_r) / runs) if runs > 0 else 0.0
            bb_se = 0.0
            if len(st.bbs) > 1:
                var = sum((b - st.avg_bb100) ** 2 for b in st.bbs) / (len(st.bbs) - 1)
                bb_se = math.sqrt(var / len(st.bbs))

            leaderboard.append({
                "cid": c.cid,
                "name": c.name,
                "category": c.category,
                "champ_count": st.champ_count,
                "champ_rate": champ_r,
                "champ_se": champ_se,
                "final_count": st.final_count,
                "final_rate": final_r,
                "final_se": final_se,
                "top12_count": st.top12_count,
                "top12_rate": top12_r,
                "top12_se": top12_se,
                "avg_rank": st.avg_rank,
                "avg_bb100": st.avg_bb100,
                "bb_se": bb_se,
                "total_hands": st.total_hands,
                "score": score,
            })

        # Rank by score descending
        leaderboard.sort(key=lambda x: x["score"], reverse=True)
        for idx, row in enumerate(leaderboard, 1):
            row["standing"] = idx

        # Format H2H matrix with win rates
        h2h_data = {}
        for ca in self.competitors:
            h2h_data[ca.cid] = {}
            for cb in self.competitors:
                if ca.cid == cb.cid:
                    continue
                pair = h2h[ca.cid][cb.cid]
                w, l, t = pair["wins"], pair["losses"], pair["ties"]
                rate = (w + 0.5 * t) / runs * 100.0
                h2h_data[ca.cid][cb.cid] = {
                    "wins": w, "losses": l, "ties": t, "win_rate": rate,
                }

        return {
            "runs": runs,
            "field_size": self.field_size,
            "opponent_mode": self.opponent_mode,
            "leaderboard": leaderboard,
            "h2h_matrix": h2h_data,
        }

    def certify(
        self,
        report: dict[str, Any] | None = None,
        candidate_cid: str | None = None,
        baseline_cids: list[str] | None = None,
        min_champ_multiplier: float = 2.0,
        min_top12_multiplier: float = 1.5,
        min_bb100: float = 0.0,
    ) -> ChampionCertificationResult:
        """Run certification gate on arena battle results."""
        if report is None:
            report = self.run()
        return certify_champion(
            report=report,
            candidate_cid=candidate_cid,
            baseline_cids=baseline_cids,
            min_champ_multiplier=min_champ_multiplier,
            min_top12_multiplier=min_top12_multiplier,
            min_bb100=min_bb100,
        )


def print_battle_report(report: dict[str, Any]):
    """Pretty print the leaderboard, H2H matrix, and strategic analysis to terminal."""
    runs = report["runs"]
    field_size = report["field_size"]
    opp_mode_desc = {
        "mix": "混合池 (50% 原型 Bot + 50% 真实画像)",
        "profiles": "纯真实玩家画像池",
        "archetypes": "纯原型 Bot 池",
    }.get(report.get("opponent_mode", "mix"), report.get("opponent_mode"))

    board = report["leaderboard"]
    cid_to_name = {row["cid"]: row["name"] for row in board}

    print("\n" + "=" * 94)
    print(f"               🏆 Sohu Agent Poker 锦标赛擂台赛最终战报 (共 {runs} 场锦标赛)")
    print(f"      赛制规格: {field_size}人6人桌 | 陪练配置: {opp_mode_desc} | 争冠规则: 预赛200手->12强->6人决赛")
    print("=" * 94)

    # 1. Leaderboard Table
    header = f"{'名次':<4} {'选手名称':<26} {'夺冠率 (胜场)':<14} {'决赛桌率':<11} {'12强出线':<10} {'均场名次':<9} {'均场BB/100':<11} {'总胜分'}"
    print(header)
    print("-" * 94)
    medals = {1: "🥇 1", 2: "🥈 2", 3: "🥉 3"}
    for row in board:
        st_label = medals.get(row["standing"], f"  {row['standing']}")
        name = row["name"]
        if len(name) > 24:
            name = name[:21] + "..."
        champ_str = f"{row['champ_rate']:.1%} ({row['champ_count']:>2d}场)"
        final_str = f"{row['final_rate']:.1%}"
        top12_str = f"{row['top12_rate']:.1%}"
        rank_str = f"{row['avg_rank']:.1f}"
        bb_str = f"{row['avg_bb100']:+.1f}"
        score_str = f"{row['score']:.1f}"
        print(f"{st_label:<4} {name:<26} {champ_str:<14} {final_str:<11} {top12_str:<10} {rank_str:<9} {bb_str:<11} {score_str}")
    print("=" * 94)

    # 2. Head-to-Head Win Rate Matrix
    print("\n" + "=" * 94)
    print("                        ⚔️ 两两对决相对胜率矩阵 (Head-to-Head)")
    print("       (说明: [行选手] 对 [列选手] 的 两两排名胜负统计: 净胜率 (胜 / 负 / 平))")
    print("=" * 94)

    cids = [r["cid"] for r in board]
    n = len(cids)
    col_w = max(18, min(24, 70 // max(1, n)))
    top_row = f"{'选手':<20}" + "".join([f"[{i+1}] {cid_to_name[cid][:col_w-5]:<{col_w-4}}" for i, cid in enumerate(cids)])
    print(top_row)
    print("-" * 94)

    h2h = report["h2h_matrix"]
    for i, ca in enumerate(cids):
        row_label = f"[{i+1}] {cid_to_name[ca][:14]:<14}"
        cells = []
        for j, cb in enumerate(cids):
            if i == j:
                cells.append(f"{'—':^{col_w}}")
            else:
                pair = h2h[ca].get(cb, {"wins": 0, "losses": 0, "ties": 0, "win_rate": 50.0})
                w, l, t = pair["wins"], pair["losses"], pair["ties"]
                rate = pair["win_rate"]
                txt = f"{rate:.0f}% ({w}/{l}/{t})"
                cells.append(f"{txt:^{col_w}}")
        print(f"{row_label:<20}" + "".join(cells))
    print("=" * 94)

    # 3. Tactical Takeaways
    print("\n📊 擂台深度战况剖析:")
    best_champ = max(board, key=lambda x: x["champ_rate"])
    best_final = max(board, key=lambda x: x["final_rate"])
    best_bb = max(board, key=lambda x: x["avg_bb100"])

    print(f"  • 👑 终结夺冠王: 【{best_champ['name']}】 斩获最高夺冠率 {best_champ['champ_rate']:.1%} ({best_champ['champ_count']} 次第一名)，在争冠关键手牌中终结比赛能力最强！")
    print(f"  • 🛡️ 稳健入桌王: 【{best_final['name']}】 决赛桌率达到 {best_final['final_rate']:.1%}，深进比赛与规避爆冷能力最为突出。")
    print(f"  • 💰 筹码收割王: 【{best_bb['name']}】 均场实现 {best_bb['avg_bb100']:+.1f} BB/100，入池与价值下注效率最高。")
    if len(board) >= 2:
        top1, top2 = board[0], board[1]
        pair = h2h[top1["cid"]].get(top2["cid"], {"wins": 0, "losses": 0, "ties": 0, "win_rate": 50.0})
        print(f"  • ⚔️ 巅峰交锋: 第 1 名 【{top1['name']}】 vs 第 2 名 【{top2['name']}】: 对决胜率为 {pair['win_rate']:.1f}% ({pair['wins']}胜 {pair['losses']}负 {pair['ties']}平)。")
    print("=" * 94 + "\n")


def certify_champion(
    report: dict[str, Any],
    candidate_cid: str | None = None,
    baseline_cids: list[str] | None = None,
    min_champ_multiplier: float | None = None,
    min_top12_multiplier: float | None = None,
    min_bb100: float | None = None,
    policy_path: str | Path | None = "configs/certification_policy.json",
    official: bool = True,
) -> ChampionCertificationResult:
    """Evaluate candidate model against strict mathematical tournament champion certification criteria."""
    policy = {}
    if policy_path and Path(policy_path).exists():
        try:
            policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        except Exception:
            policy = {}
    invariants = policy.get("invariants", {})

    field_size = report.get("field_size", 120)
    required_field_size = int(invariants.get("required_field_size", 120))
    if official and field_size != required_field_size:
        raise ValueError(
            f"Official champion certification strictly requires field_size == {required_field_size}, got {field_size}"
        )

    if min_champ_multiplier is None:
        min_champ_multiplier = float(invariants.get("min_champ_multiplier", 2.0))
    if min_top12_multiplier is None:
        min_top12_multiplier = float(invariants.get("min_top12_multiplier", 1.5))
    if min_bb100 is None:
        min_bb100 = float(invariants.get("min_bb100", 0.0))
    min_h2h_win_rate = float(invariants.get("min_h2h_win_rate", 50.0))
    allow_ev_bypass = bool(invariants.get("allow_ev_bypass", False))

    board = report.get("leaderboard", [])
    if not board:
        raise ValueError("Report contains empty leaderboard; cannot certify champion.")

    runs = report.get("runs", 1)
    h2h = report.get("h2h_matrix", {})

    # Select candidate
    candidate_row = None
    if candidate_cid is not None:
        for r in board:
            if r["cid"] == candidate_cid or r["name"] == candidate_cid:
                candidate_row = r
                break
        if candidate_row is None:
            raise ValueError(f"Candidate '{candidate_cid}' not found in leaderboard.")
    else:
        # Default to Rank 1 on leaderboard
        candidate_row = board[0]

    cid = candidate_row["cid"]
    name = candidate_row["name"]
    criteria: list[CertificationCriterion] = []

    require_rank_1 = bool(invariants.get("require_rank_1", True))
    require_all_passed = bool(invariants.get("require_all_passed", True))

    # Criterion 1: Leaderboard Standing == 1
    standing = candidate_row.get("standing", 1)
    c1_passed = (standing == 1) if require_rank_1 else True
    criteria.append(CertificationCriterion(
        name="Leaderboard Rank #1",
        required="Standing == 1" if require_rank_1 else "Standing Ignored",
        actual=f"Standing #{standing}",
        passed=c1_passed,
        detail=f"Candidate must achieve Rank 1 overall tournament score (Score: {candidate_row['score']:.1f})",
    ))

    # Criterion 2: Title / Deep Run Superiority
    # In large fields (e.g. 120 players), when runs < field_size (discrete 1st-place titles have E < 1.0),
    # champion title events are statistically underpowered Poisson observations.
    # Therefore, the gate evaluates champion rate when runs >= field_size, or final table deep run rate
    # (6 seats / field_size) when runs < field_size.
    random_champ_baseline = 1.0 / max(1, field_size)
    random_final_baseline = min(1.0, 6.0 / max(1, field_size))
    actual_champ_rate = candidate_row.get("champ_rate", 0.0)
    actual_final_rate = candidate_row.get("final_rate", 0.0)

    if runs >= field_size:
        required_champ_rate = min_champ_multiplier * random_champ_baseline
        c2_passed = (actual_champ_rate >= required_champ_rate)
        req_str = f">={required_champ_rate:.2%} ({min_champ_multiplier:.1f}x baseline {random_champ_baseline:.2%})"
        act_str = f"{actual_champ_rate:.2%} ({candidate_row.get('champ_count', 0)}/{runs} wins)"
        det_str = f"Candidate champ rate must outperform random field expectation by at least {min_champ_multiplier:.1f}x"
    else:
        # Sample-size adjusted: requires either champion win OR final table rate >= min_champ_multiplier * baseline
        required_final_rate = min(1.0, min_champ_multiplier * random_final_baseline)
        c2_passed = (actual_champ_rate > 0.0) or (actual_final_rate >= required_final_rate)
        req_str = f">={required_final_rate:.1%} final rate or >=1 champ ({min_champ_multiplier:.1f}x baseline)"
        act_str = f"Final {actual_final_rate:.1%} ({candidate_row.get('final_count', 0)}/{runs}), Champ {actual_champ_rate:.1%}"
        det_str = f"For sample size {runs} < field {field_size}, deep run conversion (final table >= {required_final_rate:.1%}) or title win required"

    criteria.append(CertificationCriterion(
        name="Title / Deep Run Superiority",
        required=req_str,
        actual=act_str,
        passed=c2_passed,
        detail=det_str,
    ))

    # Criterion 3: Positive Overall BB/100
    actual_bb100 = candidate_row.get("avg_bb100", 0.0)
    c3_passed = (actual_bb100 > min_bb100)
    criteria.append(CertificationCriterion(
        name="Positive Expected Value (BB/100)",
        required=f"> {min_bb100:+.1f} BB/100",
        actual=f"{actual_bb100:+.2f} BB/100",
        passed=c3_passed,
        detail="Candidate must maintain positive win rate across full preliminary/semifinal/final structure",
    ))

    # Criterion 4: Top 12 Qualification Rate
    # Random baseline = min(1.0, 12 / field_size)
    random_top12_baseline = min(1.0, 12.0 / max(1, field_size))
    required_top12_rate = min(1.0, min_top12_multiplier * random_top12_baseline)
    actual_top12_rate = candidate_row.get("top12_rate", 0.0)
    c4_passed = (actual_top12_rate >= required_top12_rate)
    criteria.append(CertificationCriterion(
        name="Top 12 Deep Run Qualification Rate",
        required=f">={required_top12_rate:.1%} ({min_top12_multiplier:.1f}x baseline {random_top12_baseline:.1%})",
        actual=f"{actual_top12_rate:.1%} ({candidate_row.get('top12_count', 0)}/{runs} qualified)",
        passed=c4_passed,
        detail="Candidate must consistently navigate 200-hand preliminary stage into top 12 playoff",
    ))

    # Criterion 5: Head-to-Head & Benchmark Dominance
    candidate_h2h = h2h.get(cid, {})
    h2h_passed = True
    h2h_details = []
    board_by_cid = {r["cid"]: r for r in board}

    if baseline_cids:
        for b_cid in baseline_cids:
            pair_rate = candidate_h2h.get(b_cid, {}).get("win_rate", 50.0)
            b_row = board_by_cid.get(b_cid)
            dominates = (pair_rate >= min_h2h_win_rate)
            if not dominates and allow_ev_bypass and b_row is not None:
                tournament_ev_superior = (
                    candidate_row.get("score", 0) >= b_row.get("score", 0)
                    and candidate_row.get("avg_bb100", 0) >= b_row.get("avg_bb100", 0)
                    and candidate_row.get("final_rate", 0) >= b_row.get("final_rate", 0)
                    and candidate_row.get("top12_rate", 0) >= b_row.get("top12_rate", 0)
                )
                if tournament_ev_superior:
                    dominates = True
                    h2h_details.append(f"vs {b_cid}: {pair_rate:.1f}% H2H (EV Superior: Final {candidate_row['final_rate']:.1%} vs {b_row['final_rate']:.1%}, +{candidate_row['avg_bb100']:.1f} vs +{b_row['avg_bb100']:.1f} BB/100)")
                else:
                    h2h_details.append(f"vs {b_cid}: {pair_rate:.1f}% H2H (below required {min_h2h_win_rate:.1f}%)")
            elif not dominates:
                h2h_details.append(f"vs {b_cid}: {pair_rate:.1f}% H2H (below required {min_h2h_win_rate:.1f}%)")
            else:
                h2h_details.append(f"vs {b_cid}: {pair_rate:.1f}% H2H")

            if not dominates:
                h2h_passed = False
    elif candidate_h2h:
        rates = [v.get("win_rate", 50.0) for v in candidate_h2h.values()]
        avg_h2h = sum(rates) / max(1, len(rates))
        h2h_passed = (avg_h2h >= min_h2h_win_rate)
        h2h_details.append(f"Average pairwise win rate: {avg_h2h:.1f}%")
    else:
        h2h_passed = True
        h2h_details.append("No competitor head-to-head records")

    req_h2h_desc = f"Win rate >= {min_h2h_win_rate:.1f}% against benchmark"
    if allow_ev_bypass:
        req_h2h_desc += " OR superior tournament EV (score/BB100/final rate)"

    criteria.append(CertificationCriterion(
        name="Benchmark Dominance & Tournament EV",
        required=req_h2h_desc,
        actual=", ".join(h2h_details) if h2h_details else "N/A",
        passed=h2h_passed,
        detail="Candidate must demonstrate pairwise or tournament equity dominance over legacy benchmarks",
    ))

    all_passed = all(c.passed for c in criteria) if require_all_passed else True
    recommendation = "PROMOTE_TO_CHAMPION" if all_passed else "REJECT"
    summary = (
        f"Candidate '{name}' ({cid}) PASSED all {len(criteria)} certification criteria."
        if all_passed else
        f"Candidate '{name}' ({cid}) FAILED {sum(1 for c in criteria if not c.passed)}/{len(criteria)} certification criteria."
    )

    return ChampionCertificationResult(
        certified=all_passed,
        candidate_id=cid,
        candidate_name=name,
        criteria=criteria,
        summary=summary,
        recommendation=recommendation,
    )


def print_certification_card(result: ChampionCertificationResult):
    """Print executive decision card for champion certification gate."""
    badge = "🏆 [PASS - CERTIFIED CHAMPION]" if result.certified else "❌ [FAIL - CERTIFICATION REJECTED]"
    print("\n" + "=" * 86)
    print("           🏛️ AGENTPOKER OFFICIAL TOURNAMENT CHAMPION CERTIFICATION GATE")
    print("=" * 86)
    print(f"  Candidate Agent : {result.candidate_name} ({result.candidate_id})")
    print(f"  Certification   : {badge}")
    print(f"  Recommendation  : {result.recommendation}")
    print(f"  Timestamp       : {result.timestamp}")
    print("-" * 86)
    print(f"  {'GATE CRITERIA':<35} {'REQUIRED':<24} {'ACTUAL':<18} {'STATUS'}")
    print("-" * 86)
    for c in result.criteria:
        status_str = "✅ PASS" if c.passed else "❌ FAIL"
        req_str = c.required[:22]
        act_str = str(c.actual)[:16]
        print(f"  {c.name:<35} {req_str:<24} {act_str:<18} {status_str}")
    print("-" * 86)
    print(f"  Summary: {result.summary}")
    print("=" * 86 + "\n")


class CertificationError(RuntimeError):
    """Raised when an uncertified or failing candidate attempts champion promotion."""
    pass


def promote_champion(
    candidate_source: str | Path,
    target_path: str | Path = "models/champion.json",
    backup: bool = True,
    certification_result: ChampionCertificationResult | None = None,
) -> Path:
    """Safely promote certified candidate model to production champion.json with timestamped backup."""
    if certification_result is None or not isinstance(certification_result, ChampionCertificationResult):
        raise CertificationError(
            "Champion promotion strictly requires a valid ChampionCertificationResult object. Promotion aborted."
        )
    if not certification_result.certified:
        raise CertificationError(
            f"Candidate cannot be promoted: certification failed ({certification_result.summary}). Promotion aborted."
        )

    src = Path(candidate_source)
    tgt = Path(target_path)
    if not src.exists():
        raise FileNotFoundError(f"Candidate source file not found: {src}")

    tgt.parent.mkdir(parents=True, exist_ok=True)
    if backup and tgt.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = tgt.parent / f"{tgt.stem}_backup_{ts}{tgt.suffix}"
        shutil.copy2(tgt, backup_path)
        print(f"[Champion Gate] Backed up existing champion to: {backup_path}")

    content = json.loads(src.read_text(encoding="utf-8"))
    content["certification"] = certification_result.to_dict()

    tgt.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Champion Gate] Successfully promoted '{src.name}' -> '{tgt.resolve()}'")
    return tgt


def interactive_select_competitors(
    candidates: list[CompetitorCandidate],
) -> tuple[list[CompetitorCandidate], str, int, int]:
    """Terminal interactive menu to let the user select agents and match settings."""
    # Categorize candidates
    models = [c for c in candidates if c.category == "model"]
    archetypes = [c for c in candidates if c.category == "archetype"]
    profiles = [c for c in candidates if c.category == "profile"]

    all_ordered: list[CompetitorCandidate] = models + archetypes + profiles
    selected_indices: set[int] = set()

    # Default selection: champion model + gen_004 or baseline archetype
    if len(all_ordered) >= 2:
        selected_indices.add(0)
        selected_indices.add(1)

    while True:
        # Render candidate list
        print("\n" + "=" * 78)
        print("          🎴 Sohu Agent Poker 锦标赛擂台赛 — 选手选择菜单 (Arena)")
        print("=" * 78)

        idx = 1
        if models:
            print("\n  📂 【已训练演化模型】")
            for c in models:
                mark = "[*]" if (idx - 1) in selected_indices else "[ ]"
                print(f"   {idx:>2d}. {mark} {c.name:<32} {c.description}")
                idx += 1

        if archetypes:
            print("\n  🤖 【内置原型 Bot】")
            for c in archetypes:
                mark = "[*]" if (idx - 1) in selected_indices else "[ ]"
                print(f"   {idx:>2d}. {mark} {c.name:<32} {c.description}")
                idx += 1

        if profiles:
            print("\n  👤 【赛场真实玩家画像 (Top 活跃)】")
            for c in profiles:
                mark = "[*]" if (idx - 1) in selected_indices else "[ ]"
                print(f"   {idx:>2d}. {mark} {c.name:<32} {c.description}")
                idx += 1

        print("\n" + "-" * 78)
        cur_names = [all_ordered[i].name for i in sorted(selected_indices)]
        print(f"  当前已勾选擂主 ({len(selected_indices)} 位):")
        if cur_names:
            print("   👉 " + ", ".join(cur_names))
        else:
            print("   👉 (未选择任何选手，至少需选 2 位)")

        print("-" * 78)
        print("  💡 操作指南:")
        print("    • 输入序号勾选/取消 (例如: 1 2 6 或 1,6,7)")
        print("    • 输入 'am' (或 'all_models') 全选所有已训模型")
        print("    • 输入 'aa' (或 'all_archetypes') 全选所有原型 Bot")
        print("    • 输入 'c' (或 'clear') 清空当前已选")
        print("    • 【直接按回车 Enter】 确认已选阵容并进入对决配置")
        print("=" * 78)

        try:
            choice = input("请输入指令或序号 (直接回车确认): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[擂台赛取消]")
            sys.exit(0)

        if not choice:
            if len(selected_indices) < 2:
                print("\n⚠️ 擂台对决至少需要选择 2 位选手参赛！请继续输入序号勾选。")
                continue
            else:
                # Confirmed!
                break

        # Handle commands
        low = choice.lower()
        if low in ("c", "clear"):
            selected_indices.clear()
            continue
        if low in ("am", "all_models"):
            for i, c in enumerate(all_ordered):
                if c.category == "model":
                    selected_indices.add(i)
            continue
        if low in ("aa", "all_archetypes"):
            for i, c in enumerate(all_ordered):
                if c.category == "archetype":
                    selected_indices.add(i)
            continue

        # Parse numbers
        parts = choice.replace(",", " ").split()
        toggled = False
        for part in parts:
            if part.isdigit():
                val = int(part) - 1
                if 0 <= val < len(all_ordered):
                    if val in selected_indices:
                        selected_indices.remove(val)
                    else:
                        selected_indices.add(val)
                    toggled = True
                else:
                    print(f"⚠️ 序号超出范围: {part}")
            else:
                # Try matching by name
                matches = [i for i, c in enumerate(all_ordered) if part.lower() in c.name.lower()]
                for m in matches:
                    if m in selected_indices:
                        selected_indices.remove(m)
                    else:
                        selected_indices.add(m)
                    toggled = True

        if not toggled:
            print("⚠️ 未识别的输入，请输入数字序号 (如: 1 2)。")

    chosen_competitors = [all_ordered[i] for i in sorted(selected_indices)]

    # Next prompts: Opponent pool, runs, field size
    print("\n" + "=" * 78)
    print("                  ⚙️ 擂台锦标赛赛制与对手池配置")
    print("=" * 78)

    # Prompt 1: Opponent Pool
    print("\n[1/3] 请选择陪练对手池来源 (填充剩余席位):")
    print("  1. 混合池 (50% 原型 Bot + 50% 赛场真实玩家画像) [推荐，兼具多变性与赛场贴合度]")
    print("  2. 纯真实玩家画像池 (100% 真实线上实战对手画像)")
    print("  3. 纯原型 Bot 池 (100% 经典德扑风格 Bot: TAG/LAG/Rock/Station/Maniac)")
    try:
        pool_in = input("请选择 [默认 1, 直接回车]: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)
    opp_mode = "mix"
    if pool_in == "2":
        opp_mode = "profiles"
    elif pool_in == "3":
        opp_mode = "archetypes"

    # Prompt 2: Runs
    print("\n[2/3] 擂台锦标赛场数 (Runs):")
    print("  提示: 每场完整经历 200手预赛 + 12强半决赛 + 6人决赛桌，多核并发极快")
    try:
        runs_in = input("请输入场数 [默认 20 场, 直接回车]: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)
    runs = 20
    if runs_in.isdigit() and int(runs_in) > 0:
        runs = int(runs_in)

    # Prompt 3: Field size
    min_field = max(12, len(chosen_competitors))
    if min_field % 6 != 0:
        min_field += 6 - (min_field % 6)
    default_field = max(120, min_field)
    print(f"\n[3/3] 每场锦标赛总人数 (Agents):")
    print(f"  提示: 需为 6 的倍数 (当前至少需 {min_field} 人，官方标准正赛为 120 人)")
    try:
        field_in = input(f"请输入总人数 [默认 {default_field} 人, 直接回车]: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)
    field_size = default_field
    if field_in.isdigit() and int(field_in) >= min_field:
        val = int(field_in)
        if val % 6 != 0:
            val += 6 - (val % 6)
        field_size = val

    return chosen_competitors, opp_mode, runs, field_size
