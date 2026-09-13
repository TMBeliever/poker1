from __future__ import annotations
import argparse, os, json
from pathlib import Path
from .config import TournamentConfig
from .strategy import StrategyAgent
from .training import StrategyTrainer, ArenaEvaluator, ARCHETYPES
from .tournament import LeagueSimulator, SimAgent
from .protocol import AgentPokerClient, Config
from .live import LiveRunner
from .collector import JSONLCollector, ReplayBuilder
from .profiler import OpponentProfiler
from .battle import ArenaBattle, discover_candidates, interactive_select_competitors, print_battle_report, CompetitorCandidate

def run_connect(base_url: str = "https://poker.bang.sohu.com") -> None:
    import http.server, urllib.parse, secrets, webbrowser
    state = secrets.token_urlsafe(16)
    auth_code = None

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/callback":
                qs = urllib.parse.parse_qs(parsed.query)
                ret_state = qs.get("state", [""])[0]
                code = qs.get("code", [""])[0]
                if ret_state == state and code:
                    auth_code = code
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"<h1>Agent Connected Successfully!</h1><p>Credentials received. You may close this tab now.</p>")
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Invalid state or missing authorization code.")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), CallbackHandler)
    port = server.server_address[1]
    url = f"{base_url.rstrip('/')}/cli/connect?port={port}&state={state}"
    print(f"\n[Connect] Starting temporary authorization listener on http://127.0.0.1:{port}...")
    print(f"[Connect] Opening browser for Agent authorization:\n  {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    server.handle_request()
    server.server_close()

    if not auth_code:
        print("[Connect] Authorization cancelled or no code received.")
        return

    print("[Connect] Exchanging code for Agent secretKey...")
    client = AgentPokerClient(Config(base_url=base_url))
    resp = client.exchange_code(auth_code)
    sk = resp.get("secretKey")
    agent_id = resp.get("agent_id")
    agent_name = resp.get("name", "Unnamed")

    if not sk:
        print(f"[Connect] Failed to receive secretKey: {resp}")
        return

    active_cid = ""
    try:
        disc = client.discover("active")
        comps = disc.get("competitions", [])
        if comps:
            active_cid = comps[0]["id"]
            print(f"[Connect] Discovered active competition: {comps[0].get('name')} ({active_cid})")
    except Exception:
        pass

    env_content = f"AGENTPOKER_APP={base_url.rstrip('/')}\nAGENTPOKER_KEY={sk}\nAGENTPOKER_COMPETITION_ID={active_cid}\nAGENTPOKER_TIMEOUT=10\n"
    p = Path(".env")
    p.write_text(env_content, encoding="utf-8")
    try:
        os.chmod(".env", 0o600)
    except Exception:
        pass

    masked_sk = sk[:6] + "..." + sk[-4:] if len(sk) > 10 else "[PROTECTED]"
    print(f"\n[Connect] Success! Connected as Agent '{agent_name}' (ID: {agent_id})")
    print(f"[Connect] Saved credentials to .env (mode 0600, key: {masked_sk})")
    print(f"[Connect] Start playing with: python -m agentpoker.cli live")

def main():
    p=argparse.ArgumentParser(prog='agentpoker'); sub=p.add_subparsers(dest='cmd',required=True)
    default_field = TournamentConfig.official_120().field_size
    s=sub.add_parser('simulate',help='快速模拟：让 N 个原型 bot 打一场锦标赛看结果分布（不训练）')
    s.add_argument('--agents',type=int,default=default_field,help=f'参赛人数（默认 {default_field}，官方正赛规模）')
    s.add_argument('--runs',type=int,default=10,help='模拟场数（默认 10）')
    s.add_argument('--seed',type=int,default=7,help='随机种子，同种子结果可复现')
    t=sub.add_parser('train',help='进化训练：逐代演化策略参数，产出冠军模型')
    t.add_argument('--track',choices=['universal','targeted'],default=None,help='训练轨：universal=通用原型池（泛化好）；targeted=真实画像特训（贴合当前赛场）。不指定时：有 --profiles 走 targeted，否则 universal')
    t.add_argument('--generations',type=int,default=10,help='训练代数。每代耗时取决于 --runs 和 --agents')
    t.add_argument('--population',type=int,default=16,help='每代候选策略数量，越大探索越广，但每代耗时线性增加')
    t.add_argument('--runs',type=int,default=30,help='每个候选评估的锦标赛场数。建议正式训练用 >=120')
    t.add_argument('--final-race',type=int,default=200,help='最终在留出集上验证冠军的场数，决定最后报告的 fitness 有多可信')
    t.add_argument('--agents',type=int,default=default_field,help=f'每场锦标赛参赛人数，对齐真实赛场规模（默认 {default_field}）')
    t.add_argument('--equity-samples',type=int,default=0,help='胜率蒙特卡洛采样数：0=用离线标定表查表（快，训练推荐）；>0=实时精确采样（更准但慢很多）')
    t.add_argument('--workers',type=int,default=0,help='并发进程数，0=自动（最多 8 核）')
    t.add_argument('--profiles',default=None,help='对手画像文件路径。targeted 轨默认 models/opponent_profiles.json')
    t.add_argument('--mix-profiles',dest='mix_profiles',action='store_true',help='通用训练中混入优质真实画像（与原型共同组成对手池）。不加则保持纯原型训练')
    t.add_argument('--profile-min-hands',type=int,default=100,help='画像质量门槛：手数低于此值不采用，用于剔除小样本噪声画像（默认 100 手）')
    t.add_argument('--profile-top',type=int,default=None,help='只选取手牌数排名前 N 的优质画像')
    t.add_argument('--profile-share',type=float,default=0.5,help='画像在对手池中最多占的座位比例，其余留给原型+种群+名人堂（默认 0.5，即画像最多占一半）')
    t.add_argument('--save',default=None,help='冠军模型输出路径')
    t.add_argument('--archive',default=None,help='各代存档目录，用于断点续训')
    t.add_argument('--no-resume',dest='resume',action='store_false',default=True,help='关闭续训：忽略已有存档、从第 1 代重训（默认会自动从存档继续）')
    t.add_argument('--reeval-runs',type=int,default=40,help='每代初选后用全新对手池复评的场数，用于消除「冠军只是抽样运气」的偏差')
    t.add_argument('--holdout-frac',type=float,default=0.25,help='留出集比例：这部分真实画像不参与训练，只用于最终验证泛化能力（默认 25%%）')
    t.add_argument('--base-model',default=None,help='初始底模路径（如 models/champion.json）。没有历史存档或新开特训时以此模型为起点微调，避免从零冷启动')
    t.add_argument('--self-play',action='store_true',default=False,help='启用 2.5 影子自博弈模式：在对手池中常驻历史巅峰克隆体作为守门员，强化防守平衡防反杀')
    t.add_argument('--shadow-clones',type=int,default=2,help='2.5 自博弈模式下同桌常驻的影子克隆体数量（默认 2）')
    e=sub.add_parser('evaluate',help='离线评估：让指定模型跟真实画像打 N 场锦标赛，看泛化表现（不上真实赛场）')
    e.add_argument('--strategy',default='models/champion.json',help='要评估的模型文件路径')
    e.add_argument('--runs',type=int,default=500,help='评估场数。场数越多置信区间越窄，500 场时标准误约 ±0.016')
    e.add_argument('--agents',type=int,default=default_field,help=f'对手池人数，建议对齐真实赛场规模（默认 {default_field}）')
    e.add_argument('--equity-samples',type=int,default=0,help='胜率采样数：0=查表（快）；>0=实时精确采样（准但慢）')
    e.add_argument('--profiles',default=None,help='对手画像文件，指定后对手来自真实选手而非原型')
    e.add_argument('--workers',type=int,default=0,help='并发进程数，0=自动')
    l=sub.add_parser('live',help='实战：接入真实赛场自动对局')
    l.add_argument('--competition-id',default=os.getenv('AGENTPOKER_COMPETITION_ID'),help='赛事 ID，默认读 .env')
    l.add_argument('--key',default=os.getenv('AGENTPOKER_KEY'),help='API 密钥，默认读 .env')
    l.add_argument('--app',default=os.getenv('AGENTPOKER_APP','https://poker.bang.sohu.com'),help='服务器地址')
    l.add_argument('--strategy',default=None,help='使用的模型文件路径（默认自动优选当前战力最高模型）')
    l.add_argument('--max-steps',type=int,default=0,help='最多循环多少步后退出，0=不限')
    l.add_argument('--max-hands',type=int,default=0,help='打满多少手后自动停止，0=不限（想跑一个 200 手周期就设 200）')
    l.add_argument('--round-hands',type=int,default=20,help='每轮手数，用于轮次结算提示（默认 20）')
    l.add_argument('--cycle-hands',type=int,default=200,help='每个锦标赛周期手数，用于赛季结算与压力信号（默认 200）')
    l.add_argument('--no-auto-profile',dest='auto_profile',action='store_false',default=True,help='关闭画像热更新（默认每 20 手自动把刚打完的对手写进画像库）')
    l.add_argument('--profiles',default='models/opponent_profiles.json',help='画像文件路径，实战时用于识别对手类型')
    l.add_argument('--equity-samples',type=int,default=200,help='胜率蒙特卡洛采样数（实战极高精度默认 200，单次仅 ~7ms，绝不超时）')
    r=sub.add_parser('replay-export',help='把实录事件流转换成手牌记录（供 profile 使用）')
    r.add_argument('--raw',default='data/raw/events.jsonl',help='原始事件文件（live 运行时自动写入）')
    r.add_argument('--out',default='data/processed/hands.jsonl',help='输出的手牌文件路径')
    sub.add_parser('discover',help='查询当前有哪些进行中的赛事')
    conn=sub.add_parser('connect',help='浏览器授权登录，把密钥写入 .env')
    conn.add_argument('--app',default=os.getenv('AGENTPOKER_APP','https://poker.bang.sohu.com'),help='服务器地址')
    pr=sub.add_parser('profile',help='拉取/清洗战绩，生成对手画像')
    pr.add_argument('--input',default='data/processed/hands.jsonl',help='手牌输入文件（本地已有数据时用）')
    pr.add_argument('--out',default='models/opponent_profiles.json',help='画像输出路径（会覆盖同路径旧文件）')
    pr.add_argument('--prior-weight',type=float,default=8.0,help='贝叶斯先验权重：值越大，小样本选手越向「均衡型」收缩，避免几手牌就被误判成狂徒')
    pr.add_argument('--min-hands',type=int,default=30,help='最低手数门槛：不足此手数的选手不写进画像（默认 30）')
    pr.add_argument('--no-filter-afk',dest='filter_afk',action='store_false',default=True,help='关闭挂机过滤（默认会剔除从不入池的僵尸号）')
    pr.add_argument('--competition-id',default=os.getenv('AGENTPOKER_COMPETITION_ID'),help='赛事 ID，默认读 .env')
    pr.add_argument('--pull',action='store_true',help='不从本地文件读，而是直接从赛事 API 拉取最新战绩')
    pr.add_argument('--max-hands',type=int,default=None,help='最多拉取多少手牌，不设则全量拉取')
    bt=sub.add_parser('battle',aliases=['arena'],help='锦标赛擂台对决：多模型/多Agent 同台争冠擂台赛 (终端交互勾选/命令行直选)')
    bt.add_argument('--models',nargs='*',default=None,help='参赛模型文件路径列表 (如 models/champion.json models/archive_universal/gen_004.json)')
    bt.add_argument('--archetypes',nargs='*',default=None,help='参赛内置原型 Bot 列表 (如 tight lag maniac nit station balanced)')
    bt.add_argument('--profiles',nargs='*',default=None,help='参赛真实玩家 agent_id 列表 (如 agent_2b330360a0882e97)')
    bt.add_argument('--opponents',choices=['pyramid','mix','profiles','archetypes','sharks','fish'],default='pyramid',help='陪练对手池来源: pyramid=金字塔生态(30%%鲨鱼+40%%常规+30%%鱼, 默认), mix=混合池, profiles=纯真实画像, archetypes=纯原型Bot, sharks=全鲨鱼压测, fish=全鱼收割测试')
    bt.add_argument('--profiles-file',default='models/opponent_profiles.json',help='对手画像文件路径')
    bt.add_argument('--runs',type=int,default=None,help='擂台锦标赛场数 (默认 20)')
    bt.add_argument('--agents',type=int,default=default_field,help=f'每场锦标赛总人数，6的倍数 (默认 {default_field})')
    bt.add_argument('--equity-samples',type=int,default=0,help='胜率采样数 (默认 0 查表)')
    bt.add_argument('--workers',type=int,default=0,help='并发进程数 (0 为自动多核)')
    bt.add_argument('--seed',type=int,default=42,help='随机数种子')
    bt.add_argument('--save-report',default=None,help='将完整战报与矩阵保存为 JSON 文件')
    bt.add_argument('--certify',action='store_true',help='执行冠军认证门禁 (Champion Certification Gate)')
    bt.add_argument('--candidate',default=None,help='指定参与认证的候选选手 CID、模型名称或路径')
    bt.add_argument('--promote',action='store_true',help='认证通过后自动晋升为 models/champion.json (带时间戳备份)')
    dash=sub.add_parser('dashboard',help='启动全能轻量级可视化 Web 控制台 (训练、比赛、换桌、战报复盘)')
    dash.add_argument('--port',type=int,default=8080,help='控制台监听端口，默认 8080')
    dash.add_argument('--host',default='0.0.0.0',help='监听地址，0.0.0.0 支持远程服务器访问')
    args=p.parse_args()
    if args.cmd=='simulate':
        names=list(ARCHETYPES); agents=[]
        for i in range(args.agents):
            base=ARCHETYPES[names[i%len(names)]]
            agents.append(SimAgent(f'agent_{i+1:03d}',StrategyAgent(base,seed=args.seed+i,name=f'agent_{i+1:03d}')))
        for run in range(args.runs):
            sim=LeagueSimulator(agents,seed=args.seed+run*10007); r=sim.run_event(); print(f'run={run+1} top12={[x.agent_id for x in r["qualified"][:12]]} champion={r["final"][0].agent_id if r["final"] else None}')
    elif args.cmd=='train':
        if args.track is not None:
            track = args.track
        elif args.mix_profiles:
            track = 'universal'
        else:
            track = 'targeted' if args.profiles else 'universal'

        base_model = args.base_model
        save_path = args.save or 'models/candidate.json'
        archive_dir = args.archive or 'models/archive'

        if track == 'targeted':
            profiles_src = args.profiles or 'models/opponent_profiles.json'
            if base_model is None and os.path.exists('models/champion.json'):
                base_model = 'models/champion.json'
            print(f"[Train] === 启动【赛场特训收割模式】===")
            if base_model:
                print(f"[Train] 对比/基线底模: {base_model}")
            print(f"[Train] 挂载画像: {profiles_src} | 模型保存: {save_path} | 归档: {archive_dir}")
        else:
            profiles_src = (args.profiles or 'models/opponent_profiles.json') if args.mix_profiles else None
            print(f"[Train] === 启动【自适应演化训练模式】===")
            if base_model:
                print(f"[Train] 对比/基线底模: {base_model}")
            if profiles_src:
                print(f"[Train] 混练模式: 原型池 + 优质画像 (门槛>={args.profile_min_hands}手, 占比<={args.profile_share:.0%}) | 画像源: {profiles_src}")
            else:
                print(f"[Train] 原型博弈模式 (无真实画像挂载)")
            print(f"[Train] 候选模型保存: {save_path} | 归档: {archive_dir}")

        trainer = StrategyTrainer(
            seed=7, pool_size=args.agents, equity_samples=args.equity_samples, workers=args.workers,
            profiles=profiles_src, holdout_frac=args.holdout_frac, profile_min_hands=args.profile_min_hands,
            profile_share=args.profile_share, profile_top=args.profile_top,
            self_play=args.self_play, shadow_clones=args.shadow_clones
        )
        if profiles_src:
            n_prof = trainer.n_profiles_loaded
            print(f"[Train] 画像质量筛选: {n_prof} 位通过门槛 (>={args.profile_min_hands} 手)")
            if n_prof and n_prof < 10:
                print(f"[Train] 警告: 通过门槛的画像仅 {n_prof} 位，对手池多样性偏低，建议下调 --profile-min-hands")
        champ, report = trainer.fit(args.generations, args.population, args.runs, save=save_path, archive=archive_dir, final_race=args.final_race, resume=args.resume, reeval_runs=args.reeval_runs, base_model=base_model)
        
        print(f"[Train] 训练完成！候选模型已保存到 {save_path}")
        print(f"[Train] 提示: 按照生产安全门禁规范，新演化模型必须通过擂台认证门禁后方可晋升为生产冠军:")
        print(f"       python -m agentpoker.cli battle --models models/champion.json {save_path} --certify --candidate {save_path} --promote")
    elif args.cmd=='evaluate':
        pth=StrategyAgent.load(args.strategy).params
        r=ArenaEvaluator(pool_size=args.agents,equity_samples=args.equity_samples,profiles=args.profiles,workers=args.workers).evaluate(pth,runs=args.runs,seed_offset=9911,verbose=True)
        print(json.dumps(r,ensure_ascii=False,indent=2))
    elif args.cmd=='discover': print(AgentPokerClient(Config()).discover())
    elif args.cmd=='connect': run_connect(base_url=args.app)
    elif args.cmd=='replay-export': print(f'exported {ReplayBuilder(args.raw).export(args.out)} hands -> {args.out}')
    elif args.cmd=='profile':
        profiler=OpponentProfiler()
        cid=args.competition_id or os.getenv('AGENTPOKER_COMPETITION_ID')
        if args.pull or not os.path.exists(args.input):
            if not cid: raise SystemExit('AGENTPOKER_COMPETITION_ID is required to pull hand history.')
            client=AgentPokerClient(Config(competition_id=cid))
            print(f"[Profile] Pulling hand history from competition {cid}...")
            count=profiler.pull_competition_hands(client,cid,max_hands=args.max_hands,save_hands_path=args.input)
            print(f"[Profile] Fetched and saved {count} hands to {args.input}")
        else:
            count=profiler.ingest_file(args.input)
            print(f"[Profile] Ingested {count} hands from local file {args.input}")
        res=profiler.export(args.out,prior_weight=args.prior_weight,min_hands=args.min_hands,filter_afk=args.filter_afk)
        print(f"[Profile] Generated clean profiles for {len(res)} agents -> {args.out}")
    elif args.cmd=='live':
        key = args.key or os.getenv('AGENTPOKER_KEY')
        if not key:
            raise SystemExit('Error: AGENTPOKER_KEY is required.\nRun `python -m agentpoker.cli connect` to login, or export AGENTPOKER_KEY in .env')
        cid = args.competition_id or os.getenv('AGENTPOKER_COMPETITION_ID')
        if not cid:
            # Try auto-discovery of active competition
            client_temp = AgentPokerClient(Config(base_url=args.app, key=key))
            try:
                cs = client_temp.discover('active').get('competitions', [])
                if cs:
                    cid = cs[0]['id']
                    print(f"[Live] Auto-selected active competition: {cs[0].get('name')} ({cid})")
            except Exception:
                pass
        if not cid:
            raise SystemExit('Error: AGENTPOKER_COMPETITION_ID is required.')
        prof_file = args.profiles if args.profiles and os.path.exists(args.profiles) else None
        strat_path = args.strategy or "models/champion.json"
        if not os.path.exists(strat_path):
            raise FileNotFoundError(
                f"Production strategy model file not found: '{strat_path}'. "
                f"Live play cannot run without an official certified champion model."
            )

        st = StrategyAgent.load(strat_path, profiles=prof_file)
        samples_val = getattr(args, 'equity_samples', 200) or 200
        st.params.equity_samples = samples_val
        print(f"[Live] 🎯 比赛优选模型已装载: {strat_path}")
        print(f"[Live] 🚀 胜率蒙特卡洛极致精度已开启: equity_samples={samples_val} (单次决策 ~7ms, 绝对安全不超时)")
        client=AgentPokerClient(Config(base_url=args.app, key=key, competition_id=cid))
        runner = LiveRunner(
            client,
            st,
            competition_id=cid,
            collector=JSONLCollector(),
            round_hands=args.round_hands,
            cycle_hands=args.cycle_hands,
            auto_profile=args.auto_profile,
            profiles_path=args.profiles or 'models/opponent_profiles.json',
            strategy_path=strat_path,
        )
        runner.run(max_steps=args.max_steps or None, max_hands=args.max_hands or None)
    elif args.cmd in ('battle', 'arena'):
        import sys
        candidates = discover_candidates(profiles_path=args.profiles_file)
        cand_map = {c.cid: c for c in candidates}

        explicit_models = args.models or []
        explicit_archetypes = args.archetypes or []
        explicit_profiles = args.profiles or []
        has_explicit = bool(explicit_models or explicit_archetypes or explicit_profiles)

        if not has_explicit and not args.non_interactive:
            chosen, opp_mode, runs, field_size = interactive_select_competitors(candidates)
        else:
            chosen = []
            for m in explicit_models:
                m_path = Path(m)
                if m_path.exists():
                    try:
                        agent = StrategyAgent.load(m_path)
                        chosen.append(CompetitorCandidate(
                            cid=f"model:{m_path.stem}",
                            name=str(m_path),
                            category="model",
                            params=agent.params,
                            description=f"指定模型 {m_path.name}",
                            source=str(m_path),
                        ))
                    except Exception as ex:
                        print(f"警告: 无法加载模型 {m}: {ex}")
                else:
                    print(f"警告: 模型文件不存在 {m}")

            for a in explicit_archetypes:
                a_clean = a.lower().replace("bot_", "")
                if a_clean in ARCHETYPES:
                    chosen.append(CompetitorCandidate(
                        cid=f"archetype:{a_clean}",
                        name=f"bot_{a_clean}",
                        category="archetype",
                        params=ARCHETYPES[a_clean],
                        description=f"内置原型 {a_clean}",
                        source="builtin",
                    ))
                else:
                    print(f"警告: 未知原型 {a}，支持列表: {list(ARCHETYPES.keys())}")

            for p in explicit_profiles:
                cid = f"profile:{p}"
                if cid in cand_map:
                    chosen.append(cand_map[cid])
                else:
                    print(f"警告: 未在画像库中找到选手 {p}")

            if len(chosen) < 2:
                print("错误: 命令行指定的参赛选手少于 2 位，无法开赛。请指定至少 2 位选手，或直接运行 `python -m agentpoker.cli battle` 进入终端交互选择。")
                sys.exit(1)

            opp_mode = args.opponents or 'mix'
            runs = args.runs or 20
            field_size = args.agents or TournamentConfig.official_120().field_size

        arena = ArenaBattle(
            competitors=chosen,
            opponent_mode=opp_mode,
            profiles_path=args.profiles_file,
            field_size=field_size,
            equity_samples=args.equity_samples,
            workers=args.workers,
            seed=args.seed,
        )
        report = arena.run(runs=runs, verbose=True)
        print_battle_report(report)

        if getattr(args, 'certify', False):
            from .battle import certify_champion, print_certification_card, promote_champion
            cert_res = certify_champion(report, candidate_cid=getattr(args, 'candidate', None))
            print_certification_card(cert_res)
            if getattr(args, 'promote', False):
                if cert_res.certified:
                    cand_obj = next((c for c in chosen if c.cid == cert_res.candidate_id or c.name == cert_res.candidate_name), None)
                    if cand_obj and cand_obj.source and Path(cand_obj.source).exists():
                        promote_champion(cand_obj.source, certification_result=cert_res)
                    else:
                        print(f"无法自动晋升: 候选选手 '{cert_res.candidate_name}' 无有效本地文件源")
                else:
                    print(f"拒绝晋升: 候选选手未通过冠军认证门禁 ({cert_res.recommendation})")

        if args.save_report:
            p = Path(args.save_report)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[战报已保存] {p.resolve()}")
    elif args.cmd == 'dashboard':
        from .dashboard import run_server
        run_server(host=args.host, port=args.port)
if __name__=='__main__': main()
