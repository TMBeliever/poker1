from __future__ import annotations
import os, sys, json, time, signal, subprocess, urllib.parse, datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent

class ProcessManager:
    def __init__(self):
        self.train_proc: subprocess.Popen | None = None
        self.live_proc: subprocess.Popen | None = None
        self.train_start_time: float | None = None
        self.live_start_time: float | None = None
        self.live_strategy: str | None = None
        self.logs_dir = ROOT_DIR / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_file = self.logs_dir / "training.log"
        self.live_log_file = self.logs_dir / "live.log"

    def is_training(self) -> bool:
        if self.train_proc is not None:
            if self.train_proc.poll() is None:
                return True
            self.train_proc = None
        # Check system processes
        return self._find_proc("agentpoker.cli train") is not None

    def is_live(self) -> bool:
        if self.live_proc is not None:
            if self.live_proc.poll() is None:
                return True
            self.live_proc = None
        return self._find_proc("agentpoker.cli live") is not None

    def _find_proc(self, pattern: str) -> int | None:
        try:
            out = subprocess.check_output(["ps", "-ef"], text=True)
            for line in out.splitlines():
                if pattern in line and "grep" not in line and "dashboard" not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        return int(parts[1])
        except Exception:
            pass
        return None

    def start_training(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.is_training():
            return {"status": "error", "message": "训练已在运行中，请勿重复启动"}
        
        generations = str(params.get("generations", 10))
        population = str(params.get("population", 8))
        runs = str(params.get("runs", 40))
        agents = str(params.get("agents", 120))
        workers = str(params.get("workers", 2))
        save_path = params.get("save") or "models/candidate.json"
        archive = params.get("archive") or "models/archive"
        
        start_mode = params.get("start_mode", "finetune")
        base_model = params.get("base_model") or "models/champion.json"
        
        opp_mode = params.get("opp_mode", "mix")
        min_hands = str(params.get("min_hands", 100))
        overwrite_champion = bool(params.get("overwrite_champion", False))
        self_play = bool(params.get("self_play", False))

        cmd = [
            sys.executable, "-m", "agentpoker.cli", "train",
            "--generations", generations,
            "--population", population,
            "--runs", runs,
            "--agents", agents,
            "--workers", workers,
            "--save", save_path,
            "--archive", archive,
        ]

        if start_mode == "resume":
            # 默认带 resume，不传 --no-resume
            pass
        elif start_mode == "finetune":
            cmd.extend(["--no-resume", "--base-model", base_model])
        else: # scratch
            cmd.extend(["--no-resume"])

        selected_profiles = params.get("selected_profiles") if opp_mode in ("pure_human", "mix") else None
        profiles_file = "models/opponent_profiles.json"
        if selected_profiles and isinstance(selected_profiles, list) and len(selected_profiles) > 0:
            full_p = ROOT_DIR / "models" / "opponent_profiles.json"
            if full_p.exists():
                try:
                    full_data = json.loads(full_p.read_text(encoding="utf-8"))
                    filtered = {aid: full_data[aid] for aid in selected_profiles if aid in full_data}
                    if filtered:
                        custom_p = ROOT_DIR / "models" / ".custom_selected_profiles.json"
                        custom_p.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
                        profiles_file = "models/.custom_selected_profiles.json"
                        # 用户显式勾选的选手，免受 min_hands 门槛意外剔除，确保100%参训
                        min_hands = "0"
                except Exception:
                    pass

        if opp_mode == "pure_human":
            cmd.extend(["--track", "targeted", "--profiles", profiles_file, "--profile-min-hands", min_hands, "--profile-share", "1.0"])
        elif opp_mode == "mix":
            cmd.extend(["--mix-profiles", "--profiles", profiles_file, "--profile-min-hands", min_hands, "--profile-share", "0.5"])
        else: # archetypes
            cmd.extend(["--track", "universal"])

        if self_play:
            cmd.extend(["--self-play", "--shadow-clones", "2"])

        f_log = open(self.train_log_file, "a", encoding="utf-8")
        f_log.write(f"\n=== [Dashboard] 训练启动于 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f_log.write(f"配置: 起步={start_mode}, 对手池={opp_mode}, 2.5影子自博弈={'开启' if self_play else '关闭'}, 保存={save_path}, 覆盖主模型={'是' if overwrite_champion else '否'}\n")
        f_log.write(f"命令: {' '.join(cmd)}\n\n")
        f_log.flush()

        self.train_proc = subprocess.Popen(
            cmd, cwd=str(ROOT_DIR), stdout=f_log, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid
        )
        self.train_start_time = time.time()

        if overwrite_champion:
            import threading
            def _watch_and_promote(proc, sp):
                proc.wait()
                if proc.returncode == 0:
                    src = ROOT_DIR / sp
                    dst = ROOT_DIR / "models" / "champion.json"
                    with open(self.train_log_file, "a", encoding="utf-8") as fl:
                        fl.write(f"\n[Dashboard] 🏁 训练成功完成！产出模型已保存至: {sp}\n")
                        try:
                            content = json.loads(src.read_text(encoding="utf-8")) if src.exists() else {}
                            cert = content.get("certification", {})
                            if cert.get("certified"):
                                from agentpoker.battle import ChampionCertificationResult, CertificationCriterion, promote_champion
                                cert_res = ChampionCertificationResult(
                                    certified=True,
                                    candidate_id=cert.get("candidate_id", src.stem),
                                    candidate_name=cert.get("candidate_name", src.stem),
                                    criteria=[CertificationCriterion(**c) for c in cert.get("criteria", [])],
                                    summary=cert.get("summary", ""),
                                    recommendation=cert.get("recommendation", "PROMOTE_TO_CHAMPION"),
                                    timestamp=cert.get("timestamp", "")
                                )
                                promote_champion(candidate_source=src, target_path=dst, backup=True, certification_result=cert_res)
                                fl.write(f"[Dashboard] 🏆 模型已通过认证并安全晋升为主战模型: {dst}\n")
                            else:
                                fl.write("[Dashboard] ⚠️ [Gate 4 认证保护] 新产出模型为训练演化检查点。根据 V2.2 生产级规范，必须通过 120 人官方认证评估 (python -m agentpoker.cli battle --certify --promote) 验证 5 项金标后方可晋升为 champion.json，已阻止未认证直接覆盖。\n")
                        except Exception as e:
                            fl.write(f"[Dashboard] 警告: 自动晋升检查失败: {e}\n")
            threading.Thread(target=_watch_and_promote, args=(self.train_proc, save_path), daemon=True).start()

        return {"status": "success", "message": f"训练已成功启动 (PID: {self.train_proc.pid})"}

    def stop_training(self) -> dict[str, Any]:
        stopped = False
        if self.train_proc and self.train_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.train_proc.pid), signal.SIGTERM)
                stopped = True
            except Exception:
                pass
            self.train_proc = None

        pid = self._find_proc("agentpoker.cli train")
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                stopped = True
            except Exception:
                pass

        return {"status": "success", "message": "训练已终止" if stopped else "没有运行中的训练"}

    def start_live(self, strategy: str = "models/champion.json") -> dict[str, Any]:
        if self.is_live():
            return {"status": "error", "message": "比赛已在进行中"}
        
        cmd = [sys.executable, "-m", "agentpoker.cli", "live", "--strategy", strategy]
        f_log = open(self.live_log_file, "a", encoding="utf-8")
        f_log.write(f"\n=== [Dashboard] 比赛对战启动于 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f_log.write(f"使用策略: {strategy}\n\n")
        f_log.flush()

        self.live_proc = subprocess.Popen(
            cmd, cwd=str(ROOT_DIR), stdout=f_log, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid
        )
        self.live_start_time = time.time()
        self.live_strategy = strategy
        return {"status": "success", "message": f"比赛已成功启动 (PID: {self.live_proc.pid})"}

    def stop_live(self) -> dict[str, Any]:
        stopped = False
        if self.live_proc and self.live_proc.poll() is None:
            try:
                # Send SIGINT so live runner exits cleanly with self._leave()
                os.killpg(os.getpgid(self.live_proc.pid), signal.SIGINT)
                stopped = True
            except Exception:
                pass
            self.live_proc = None

        pid = self._find_proc("agentpoker.cli live")
        if pid:
            try:
                os.kill(pid, signal.SIGINT)
                stopped = True
            except Exception:
                pass

        return {"status": "success", "message": "已安全发送离桌指令并停止比赛" if stopped else "当前没有运行中的比赛"}

    def switch_table(self) -> dict[str, Any]:
        flag = ROOT_DIR / "data" / ".switch_table_flag"
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(str(time.time()), encoding="utf-8")
        return {"status": "success", "message": "已发送换桌指令！将在本手结束后安全离桌并重新排队进新桌"}

    def get_logs(self, which: str = "train", max_lines: int = 120) -> str:
        fpath = self.train_log_file if which == "train" else self.live_log_file
        if not fpath.exists():
            return "暂无日志输出。"
        try:
            lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()
            return "\n".join(lines[-max_lines:])
        except Exception as e:
            return f"读取日志出错: {e}"

pm = ProcessManager()

def get_archive_data() -> list[dict[str, Any]]:
    # Find active archive
    candidates = ["models/archive", "models/archive_v2"]
    archive_dir = None
    for c in candidates:
        p = ROOT_DIR / c
        if p.exists() and list(p.glob("gen_*.json")):
            archive_dir = p
            break
    if not archive_dir:
        return []

    results = []
    for gf in sorted(archive_dir.glob("gen_*.json")):
        try:
            d = json.loads(gf.read_text(encoding="utf-8"))
            gen_no = d.get("generation")
            m = d.get("metrics") or d.get("training_metrics") or {}
            p = d.get("parameters") or d.get("champion") or d.get("params") or {}
            results.append({
                "generation": gen_no,
                "fitness": round(float(m.get("fitness", 0)), 4),
                "top12_rate": round(float(m.get("top12_rate", 0)) * 100, 1),
                "champion_rate": round(float(m.get("champion_rate", 0)) * 100, 1),
                "avg_bb100": round(float(m.get("avg_bb100", 0)), 1),
                "params": {
                    "vpip": round(float(p.get("vpip", 0)), 3),
                    "cbet_frequency": round(float(p.get("cbet_frequency", 0)), 3),
                    "flop_value_threshold": round(float(p.get("flop_value_threshold", 0.58)), 3),
                    "turn_value_threshold": round(float(p.get("turn_value_threshold", 0.65)), 3),
                    "river_value_threshold": round(float(p.get("river_value_threshold", 0.74)), 3),
                    "dry_board_bet_size": round(float(p.get("dry_board_bet_size", 0.33)), 3),
                    "wet_board_bet_size": round(float(p.get("wet_board_bet_size", 0.75)), 3),
                }
            })
        except Exception:
            pass
    return sorted(results, key=lambda x: x["generation"])

def get_models_list() -> list[str]:
    out = []
    models_dir = ROOT_DIR / "models"
    if models_dir.exists():
        for p in sorted(models_dir.glob("*.json")):
            if "profile" not in p.name and not p.name.startswith("."):
                out.append(f"models/{p.name}")
    return out or ["models/champion.json"]

def get_models_details() -> list[dict[str, Any]]:
    models_dir = ROOT_DIR / "models"
    if not models_dir.exists():
        return []
    
    out = []
    for p in sorted(models_dir.glob("*.json")):
        if "profile" in p.name or p.name.startswith("."):
            continue
        try:
            stat = p.stat()
            d = json.loads(p.read_text(encoding="utf-8"))
            
            m = d.get("metrics") or d.get("training_metrics") or {}
            history = d.get("history") or []
            if not m and history:
                m = history[-1].get("metrics") or {}
            
            cert = d.get("certification") or {}
            is_certified = bool(cert.get("certified", False))
            cert_summary = cert.get("summary", "")
            
            params = d.get("parameters") or d.get("champion") or d.get("params") or {}
            vpip = float(params.get("vpip", 0.15))
            open_freq = float(params.get("open_frequency", 0.5))
            threebet_freq = float(params.get("threebet_frequency", 0.05))
            cbet_freq = float(params.get("cbet_frequency", 0.5))
            turn_barrel = float(params.get("turn_barrel_frequency", 0.5))
            river_bluff = float(params.get("river_bluff_frequency", 0.03))
            safety = float(params.get("safety", 0.5))
            attack = float(params.get("attack", 0.5))
            flop_val = float(params.get("flop_value_threshold", 0.58))
            turn_val = float(params.get("turn_value_threshold", 0.65))
            river_val = float(params.get("river_value_threshold", 0.74))
            dry_size = float(params.get("dry_board_bet_size", 0.33))
            wet_size = float(params.get("wet_board_bet_size", 0.75))

            fitness = round(float(m.get("fitness", 0.0)), 4)
            top12_rate = round(float(m.get("top12_rate", 0.0)) * 100, 1)
            final_rate = round(float(m.get("final_rate", 0.0)) * 100, 1)
            champion_rate = round(float(m.get("champion_rate", 0.0)) * 100, 1)
            avg_bb100 = round(float(m.get("avg_bb100", 0.0)), 1)
            avg_rank = round(float(m.get("avg_rank", 18.0)), 1)
            gen = int(d.get("generation") or (history[-1].get("generation") if history else 0))

            if not m and cert:
                for crit in cert.get("criteria", []):
                    act = crit.get("actual", "")
                    detail = crit.get("detail", "")
                    cname = crit.get("name", "")
                    if "Top 12" in cname:
                        try:
                            top12_rate = float(act.split("%")[0].strip())
                        except Exception:
                            pass
                    elif "Title" in cname or "Deep Run" in cname:
                        try:
                            if "Final" in act:
                                final_rate = float(act.split("Final")[1].split("%")[0].strip())
                            if "Champ" in act:
                                champion_rate = float(act.split("Champ")[1].split("%")[0].strip())
                        except Exception:
                            pass
                    elif "BB/100" in cname:
                        try:
                            clean_bb = act.replace("+", "").replace("BB/100", "").strip()
                            avg_bb100 = round(float(clean_bb), 1)
                        except Exception:
                            pass
                    elif "Rank" in cname:
                        try:
                            if "#1" in act:
                                avg_rank = 1.0
                            if "Score:" in detail:
                                fitness = round(float(detail.split("Score:")[1].replace(")", "").strip()), 2)
                        except Exception:
                            pass

            if vpip < 0.07:
                archetype = "🛡️ 极紧反剥削 (Ultra-Nit)"
                arch_desc = "极其自律精炼的入池范围，专注于坚果与顶端牌力；在多人混战局中极难被清空，具备深赛程反杀与抗剥削能力。"
            elif vpip < 0.15:
                archetype = "🎯 严谨紧凶 (Disciplined TAG)"
                arch_desc = "平衡教科书式紧凶打法，翻前位次严明、后位积极偷盲，价值注下注极其清晰。"
            elif vpip < 0.22:
                archetype = "⚔️ 全能压制 (Universal TAG/LAG)"
                arch_desc = "宽入池配合中高频翻后持续下注，兼顾底池收割与强力价值注，对被动弱手与跟注站具有极强压榨效率。"
            else:
                archetype = "💥 激进松凶 (Aggressive LAG)"
                arch_desc = "极具攻击侵略性，频繁抢夺无主底池，单手筹码爆发力强。"

            strengths = []
            if open_freq >= 0.68:
                strengths.append(f"翻前高频偷盲 ({round(open_freq*100, 1)}%)")
            if threebet_freq >= 0.06:
                strengths.append(f"3-Bet 激进反打 ({round(threebet_freq*100, 1)}%)")
            if safety >= 0.7:
                strengths.append(f"超高防守容错 ({round(safety*100, 1)}%)")
            elif attack >= 0.55:
                strengths.append(f"高压进攻导向 ({round(attack*100, 1)}%)")
            if cbet_freq >= 0.5:
                strengths.append(f"连贯 C-Bet 开枪 ({round(cbet_freq*100, 1)}%)")
            if champion_rate >= 8.0:
                strengths.append(f"决赛夺冠爆发 ({champion_rate}%)")
            if top12_rate >= 50.0:
                strengths.append(f"深赛程晋级稳 ({top12_rate}%)")
            if avg_bb100 >= 60.0:
                strengths.append(f"大盲暴利收割 (+{avg_bb100} BB/100)")
            strengths.append("2.0干湿板平滑插值")

            score_burst = min(100, int(champion_rate * 9.0 + 10))
            score_deep = min(100, int(top12_rate * 1.5 + 10))
            score_profit = min(100, max(20, int(avg_bb100 * 0.8 + 20)))
            score_defense = min(100, max(10, int(safety * 60 + (1 - min(0.3, vpip)) * 130 - 30)))
            score_pressure = min(100, int(attack * 50 + cbet_freq * 40 + open_freq * 20))

            mtime_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
            size_kb = round(stat.st_size / 1024, 1)

            out.append({
                "name": p.name,
                "path": f"models/{p.name}",
                "is_main": (p.name == "champion.json"),
                "is_certified": is_certified,
                "cert_summary": cert_summary,
                "schema_version": d.get("schema_version", 1),
                "version": d.get("model_version") or d.get("version", "2.0"),
                "size_kb": size_kb,
                "mtime": mtime_str,
                "generation": gen,
                "fitness": fitness,
                "top12_rate": top12_rate,
                "final_rate": final_rate,
                "champion_rate": champion_rate,
                "avg_bb100": avg_bb100,
                "avg_rank": avg_rank,
                "archetype": archetype,
                "arch_desc": arch_desc,
                "strengths": strengths,
                "scores": {
                    "burst": score_burst,
                    "deep": score_deep,
                    "profit": score_profit,
                    "defense": score_defense,
                    "pressure": score_pressure
                },
                "params": {
                    "vpip": round(vpip * 100, 1),
                    "open": round(open_freq * 100, 1),
                    "threebet": round(threebet_freq * 100, 1),
                    "cbet": round(cbet_freq * 100, 1),
                    "barrel": round(turn_barrel * 100, 1),
                    "bluff": round(river_bluff * 100, 1),
                    "safety": round(safety * 100, 1),
                    "attack": round(attack * 100, 1),
                    "flop_val": round(flop_val, 2),
                    "turn_val": round(turn_val, 2),
                    "river_val": round(river_val, 2),
                    "dry_size": round(dry_size * 100, 0),
                    "wet_size": round(wet_size * 100, 0),
                }
            })
        except Exception:
            pass

    return sorted(out, key=lambda x: (not x["is_main"], -x["fitness"]))

def delete_model(model_rel_path: str) -> dict[str, Any]:
    if not model_rel_path:
        return {"status": "error", "message": "未指定要删除的模型文件"}
    norm = os.path.normpath(model_rel_path)
    if ".." in norm or not (norm.startswith("models/") or norm.startswith("models\\") or norm.startswith("models")):
        return {"status": "error", "message": "非法路径，仅允许操作 models/ 目录下的模型"}
    
    target = ROOT_DIR / norm
    if not target.exists() or not target.is_file():
        return {"status": "error", "message": f"文件不存在: {model_rel_path}"}
    
    if target.name == "champion.json":
        return {"status": "error", "message": "【安全保护】champion.json 是系统核心主战模型，禁止直接删除！如需更换，请将其他模型设为主战。"}
    if target.name == "opponent_profiles.json":
        return {"status": "error", "message": "opponent_profiles.json 是赛场对手画像库，禁止删除！"}

    try:
        target.unlink()
        return {"status": "success", "message": f"模型 {target.name} 已彻底删除！"}
    except Exception as e:
        return {"status": "error", "message": f"删除失败: {e}"}

def promote_model(model_rel_path: str) -> dict[str, Any]:
    norm = os.path.normpath(model_rel_path)
    if ".." in norm or not (norm.startswith("models/") or norm.startswith("models\\") or norm.startswith("models")):
        return {"status": "error", "message": "非法路径"}
    target = ROOT_DIR / norm
    if not target.exists() or not target.is_file():
        return {"status": "error", "message": f"文件不存在: {model_rel_path}"}
    
    dst = ROOT_DIR / "models" / "champion.json"
    if target.resolve() == dst.resolve():
        return {"status": "error", "message": "该模型已经是当前主战模型"}
    
    try:
        content = json.loads(target.read_text(encoding="utf-8"))
        cert = content.get("certification", {})
        if not cert or not cert.get("certified"):
            return {
                "status": "error",
                "message": (
                    f"❌ 【Gate 4 认证拦截】模型 {target.name} 尚未通过 120 人官方生产级认证！\n"
                    "根据系统规范，未认证模型禁止直接晋升为主战模型。\n"
                    "请在终端运行: python -m agentpoker.cli battle --certify --promote\n"
                    "通过 120 人对抗与 5 项严苛指标考核后方可安全晋升。"
                )
            }
        
        from agentpoker.battle import ChampionCertificationResult, CertificationCriterion, promote_champion
        criteria_objs = [
            CertificationCriterion(
                name=c.get("name", ""),
                required=c.get("required", ""),
                actual=c.get("actual", ""),
                passed=c.get("passed", False),
                detail=c.get("detail", "")
            )
            for c in cert.get("criteria", [])
        ]
        cert_res = ChampionCertificationResult(
            certified=True,
            candidate_id=cert.get("candidate_id", target.stem),
            candidate_name=cert.get("candidate_name", target.stem),
            criteria=criteria_objs,
            summary=cert.get("summary", "Certified via battle"),
            recommendation=cert.get("recommendation", "PROMOTE_TO_CHAMPION"),
            timestamp=cert.get("timestamp", datetime.datetime.now().isoformat())
        )
        promote_champion(candidate_source=target, target_path=dst, backup=True, certification_result=cert_res)
        return {"status": "success", "message": f"🏆 已安全晋升 {target.name} 为当前主战模型 (champion.json)！原模型已生成时间戳备份。"}
    except Exception as e:
        return {"status": "error", "message": f"设为主战模型失败: {e}"}

def get_model_content(model_rel_path: str) -> dict[str, Any]:
    norm = os.path.normpath(model_rel_path)
    if ".." in norm or not (norm.startswith("models/") or norm.startswith("models\\") or norm.startswith("models")):
        return {"status": "error", "message": "非法路径"}
    target = ROOT_DIR / norm
    if not target.exists() or not target.is_file():
        return {"status": "error", "message": f"文件不存在: {model_rel_path}"}
    try:
        return {"status": "success", "content": target.read_text(encoding="utf-8")}
    except Exception as e:
        return {"status": "error", "message": f"读取模型失败: {e}"}


def get_profiles_data() -> list[dict[str, Any]]:
    p = ROOT_DIR / "models" / "opponent_profiles.json"
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        out = []
        for aid, d in raw.items():
            out.append({
                "id": aid,
                "name": d.get("name") or aid[:8],
                "hands": int(d.get("hands", 0)),
                "bb_100": round(float(d.get("bb_100", 0.0)), 1),
                "vpip": round(float(d.get("vpip", 0.0)) * 100, 1),
                "pfr": round(float(d.get("pfr", 0.0)) * 100, 1),
                "af": round(float(d.get("af", 0.0)), 2),
                "archetype": d.get("archetype", "未知风格"),
                "advice": d.get("exploit_advice", "")
            })
        return sorted(out, key=lambda x: x["bb_100"], reverse=True)
    except Exception:
        return []

def get_recent_hands(limit: int = 30) -> list[dict[str, Any]]:
    hands_file = ROOT_DIR / "data" / "processed" / "hands.jsonl"
    if not hands_file.exists():
        # Fallback to replay export if raw exists
        raw_events = ROOT_DIR / "data" / "raw" / "events.jsonl"
        if raw_events.exists():
            try:
                from agentpoker.collector import ReplayBuilder
                ReplayBuilder(raw_events).export(str(hands_file))
            except Exception:
                return []
        else:
            return []

    lines = []
    try:
        with open(hands_file, "r", encoding="utf-8", errors="ignore") as f:
            for l in f:
                if l.strip():
                    lines.append(l)
    except Exception:
        return []

    recent = lines[-limit:]
    recent.reverse()
    results = []
    for line in recent:
        try:
            h = json.loads(line)
            table_id = h.get("tableId", "Unknown")
            completed_at = h.get("completedAt", "")
            pot = h.get("pot", 0)
            board = h.get("communityCards", [])
            players = h.get("players", [])
            hero = next((p for p in players if p.get("agentId") and ("hero" in p.get("name", "").lower() or p.get("netChange") is not None)), None)
            
            results.append({
                "id": h.get("id"),
                "tableId": table_id[:8],
                "time": completed_at[11:19] if len(completed_at) >= 19 else completed_at,
                "pot": pot,
                "board": board,
                "players_count": len(players),
                "players": [{
                    "name": p.get("name", p.get("agentId", "P")[:6]),
                    "seat": p.get("seatIndex", 0),
                    "stack": p.get("startingStack", 0),
                    "cards": p.get("holeCards", []),
                    "net": p.get("netChange", 0)
                } for p in players],
                "hero_net": hero.get("netChange", 0) if hero else 0,
                "actions_count": len(h.get("actions", []))
            })
        except Exception:
            pass
    return results

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgentPoker V2.2 全能可视化控制台</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { background-color: #0b0f19; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .card-dark { background: #131b2e; border: 1px solid #1e293b; }
    .poker-card {
      display: inline-flex; align-items: center; justify-content: center;
      width: 32px; height: 44px; border-radius: 4px; font-weight: bold; font-size: 14px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.5); margin-right: 4px; background: white;
    }
    .suit-red { color: #dc2626; }
    .suit-black { color: #1e293b; }
  </style>
</head>
<body class="text-slate-200">
  <!-- Top Navbar -->
  <header class="bg-slate-900 border-b border-slate-800 px-6 py-4 flex flex-wrap items-center justify-between shadow-lg">
    <div class="flex items-center space-x-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-red-600 flex items-center justify-center text-xl shadow-md">
        🃏
      </div>
      <div>
        <h1 class="text-xl font-bold text-white tracking-wide flex items-center gap-2">
          AgentPoker <span class="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">V2.2 TOURNAMENT READY</span>
        </h1>
        <p class="text-xs text-slate-400">德州扑克自适应博弈训练引擎 & 120人实战云控中心</p>
      </div>
    </div>
    <!-- Quick Status Bar -->
    <div class="flex items-center space-x-4 mt-2 sm:mt-0">
      <div id="trainBadge" class="flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-medium bg-slate-800 border border-slate-700 text-slate-400">
        <span class="w-2 h-2 rounded-full bg-slate-500"></span>
        <span>训练: 待命</span>
      </div>
      <div id="liveBadge" class="flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-medium bg-slate-800 border border-slate-700 text-slate-400">
        <span class="w-2 h-2 rounded-full bg-slate-500"></span>
        <span>比赛: 未连接</span>
      </div>
      <button onclick="switchTable()" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-600/80 hover:bg-amber-600 text-white flex items-center gap-1.5 transition">
        <i class="fa-solid fa-arrows-rotate"></i> 换桌
      </button>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 py-6">
    <!-- Navigation Tabs -->
    <div class="flex border-b border-slate-800 mb-6 space-x-6 text-sm font-medium">
      <button onclick="switchTab('tabTrain')" id="tabBtnTrain" class="pb-3 border-b-2 border-emerald-500 text-emerald-400 flex items-center gap-2">
        <i class="fa-solid fa-dumbbell"></i> 演化训练控制
      </button>
      <button onclick="switchTab('tabLive')" id="tabBtnLive" class="pb-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center gap-2">
        <i class="fa-solid fa-trophy"></i> 在线赛事实战
      </button>
      <button onclick="switchTab('tabHands')" id="tabBtnHands" class="pb-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center gap-2">
        <i class="fa-solid fa-clock-rotate-left"></i> 对局复盘观战
      </button>
      <button onclick="switchTab('tabModels')" id="tabBtnModels" class="pb-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center gap-2">
        <i class="fa-solid fa-microchip"></i> 策略模型仓库
      </button>
    </div>

    <!-- TAB 1: 训练控制台 -->
    <div id="tabTrain" class="space-y-6">
      <!-- 指标卡片 -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="card-dark rounded-xl p-4 shadow">
          <div class="text-xs text-slate-400 font-medium">当前最高 Fitness</div>
          <div class="text-2xl font-bold text-emerald-400 mt-1" id="statFitness">--</div>
          <div class="text-xs text-slate-500 mt-1">加权适应度评估</div>
        </div>
        <div class="card-dark rounded-xl p-4 shadow">
          <div class="text-xs text-slate-400 font-medium">预赛出线率 (Top 12)</div>
          <div class="text-2xl font-bold text-amber-400 mt-1" id="statTop12">--%</div>
          <div class="text-xs text-slate-500 mt-1">半决赛突围概率</div>
        </div>
        <div class="card-dark rounded-xl p-4 shadow">
          <div class="text-xs text-slate-400 font-medium">冠军夺冠率 (Champion)</div>
          <div class="text-2xl font-bold text-purple-400 mt-1" id="statChamp">--%</div>
          <div class="text-xs text-slate-500 mt-1">决赛第 1 名胜率</div>
        </div>
        <div class="card-dark rounded-xl p-4 shadow">
          <div class="text-xs text-slate-400 font-medium">平均单手收益 BB/100</div>
          <div class="text-2xl font-bold text-blue-400 mt-1" id="statBB">--</div>
          <div class="text-xs text-slate-500 mt-1">大盲/百手盈利能力</div>
        </div>
      </div>

      <!-- 训练参数控制与图表 -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- 控制表单 -->
        <div class="card-dark rounded-xl p-5 shadow space-y-3.5">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-sliders text-emerald-400"></i> 演化训练高级配置
            </h2>
            <span class="text-[10px] text-emerald-400/80 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">2.0 引擎</span>
          </div>

          <!-- 1. 起步方式与底模 -->
          <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800 space-y-2 text-xs">
            <div>
              <label class="block text-slate-400 mb-1 font-medium">起步演化方式</label>
              <select id="selStartMode" onchange="onStartModeChange(); updateEstimates();" class="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-white">
                <option value="finetune">🔥 基于底模热启动微调 (稳健推荐)</option>
                <option value="resume">⚡ 历史断点续训 (从选定归档最新代数继续)</option>
                <option value="scratch">🌱 从零冷启动演化 (无底模自主摸索)</option>
              </select>
            </div>
            <div id="boxBaseModel">
              <label class="block text-slate-400 mb-1">选择微调底模 (Base Model)</label>
              <select id="selBaseModel" onchange="updateEstimates();" class="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-white"></select>
            </div>
          </div>

          <!-- 2. 对手池与真人画像配置 -->
          <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800 space-y-2 text-xs">
            <div>
              <label class="block text-slate-400 mb-1 font-medium">对抗对手池来源</label>
              <select id="selOppMode" onchange="onOppModeChange(); updateEstimates();" class="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-white">
                <option value="mix">🛡️ 混合模式 (50% 真实用户模型 + 50% 经典原型)</option>
                <option value="pure_human">🎯 真人模式 (100% 真实用户模型特训收割)</option>
                <option value="archetypes">⚖️ 原型模式 (0% 真人，纯经典原型自博弈)</option>
              </select>
            </div>
            <div class="grid grid-cols-2 gap-2">
              <div id="boxMinHands">
                <label class="block text-slate-400 mb-1">画像入选门槛 (手)</label>
                <input type="number" id="inpMinHands" value="100" oninput="updateEstimates()" class="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-white" title="剔除低于该手数的样本噪声画像">
              </div>
              <div id="boxAgents">
                <label class="block text-slate-400 mb-1">每场总人数 (Agents)</label>
                <input type="number" id="inpAgents" value="120" oninput="updateEstimates()" class="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-white">
              </div>
            </div>
            <div class="pt-1.5 border-t border-slate-800/60 flex items-center space-x-2">
              <input type="checkbox" id="chkSelfPlay" onchange="updateEstimates()" class="rounded bg-slate-900 border-slate-700 text-purple-500">
              <label for="chkSelfPlay" class="text-purple-300 font-medium text-[11px] flex items-center gap-1 cursor-pointer">
                <i class="fa-solid fa-clone text-purple-400"></i> 激活 2.5 影子自博弈 (常驻历史巅峰镜像互博防守)
              </label>
            </div>

            <!-- 原型模式说明提示块 (当选择原型模式时展示，明确提示不支持指定用户模型) -->
            <div id="boxArchetypesNotice" class="hidden p-2.5 bg-indigo-950/40 border border-indigo-800/60 rounded-lg text-indigo-300 space-y-1">
              <div class="flex items-center gap-1.5 font-bold text-xs">
                <i class="fa-solid fa-scale-balanced text-indigo-400"></i>
                <span>当前为【原型模式】</span>
              </div>
              <p class="text-[11px] text-slate-400 leading-relaxed">
                此模式由系统 8 大经典纳什博弈原型互博自演化，<span class="text-amber-300 font-semibold">无需且不挂载真人用户模型</span>。
              </p>
              <p class="text-[10px] text-slate-500">
                💡 提示：如需指定参训的真实用户模型（针对性特训或混练），请将上方对手池切换为 <span class="text-emerald-400 font-semibold">【真人模式】</span> 或 <span class="text-indigo-400 font-semibold">【混合模式】</span>。
              </p>
            </div>

            <!-- 真实用户模型挑选面板 (仅在真人模式或混合模式下显示与激活) -->
            <div id="boxUserProfiles" class="pt-2 border-t border-slate-800/80 space-y-2">
              <div class="flex items-center justify-between">
                <label class="text-xs text-amber-400 font-bold flex items-center gap-1.5">
                  <i class="fa-solid fa-users text-amber-400"></i> 
                  <span>指定真实用户模型 (参训对手定制)</span>
                </label>
                <span id="lblSelectedCount" class="text-[10px] px-2 py-0.5 rounded font-bold bg-emerald-950 border border-emerald-700 text-emerald-300">加载中...</span>
              </div>

              <!-- 快捷预设按钮组 -->
              <div class="flex flex-wrap items-center gap-1 text-[10px]">
                <button type="button" onclick="selectProfiles('pyramid')" class="px-2 py-1 rounded bg-indigo-950/90 border border-indigo-500/80 hover:bg-indigo-900 text-indigo-200 transition font-bold shadow-sm" title="按 30% 顶级鲨鱼 + 40% 中游稳健 + 30% 提款机弱鱼 智能配比"><i class="fa-solid fa-layer-group text-[9px] mr-1"></i>金字塔生态(36人:强中弱)</button>
                <button type="button" onclick="selectProfiles('all')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 transition">全选(63人)</button>
                <button type="button" onclick="selectProfiles('top20')" class="px-2 py-1 rounded bg-emerald-950/80 border border-emerald-700 hover:bg-emerald-900 text-emerald-300 transition">前20强劲敌</button>
                <button type="button" onclick="selectProfiles('station')" class="px-2 py-1 rounded bg-amber-950/80 border border-amber-700 hover:bg-amber-900 text-amber-300 transition">跟注站/鱼群</button>
                <button type="button" onclick="selectProfiles('maniac')" class="px-2 py-1 rounded bg-red-950/80 border border-red-700 hover:bg-red-900 text-red-300 transition">狂徒/松凶</button>
                <button type="button" onclick="selectProfiles('none')" class="px-1.5 py-1 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 transition">清空</button>
              </div>

              <!-- 大弹窗挑选入口按钮 + 折叠切换按钮 -->
              <div class="flex items-center gap-1.5">
                <button type="button" onclick="openRosterModal()" class="flex-1 py-2 px-2.5 rounded-lg bg-gradient-to-r from-amber-600/30 via-slate-800 to-amber-600/30 hover:from-amber-600/50 hover:to-amber-600/50 border border-amber-500/60 text-amber-300 text-xs font-bold flex items-center justify-center gap-1.5 transition shadow">
                  <i class="fa-solid fa-users-viewfinder"></i>
                  <span>全景大屏挑选 (展示完整名字/战力)</span>
                  <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i>
                </button>
                <button type="button" onclick="toggleRosterDrawer()" id="btnToggleDrawer" class="py-2 px-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs font-medium flex items-center gap-1 transition" title="展开/收起就地列表">
                  <span id="lblDrawerAction">展开</span>
                  <i id="icoRosterToggle" class="fa-solid fa-chevron-down text-[10px] transition-transform"></i>
                </button>
              </div>

              <!-- 优化后的就地折叠列表 -->
              <div id="rosterDrawer" class="hidden mt-2 p-2 bg-slate-950/95 rounded-lg border border-slate-800 space-y-2 shadow-inner">
                <div class="flex items-center justify-between gap-1 pb-1.5 border-b border-slate-800 text-[10px]">
                  <span class="text-slate-400">就地快速勾选:</span>
                  <select id="selRosterSort" onchange="renderRosterList()" class="bg-slate-900 border border-slate-700 rounded px-1.5 py-0.5 text-slate-300 text-[10px]">
                    <option value="bb">按 战力(BB/100) 降序</option>
                    <option value="hands">按 对战手数 降序</option>
                    <option value="vpip">按 入池激进度 降序</option>
                  </select>
                </div>
                <div id="rosterList" class="max-h-[300px] overflow-y-auto space-y-2 pr-1 text-xs">
                  <div class="text-center py-4 text-slate-500 text-xs">正在加载赛场画像库...</div>
                </div>
              </div>
            </div>
          </div>

          <!-- 3. 训练规模参数 -->
          <div class="grid grid-cols-2 gap-2 text-xs">
            <div>
              <label class="block text-slate-400 mb-1">训练代数</label>
              <input type="number" id="inpGens" value="10" oninput="updateEstimates()" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-white">
            </div>
            <div>
              <label class="block text-slate-400 mb-1">种群候选数</label>
              <input type="number" id="inpPop" value="8" oninput="updateEstimates()" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-white">
            </div>
            <div>
              <label class="block text-slate-400 mb-1">初评场数/候选</label>
              <input type="number" id="inpRuns" value="40" oninput="updateEstimates()" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-white">
            </div>
            <div>
              <label class="block text-slate-400 mb-1">并发 Workers (2h2g设2)</label>
              <input type="number" id="inpWorkers" value="2" oninput="updateEstimates()" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-white">
            </div>
          </div>

          <!-- 4. 模型保存与覆盖策略 -->
          <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800 space-y-2 text-xs">
            <div class="grid grid-cols-2 gap-2">
              <div>
                <label class="block text-slate-400 mb-1">产出模型文件</label>
                <input type="text" id="inpSavePath" value="models/candidate.json" oninput="updateEstimates()" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-white font-mono text-[11px]">
              </div>
              <div>
                <label class="block text-slate-400 mb-1">归档目录</label>
                <input type="text" id="inpArchiveDir" value="models/archive" oninput="updateEstimates()" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-white font-mono text-[11px]">
              </div>
            </div>
            <div class="flex items-start space-x-2 pt-1">
              <input type="checkbox" id="chkOverwriteMain" onchange="updateEstimates()" class="mt-0.5 rounded bg-slate-900 border-slate-700 text-emerald-500">
              <label for="chkOverwriteMain" class="text-slate-300 text-[11px] leading-tight">
                训练终局胜出后，<span class="text-amber-400 font-semibold">自动同步覆盖主战模型</span> (models/champion.json，旧模型自动生成 .bak 备份)
              </label>
            </div>
          </div>

          <!-- 操作按钮 -->
          <div class="pt-1 flex gap-3">
            <button onclick="startTrain()" id="btnStartTrain" class="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-2.5 rounded-lg text-xs transition flex items-center justify-center gap-1.5 shadow-lg shadow-emerald-950">
              <i class="fa-solid fa-play"></i> 启动演化训练
            </button>
            <button onclick="stopTrain()" id="btnStopTrain" class="bg-red-600/80 hover:bg-red-600 text-white font-medium px-4 py-2.5 rounded-lg text-xs transition flex items-center justify-center gap-1.5">
              <i class="fa-solid fa-stop"></i> 停止
            </button>
          </div>
        </div>

        <!-- 旁边预估面板 & 演化趋势折线图 -->
        <div class="lg:col-span-2 space-y-4 flex flex-col">
          <!-- ⚡ 训练开销与收益实时智能预估看板 -->
          <div class="card-dark rounded-xl p-4 shadow border border-emerald-500/20 bg-gradient-to-r from-slate-900 via-slate-900 to-emerald-950/20">
            <div class="flex items-center justify-between border-b border-slate-800 pb-2 mb-3">
              <h2 class="text-xs font-bold text-white flex items-center gap-2">
                <i class="fa-solid fa-bolt text-amber-400"></i> 参数实时动态预估与硬件评估
              </h2>
              <span class="text-[10px] text-slate-400">基于 2h2g 算力模型实时演算</span>
            </div>
            
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800/80">
                <div class="text-[11px] text-slate-400">单代预估耗时</div>
                <div class="text-base font-black text-emerald-400 mt-0.5" id="estGenTime">~3 分钟</div>
                <div class="text-[10px] text-slate-500 mt-0.5">双核并行评估</div>
              </div>

              <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800/80">
                <div class="text-[11px] text-slate-400">整轮总耗时</div>
                <div class="text-base font-black text-amber-400 mt-0.5" id="estTotalTime">~34 分钟</div>
                <div class="text-[10px] text-slate-500 mt-0.5">含终局 3 方大验证</div>
              </div>

              <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800/80">
                <div class="text-[11px] text-slate-400">锦标赛对抗总量</div>
                <div class="text-base font-black text-blue-400 mt-0.5" id="estTotalMatches">3,700 场</div>
                <div class="text-[10px] text-slate-500 mt-0.5">全手牌博弈样本</div>
              </div>

              <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800/80">
                <div class="text-[11px] text-slate-400">2h2g 内存预计占用</div>
                <div class="text-xs font-bold text-slate-200 mt-1" id="estRam">~75 MB (3.6%)</div>
                <div class="text-[10px] text-emerald-400 mt-0.5">安全余量 > 1.8GB</div>
              </div>
            </div>

            <div class="mt-3 pt-2.5 border-t border-slate-800/80 flex flex-col sm:flex-row items-start sm:items-center justify-between text-[11px] text-slate-400 gap-1">
              <div><i class="fa-solid fa-bullseye text-purple-400 mr-1"></i> 置信度评级: <span id="estConfidence" class="text-emerald-400 font-semibold">稳健平衡 (标准误 ±0.05)</span></div>
              <div class="text-slate-500" id="estStrategySummary">底模微调 · 50% 混合实战 · 仅存新模型</div>
            </div>
          </div>

          <!-- 演化趋势折线图 -->
          <div class="card-dark rounded-xl p-5 shadow flex-1 flex flex-col">
            <div class="flex items-center justify-between mb-3">
              <h2 class="text-base font-semibold text-white flex items-center gap-2">
                <i class="fa-solid fa-chart-line text-blue-400"></i> Fitness 与盈利演化曲线
              </h2>
              <span class="text-xs text-slate-400" id="lblGenCount">累计 0 代数据</span>
            </div>
            <div class="flex-1 min-h-[220px]">
              <canvas id="fitnessChart"></canvas>
            </div>
          </div>
        </div>
      </div>

      <!-- 2.0 参数架构与实时日志 -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- 2.0 基因参数面板 -->
        <div class="card-dark rounded-xl p-5 shadow space-y-3">
          <h2 class="text-base font-semibold text-white flex items-center gap-2">
            <i class="fa-solid fa-dna text-purple-400"></i> 2.0 策略基因指纹 (Champion)
          </h2>
          <div class="space-y-2 text-xs">
            <div class="flex justify-between border-b border-slate-800 pb-1">
              <span class="text-slate-400">入池率 VPIP</span>
              <span class="font-bold text-white" id="valVPIP">--</span>
            </div>
            <div class="flex justify-between border-b border-slate-800 pb-1">
              <span class="text-slate-400">翻牌价值门槛 Flop</span>
              <span class="font-bold text-emerald-400" id="valFlopVal">--</span>
            </div>
            <div class="flex justify-between border-b border-slate-800 pb-1">
              <span class="text-slate-400">转牌价值门槛 Turn</span>
              <span class="font-bold text-amber-400" id="valTurnVal">--</span>
            </div>
            <div class="flex justify-between border-b border-slate-800 pb-1">
              <span class="text-slate-400">河牌价值门槛 River</span>
              <span class="font-bold text-red-400" id="valRiverVal">--</span>
            </div>
            <div class="flex justify-between border-b border-slate-800 pb-1">
              <span class="text-slate-400">干燥板面下注尺寸</span>
              <span class="font-bold text-blue-400" id="valDrySize">--</span>
            </div>
            <div class="flex justify-between pb-1">
              <span class="text-slate-400">潮湿板面下注尺寸</span>
              <span class="font-bold text-purple-400" id="valWetSize">--</span>
            </div>
          </div>
        </div>

        <!-- 训练日志流 -->
        <div class="card-dark rounded-xl p-5 shadow lg:col-span-2 flex flex-col">
          <div class="flex items-center justify-between mb-2">
            <h2 class="text-base font-semibold text-white flex items-center gap-2">
              <i class="fa-solid fa-terminal text-emerald-400"></i> 训练引擎实时日志 (Console)
            </h2>
            <button onclick="refreshLogs()" class="text-xs text-slate-400 hover:text-white transition">
              <i class="fa-solid fa-rotate-right"></i> 刷新日志
            </button>
          </div>
          <pre id="trainLogBox" class="bg-black/80 rounded-lg p-3 text-xs font-mono text-emerald-400/90 overflow-y-auto h-48 border border-slate-800 whitespace-pre-wrap leading-relaxed"></pre>
        </div>
      </div>
    </div>

    <!-- TAB 2: 在线赛事实战 -->
    <div id="tabLive" class="hidden space-y-6">
      <div class="card-dark rounded-xl p-6 shadow space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <h2 class="text-lg font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-satellite-dish text-blue-400"></i> 赛场接入与桌况调度
            </h2>
            <p class="text-xs text-slate-400 mt-1">自动接入平台比赛，执行自适应决策与对手画像贝叶斯识别</p>
          </div>
          <div class="flex items-center gap-3">
            <button onclick="startLive()" class="bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition flex items-center gap-2">
              <i class="fa-solid fa-play"></i> 开始比赛
            </button>
            <button onclick="switchTable()" class="bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition flex items-center gap-2">
              <i class="fa-solid fa-arrows-rotate"></i> 立即换桌
            </button>
            <button onclick="stopLive()" class="bg-red-600 hover:bg-red-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition flex items-center gap-2">
              <i class="fa-solid fa-door-open"></i> 安全离桌停止
            </button>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div>
            <label class="block text-slate-400 mb-1">参赛模型选择</label>
            <select id="selLiveModel" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white"></select>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">换桌策略说明</label>
            <div class="bg-slate-900/60 border border-slate-800 rounded p-2 text-slate-400">
              点击换桌后，Agent 将在本手结算间隙向服务器发送离桌协议，并自动重新排队分配到新桌。
            </div>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">热重载保护</label>
            <div class="bg-slate-900/60 border border-slate-800 rounded p-2 text-emerald-400/80">
              ⚡ 策略热重载激活：比赛进行中若后台训练产出新 Champion，系统将自动热更新参数，无需退桌。
            </div>
          </div>
        </div>
      </div>

      <!-- 实战日志流 -->
      <div class="card-dark rounded-xl p-5 shadow">
        <h2 class="text-base font-semibold text-white mb-3 flex items-center gap-2">
          <i class="fa-solid fa-bolt text-amber-400"></i> 实战比赛实时日志 (Live Stream)
        </h2>
        <pre id="liveLogBox" class="bg-black/80 rounded-lg p-3 text-xs font-mono text-amber-400/90 overflow-y-auto h-72 border border-slate-800 whitespace-pre-wrap leading-relaxed"></pre>
      </div>
    </div>

    <!-- TAB 3: 对局复盘观战 -->
    <div id="tabHands" class="hidden space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-base font-semibold text-white flex items-center gap-2">
          <i class="fa-solid fa-list-check text-emerald-400"></i> 最近对战手牌记录 (Hand Replays)
        </h2>
        <button onclick="loadHands()" class="text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3 py-1.5 rounded-lg text-slate-300 transition">
          <i class="fa-solid fa-rotate"></i> 刷新对局
        </button>
      </div>
      <div id="handsList" class="space-y-3">
        <div class="text-center py-12 text-slate-500 text-xs">正在加载历史手牌...</div>
      </div>
    </div>

    <!-- TAB 4: 策略模型仓库与能力档案 -->
    <div id="tabModels" class="hidden space-y-6">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900/70 p-4 rounded-xl border border-slate-800 shadow">
        <div>
          <h2 class="text-base font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-microchip text-emerald-400"></i> 策略模型全景仓库与能力档案
          </h2>
          <p class="text-xs text-slate-400 mt-1">全景深度透视所有策略模型的胜率、战术特长、五维战力雷达与核心参数，支持一键切换主战、实战调用与安全删除清理。</p>
        </div>
        <div class="flex items-center gap-3">
          <button onclick="loadModelsDetails()" class="text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3.5 py-2 rounded-lg text-slate-200 transition flex items-center gap-1.5 shadow">
            <i class="fa-solid fa-rotate"></i> 刷新模型档案
          </button>
        </div>
      </div>

      <!-- 模型网格容器 -->
      <div id="modelsGrid" class="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div class="text-center py-16 text-slate-500 text-xs col-span-full">正在深度解析模型能力与战术特征...</div>
      </div>
    </div>
  </main>

  <script>
    let chartInstance = null;
    let currentTab = "tabTrain";

    function switchTab(tabId) {
      currentTab = tabId;
      document.getElementById("tabTrain").classList.add("hidden");
      document.getElementById("tabLive").classList.add("hidden");
      document.getElementById("tabHands").classList.add("hidden");
      document.getElementById("tabModels").classList.add("hidden");
      document.getElementById(tabId).classList.remove("hidden");

      document.getElementById("tabBtnTrain").className = "pb-3 border-b-2 flex items-center gap-2 " + (tabId === "tabTrain" ? "border-emerald-500 text-emerald-400" : "border-transparent text-slate-400 hover:text-slate-200");
      document.getElementById("tabBtnLive").className = "pb-3 border-b-2 flex items-center gap-2 " + (tabId === "tabLive" ? "border-emerald-500 text-emerald-400" : "border-transparent text-slate-400 hover:text-slate-200");
      document.getElementById("tabBtnHands").className = "pb-3 border-b-2 flex items-center gap-2 " + (tabId === "tabHands" ? "border-emerald-500 text-emerald-400" : "border-transparent text-slate-400 hover:text-slate-200");
      document.getElementById("tabBtnModels").className = "pb-3 border-b-2 flex items-center gap-2 " + (tabId === "tabModels" ? "border-emerald-500 text-emerald-400" : "border-transparent text-slate-400 hover:text-slate-200");

      if (tabId === "tabHands") loadHands();
      if (tabId === "tabModels") loadModelsDetails();
    }

    function renderCard(c) {
      if (!c || c.length < 2) return "";
      const rank = c.slice(0, -1);
      const suit = c.slice(-1).toLowerCase();
      let icon = "";
      let isRed = false;
      if (suit === "h") { icon = "♥"; isRed = true; }
      else if (suit === "d") { icon = "♦"; isRed = true; }
      else if (suit === "s") { icon = "♠"; isRed = false; }
      else if (suit === "c") { icon = "♣"; isRed = false; }
      return `<div class="poker-card ${isRed ? "suit-red" : "suit-black"}">${rank}${icon}</div>`;
    }

    async function fetchStatus() {
      try {
        const res = await fetch("/api/status");
        const data = await res.json();
        
        // Update badges
        const trainBadge = document.getElementById("trainBadge");
        if (data.training.running) {
          trainBadge.className = "flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-medium bg-emerald-950/80 border border-emerald-600 text-emerald-400";
          trainBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span><span>训练进行中 (PID: ${data.training.pid})</span>`;
        } else {
          trainBadge.className = "flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-medium bg-slate-800 border border-slate-700 text-slate-400";
          trainBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-slate-500"></span><span>训练: 空闲待命</span>`;
        }

        const liveBadge = document.getElementById("liveBadge");
        if (data.live.running) {
          liveBadge.className = "flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-medium bg-blue-950/80 border border-blue-600 text-blue-400";
          liveBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span><span>比赛对局中 (PID: ${data.live.pid})</span>`;
        } else {
          liveBadge.className = "flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs font-medium bg-slate-800 border border-slate-700 text-slate-400";
          liveBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-slate-500"></span><span>比赛: 未连接</span>`;
        }

        // Update archive charts and cards
        if (data.archive && data.archive.length > 0) {
          const last = data.archive[data.archive.length - 1];
          document.getElementById("statFitness").innerText = last.fitness.toFixed(3);
          document.getElementById("statTop12").innerText = last.top12_rate.toFixed(1) + "%";
          document.getElementById("statChamp").innerText = last.champion_rate.toFixed(1) + "%";
          document.getElementById("statBB").innerText = (last.avg_bb100 >= 0 ? "+" : "") + last.avg_bb100.toFixed(1);
          document.getElementById("lblGenCount").innerText = `累计到达 Gen ${last.generation}`;

          if (last.params) {
            document.getElementById("valVPIP").innerText = (last.params.vpip * 100).toFixed(1) + "%";
            document.getElementById("valFlopVal").innerText = last.params.flop_value_threshold.toFixed(2);
            document.getElementById("valTurnVal").innerText = last.params.turn_value_threshold.toFixed(2);
            document.getElementById("valRiverVal").innerText = last.params.river_value_threshold.toFixed(2);
            document.getElementById("valDrySize").innerText = (last.params.dry_board_bet_size * 100).toFixed(0) + "% 底池";
            document.getElementById("valWetSize").innerText = (last.params.wet_board_bet_size * 100).toFixed(0) + "% 底池";
          }
          updateChart(data.archive);
        }
      } catch (e) {
        console.error("Status error:", e);
      }
    }

    function updateChart(archive) {
      if (typeof Chart === 'undefined') return;
      const labels = archive.map(a => `Gen ${a.generation}`);
      const fitnessData = archive.map(a => a.fitness);
      const bbData = archive.map(a => a.avg_bb100);

      if (!chartInstance) {
        const ctx = document.getElementById("fitnessChart").getContext("2d");
        chartInstance = new Chart(ctx, {
          type: "line",
          data: {
            labels: labels,
            datasets: [
              {
                label: "Fitness 适应度",
                data: fitnessData,
                borderColor: "#10b981",
                backgroundColor: "rgba(16, 185, 129, 0.15)",
                fill: true,
                tension: 0.3,
                yAxisID: "y"
              },
              {
                label: "BB/100 净收益",
                data: bbData,
                borderColor: "#3b82f6",
                backgroundColor: "transparent",
                borderDash: [4, 4],
                tension: 0.3,
                yAxisID: "y1"
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            scales: {
              x: { grid: { color: "#1e293b" }, ticks: { color: "#94a3b8" } },
              y: {
                grid: { color: "#1e293b" }, ticks: { color: "#10b981" },
                title: { display: true, text: "Fitness", color: "#10b981" }
              },
              y1: {
                position: "right",
                grid: { drawOnChartArea: false }, ticks: { color: "#3b82f6" },
                title: { display: true, text: "BB/100", color: "#3b82f6" }
              }
            },
            plugins: { legend: { labels: { color: "#cbd5e1" } } }
          }
        });
      } else {
        chartInstance.data.labels = labels;
        chartInstance.data.datasets[0].data = fitnessData;
        chartInstance.data.datasets[1].data = bbData;
        chartInstance.update();
      }
    }

    async function loadModels() {
      try {
        const res = await fetch("/api/models");
        const list = await res.json();
        const sel1 = document.getElementById("selBaseModel");
        const sel2 = document.getElementById("selLiveModel");
        sel1.innerHTML = "";
        sel2.innerHTML = "";
        list.forEach(m => {
          const opt1 = document.createElement("option");
          opt1.value = m; opt1.innerText = m;
          if (m === "models/champion.json" || m.endsWith("/champion.json")) opt1.selected = true;
          sel1.appendChild(opt1);

          const opt2 = document.createElement("option");
          opt2.value = m; opt2.innerText = m;
          if (m === "models/champion.json" || m.endsWith("/champion.json")) opt2.selected = true;
          sel2.appendChild(opt2);
        });
      } catch (e) {}
    }

    async function loadHands() {
      const container = document.getElementById("handsList");
      try {
        const res = await fetch("/api/hands");
        const hands = await res.json();
        if (!hands || hands.length === 0) {
          container.innerHTML = `<div class="text-center py-10 text-slate-500 text-xs">暂无对战手牌记录，启动实战比赛后即可实时记录。</div>`;
          return;
        }
        let html = "";
        hands.forEach(h => {
          const isWin = h.hero_net > 0;
          const isLoss = h.hero_net < 0;
          const badgeColor = isWin ? "bg-emerald-950 text-emerald-400 border-emerald-800" : (isLoss ? "bg-red-950 text-red-400 border-red-800" : "bg-slate-800 text-slate-400 border-slate-700");
          const netText = (h.hero_net >= 0 ? "+" : "") + h.hero_net;
          
          let cardsHtml = "";
          (h.board || []).forEach(c => { cardsHtml += renderCard(c); });
          if (!cardsHtml) cardsHtml = `<span class="text-slate-500 text-xs italic">翻前决战 (Preflop)</span>`;

          html += `
            <div class="card-dark rounded-xl p-4 shadow border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div class="flex items-center space-x-3">
                <div class="text-center px-3 py-1 rounded-lg border ${badgeColor}">
                  <div class="text-[10px] uppercase font-bold">HERO</div>
                  <div class="text-sm font-black">${netText}</div>
                </div>
                <div>
                  <div class="text-xs text-slate-400 flex items-center gap-2">
                    <span><i class="fa-regular fa-clock"></i> ${h.time}</span>
                    <span>• 桌号: <span class="font-mono text-slate-300">${h.tableId}</span></span>
                    <span>• 底池: <span class="font-bold text-amber-400">${h.pot}</span></span>
                  </div>
                  <div class="mt-2 flex items-center">
                    <span class="text-xs text-slate-400 mr-2">公牌:</span>
                    <div class="flex">${cardsHtml}</div>
                  </div>
                </div>
              </div>
              <div class="text-xs text-slate-400 flex flex-wrap gap-2">
                ${h.players.map(p => `<span class="px-2 py-1 rounded bg-slate-900 border border-slate-800">${p.name}: ${(p.net>=0?"+":"")+p.net}</span>`).join("")}
              </div>
            </div>
          `;
        });
        container.innerHTML = html;
      } catch (e) {
        container.innerHTML = `<div class="text-red-400 text-xs py-4">加载手牌出错: ${e}</div>`;
      }
    }

    async function refreshLogs() {
      try {
        const r1 = await fetch("/api/train/logs");
        const t1 = await r1.text();
        const b1 = document.getElementById("trainLogBox");
        b1.innerText = t1;
        b1.scrollTop = b1.scrollHeight;

        const r2 = await fetch("/api/live/logs");
        const t2 = await r2.text();
        const b2 = document.getElementById("liveLogBox");
        b2.innerText = t2;
        b2.scrollTop = b2.scrollHeight;
      } catch (e) {}
    }

    function onStartModeChange() {
      const mode = document.getElementById("selStartMode").value;
      const box = document.getElementById("boxBaseModel");
      if (mode === "scratch" || mode === "resume") {
        box.classList.add("opacity-40", "pointer-events-none");
      } else {
        box.classList.remove("opacity-40", "pointer-events-none");
      }
    }

    function onOppModeChange() {
      const mode = document.getElementById("selOppMode").value;
      const boxProfiles = document.getElementById("boxUserProfiles");
      const boxNotice = document.getElementById("boxArchetypesNotice");
      const boxMinHands = document.getElementById("boxMinHands");

      if (mode === "pure_human" || mode === "mix") {
        if (boxProfiles) boxProfiles.classList.remove("hidden");
        if (boxNotice) boxNotice.classList.add("hidden");
        if (boxMinHands) boxMinHands.classList.remove("opacity-40", "pointer-events-none");
      } else {
        // archetypes mode: hide user models selection panel and show clear prompt
        if (boxProfiles) boxProfiles.classList.add("hidden");
        if (boxNotice) boxNotice.classList.remove("hidden");
        if (boxMinHands) boxMinHands.classList.add("opacity-40", "pointer-events-none");
      }
    }

    let cachedProfiles = [];
    let selectedProfileIds = new Set();
    let rosterDrawerOpen = false;

    function toggleRosterDrawer() {
      const mode = document.getElementById("selOppMode").value;
      if (mode === "archetypes") {
        alert(`当前对手池为【原型模式】，基于纳什博弈经典原型自博弈，不挂载真人用户模型。\n如需指定参训用户模型，请将对手池切换为【真人模式】或【混合模式】。`);
        return;
      }
      const drawer = document.getElementById("rosterDrawer");
      const icon = document.getElementById("icoRosterToggle");
      const actionText = document.getElementById("lblDrawerAction");
      rosterDrawerOpen = !rosterDrawerOpen;
      if (rosterDrawerOpen) {
        drawer.classList.remove("hidden");
        if (icon) icon.classList.add("rotate-180");
        if (actionText) actionText.innerText = "收起";
      } else {
        drawer.classList.add("hidden");
        if (icon) icon.classList.remove("rotate-180");
        if (actionText) actionText.innerText = "展开";
      }
    }

    function openRosterModal() {
      const mode = document.getElementById("selOppMode").value;
      if (mode === "archetypes") {
        alert(`当前对手池为【原型模式】，基于纳什博弈经典原型自博弈，不挂载真人用户模型。\n如需指定参训用户模型，请将对手池切换为【真人模式】或【混合模式】。`);
        return;
      }
      const modal = document.getElementById("rosterModal");
      if (modal) {
        modal.classList.remove("hidden");
        renderRosterList();
      }
    }

    function closeRosterModal() {
      const modal = document.getElementById("rosterModal");
      if (modal) modal.classList.add("hidden");
    }

    function syncAndRenderRoster(val) {
      const selInline = document.getElementById("selRosterSort");
      const selModal = document.getElementById("selModalRosterSort");
      if (selInline) selInline.value = val;
      if (selModal) selModal.value = val;
      renderRosterList();
    }

    function getArchetypeBadge(archetype) {
      let colorClass = "bg-slate-800 text-slate-300 border-slate-700";
      let shortName = archetype || "常规选手";
      if (archetype.includes("Maniac") || archetype.includes("狂徒")) {
        colorClass = "bg-red-950/80 text-red-400 border-red-700";
        shortName = "狂徒 Maniac";
      } else if (archetype.includes("LAG") || archetype.includes("松凶")) {
        colorClass = "bg-orange-950/80 text-orange-400 border-orange-700";
        shortName = "松凶 LAG";
      } else if (archetype.includes("TAG") || archetype.includes("紧凶")) {
        colorClass = "bg-blue-950/80 text-blue-400 border-blue-700";
        shortName = "紧凶 TAG";
      } else if (archetype.includes("Station") || archetype.includes("跟注站")) {
        colorClass = "bg-amber-950/80 text-amber-400 border-amber-700";
        shortName = "跟注站 Station";
      } else if (archetype.includes("Passive") || archetype.includes("被动鱼")) {
        colorClass = "bg-yellow-950/80 text-yellow-300 border-yellow-700";
        shortName = "被动鱼 Passive";
      } else if (archetype.includes("Nit") || archetype.includes("岩石")) {
        colorClass = "bg-slate-800 text-slate-400 border-slate-600";
        shortName = "岩石 Nit";
      } else if (archetype.includes("Balanced") || archetype.includes("均衡")) {
        colorClass = "bg-emerald-950/80 text-emerald-400 border-emerald-700";
        shortName = "均衡 Balanced";
      }
      return `<span class="px-1.5 py-0.5 rounded text-[10px] font-medium border ${colorClass}">${shortName}</span>`;
    }

    async function loadProfilesList() {
      const listEl = document.getElementById("rosterList");
      try {
        const res = await fetch("/api/profiles");
        cachedProfiles = await res.json();
        selectedProfileIds = new Set(cachedProfiles.map(p => p.id));
        updateRosterCount();
        renderRosterList();
      } catch (e) {
        if (listEl) listEl.innerHTML = `<div class="text-red-400 text-xs py-3 text-center">加载选手花名册失败: ${e}</div>`;
      }
    }

    function updateRosterCount() {
      const lbl = document.getElementById("lblSelectedCount");
      const modalLbl = document.getElementById("lblModalSelectedCount");
      if (cachedProfiles) {
        let nShark = 0, nMid = 0, nFish = 0;
        selectedProfileIds.forEach(id => {
          const p = cachedProfiles.find(x => x.id === id);
          if (p) {
            if (p.bb_100 >= 30) nShark++;
            else if (p.bb_100 <= -30) nFish++;
            else nMid++;
          }
        });
        const breakdown = selectedProfileIds.size > 0 ? ` (强${nShark}:中${nMid}:弱${nFish})` : '';
        const txt = `已选 ${selectedProfileIds.size} / ${cachedProfiles.length} 人${breakdown}`;
        if (lbl) {
          lbl.innerText = txt;
          if (selectedProfileIds.size === 0) {
            lbl.className = "text-[10px] px-2 py-0.5 rounded font-bold bg-red-950 border border-red-700 text-red-300";
          } else if (selectedProfileIds.size === cachedProfiles.length) {
            lbl.className = "text-[10px] px-2 py-0.5 rounded font-bold bg-emerald-950 border border-emerald-700 text-emerald-300";
          } else {
            lbl.className = "text-[10px] px-2 py-0.5 rounded font-bold bg-indigo-950 border border-indigo-700 text-indigo-300";
          }
        }
        if (modalLbl) {
          modalLbl.innerText = txt;
          modalLbl.className = selectedProfileIds.size === 0
            ? "text-xs font-bold px-3 py-1 rounded-full bg-red-950 text-red-300 border border-red-700 shadow-sm"
            : (selectedProfileIds.size === cachedProfiles.length
              ? "text-xs font-bold px-3 py-1 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-700 shadow-sm"
              : "text-xs font-bold px-3 py-1 rounded-full bg-indigo-950 text-indigo-300 border border-indigo-700 shadow-sm");
        }
      }
    }

    function renderRosterList() {
      const inlineContainer = document.getElementById("rosterList");
      const modalContainer = document.getElementById("rosterModalList");
      if (!cachedProfiles || cachedProfiles.length === 0) return;

      const searchInput = document.getElementById("inpRosterSearch");
      const query = searchInput ? searchInput.value.trim().toLowerCase() : "";

      const selModal = document.getElementById("selModalRosterSort");
      const selInline = document.getElementById("selRosterSort");
      const sortMode = (selModal && selModal.value) || (selInline ? selInline.value : "bb");

      const sorted = [...cachedProfiles].sort((a, b) => {
        if (sortMode === "hands") return b.hands - a.hands;
        if (sortMode === "vpip") return b.vpip - a.vpip;
        return b.bb_100 - a.bb_100;
      });

      const filtered = sorted.filter(p => {
        if (!query) return true;
        return (p.name && p.name.toLowerCase().includes(query)) ||
               (p.archetype && p.archetype.toLowerCase().includes(query)) ||
               (p.advice && p.advice.toLowerCase().includes(query));
      });

      let inlineHtml = "";
      let modalHtml = "";

      if (filtered.length === 0) {
        const emptyMsg = `<div class="text-center py-8 text-slate-500 text-xs col-span-full">未找到匹配 "${query}" 的选手</div>`;
        if (inlineContainer) inlineContainer.innerHTML = emptyMsg;
        if (modalContainer) modalContainer.innerHTML = emptyMsg;
        return;
      }

      filtered.forEach((p, idx) => {
        const isChecked = selectedProfileIds.has(p.id);
        const badgeHtml = getArchetypeBadge(p.archetype);
        const bbColor = p.bb_100 >= 0 ? "text-emerald-400" : "text-red-400";
        const bbSign = p.bb_100 >= 0 ? "+" : "";
        const rankNo = idx + 1;
        const rankClass = rankNo <= 3 ? "text-amber-400 font-bold" : (rankNo <= 10 ? "text-slate-300 font-medium" : "text-slate-500");

        // 1. Modal Spacious Card Grid View (完整大名字、不截断)
        modalHtml += `
          <div onclick="toggleProfileSelection('${p.id}')" class="p-3 rounded-xl border transition cursor-pointer flex flex-col justify-between space-y-2.5 ${isChecked ? 'bg-slate-900 border-amber-500/60 shadow-md shadow-amber-950/20 ring-1 ring-amber-500/30' : 'bg-slate-950/60 border-slate-800/80 opacity-55 hover:opacity-90'}">
            <div class="flex items-start justify-between gap-2">
              <div class="flex items-center space-x-2.5 min-w-0">
                <input type="checkbox" ${isChecked ? 'checked' : ''} onclick="event.stopPropagation(); toggleProfileSelection('${p.id}');" class="rounded bg-slate-950 border-slate-700 text-amber-500 cursor-pointer w-4 h-4 shrink-0">
                <span class="font-mono text-xs font-bold ${rankClass}">#${rankNo}</span>
                <span class="text-sm font-bold text-white tracking-wide break-words" title="${p.name} (${p.id})">${p.name}</span>
              </div>
              <div class="shrink-0">
                ${badgeHtml}
              </div>
            </div>
            <div class="flex items-center justify-between text-xs pt-1 border-t border-slate-800/60 font-mono">
              <span class="font-bold ${bbColor}" title="大盲百手收益">${bbSign}${p.bb_100} BB/100</span>
              <span class="text-slate-400" title="样本手数">${p.hands}手</span>
              <span class="text-slate-400" title="入池率">VPIP ${p.vpip}%</span>
              <span class="text-slate-400" title="激进度">AF ${p.af}</span>
            </div>
            <div class="text-[11px] text-amber-300/90 bg-slate-950 p-2 rounded-lg border border-slate-800/80 leading-relaxed flex items-start gap-1.5">
              <i class="fa-solid fa-lightbulb text-amber-400 mt-0.5 shrink-0 text-xs"></i>
              <span>${p.advice || '常规防守与价值下注'}</span>
            </div>
          </div>
        `;

        // 2. Inline Collapsible Drawer Item (两行式布局，名字完整不截断)
        inlineHtml += `
          <div onclick="toggleProfileSelection('${p.id}')" class="p-2 rounded-lg border transition cursor-pointer space-y-1 ${isChecked ? 'bg-slate-900 border-amber-500/40' : 'bg-slate-950/60 border-slate-800 opacity-50'}">
            <div class="flex items-center justify-between gap-1.5">
              <div class="flex items-center space-x-2 min-w-0">
                <input type="checkbox" ${isChecked ? 'checked' : ''} onclick="event.stopPropagation(); toggleProfileSelection('${p.id}');" class="rounded bg-slate-950 border-slate-700 text-amber-500 cursor-pointer shrink-0">
                <span class="font-mono text-[10px] ${rankClass}">#${rankNo}</span>
                <span class="font-bold text-white text-xs break-words" title="${p.name} (${p.id})">${p.name}</span>
              </div>
              <div class="shrink-0">
                ${badgeHtml}
              </div>
            </div>
            <div class="flex items-center justify-between text-[10px] text-slate-400 pl-6">
              <span class="font-bold ${bbColor}">${bbSign}${p.bb_100} BB</span>
              <span>${p.hands}手</span>
              <span>VPIP ${p.vpip}%</span>
              <span>AF ${p.af}</span>
              <span class="cursor-help text-slate-400 hover:text-amber-300" title="💡 针对建议: ${p.advice}"><i class="fa-solid fa-circle-info text-amber-400"></i></span>
            </div>
          </div>
        `;
      });

      if (inlineContainer) inlineContainer.innerHTML = inlineHtml;
      if (modalContainer) modalContainer.innerHTML = modalHtml;
    }

    function toggleProfileSelection(id) {
      if (selectedProfileIds.has(id)) {
        selectedProfileIds.delete(id);
      } else {
        selectedProfileIds.add(id);
      }
      updateRosterCount();
      renderRosterList();
      updateEstimates();
    }

    function selectProfiles(type) {
      const mode = document.getElementById("selOppMode").value;
      if (mode === "archetypes") {
        alert(`当前对手池为【原型模式】，基于纳什博弈经典原型自博弈，无需指定真人用户模型。\n如需指定参训选手画像，请先将对手池切换为【真人模式】或【混合模式】。`);
        return;
      }
      if (!cachedProfiles || cachedProfiles.length === 0) return;
      if (type === "all") {
        selectedProfileIds = new Set(cachedProfiles.map(p => p.id));
      } else if (type === "none") {
        selectedProfileIds.clear();
      } else if (type === "pyramid") {
        // 金字塔生态配平: 30% 顶级鲨鱼 + 40% 中游稳健 + 30% 提款机弱鱼 (选出 36 位代表性选手)
        const sorted = [...cachedProfiles].sort((a, b) => b.bb_100 - a.bb_100);
        const total = sorted.length;
        if (total > 0) {
          const targetTotal = Math.min(36, total);
          const nTop = Math.max(1, Math.round(targetTotal * 0.30));     // 约 11 位强手
          const nBottom = Math.max(1, Math.round(targetTotal * 0.30));  // 约 11 位弱鱼
          const nMid = targetTotal - nTop - nBottom;                    // 约 14 位中游
          
          const topSharks = sorted.slice(0, nTop);
          const bottomFish = sorted.slice(total - nBottom);
          const midStart = Math.max(nTop, Math.floor((total - nMid) / 2));
          const middleRegulars = sorted.slice(midStart, midStart + nMid);
          
          const balanced = [...topSharks, ...middleRegulars, ...bottomFish];
          selectedProfileIds = new Set(balanced.map(p => p.id));
        }
      } else if (type === "top20") {
        const top = [...cachedProfiles].sort((a, b) => b.bb_100 - a.bb_100).slice(0, 20);
        selectedProfileIds = new Set(top.map(p => p.id));
      } else if (type === "station") {
        const stations = cachedProfiles.filter(p => 
          p.archetype.includes("Station") || p.archetype.includes("跟注站") || 
          p.archetype.includes("Passive") || p.archetype.includes("被动鱼")
        );
        selectedProfileIds = new Set(stations.map(p => p.id));
      } else if (type === "maniac") {
        const maniacs = cachedProfiles.filter(p => 
          p.archetype.includes("Maniac") || p.archetype.includes("狂徒") || 
          p.archetype.includes("LAG") || p.archetype.includes("松凶")
        );
        selectedProfileIds = new Set(maniacs.map(p => p.id));
      }
      updateRosterCount();
      renderRosterList();
      updateEstimates();
    }

    async function startTrain() {
      const oppMode = document.getElementById("selOppMode").value;
      if ((oppMode === "pure_human" || oppMode === "mix") && selectedProfileIds.size === 0) {
        alert(`⚠️ 当前选择了【${oppMode === "pure_human" ? "真人模式" : "混合模式"}】，但您尚未指定任何参训真实用户模型！\n请在下方【指定真实用户模型】中挑选至少 1 位选手画像，或将对手池切换为【原型模式】。`);
        return;
      }

      const payload = {
        generations: document.getElementById("inpGens").value,
        population: document.getElementById("inpPop").value,
        runs: document.getElementById("inpRuns").value,
        agents: document.getElementById("inpAgents").value,
        workers: document.getElementById("inpWorkers").value,
        start_mode: document.getElementById("selStartMode").value,
        base_model: document.getElementById("selBaseModel").value,
        opp_mode: oppMode,
        selected_profiles: (oppMode === "pure_human" || oppMode === "mix") ? Array.from(selectedProfileIds) : [],
        min_hands: document.getElementById("inpMinHands").value,
        self_play: document.getElementById("chkSelfPlay").checked,
        save: document.getElementById("inpSavePath").value,
        archive: document.getElementById("inpArchiveDir").value,
        overwrite_champion: document.getElementById("chkOverwriteMain").checked
      };
      const res = await fetch("/api/train/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const d = await res.json();
      alert(d.message);
      fetchStatus();
    }

    async function stopTrain() {
      if (!confirm("确认终止当前正在运行的训练进程？")) return;
      const res = await fetch("/api/train/stop", { method: "POST" });
      const d = await res.json();
      alert(d.message);
      fetchStatus();
    }

    async function startLive() {
      const model = document.getElementById("selLiveModel").value;
      const res = await fetch("/api/live/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ strategy: model }) });
      const d = await res.json();
      alert(d.message);
      fetchStatus();
    }

    async function stopLive() {
      if (!confirm("确认停止比赛并离桌？")) return;
      const res = await fetch("/api/live/stop", { method: "POST" });
      const d = await res.json();
      alert(d.message);
      fetchStatus();
    }

    function updateEstimates() {
      const gens = parseInt(document.getElementById("inpGens").value) || 10;
      const pop = parseInt(document.getElementById("inpPop").value) || 8;
      const runs = parseInt(document.getElementById("inpRuns").value) || 40;
      const workers = Math.max(1, parseInt(document.getElementById("inpWorkers").value) || 2);
      const startMode = document.getElementById("selStartMode").value;
      const oppMode = document.getElementById("selOppMode").value;
      const selfPlay = document.getElementById("chkSelfPlay").checked;
      const overwrite = document.getElementById("chkOverwriteMain").checked;

      const genMatches = pop * runs;
      const totalMatches = gens * genMatches + 500;
      
      const secPerGen = Math.round((genMatches * 0.55) / workers + 15);
      const totalSec = secPerGen * gens + 90;

      let genTimeStr = "";
      if (secPerGen < 60) {
        genTimeStr = `~${secPerGen} 秒`;
      } else {
        const m = Math.floor(secPerGen / 60);
        const s = secPerGen % 60;
        genTimeStr = s > 0 ? `~${m}分${s}秒` : `~${m} 分钟`;
      }

      let totalTimeStr = "";
      if (totalSec < 60) {
        totalTimeStr = `~${totalSec} 秒`;
      } else if (totalSec < 3600) {
        totalTimeStr = `~${Math.round(totalSec / 60)} 分钟`;
      } else {
        const h = Math.floor(totalSec / 3600);
        const m = Math.round((totalSec % 3600) / 60);
        totalTimeStr = m > 0 ? `~${h} 小时 ${m} 分钟` : `~${h} 小时`;
      }

      let seLevel = "";
      let seColor = "";
      if (selfPlay) {
        seLevel = "🟣 2.5 影子自博弈 (攻防兼备·防反杀)";
        seColor = "text-purple-400";
      } else if (runs < 25) {
        seLevel = "粗略初筛 (标准误 ±0.08，方差略大)";
        seColor = "text-amber-400";
      } else if (runs <= 50) {
        seLevel = "稳健平衡 (标准误 ±0.05，推荐)";
        seColor = "text-emerald-400";
      } else {
        seLevel = "高精度极佳 (标准误 ±0.035，收敛极强)";
        seColor = "text-purple-400";
      }

      const ramMB = 25 + workers * 25;
      const ramPercent = ((ramMB / 2048) * 100).toFixed(1);

      const selCount = selectedProfileIds ? selectedProfileIds.size : 0;
      const totalCount = cachedProfiles ? cachedProfiles.length : 0;
      let oppText = "";
      if (oppMode === "pure_human") {
        oppText = `100% 纯真人特训 (${selCount}/${totalCount}人)`;
      } else if (oppMode === "mix") {
        oppText = `50% 混合实战池 (${selCount}人真实画像)`;
      } else {
        oppText = "0% 真人，纯原型自博弈";
      }

      let startText = "";
      if (startMode === "finetune") startText = "底模微调";
      else if (startMode === "resume") startText = "断点续训";
      else startText = "从零冷启动";

      const spText = selfPlay ? " · ⚡2.5影子守门员" : "";

      document.getElementById("estGenTime").innerText = genTimeStr;
      document.getElementById("estTotalTime").innerText = totalTimeStr;
      document.getElementById("estTotalMatches").innerText = totalMatches.toLocaleString() + " 场";
      
      const seEl = document.getElementById("estConfidence");
      seEl.innerText = seLevel;
      seEl.className = "font-semibold text-xs " + seColor;

      document.getElementById("estRam").innerText = `~${ramMB} MB (${ramPercent}%)`;
      document.getElementById("estStrategySummary").innerText = `${startText} · ${oppText}${spText} · ${overwrite ? "终局自动覆盖主模型" : "仅存新模型"}`;
    }

    async function switchTable() {
      const res = await fetch("/api/live/switch_table", { method: "POST" });
      const d = await res.json();
      alert(d.message);
    }

    let cachedModelsDetails = [];

    async function loadModelsDetails() {
      const grid = document.getElementById("modelsGrid");
      try {
        const res = await fetch("/api/models/details");
        const list = await res.json();
        cachedModelsDetails = list;
        renderModelsGrid(list);
      } catch (e) {
        if (grid) grid.innerHTML = `<div class="text-red-400 text-xs py-10 text-center col-span-full">加载模型档案失败: ${e}</div>`;
      }
    }

    function renderModelsGrid(models) {
      const grid = document.getElementById("modelsGrid");
      if (!grid) return;
      if (!models || models.length === 0) {
        grid.innerHTML = `<div class="text-center py-16 text-slate-500 text-xs col-span-full">暂无已保存的策略模型，可在【演化训练控制】中开始训练生成。</div>`;
        return;
      }

      let html = "";
      models.forEach(m => {
        const borderClass = m.is_main 
          ? "border-amber-500/50 shadow-amber-950/20 bg-gradient-to-b from-slate-900 via-slate-900 to-amber-950/10" 
          : "border-slate-800";
        
        const mainBadge = m.is_main
          ? `<span class="px-2 py-0.5 rounded text-[10px] font-black bg-amber-500/20 text-amber-300 border border-amber-500/40 flex items-center gap-1 shadow-sm"><i class="fa-solid fa-crown text-amber-400"></i> 当前主战模型 (Active)</span>`
          : "";

        const certBadge = m.is_certified
          ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-1 shadow-sm" title="通过 120 人全环境官方 Gate 4/5 认证"><i class="fa-solid fa-certificate text-emerald-400"></i> 官方认证通过</span>`
          : `<span class="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400 border border-slate-700 flex items-center gap-1" title="未经 120 人官方认证，受 Gate 4 保护"><i class="fa-solid fa-flask text-slate-500"></i> 未认证候选</span>`;

        const deleteBtn = m.is_main
          ? `<span class="text-[10px] text-slate-500 px-2 py-1 rounded bg-slate-950 border border-slate-800 flex items-center gap-1" title="主战模型受保护，不可删除"><i class="fa-solid fa-shield-halved text-amber-400/80"></i> 主模型保护中</span>`
          : `<button onclick="deleteModel('${m.path}', '${m.name}')" class="text-xs px-2.5 py-1 rounded bg-red-950/60 border border-red-800 hover:bg-red-900 text-red-300 transition flex items-center gap-1 shadow" title="彻底删除模型"><i class="fa-solid fa-trash-can"></i> 删除</button>`;

        const promoteBtn = m.is_main
          ? `<div class="text-xs font-semibold text-amber-400 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg bg-amber-950/40 border border-amber-800/60 flex-1"><i class="fa-solid fa-check"></i> 当前默认实战主模</div>`
          : (m.is_certified
            ? `<button onclick="promoteModel('${m.path}', '${m.name}')" class="flex-1 px-3 py-2 rounded-lg text-xs font-semibold bg-amber-600/90 hover:bg-amber-500 text-white flex items-center justify-center gap-1.5 transition shadow"><i class="fa-solid fa-crown"></i> 设为当前主战模型</button>`
            : `<button onclick="promoteModel('${m.path}', '${m.name}')" class="flex-1 px-3 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 flex items-center justify-center gap-1.5 transition shadow" title="未通过 Gate 4 认证的模型晋升将被安全拦截"><i class="fa-solid fa-shield-halved text-amber-400"></i> 设为主战 (需 Gate 4)</button>`
          );

        const strengthsBadges = m.strengths.map(s => 
          `<span class="text-[10px] px-2 py-0.5 rounded bg-slate-900/90 text-amber-300/90 border border-slate-800">${s}</span>`
        ).join("");

        const bbColor = m.avg_bb100 >= 0 ? "text-emerald-400" : "text-red-400";
        const bbSign = m.avg_bb100 >= 0 ? "+" : "";

        html += `
          <div class="card-dark rounded-xl p-5 shadow-lg border ${borderClass} flex flex-col justify-between space-y-4">
            <!-- 头部基本信息 -->
            <div>
              <div class="flex items-start justify-between gap-2 pb-3 border-b border-slate-800">
                <div>
                  <div class="flex items-center flex-wrap gap-2">
                    <span class="font-bold text-white text-base font-mono">${m.name}</span>
                    ${mainBadge}
                    ${certBadge}
                    <span class="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 border border-slate-700">Gen ${m.generation}</span>
                    <span class="px-1.5 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800">v${m.version}</span>
                  </div>
                  <div class="text-[10px] text-slate-400 mt-1 flex items-center gap-2">
                    <span><i class="fa-regular fa-hard-drive"></i> ${m.size_kb} KB</span>
                    <span>•</span>
                    <span><i class="fa-regular fa-clock"></i> ${m.mtime}</span>
                  </div>
                </div>
                <div class="shrink-0">
                  ${deleteBtn}
                </div>
              </div>

              <!-- 战术定位与特长 -->
              <div class="mt-3.5 bg-slate-950/80 p-3 rounded-lg border border-slate-800/80 space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-bold text-emerald-400">${m.archetype}</span>
                  <span class="text-[10px] text-slate-400">综合 Fitness: <span class="font-mono font-bold text-emerald-300">${m.fitness.toFixed(3)}</span></span>
                </div>
                <p class="text-[11px] text-slate-400 leading-relaxed">${m.arch_desc}</p>
                <div class="flex flex-wrap gap-1.5 pt-1">
                  ${strengthsBadges}
                </div>
                ${m.cert_summary ? `
                <div class="mt-2 text-[10px] text-emerald-300/90 bg-emerald-950/40 border border-emerald-800/50 p-2 rounded-lg flex items-start gap-1.5">
                  <i class="fa-solid fa-circle-check text-emerald-400 mt-0.5 shrink-0 text-xs"></i>
                  <span class="leading-relaxed font-mono">${m.cert_summary}</span>
                </div>` : ''}
              </div>

              <!-- 4项核心胜率与收益指标卡片 -->
              <div class="grid grid-cols-4 gap-2 mt-3.5 text-center">
                <div class="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                  <div class="text-[10px] text-slate-400">🏆 夺冠胜率</div>
                  <div class="text-sm font-black text-purple-400 mt-0.5">${m.champion_rate.toFixed(1)}%</div>
                </div>
                <div class="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                  <div class="text-[10px] text-slate-400">🎯 前12出线</div>
                  <div class="text-sm font-black text-amber-400 mt-0.5">${m.top12_rate.toFixed(1)}%</div>
                </div>
                <div class="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                  <div class="text-[10px] text-slate-400">🏅 决赛入围</div>
                  <div class="text-sm font-black text-indigo-400 mt-0.5">${m.final_rate.toFixed(1)}%</div>
                </div>
                <div class="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                  <div class="text-[10px] text-slate-400">📈 单手收益</div>
                  <div class="text-sm font-black ${bbColor} mt-0.5">${bbSign}${m.avg_bb100.toFixed(1)}</div>
                </div>
              </div>

              <!-- 五维能力评分条 -->
              <div class="mt-3.5 space-y-1.5 bg-slate-950/50 p-2.5 rounded-lg border border-slate-800/60 text-[11px]">
                <div class="flex items-center justify-between text-slate-400">
                  <span class="flex items-center gap-1.5"><i class="fa-solid fa-fire text-purple-400"></i> 锦标赛夺冠爆发力</span>
                  <div class="w-1/2 flex items-center gap-2">
                    <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div class="bg-purple-500 h-1.5 rounded-full" style="width: ${m.scores.burst}%"></div>
                    </div>
                    <span class="w-6 text-right font-mono text-[10px] text-slate-300">${m.scores.burst}</span>
                  </div>
                </div>

                <div class="flex items-center justify-between text-slate-400">
                  <span class="flex items-center gap-1.5"><i class="fa-solid fa-bullseye text-amber-400"></i> 预赛深赛程晋级率</span>
                  <div class="w-1/2 flex items-center gap-2">
                    <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div class="bg-amber-500 h-1.5 rounded-full" style="width: ${m.scores.deep}%"></div>
                    </div>
                    <span class="w-6 text-right font-mono text-[10px] text-slate-300">${m.scores.deep}</span>
                  </div>
                </div>

                <div class="flex items-center justify-between text-slate-400">
                  <span class="flex items-center gap-1.5"><i class="fa-solid fa-coins text-emerald-400"></i> 大盲筹码收割能力</span>
                  <div class="w-1/2 flex items-center gap-2">
                    <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div class="bg-emerald-500 h-1.5 rounded-full" style="width: ${m.scores.profit}%"></div>
                    </div>
                    <span class="w-6 text-right font-mono text-[10px] text-slate-300">${m.scores.profit}</span>
                  </div>
                </div>

                <div class="flex items-center justify-between text-slate-400">
                  <span class="flex items-center gap-1.5"><i class="fa-solid fa-shield text-blue-400"></i> 防守容错与抗反杀</span>
                  <div class="w-1/2 flex items-center gap-2">
                    <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div class="bg-blue-500 h-1.5 rounded-full" style="width: ${m.scores.defense}%"></div>
                    </div>
                    <span class="w-6 text-right font-mono text-[10px] text-slate-300">${m.scores.defense}</span>
                  </div>
                </div>

                <div class="flex items-center justify-between text-slate-400">
                  <span class="flex items-center gap-1.5"><i class="fa-solid fa-bolt text-red-400"></i> 进攻压迫与下注施压</span>
                  <div class="w-1/2 flex items-center gap-2">
                    <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div class="bg-red-500 h-1.5 rounded-full" style="width: ${m.scores.pressure}%"></div>
                    </div>
                    <span class="w-6 text-right font-mono text-[10px] text-slate-300">${m.scores.pressure}</span>
                  </div>
                </div>
              </div>

              <!-- 核心参数速览 -->
              <div class="mt-3 p-2.5 bg-slate-900/60 rounded-lg border border-slate-800 text-[10px] text-slate-400 space-y-1">
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-1.5 font-mono">
                  <div>VPIP: <span class="text-white">${m.params.vpip}%</span></div>
                  <div>Open: <span class="text-white">${m.params.open}%</span></div>
                  <div>3-Bet: <span class="text-white">${m.params.threebet}%</span></div>
                  <div>C-Bet: <span class="text-white">${m.params.cbet}%</span></div>
                </div>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-1.5 font-mono pt-1 border-t border-slate-800/60">
                  <div>防守 Safety: <span class="text-white">${m.params.safety}%</span></div>
                  <div>进攻 Attack: <span class="text-white">${m.params.attack}%</span></div>
                  <div>门槛: <span class="text-white">${m.params.flop_val}/${m.params.turn_val}/${m.params.river_val}</span></div>
                  <div>注码: <span class="text-white">干${m.params.dry_size}%/湿${m.params.wet_size}%</span></div>
                </div>
              </div>
            </div>

            <!-- 卡片底部操作按钮 -->
            <div class="pt-2 border-t border-slate-800/80 flex items-center gap-2">
              ${promoteBtn}
              <button onclick="useModelForLive('${m.path}', '${m.name}')" class="px-3 py-2 rounded-lg text-xs font-semibold bg-blue-700/80 hover:bg-blue-600 text-white flex items-center justify-center gap-1.5 transition shadow" title="在比赛对战中使用该模型"><i class="fa-solid fa-play"></i> 选定实战</button>
              <button onclick="viewJsonModal('${m.path}', '${m.name}')" class="px-3 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center gap-1.5 transition border border-slate-700" title="查看完整 JSON 参数"><i class="fa-solid fa-code"></i> JSON</button>
            </div>
          </div>
        `;
      });
      grid.innerHTML = html;
    }

    async function deleteModel(path, name) {
      if (!confirm(`⚠️ 危险操作确认：\n确认彻底删除策略模型【${name}】？\n此操作不可撤销！`)) {
        return;
      }
      try {
        const res = await fetch("/api/models/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: path })
        });
        const d = await res.json();
        alert(d.message);
        loadModelsDetails();
        loadModels();
      } catch (e) {
        alert("删除请求失败: " + e);
      }
    }

    async function promoteModel(path, name) {
      if (!confirm(`🏆 确认将【${name}】晋升为当前主战模型 (champion.json)？\n旧主战模型将自动备份为 champion.json.bak。`)) {
        return;
      }
      try {
        const res = await fetch("/api/models/promote", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: path })
        });
        const d = await res.json();
        alert(d.message);
        loadModelsDetails();
        loadModels();
      } catch (e) {
        alert("设置主战模型失败: " + e);
      }
    }

    function useModelForLive(path, name) {
      const sel = document.getElementById("selLiveModel");
      if (sel) {
        sel.value = path;
      }
      switchTab("tabLive");
      alert(`已为您切换至【在线赛事实战】面板，并选定策略模型:【${name}】！\n点击“启动比赛对战”即可开始实战。`);
    }

    async function viewJsonModal(path, name) {
      const modal = document.getElementById("jsonModal");
      const title = document.getElementById("jsonModalTitle");
      const content = document.getElementById("jsonModalContent");
      title.innerHTML = `<i class="fa-solid fa-code text-amber-400"></i> 模型 JSON 明细: <span class="font-mono text-emerald-400 ml-1">${name}</span>`;
      content.innerText = "正在读取文件内容...";
      modal.classList.remove("hidden");

      try {
        const res = await fetch(`/api/models/content?model=${encodeURIComponent(path)}`);
        const d = await res.json();
        if (d.status === "success") {
          content.innerText = d.content;
        } else {
          content.innerText = "读取失败: " + d.message;
        }
      } catch (e) {
        content.innerText = "请求出错: " + e;
      }
    }

    function closeJsonModal() {
      document.getElementById("jsonModal").classList.add("hidden");
    }

    function copyJsonModalContent() {
      const content = document.getElementById("jsonModalContent").innerText;
      navigator.clipboard.writeText(content).then(() => {
        const btn = document.getElementById("btnCopyJson");
        const orig = btn.innerHTML;
        btn.innerHTML = `<i class="fa-solid fa-check text-emerald-400"></i> 已复制！`;
        setTimeout(() => { btn.innerHTML = orig; }, 2000);
      }).catch(e => {
        alert("复制失败，请手动选取复制");
      });
    }

    window.onload = () => {
      loadModels();
      loadProfilesList();
      loadModelsDetails();
      onStartModeChange();
      onOppModeChange();
      fetchStatus();
      loadHands();
      refreshLogs();
      updateEstimates();
      setInterval(fetchStatus, 3000);
      setInterval(refreshLogs, 4000);
    };
  </script>

  <!-- Modal for full-view roster selection -->
  <div id="rosterModal" class="hidden fixed inset-0 bg-black/85 backdrop-blur-md z-50 flex items-center justify-center p-3 sm:p-6">
    <div class="bg-slate-900 border border-slate-700 rounded-2xl max-w-5xl w-full max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
      <!-- Modal Header -->
      <div class="p-4 sm:p-5 border-b border-slate-800 bg-slate-950/80 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 class="text-base font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-users-viewfinder text-amber-400 text-lg"></i>
            <span>赛场真实选手全景花名册与能力排行榜</span>
          </h3>
          <p class="text-xs text-slate-400 mt-0.5">每位真实选手的战术风格、盈利战力与剥削建议均已深度标注，自由勾选参训对手以打造定制化陪练池。</p>
        </div>
        <div class="flex items-center gap-2">
          <span id="lblModalSelectedCount" class="text-xs font-bold px-3 py-1 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-700 shadow-sm">
            已勾选: 63 / 63 人
          </span>
          <button onclick="closeRosterModal()" class="text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-slate-800 transition">
            <i class="fa-solid fa-xmark text-lg"></i>
          </button>
        </div>
      </div>

      <!-- Controls & Search Toolbar -->
      <div class="p-3 sm:px-5 py-3 border-b border-slate-800/80 bg-slate-900 flex flex-wrap items-center justify-between gap-2.5 text-xs">
        <div class="flex flex-wrap items-center gap-1.5">
          <button type="button" onclick="selectProfiles('pyramid')" class="px-2.5 py-1 rounded bg-indigo-950/90 border border-indigo-500/80 hover:bg-indigo-900 text-indigo-200 transition font-bold shadow-sm" title="自动挑选 11 位顶级强手 + 14 位中游稳健 + 11 位提款机弱鱼"><i class="fa-solid fa-layer-group text-[10px] mr-1"></i>金字塔生态 (强:中:弱 = 11:14:11)</button>
          <button type="button" onclick="selectProfiles('all')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 transition font-medium">全选(63人)</button>
          <button type="button" onclick="selectProfiles('top20')" class="px-2.5 py-1 rounded bg-emerald-950/80 border border-emerald-700 hover:bg-emerald-900 text-emerald-300 transition font-medium">前20强劲敌</button>
          <button type="button" onclick="selectProfiles('station')" class="px-2.5 py-1 rounded bg-amber-950/80 border border-amber-700 hover:bg-amber-900 text-amber-300 transition font-medium">跟注站/被动鱼</button>
          <button type="button" onclick="selectProfiles('maniac')" class="px-2.5 py-1 rounded bg-red-950/80 border border-red-700 hover:bg-red-900 text-red-300 transition font-medium">狂徒/松凶高手</button>
          <button type="button" onclick="selectProfiles('none')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-400 transition font-medium">清空</button>
        </div>

        <div class="flex items-center gap-2 flex-1 sm:flex-initial justify-end">
          <div class="relative min-w-[180px]">
            <input type="text" id="inpRosterSearch" oninput="renderRosterList()" placeholder="🔍 快速搜索选手姓名..." class="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-amber-500">
          </div>
          <select id="selModalRosterSort" onchange="syncAndRenderRoster(this.value)" class="bg-slate-950 border border-slate-700 rounded-lg px-2 py-1 text-slate-300 text-xs">
            <option value="bb">按 战力(BB/100) 降序</option>
            <option value="hands">按 对战手数 降序</option>
            <option value="vpip">按 入池激进度 降序</option>
          </select>
        </div>
      </div>

      <!-- Modal Card Grid Container -->
      <div class="p-4 sm:p-5 overflow-y-auto flex-1 bg-slate-950/40">
        <div id="rosterModalList" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          <!-- Dynamic cards rendered by renderRosterList -->
        </div>
      </div>

      <!-- Modal Footer -->
      <div class="p-3 sm:px-5 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between">
        <div class="text-xs text-slate-400">
          💡 点击整张选手卡片即可快速勾选 / 取消，所选对手将即时同步至训练对抗池。
        </div>
        <button onclick="closeRosterModal()" class="px-5 py-1.5 bg-amber-600 hover:bg-amber-500 text-white font-semibold text-xs rounded-lg transition shadow">
          确定并保存勾选
        </button>
      </div>
    </div>
  </div>

  <!-- Modal for viewing model JSON -->
  <div id="jsonModal" class="hidden fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-700 rounded-2xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl">
      <div class="flex items-center justify-between p-4 border-b border-slate-800">
        <h3 id="jsonModalTitle" class="text-sm font-bold text-white flex items-center gap-2">
          <i class="fa-solid fa-code text-amber-400"></i> 模型 JSON 参数明细
        </h3>
        <button onclick="closeJsonModal()" class="text-slate-400 hover:text-white p-1">
          <i class="fa-solid fa-xmark text-base"></i>
        </button>
      </div>
      <div class="p-4 overflow-y-auto flex-1 font-mono text-xs text-emerald-400 bg-slate-950 rounded-lg mx-4 my-2 border border-slate-800">
        <pre id="jsonModalContent" class="whitespace-pre-wrap break-all"></pre>
      </div>
      <div class="p-3 border-t border-slate-800 flex justify-end gap-2 px-4">
        <button onclick="copyJsonModalContent()" id="btnCopyJson" class="text-xs px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition flex items-center gap-1.5">
          <i class="fa-regular fa-copy"></i> 复制完整 JSON
        </button>
        <button onclick="closeJsonModal()" class="text-xs px-3.5 py-1.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition">关闭</button>
      </div>
    </div>
  </div>
</body>
</html>"""

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress access logs

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            b = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return

        if path == "/api/status":
            data = {
                "training": {"running": pm.is_training(), "pid": pm.train_proc.pid if pm.train_proc else None},
                "live": {"running": pm.is_live(), "pid": pm.live_proc.pid if pm.live_proc else None},
                "archive": get_archive_data()
            }
            self._send_json(data)
            return

        if path == "/api/models":
            self._send_json(get_models_list())
            return

        if path == "/api/models/details":
            self._send_json(get_models_details())
            return

        if path == "/api/models/content":
            query = urllib.parse.parse_qs(parsed.query)
            m_path = query.get("model", [""])[0]
            self._send_json(get_model_content(m_path))
            return

        if path == "/api/profiles":
            self._send_json(get_profiles_data())
            return

        if path == "/api/hands":
            self._send_json(get_recent_hands(limit=30))
            return

        if path == "/api/train/logs":
            self._send_text(pm.get_logs("train"))
            return

        if path == "/api/live/logs":
            self._send_text(pm.get_logs("live"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            params = json.loads(body) if body else {}
        except Exception:
            params = {}

        if path == "/api/train/start":
            self._send_json(pm.start_training(params))
            return
        if path == "/api/train/stop":
            self._send_json(pm.stop_training())
            return
        if path == "/api/live/start":
            strategy = params.get("strategy", "models/champion.json")
            self._send_json(pm.start_live(strategy))
            return
        if path == "/api/live/stop":
            self._send_json(pm.stop_live())
            return
        if path == "/api/live/switch_table":
            self._send_json(pm.switch_table())
            return
        if path == "/api/models/delete":
            model = params.get("model", "")
            self._send_json(delete_model(model))
            return
        if path == "/api/models/promote":
            model = params.get("model", "")
            self._send_json(promote_model(model))
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, obj: Any):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def _send_text(self, text: str):
        b = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

def run_server(host: str = "0.0.0.0", port: int = 8080):
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"\n" + "="*68)
    print(f" 🃏 AgentPoker 2.0 全能可视化控制台已启动！")
    print(f" • 本地访问: http://127.0.0.1:{port}")
    print(f" • 局域网/公网: http://{host}:{port}")
    print(f" • 功能: 训练启停、比赛启停、一键换桌、折线走势图、实时对局复盘")
    print(f"="*68 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] 正在停止控制台...")
        server.server_close()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    run_server(port=port)
