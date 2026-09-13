# ApexPoker Project Map & Codebase Blueprint

## 1. 原项目结构与目录清单 (Repository Structure)

```text
deepcfr-texas-no-limit-holdem-6-players/
├── configs/                     # 配置目录（原项目为空 __init__.py）
├── docs/                        # 项目工程与架构文档
├── models/                      # 训练检查点保存目录
│   └── phase0_test/             # Phase 0 冒烟测试生成的测试模型
├── notebooks/                   # Jupyter 实验分析脚本
├── scripts/                     # CLI 运行、对战、评估与 GUI 工具
│   ├── evaluate_models.py       # 跨模型、随机对手评估基准脚本
│   ├── play.py                  # 人机终端对战入口
│   ├── poker_gui.py             # PyQt5 可视化对战界面
│   ├── run_regression_suite.py  # 回归测试汇总启动器
│   ├── telegram_notifier.py     # 训练消息推送工具
│   └── visualize_tournament.py  # 历史多模型锦标赛对决可视化
├── src/                         # 扑克算法核心实现
│   ├── agents/                  # 基础 Agent 实现
│   │   └── random_agent.py      # 随机决策智能体 (用于基线与环境填充)
│   ├── core/                    # Deep CFR 算法核心与网络结构
│   │   ├── deep_cfr.py          # CFR 树遍历、优先经验回放与网络更新
│   │   └── model.py             # PokerNetwork (双头网络) 与扑克状态编码
│   ├── opponent_modeling/       # 对手建模扩展
│   │   ├── deep_cfr_with_opponent_modeling.py # 集成 GRU 对手建模的 CFR Agent
│   │   └── opponent_model.py    # GRU 历史编码器与对手模型
│   ├── training/                # 训练管线
│   │   ├── train.py             # 标准 Deep CFR 训练主入口 (Phase 1/2/3)
│   │   ├── train_opponent_modeling.py # 对手建模训练主入口
│   │   ├── train_mixed_with_opponent_modeling.py # 混合对手建模内部实现
│   │   └── train_with_opponent_modeling.py       # 固定对手建模内部实现
│   └── utils/                   # 辅助工具模块
│       ├── actions.py           # 动作空间离散化、连续加注边界与清洗
│       ├── agents.py            # CheckpointAgent 封装与加载
│       ├── checkpoints.py       # 模型保存、发现与元数据管理
│       ├── evaluation.py        # 评测基准引擎、收益统计与动作分布统计
│       ├── logging.py           # 动作执行安全封装与日志
│       ├── settings.py          # 全局配置 (strict checking 等)
│       ├── training_diagnostics.py  # 梯度与训练诊断监控
│       └── traversal_diagnostics.py # CFR 树遍历故障排查
└── tests/                       # 自动化单元测试与回归测试集 (50 项测试)
```

---

## 2. 关键模块职责划分

| 模块 | 核心职责 | 保留/扩展/禁止修改 |
|---|---|---|
| `pokers` (Rust C-Ext) | 底层扑克牌局引擎，包含洗牌、发牌、彩池分配与合法动作校验 | **保留，禁止直接修改底层 C/Rust 库源码** |
| `src/core/model.py` | 状态特征向量化 (`encode_state`) 与 3 动作+连续加注尺度网络 (`PokerNetwork`) | **保留基础网络，通过外部 Wrapper / Adapter 扩展融合层** |
| `src/core/deep_cfr.py` | Advantage Net / Strategy Net 训练逻辑、优先经验回放 (`PrioritizedMemory`)、线性 CFR 权重 | **保留核心算法，禁止直接入侵修改 CFR 树遍历底层** |
| `src/utils/actions.py` | 动作清洗 (`sanitize_action`)、合法动作范围计算 (`raise_bounds`) | **保留，确保所有与环境交互的动作均经过清洗** |
| `src/utils/evaluation.py` | 单桌 6-Max 环境实例化 (`pkrs.State.from_seed`) 与对局推进循环 | **保留单桌评测基础，扩展为赛事级评测系统** |
| `src/tournament/` (新增) | 120 人多阶段赛事状态机、瑞士轮配对、蛇形分桌、自动 Rebuy 管理、排名引擎 | **新增核心层 (Phase 1)** |
| `src/observation/` (新增) | 扑克特征与赛事特征 (Stage, Hand, Cutoff, Margin, Rank) 的特征融合编码器 | **新增观察层 (Phase 3)** |
| `src/training/tournament_reward.py` (新增) | 综合 Net BB、晋级概率、决赛冠军概率的多目标奖励接口 | **新增奖励层 (Phase 5)** |
| `src/league/` (新增) | 联赛对弈框架 (Main, Exploiters, Frozen Pool, Rule Agents) | **新增自博弈框架 (Phase 8)** |

---

## 3. 当前模型输入输出规范

### 3.1 状态编码输入 (`encode_state`, 维度分析)
输入为单桌 6-Max NLHE 局内状态：
1. **玩家手牌 (Hole Cards)**: 52 维 One-hot (玩家自己拥有的两张私有牌)
2. **公共牌 (Community Cards)**: 52 维 One-hot (当前公共翻牌、转牌、河牌)
3. **游戏阶段 (Stage)**: 5 维 One-hot (Preflop, Flop, Turn, River, Showdown)
4. **当前底池 (Pot)**: 1 维 (除以初始筹码 stake 归一化)
5. **庄家位置 (Button)**: 6 维 One-hot (庄家座位号 0-5)
6. **行动玩家 (Current Player)**: 6 维 One-hot
7. **玩家状态 (Player States)**: 6 玩家 × 4 维 = 24 维
   - `active`: 存活状态 (1.0 / 0.0)
   - `bet_chips`: 当前街已下注额 (归一化)
   - `pot_chips`: 已投入底池总额 (归一化)
   - `stake`: 剩余筹码 (归一化)
8. **最小下注额 (Min Bet)**: 1 维 (归一化)
9. **合法动作集 (Legal Actions)**: 4 维 (Fold, Check, Call, Raise)
10. **前序动作 (Previous Action)**: 5 维 (4 动作类型 One-hot + 1 动作下注额)
- **总基准输入维度**: $52 + 52 + 5 + 1 + 6 + 6 + 24 + 1 + 4 + 5 = 156$ 维 (通过默认 `input_size=500` 支持后续扩展)。

### 3.2 模型结构与输出
- **主干网络 (Base)**: 3 层 MLP ($Linear \to ReLU$), 隐藏层大小 256。
- **Action Head**: 输出 3 维 logits (`Fold`, `Check/Call`, `Raise`)。
- **Sizing Head**: 输出连续加注倍率标量，映射到 $[0.1, 3.0]$ 倍当前底池。

---

## 4. 当前训练与评估流程

### 4.1 训练流程 (Deep CFR)
1. **Traversals**: 每轮迭代针对每个目标玩家进行环境遍历，随机发牌并递归展开博弈树。
2. **Advantage Memory**: 收集后悔值经验 $(s, a, R(s,a))$ 存入带优先级的经验池 (`PrioritizedMemory`)。
3. **Strategy Memory**: 记录目标玩家的正优势策略采样存入策略回放池 (`ReservoirMemory`)。
4. **Network Updates**:
   - `AdvantageNet`: 采样加权 MSE 损失更新各动作优势值。
   - `StrategyNet`: 采样 Cross-Entropy 更新平均策略，收敛于纳什均衡。

### 4.2 Checkpoint 格式与元数据
使用 `torch.save` 序列化为 `.pt` 文件，标准字典结构包含：
- `schema_version`: 1
- `agent_type`: `"standard"` 或 `"opponent_modeling"`
- `num_players`: 6
- `player_id`: 0
- `iteration`: 迭代步数
- `advantage_net`: 网络 state_dict
- `strategy_net`: 网络 state_dict
- `min_bet_size`: 0.1
- `max_bet_size`: 3.0
- `metadata`: 运行时附加信息

### 4.3 当前测试机制
- 采用 `pytest` 执行 `tests/` 目录下的 10 个测试套件，共 50 个单元测试和回归测试。
- 覆盖动作清洗、检查点序列化与版本推断、CLI 参数解析、状态异常安全恢复、`pokers` 引擎边界情况、GRU 对手建模特征等。
