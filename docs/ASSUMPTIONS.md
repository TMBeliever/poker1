# ApexPoker Architectural Assumptions & Tournament Invariants

## 1. 赛事阶段与轮次结构 (Stage & Round Structure)

1. **预赛结构 (Preliminary)**:
   - 选手总数: 120 名。
   - 参赛形式: 20 张 6-Max 桌。
   - 总手数: 200 手。
   - 轮次划分: 共 10 轮 (Round 1 至 Round 10)，每轮每桌进行 20 手牌。
   - 分桌策略:
     - **R1 - R3**: 完全随机分桌 (Random Seeding)。
     - **R4 - R10**: 依据累计净收益 (Cumulative Net BB / BB100) 进行**瑞士轮配对 (Swiss Pairing)**，成绩相近的选手分在同一桌。
   - 晋级人数: 预赛结束后排名前 12 名 (Top 12) 晋级半决赛。

2. **半决赛结构 (Semifinal)**:
   - 选手总数: 12 名。
   - 筹码重置: 独立重新获得 100 BB 筹码。
   - 蛇形分桌 (Snake Seeding):
     - Table A: Rank 1, 4, 5, 8, 9, 12
     - Table B: Rank 2, 3, 6, 7, 10, 11
   - 总手数: 20 手。
   - 晋级人数: 每桌前 3 名 (Top 3 each) 共 6 名选手晋级决赛。

3. **决赛结构 (Final)**:
   - 选手总数: 6 名 (单桌 6-Max)。
   - 筹码重置: 独立重新获得 100 BB 筹码。
   - 总手数: 30 手。
   - 决胜标准: 决赛 30 手累计净收益第 1 名夺得冠军。

---

## 2. 筹码、Rebuy 与净收益核算规则 (Rebuy & Net BB Accounting)

1. **初始筹码与盲注基准**:
   - 默认 Big Blind (BB) = 2.0，Small Blind (SB) = 1.0。
   - 初始筹码量 = 100 BB (即 200.0 筹码)。
2. **Auto-Rebuy 机制**:
   - 选手在任何一手牌局中发生破产 (筹码 $\le 0$ 或不足 1 BB) 时，不会被永久淘汰，触发 Auto-Rebuy，筹码量重置为 100 BB。
   - Rebuy 次数 `rebuy_count` 加 1。
3. **累计净收益 (Cumulative Net BB) 公式**:
   - $\text{Net BB} = (\text{Current Stack in BB} - 100.0) - (\text{Rebuy Count} \times 100.0)$
   - 该核算方法确保全场总 Net BB 严格守恒为 0（零和博弈公理）。
4. **阶段筹码重置与收益统计**:
   - 半决赛和决赛开始时，选手当前桌内筹码恢复为 100 BB，且半决赛/决赛内部的排名依据该阶段内的 Net BB 独立计算（预赛成绩作为晋级门槛，不带入半决赛计算）。

---

## 3. 平局决胜规则 (Tie-break Rule)

当多名选手的阶段净收益相同时，按以下优先级确定位次：
1. **累计净 BB** (Cumulative Net BB) 更高者优先。
2. **Rebuy 次数** (Rebuy Count) 更少者优先（代表更稳健的筹码控制能力）。
3. **完成手数率** (Completion Rate) 更高者优先。
4. **确定性选手 ID** (Player ID) 较小者优先（保证确定性与可复现性）。

---

## 4. 观察与决策解耦原则

- 赛事引擎 (`TournamentEngine`, `TournamentEnv`, `RankingEngine`, `SwissPairing`, `RebuyManager`, `AdvancementManager`) 为纯确定性规则状态机，不依赖任何神经网络参数。
- 智能体决策仅能根据自身在当前时刻合法接收到的 `TournamentState` 与 `PokerState` 特征进行推理，禁止前瞻或读取其他选手手牌信息。
