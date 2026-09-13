# ApexPoker: 120-Player Multi-Stage Tournament AI System

ApexPoker is a reinforcement learning and game-theoretic AI platform that scales 6-player No-Limit Texas Hold'em (NLHE) Deep Counterfactual Regret Minimization (Deep CFR) to an elite **120-player, multi-stage competitive tournament system**.

Built on a decoupled, mathematically verified architecture, ApexPoker preserves underlying poker engine invariants while introducing macro-tournament state fusion, multi-objective survival rewards, dynamic opponent modeling, and a competitive archetype league.

---

## Tournament Format & Structure

```
120 Players across 20 Tables (6-Max)
  │
  ▼ [Preliminary Stage - 200 Hands]
  ├─ Rounds 1–3: Random Seeding (60 hands)
  └─ Rounds 4–10: Swiss Pairing by Net BB (140 hands)
  │
  ▼ [Top 12 Qualify]
  │  Snake Seeding into Tables A & B (100 BB Stack Reset)
  ▼ [Semifinal Stage - 20 Hands]
  ├─ Table A: 6 players, 20 hands
  └─ Table B: 6 players, 20 hands
  │
  ▼ [Top 3 from Table A & Top 3 from Table B Qualify]
  │  100 BB Stack Reset
  ▼ [Final Table - 30 Hands]
  └─ 6 Finalists, 30 hands
  │
  ▼
🏆 Champion: Rank 1 by Final Stage Net BB
```

### Key Tournament Rules & Invariants
1. **Zero-Sum Accounting Invariant:** Chips and net BB sum to zero across all participants at all times: $\sum \text{net\_bb} \equiv 0.0$.
2. **Auto-Rebuy:** Bankruptcy ($stack \le 0$) does not eliminate a player. The system automatically rebuys 100 BB at a cost of 100 BB to net balance.
3. **Stage Isolation:** Each qualifying stage resets active stacks to 100 BB to eliminate chip runaway. Qualification rankings depend strictly on performance in that stage.
4. **Deterministic Tie-Breaking:** In order: (1) higher Net BB, (2) fewer rebuys, (3) higher current stack, (4) player ID.

---

## Quickstart Guide

### 1. Installation

```bash
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git
cd deepcfr-texas-no-limit-holdem-6-players

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Single Table Poker Simulation

Simulate isolated 6-Max NLHE hands using random or neural agents:

```bash
# Run 100 hands on a single table
python scripts/run_poker.py --hands 100 --seed 42

# Run with checkpoint
python scripts/run_poker.py --hands 50 --checkpoint models/base/base_model.pt
```

### 3. Run Full 120-Player Tournament

Simulate the complete multi-stage tournament (Preliminary $\to$ Semifinals $\to$ Final Table):

```bash
# Run 120-player tournament with full logging
python scripts/run_tournament.py --players 120 --seed 42

# Save tournament summary to JSON
python scripts/run_tournament.py --players 120 --output tournament_summary.json
```

### 4. Train Base Deep CFR Agent

Train a cash-game Deep CFR baseline model on isolated 6-Max NLHE:

```bash
python scripts/train_base.py \
  --iterations 1000 \
  --traversals 200 \
  --save-dir models/base \
  --log-dir logs/base
```

### 5. Curriculum Training for Tournament Deep CFR

Train a tournament-aware agent through progressive curriculum stages (Cash EV $\to$ Mini-Tournament $\to$ Full Tournament):

```bash
python scripts/train_tournament.py \
  --stage 1 \
  --iterations 500 \
  --save-dir models/tournament \
  --log-dir logs/tournament
```

### 6. Train Dynamic Opponent Model

Train empirical tracking and representation learning for opponent styles:

```bash
python scripts/train_opponent_model.py \
  --hands 5000 \
  --save-path models/opponent/opponent_encoder.pt
```

### 7. Run Archetype League

Execute a 120-player tournament featuring 8 diverse player archetypes (Nit, TAG, LAG, Station, Maniac, Overfolder, Overbluffer, Adaptive Reg) alongside trained CFR models:

```bash
python scripts/run_league.py \
  --config configs/league.yaml \
  --seed 42 \
  --output league_results.json
```

### 8. Evaluate Checkpoints & Run Benchmarks

Evaluate model performance or run the system-wide throughput benchmark suite:

```bash
# Evaluate checkpoint against baseline pool
python scripts/evaluate_checkpoint.py \
  --checkpoint models/base/checkpoint_iter_2.pt \
  --hands 200

# Run unified performance benchmark (hands/s, steps/s, tournaments/s, inference latency)
python scripts/run_benchmark.py --output benchmark_report.json
```

### 9. Join Real Online Competition (Sohu Agent Poker)

Directly connect trained Deep CFR neural models to the official Sohu Agent Poker platform adhering strictly to [`skill.md`](https://poker.bang.sohu.com/skill.md) using credentials from `.env`:

```bash
# 1. Check environment configuration & platform connectivity
python scripts/run_competition.py status

# 2. Discover active competitions on the platform
python scripts/run_competition.py discover

# 3. Enter competition and run AI watchman loop with trained model (uses .env key & comp ID by default)
python scripts/run_competition.py join

# Or specify custom competition ID and model path:
python scripts/run_competition.py join \
  --competition-id <COMPETITION_ID> \
  --model models/base/base_checkpoint_iter_2000.pt

# 4. Check real-time leaderboard / standings
python scripts/run_competition.py standings
```

**Key Features of the Competition Module (`src/competition/`):**
- **Protocol & Error Handling (`protocol.py`):** Strictly adheres to API contract, status handling (400, 401, 404, 409 `stale_table` / `registration_required`, 502/503 `temporarily_unavailable`), and exponential backoff respecting `Retry-After`.
- **Observation Translation & Clamping (`cfr_bridge.py`):** Encodes live JSON table states into 156-dim input tensors for our PyTorch Deep CFR networks, automatically clamping decisions to allowed ranges (`[minAmount, maxAmount]`).
- **Autonomous Watchman Loop (`live_watchman.py`):** Implements `idle` (30s polling), `queued` (5s polling), `seated` (2s polling), and `actionRequest` handling with exact HTTP body preservation on network retries.
- **Graceful Leave Play:** Handles SIGINT (Ctrl+C) and hand completions by calling `POST /api/competitions/leave` to cleanly exit tables.
- **SOUL.md & Table Chat (`soul.py`):** Generates compliant, character-bounded table chats ($\le 140$ characters) adhering to persona specifications.


---

## System Architecture

ApexPoker separates the deterministic tournament rules from poker-level neural decision-making:

- **Tournament Core (`src/tournament/`):** Implements state machines ([`state.py`](src/tournament/state.py)), auto-rebuy logic ([`rebuy.py`](src/tournament/rebuy.py)), Swiss pairings ([`swiss.py`](src/tournament/swiss.py)), rankings ([`ranking.py`](src/tournament/ranking.py)), and multi-table environments ([`tournament_env.py`](src/tournament/tournament_env.py)).
- **Observation & State Fusion (`src/observation/`):** Combines 156-dim intra-hand poker states with 13-dim macro tournament context into a 220-dim fused observation tensor via [`fusion.py`](src/observation/fusion.py).
- **Tournament Deep CFR (`src/training/` & `src/tournament/`):** Incorporates multi-objective rewards ([`tournament_reward.py`](src/training/tournament_reward.py)) balancing cash EV, survival incentives, ICM rank bonuses, and rebuy penalties.
- **Counterfactual Replay & Transformer V2 (`src/training/` & `src/core/`):** High-leverage replay buffers ([`counterfactual_replay.py`](src/training/counterfactual_replay.py)) and attention-based policy networks ([`transformer_v2.py`](src/core/transformer_v2.py)).
- **Archetypes & League (`src/agents/` & `src/league/`):** Behavioral bots and 120-seat league manager ([`league.py`](src/league/league.py)).

Detailed architectural diagrams and mathematical formulations are available in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Configuration Files

Centralized YAML configurations are located in `configs/`:
- [`configs/poker.yaml`](configs/poker.yaml): Blinds, stack sizes, betting limits, and table parameters.
- [`configs/tournament.yaml`](configs/tournament.yaml): 120 players, 20 tables, stage hand counts, Swiss rounds, and advancement cutoffs.
- [`configs/training.yaml`](configs/training.yaml): Learning rates, buffer sizes, traversals, and batch sizes.
- [`configs/league.yaml`](configs/league.yaml): Player distribution across the 8 canonical archetypes.
- [`configs/evaluation.yaml`](configs/evaluation.yaml): Evaluation games, seeds, metrics, and report outputs.

---

## Verification & Test Suite

All system modules are thoroughly tested with 107 automated unit, integration, and rule validation tests:

```bash
# Run all tests
pytest tests/

# Run rule validation suite
pytest tests/tournament/test_rule_validation.py

# Run benchmark verification
pytest tests/benchmark/test_benchmark.py
```

Test coverage includes:
- Strict zero-sum balance checks across thousands of hands.
- Auto-rebuy logic and bankruptcy resilience.
- Swiss pairing bracket alignment and snake seeding correctness.
- Observation fusion shapes and backpropagation gradients.
- Transformer V2 vs. MLP V1 forward and backward passes.

---

## Performance Benchmark

Measured on local hardware (Apple Silicon):
- **Poker Engine Simulation:** ~1,960 hands/sec
- **Tournament Decision Steps:** ~19,590 steps/sec
- **Full 120-Player Tournaments:** ~0.91 tournaments/sec (~5,000 hands/tourn)
- **Neural Forward Inference:** ~0.088 ms/step (~11,340 forward passes/sec)

---

## Documentation Index

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): Complete architecture manual, state machine diagrams, and neural network dataflow.
- [`docs/PHASE_COMPLETION_REPORT.md`](docs/PHASE_COMPLETION_REPORT.md): Detailed audit and verification report across all 12 project phases.
- [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md): Formal tournament rules, mathematical invariants, and tie-breaking specifications.
- [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md): Repository structure, module dependencies, and file layout.
- [`docs/PHASE_0_REPORT.md`](docs/PHASE_0_REPORT.md): Baseline environment setup and verification report.

---

## License

MIT. See [LICENSE.txt](./LICENSE.txt).
