# Thesis Verification Report

**Thesis:** Adaptation and Integration of MARL Algorithms in a Turn-Based Multiagent Combat Environment Using OpenSpiel  
**Author:** Antonio Jiménez Godínez | **Supervisor:** Eduardo Guzmán de los Riscos  
**Anteproyecto:** 17 May 2024 | **Canonical dataset:** `production_final/`

Generated: 2026-06-13  
GPU: NVIDIA GeForce RTX 3060 Ti (`--device=auto`)  
**Canonical tournament eval:** fixed team slots (team~1 = P0/P1, team~2 = P2/P3; no team swapping)

## Phase summary

| Phase | Result |
|-------|--------|
| 0 Build/GPU | PASS |
| 1 Foundation tests | PASS |
| 2 Algo smoke + checkpoints | PASS |
| 3 Compare mode | PASS |
| 4 Design checks | PASS |
| 5 Pilot tournament | PASS |
| 6 Mini multi-seed | PASS |
| 7 Production (3 seeds × 300 eps) | PASS |
| 8 Analysis (interim, superseded) | PASS |
| 9 Reproducibility audit | PASS |
| **10 Final pipeline** | **PASS** |

## Phase 10 — canonical production run

### Pipeline

| Algorithm | Training | Checkpoints |
|-----------|----------|-------------|
| AlphaZero | Reused from Phase 7 (`production_300`) | `checkpoints/alphazero/` |
| Deep CFR | Reused from Phase 7 | `checkpoints/deep_cfr/` |
| NFSP | **Retrained** (role-shared, 300 eps) | `role_defender/`, `role_attacker/`, `role_selection.json` |
| QPG | **Retrained** (role-shared, 300 eps) | same layout as NFSP |

Role-shared training pools experience across defender seats (0/2) and attacker seats (1/3).  
Before the tournament, `select_best_role_seats()` picks the best defender/attacker seat pair per seed.

### Command

```bash
# 1) Copy reused checkpoints per seed (AZ + DCFR from Phase 7)
for seed in 42 43 44; do
  OUT=turn_battle_results/production_final/seed_${seed}
  mkdir -p "${OUT}/checkpoints"
  cp -a turn_battle_results/production_300/seed_${seed}/checkpoints/{alphazero,deep_cfr} \
        "${OUT}/checkpoints/"
done

# 2) Partial retrain + tournament + aggregate
python open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=multi_seed \
  --results_root=turn_battle_results/production_final \
  --seeds=42,43,44 \
  --train_episodes=300 \
  --eval_episodes=50 \
  --eval_every=50 \
  --retrain_algorithms=nfsp,qpg \
  --az_cpp_actors=1 \
  --az_cpp_evaluators=0 \
  --device=auto
```

**Wall-clock:** 589 s (~9.8 min). Log: `production_final_run.log`  
Checkpoints + tournament JSON: `production_final/seed_*/`

### Aggregated results (`production_final/aggregated_results.json`)

| Algorithm | Seeding mean [95% CI] | RR points | Bracket wins |
|-----------|----------------------|-----------|--------------|
| AlphaZero | 0.747 [0.674, 0.819] | 3.0 | 3/3 |
| Deep CFR | 0.273 [0.216, 0.330] | 0.67 | 0/3 |
| NFSP | 0.320 [0.320, 0.320] | 0.83 | 0/3 |
| QPG | 0.280 [0.241, 0.319] | 1.50 | 0/3 |

**Pairwise highlights (team~1 vs team~2, 3 seeds):**

| Matchup | Mean win rate [95% CI] |
|---------|------------------------|
| AlphaZero vs Deep CFR | 0.613 [0.579, 0.648] |
| AlphaZero vs NFSP | 0.693 [0.659, 0.728] |
| AlphaZero vs QPG | 0.687 [0.652, 0.721] |
| QPG vs Deep CFR | 0.433 [0.309, 0.558] |
| NFSP vs Deep CFR | 0.347 [0.281, 0.412] |
| NFSP vs QPG | 0.340 [0.295, 0.385] |

**Per-seed seeding:**

| Algorithm | Seed 42 | Seed 43 | Seed 44 |
|-----------|---------|---------|---------|
| AlphaZero | 0.72 | 0.72 | 0.82 |
| Deep CFR | 0.28 | 0.32 | 0.22 |
| NFSP | 0.32 | 0.32 | 0.32 |
| QPG | 0.30 | 0.30 | 0.24 |

**Role seat selection (NFSP / QPG, per seed):**

| Seed | NFSP (def, atk) | QPG (def, atk) |
|------|-----------------|----------------|
| 42 | (0, 1) | (2, 3) |
| 43 | (0, 3) | (0, 3) |
| 44 | (2, 3) | (2, 1) |

### Code fixes in Phase 10

1. **`--retrain_algorithms`** — hybrid train/load in `tournament.py` for partial reruns.
2. **`_maybe_finalize_rl_agents`** — only finalize `RoleSharedTeam` on checkpoint load (not AZ/DCFR lists).
3. **`_remap_timestep`** — remap `current_player` to canonical role index so QPG acts correctly in turn-based eval.

## Research hypotheses

*Dataset: `production_final/` (canonical). Aligned with anteproyecto RQs (May 2024).*

| ID | Hypothesis | Outcome |
|----|------------|---------|
| H1 | AlphaZero best with sufficient budget | **Supported** — wins seeding, RR (3.0 pts), brackets (3/3) |
| H2 | Deep CFR more stable than QPG | **Partial** — similar seeding means (0.27 vs 0.28); QPG wins RR (1.5 vs 0.67 pts) |
| H3 | NFSP competitive in tournaments | **Partial** — RR 0.83 pts, beats DCFR head-to-head; loses to QPG in RR ordering |
| H4 | QPG trains fast but plateaus | **Mixed** — role-shared QPG no longer leads model-free seeding; strongest model-free RR (1.5 pts) |

Use **`production_final/`** as the primary results table for the thesis chapter. Document that AZ/DCFR checkpoints originate from Phase 7 training while NFSP/QPG use the role-shared pipeline (same episode budget and eval protocol).

## Reproducibility

### Reproduce from checkpoints

```bash
python open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=tournament --train_episodes=0 \
  --eval_episodes=50 --seed=42 \
  --output_dir=turn_battle_results/production_final/seed_42

python open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=aggregate \
  --results_root=turn_battle_results/production_final \
  --seeds=42,43,44
```

### Limitations

1. **Dual evaluation stacks** — QPG/NFSP use RL env; AlphaZero/Deep CFR use turn-based bots.
2. **AlphaZero training view** — 2-team `turn_battle_teams` training vs 4-player tournament eval.
3. **Sample equivalence ≠ wall-clock** — Deep CFR dominates GPU time when retrained (~2–3 h/seed).
4. **LibTorch/PyTorch bridge** — AlphaZero C++ checkpoints loaded into Python `VpNetMlp` for MCTS eval.
5. **Scaled budget** — 3 seeds × 300 episodes on consumer GPU (RTX 3060 Ti).
6. **Mixed pipeline revision** — AZ/DCFR from Phase 7; NFSP/QPG from role-shared Phase 10 retrain.
7. **Small N** — 3 seeds; NFSP seeding has zero variance in this run (all seeds 0.32).

### Documentation

| Document | Path |
|----------|------|
| Anteproyecto TFM (May 2024) | `master's degree docs/Anteproyecto TFM Antonio Jiménez Godínez.pdf` |
| Thesis LaTeX project | `master's degree docs/thesis/main.tex` |
| Thesis plan | `master's degree docs/THESIS_PLAN.md` |
| Experiment README | `open_spiel/python/examples/turn_battle_study/README_experiments.md` |
| Default config | `experiments/configs/thesis_default.json` |

## Deep CFR tuning

Hyperparameters were tuned for the 300-episode budget (RTX 3060 Ti, `--train_episodes=300`):

| Setting | Before | After | Rationale |
|---------|--------|-------|-----------|
| `dcfr_traversals` | 3 | **10** | More tree coverage per CFR iteration |
| `dcfr_batch_size` | 128 | **32** | Learning starts before buffer reaches 128 samples |
| `dcfr_learning_rate` | 1e-3 | **1e-4** | Match OpenSpiel Deep CFR default |
| `dcfr_advantage_steps` | 3 | **10** | More advantage-net updates per player |
| `dcfr_policy_steps` | 25 | **10** | Strategy net refresh at eval checkpoints |
| `dcfr_reinitialize_advantage_networks` | true | **false** | Preserve regret nets across iterations at 300-eps budget |
| Strategy training | End only | **Before each eval + final** | Tournament bots use trained average policy |

`dcfr_reinitialize_advantage_networks=false` is the default; `production_final` Deep CFR checkpoints use these settings.  
Canonical seeding result (Deep CFR, 3 seeds × 300 eps): **0.273 [0.216, 0.330]**; RR points: 0.67; bracket champion: 0/3.  
Use **`production_final/aggregated_results.json`** for all thesis claims.
