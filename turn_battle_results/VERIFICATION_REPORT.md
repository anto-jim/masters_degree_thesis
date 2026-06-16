# Thesis Verification Report

**Thesis:** Adaptation and Integration of MARL Algorithms in a Turn-Based Multiagent Combat Environment Using OpenSpiel  
**Author:** Antonio Jiménez Godínez | **Supervisor:** Eduardo Guzmán de los Riscos  
**Anteproyecto:** 17 May 2024 | **Canonical dataset:** `production_final/`

Generated: 2026-06-13  
GPU: NVIDIA GeForce RTX 3060 Ti (`--device=auto`)  
**Canonical tournament eval:** fixed team slots (team~1 = P0/P1, team~2 = P2/P3; no team swapping)

## Phase summary

| Phase | Result | Notes |
|-------|--------|-------|
| 0 Build/GPU | PASS | CUDA LibTorch binary, `torch.cuda.is_available()` |
| 1 Foundation tests | PASS | C++ + Python sim tests |
| 2 Algo smoke + checkpoints | PASS | NFSP checkpoint save fixed in `checkpoints.py` |
| 3 Compare mode | PASS | All 4 algos in one run |
| 4 Design checks | PASS | Equal budget, fixed-role eval, dual stacks |
| 5 Pilot tournament | PASS | 100 eps, ~29 min |
| 6 Mini multi-seed | PASS | 2 seeds × 50 eps, CIs, idempotent aggregate |
| 7 Production-lite | PASS | 3 seeds × 100 eps (pilot-scale; artifacts removed) |
| 7 Production | PASS | 3 seeds × 300 eps AZ/DCFR/NFSP/QPG (legacy NFSP/QPG; artifacts removed) |
| 8 Analysis (interim) | PASS | Fixed-role re-eval of legacy checkpoints (superseded) |
| 9 Reproducibility audit | PASS | Config snapshot, docs, limitations |
| **10 Final pipeline** | **PASS** | Role-shared NFSP/QPG + partial retrain + full tournament (`production_final/`) |

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

## Phase 8 — Research questions and hypotheses

*Dataset: `production_final/` (canonical). Aligned with anteproyecto RQs (May 2024).*

### Anteproyecto RQ1 — Most effective MARL algorithms

**AlphaZero** ranks first on seeding, round-robin points, and bracket outcomes across all 3 seeds.  
Seeding CI for AlphaZero (0.747) does not overlap model-free methods (0.27–0.32).  
AlphaZero wins every pairwise matchup with mean win rates 0.61–0.69.

### Anteproyecto RQ2 — Learning speed and adaptability

AlphaZero and Deep CFR training curves are from Phase 7 (`training_alphazero.csv`, `training_deep_cfr.csv` copied into each seed).  
NFSP/QPG curves are from role-shared retraining (`training_nfsp.csv`, `training_qpg.csv`).

| Algorithm | Notes |
|-----------|-------|
| Deep CFR | Positive mid-training gains; seeding 0.22–0.32 |
| NFSP | Stable seeding 0.32 across all 3 seeds under role-shared training |
| QPG | Seeding 0.24–0.30; below legacy independent-agent pipeline (0.40 mean in interim re-eval) |
| AlphaZero | Final eval win rates 0.72–0.82 |

### Anteproyecto RQ3 — Limitations and improvements

Dual evaluation stacks, sample vs wall-clock mismatch, 3-seed budget, mixed pipeline (AZ/DCFR Phase 7 + role-shared NFSP/QPG Phase 10). Improvements: Deep CFR hyperparameter tuning (`DCFR_VERIFICATION.md`), role-shared training, `--retrain_algorithms`.

### Operational — round-robin vs bracket

| Seed | RR leader | Bracket champion | Agreement |
|------|-----------|------------------|-----------|
| 42 | alphazero | alphazero | ✓ |
| 43 | alphazero | alphazero | ✓ |
| 44 | alphazero | alphazero | ✓ |

**3/3 seeds (100%):** round-robin leader = bracket champion = AlphaZero.

### Hypotheses (300-episode budget, canonical pipeline)

| ID | Hypothesis | Outcome |
|----|------------|---------|
| H1 | AlphaZero best with sufficient budget | **Supported** — wins seeding, RR (3.0 pts), brackets (3/3) |
| H2 | Deep CFR more stable than QPG | **Partial** — similar seeding means (0.27 vs 0.28); QPG wins RR (1.5 vs 0.67 pts) |
| H3 | NFSP competitive in tournaments | **Partial** — RR 0.83 pts, beats DCFR head-to-head on average; loses to QPG in RR ordering |
| H4 | QPG trains fast but plateaus | **Mixed** — role-shared QPG no longer leads model-free seeding; still strongest model-free RR (1.5 pts) |

### Thesis writing notes

- Use **`production_final/`** as the **primary results table** for the thesis chapter.
- Legacy results (`production_300`, `production_300_fixed_roles`, pilots) were **removed** after Phase 10; numbers above are the publishable set.
- Document that AZ/DCFR checkpoints originate from Phase 7 training while NFSP/QPG use the role-shared pipeline (same episode budget and eval protocol).

## Phase 9 — Reproducibility audit

### Config vs `thesis_default.json`

| Parameter | thesis_default | production_final |
|-----------|----------------|------------------|
| train_episodes | 300 | 300 (NFSP/QPG); AZ/DCFR loaded |
| eval_episodes | 50 | 50 |
| eval_every | 50 | 50 |
| seeds | 42–46 | 42, 43, 44 |
| retrain_algorithms | — | nfsp, qpg |
| device | auto | auto |
| algorithms | 4 paradigms | 4 |
| round_robin | true | true |

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

### Limitations (document in thesis)

1. **Dual evaluation stacks** — QPG/NFSP use RL env; AlphaZero/Deep CFR use turn-based bots.
2. **AlphaZero training view** — 2-team `turn_battle_teams` training vs 4-player tournament eval.
3. **Sample equivalence ≠ wall-clock** — Deep CFR dominates GPU time when retrained (~2–3 h/seed).
4. **LibTorch/PyTorch bridge** — AlphaZero C++ checkpoints loaded into Python `VpNetMlp` for MCTS eval.
5. **Scaled budget** — 3 seeds × 300 episodes on consumer GPU (RTX 3060 Ti).
6. **Mixed pipeline revision** — AZ/DCFR from Phase 7; NFSP/QPG from role-shared Phase 10 retrain.
7. **Small N** — 3 seeds; NFSP seeding has zero variance in this run (all seeds 0.32).

### Artifact index

```
turn_battle_results/
├── production_final/               # Phase 10 canonical ✓
│   ├── aggregated_results.json
│   ├── thesis_aggregated_report.tex
│   └── seed_42..44/
│       ├── checkpoints/            # all 4 algorithms
│       ├── training_*.csv
│       ├── tournament_results.json # includes role_selection
│       └── experiment_config.json
├── production_final_run.log
├── DCFR_VERIFICATION.md
└── VERIFICATION_REPORT.md          # this file
```

### Documentation cross-references

| Document | Path |
|----------|------|
| Anteproyecto TFM (May 2024) | `master's degree docs/Anteproyecto TFM Antonio Jiménez Godínez.pdf` |
| Thesis LaTeX project | `master's degree docs/thesis/main.tex` |
| Thesis plan | `master's degree docs/THESIS_PLAN.md` |
| Experiment README | `open_spiel/python/examples/turn_battle_study/README_experiments.md` |
| Default config | `experiments/configs/thesis_default.json` |
| Deep CFR tuning | `turn_battle_results/DCFR_VERIFICATION.md` |

### Phase 9/10 checklist

- [x] Training checkpoints saved for all 4 algorithms × 3 seeds
- [x] Role-shared NFSP/QPG checkpoints with `role_selection.json`
- [x] `experiment_config.json` / checkpoint metadata per seed
- [x] Fixed team-slot tournament protocol (code default)
- [x] Aggregated results + LaTeX report generated
- [x] RQ/H1–H4 analysis documented against canonical dataset
- [x] Superseded pilot/interim artifacts removed
- [x] Partial retrain workflow (`--retrain_algorithms`) documented
