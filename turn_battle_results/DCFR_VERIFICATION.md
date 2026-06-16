# Deep CFR Training Improvements — Verification

**Thesis:** Adaptation and Integration of MARL Algorithms in a Turn-Based Multiagent Combat Environment Using OpenSpiel  
**Author:** Antonio Jiménez Godínez

Date: 2026-06-13  
GPU: RTX 3060 Ti, `--device=auto`, `train_episodes=300` (default)

## Code changes

| Setting | Before | After | Rationale |
|---------|--------|-------|-----------|
| `dcfr_traversals` | 3 | **10** | More tree coverage per CFR iteration |
| `dcfr_batch_size` | 128 | **32** | Learning starts before buffer reaches 128 samples |
| `dcfr_learning_rate` | 1e-3 | **1e-4** | Match OpenSpiel Deep CFR default |
| `dcfr_advantage_steps` | 3 | **10** | More advantage-net updates per player |
| `dcfr_policy_steps` | 25 | **10** | Strategy net refresh at eval checkpoints |
| `dcfr_reinitialize_advantage_networks` | true | **false** | Preserve regret nets across iterations at 300-eps budget |
| Strategy training | End only | **Before each eval + final** | Tournament bots use trained average policy |
| `DEFAULT_TRAIN_EPISODES` | 3000 | **300** | GPU-feasible thesis standard |

Same `train_episodes` count still applies to all algorithms (sample equivalence unchanged).

## Verification runs (Deep CFR only)

### Baseline (old defaults, 100 eps, seed 42)

From legacy `production_lite/seed_42/training_deep_cfr.csv` (artifact removed):

- Step 100 bidirectional win vs random: **0.15**
- Aggregated seeding mean (3 seeds, old pipeline): **0.21**

### New defaults, 100 eps, seed 42 (`/tmp/dcfr_reinit_false_100`)

- Step 100 bidirectional win vs random: **0.233**
- Final eval vs random (30 eps): team1 win rate **0.33**
- Wall-clock: **~70 min**

### New defaults (reinit still true), 300 eps, seed 42 (`/tmp/verify_dcfr_improved`)

- Step 300 bidirectional win vs random: **0.133**
- Wall-clock: **~3.2 h**
- Note: run started before `dcfr_reinitialize_advantage_networks=false` was set.

### Checkpoint round-trip (50 eps, seed 99)

- `save_trained_checkpoint` + `load_trained_agents` + eval: **PASS**

## Canonical production results

Deep CFR checkpoints in **`production_final/`** were **reused from Phase 7** (not retrained in Phase 10).  
NFSP and QPG were retrained with the role-shared pipeline; AZ/DCFR training settings are unchanged.

| Metric | Old pipeline (100 eps) | Phase 7 / production_final (300 eps) |
|--------|------------------------|--------------------------------------|
| Seeding mean (Deep CFR) | 0.21 | **0.273** [0.216, 0.330] |
| RR points | 0.0 | **0.67** |
| Champion | 0/3 | 0/3 |

`dcfr_reinitialize_advantage_networks=false` is the default; `production_final` Deep CFR checkpoints use these settings.

## Interpretation

Training is aligned with Deep CFR best practices (more traversals, usable batch size, strategy net synced to eval).  
Deep CFR remains below NFSP/QPG in round-robin ordering but shows improved seeding vs the 100-eps pilot.  
Use **`production_final/aggregated_results.json`** for thesis claims.
