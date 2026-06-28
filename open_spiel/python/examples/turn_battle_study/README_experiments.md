# Turn Battle MARL Thesis Experiments

**Thesis:** Adaptation and Integration of MARL Algorithms in a Turn-Based Multiagent Combat Environment Using OpenSpiel  
**Author:** Antonio Jiménez Godínez | **Supervisor:** Eduardo Guzmán de los Riscos

Comparative study of **AlphaZero**, **Deep CFR**, **NFSP**, and **QPG** on the
`turn_battle` OpenSpiel game.

**Related docs:** [Thesis docs hub](../../../../../master's%20degree%20docs/README.md) ·
[THESIS_PLAN.md](../../../../../master's%20degree%20docs/THESIS_PLAN.md) ·
[CODE_DOCUMENTATION.md](../../../../../master's%20degree%20docs/CODE_DOCUMENTATION.md) ·
[VERIFICATION_REPORT.md](../../../../turn_battle_results/VERIFICATION_REPORT.md)

## Package layout

| Module | Role |
|--------|------|
| `turn_battle_marl_study.py` | CLI entry point (modes, flags) |
| `config.py` | Algorithm registry, thesis defaults |
| `game.py` / `device.py` | Game loading, GPU/CPU resolution |
| `agents.py` / `trainers.py` | RL factories and training loops |
| `az_cpp.py` | C++ AlphaZero driver + tournament eval bots |
| `bots.py` / `evaluation.py` | Bot adapters, match runners |
| `role_shared.py` | Shared defender/attacker policies (NFSP, QPG) |
| `tournament.py` | Train → seed → round-robin → bracket |
| `checkpoints.py` | Save/load trained agents |
| `storage.py` / `report.py` / `aggregate.py` | Artifacts and LaTeX reports |

Full reference: [`master's degree docs/CODE_DOCUMENTATION.md`](../../../../../master's%20degree%20docs/CODE_DOCUMENTATION.md).

## AlphaZero (C++ LibTorch)

AlphaZero uses OpenSpiel's official C++ implementation
(`open_spiel/algorithms/alpha_zero_torch`), not a custom Python trainer.

Because C++ AlphaZero only supports **2-player sequential** games, training runs
on `turn_battle_teams` — a wrapper where each player controls both members of a
team (Team 0 = players 0+1, Team 1 = players 2+3). Tournament evaluation still
uses the 4-player `turn_based_simultaneous_game` wrapper with shared team
policies (`az_cpp.py`).

### Build prerequisite

```bash
# From repository root
./install.sh
source open_spiel/scripts/global_variables.sh
mkdir -p build && cd build
cmake ../open_spiel
cmake --build . --target alpha_zero_torch_example turn_battle_teams_test pyspiel -j$(nproc)
```

The training subprocess expects `build/examples/alpha_zero_torch_example`.

`global_variables.sh` defaults to **CUDA LibTorch** (`cu126`, `2.6.0+cu126`). You need a working NVIDIA driver at runtime. If CMake cannot find CUDA libraries, install a matching toolkit (e.g. `conda install -c nvidia cuda-toolkit=12.6`) before re-running `cmake`.

Verify the build is GPU-enabled:

```bash
ldd build/examples/alpha_zero_torch_example | grep -E 'cuda|c10'
```

## GPU training (all algorithms)

All four algorithms share one device flag: `--device=auto` (default). With CUDA available:

| Algorithm | GPU path |
|-----------|----------|
| AlphaZero | C++ subprocess `--devices=cuda:0` (via `resolve_cpp_az_devices`) |
| Deep CFR | `DeepCFRSolver(device="cuda")` |
| NFSP / QPG | PyTorch agents created with `device="cuda"` |

AlphaZero tournament eval loads C++ checkpoints into a Python `VpNetMlp` on the same device for MCTS inference.

Confirm GPU usage in logs:

- Startup: `Training device: cuda (<GPU name>)`
- Per trainer: `backend: … on cuda (…)` or `train --devices=cuda:0`
- C++ learner: `Running the learner on device 0: cuda:0` in `<output_dir>/az_cpp_run/log-learner.txt`

Force CPU for every algorithm: `--device=cpu`.

### GPU smoke test

```bash
python open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=compare \
  --algorithms=alphazero,deep_cfr,nfsp,qpg \
  --train_episodes=3 \
  --eval_every=2 \
  --eval_episodes=4 \
  --az_cpp_actors=1 \
  --az_cpp_evaluators=0 \
  --device=auto \
  --output_dir=/tmp/turn_battle_gpu_pilot
```

## Training budget

All four algorithms train for the same `train_episodes` count (default: **300**).
This is the only training budget mechanism in the pipeline — there is no
per-algorithm cap or alternate budget.

| Algorithm | What one episode means |
|-----------|------------------------|
| AlphaZero | One C++ LibTorch learner step (`--max_steps`) |
| Deep CFR | One CFR iteration (traversals + advantage learning) |
| NFSP | One RL environment episode |
| QPG | One RL environment episode |

`train_episodes` is stored in `experiment_config.json` and
`tournament_results.json`.

Training curves (`training_<algo>.csv`) log win rate vs `random` on team~1 every
`eval_every` steps **and** at the final episode (RL and Deep CFR trainers).

**Thesis note:** Equal episode counts do not imply equal wall-clock time or
gradient steps. Report training curves to show convergence differences.

## Quick Start

From the repository root:

```bash
# Single-seed tournament (pilot)
python open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=tournament \
  --train_episodes=300 \
  --eval_episodes=50 \
  --bracket_size=4 \
  --seed=42 \
  --output_dir=turn_battle_results/seed_42

# Multi-seed campaign (default seeds 42–46; use --seeds=42,43,44 for canonical 3-seed run)
python open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=multi_seed \
  --train_episodes=300 \
  --eval_episodes=50 \
  --bracket_size=4 \
  --results_root=turn_battle_results

# Re-aggregate after manual runs
python open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=aggregate \
  --results_root=turn_battle_results
```

## Output Layout

```
turn_battle_results/
├── seed_42/
│   ├── experiment_config.json
│   ├── tournament_results.json
│   ├── training_<algo>.csv
│   ├── checkpoints/<algo>/
│   ├── figures/
│   └── turn_battle_report.tex
├── seed_43/
│   └── ...
├── aggregated_results.json
├── thesis_aggregated_report.tex
└── figures/
    ├── aggregated_seeding.pdf
    └── champion_frequency.pdf
```

## Modes

| Mode | Purpose |
|------|---------|
| `train` | Train one algorithm |
| `evaluate` | Head-to-head matchup |
| `compare` | Train all listed algos vs random |
| `tournament` | Train + round-robin + bracket (single seed) |
| `multi_seed` | Run `tournament` for each seed, then aggregate |
| `aggregate` | Recompute cross-seed statistics from existing runs |

## Default Thesis Configuration

See `experiments/configs/thesis_default.json` for the canonical flag snapshot
(`num_turns=10`, seeds 42–44).

**Note:** Prior `production_final/` results at `num_turns=5` were removed during
the thesis refactor. Run `--mode=multi_seed` to regenerate at `num_turns=10`
(see `turn_battle_results/VERIFICATION_REPORT.md`).

## Checkpoints

Trained models are saved under `checkpoints/<algorithm>/` when
`--save_checkpoints=true` (default). Implemented in `checkpoints.py`.

| Algorithm | Checkpoint layout |
|-----------|---------------------|
| AlphaZero | C++ run dir: `vpnet.pb`, `checkpoint--1.pt`, `config.json`, `metadata.json`, … |
| Deep CFR | `model.pt` (policy + advantage nets + iteration) |
| NFSP / QPG (role-shared) | `role_defender/`, `role_attacker/`, optional `role_selection.json` |

NFSP uses custom save/load helpers (`model_state_dict` keys) because OpenSpiel's
built-in `NFSP.save` / `NFSP.restore` keys do not match.

Reload with:

```python
import numpy as np
from open_spiel.python.examples.turn_battle_study.checkpoints import load_trained_agents

rng = np.random.RandomState(42)
agents = load_trained_agents(
    "alphazero", "turn_battle_results/seed_42/checkpoints/alphazero", rng)
```

### Partial retrain (reuse checkpoints)

When only some algorithms need retraining after a pipeline change, copy unchanged
checkpoints into each seed directory, then:

```bash
python open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=multi_seed \
  --results_root=turn_battle_results \
  --seeds=42,43,44 \
  --train_episodes=300 \
  --retrain_algorithms=nfsp,qpg
```

Algorithms not listed in `--retrain_algorithms` are loaded from
`<seed>/checkpoints/<algo>/` instead of retrained.

## Evaluation Design

- **Fixed role slots:** tournament matchups use team~1 (players 0, 1) vs team~2
  (players 2, 3) without swapping teams.
- **Role-shared training (NFSP, QPG):** one defender policy and one attacker
  policy pool experience from both teams. Before the tournament,
  `select_best_role_seats()` evaluates four facade pairs `(0,1)`, `(0,3)`,
  `(2,1)`, `(2,3)` and deploys the winner via `_lineup_with_seats()`.
- **NFSP tournament play:** evaluation uses the average policy
  (`AVERAGE_POLICY` mode), including when NFSP is wrapped in `RoleSeatFacade`
  or `RlAgentTurnBasedBot`.
- **Round-robin:** all 6 pairwise comparisons before the bracket.
- **Bracket:** single elimination seeded by win rate vs random.

## Deep CFR defaults

One `train_episodes` step is one CFR iteration. Defaults are tuned for Turn Battle
(same episode budget as other algorithms):

| Flag | Default | Purpose |
|------|---------|---------|
| `dcfr_traversals` | 10 | Tree walks per player per iteration |
| `dcfr_batch_size` | 32 | Avoid skipped updates on small buffers |
| `dcfr_advantage_steps` | 10 | Advantage net SGD steps per player |
| `dcfr_policy_steps` | 10 | Average-policy net steps before eval/final |
| `dcfr_train_strategy_each_iteration` | false | Strategy refresh at eval checkpoints only |
| `dcfr_reinitialize_advantage_networks` | false | Keep advantage nets across iterations at 300-eps budget |

## Key flags

| Flag | Default | Purpose |
|------|---------|---------|
| `device` | `auto` | **Shared** device for all algorithms (`auto` → CUDA when available; maps to `cuda:0` for C++ AZ) |
| `az_cpp_nn_model` | `mlp` | C++ network type |
| `az_cpp_nn_width` | 128 | MLP width |
| `az_cpp_nn_depth` | 2 | MLP depth |
| `az_cpp_actors` | 2 | Self-play actor threads |
| `mcts_simulations` | 50 | MCTS sims per move (eval) |
| `retrain_algorithms` | (empty) | Only train listed algos; load others from `checkpoints/` |

AlphaZero checkpoints are TorchScript archives (`checkpoint--1.pt`). Python eval copies weights into `VpNetMlp` via `state_dict()` for compatibility with the installed PyTorch version.
