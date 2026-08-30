# Thesis Verification Report

**Thesis:** Adaptation and Integration of MARL Algorithms in a Turn-Based Multiagent Combat Environment Using OpenSpiel
**Author:** Antonio Jiménez Godínez | **Supervisor:** Eduardo Guzmán de los Riscos
**Game config:** `num_turns=5`, shared by every algorithm in both training and evaluation

## Status: CAMPAIGN COMPLETE (30 Aug 2026, 02:32)

All three seeds finished in one uninterrupted run: **exit code 0**, wall clock
**23,883 s (6 h 38 m)**, ~2 h 12 m per seed. Artifacts are in `seed_42/`,
`seed_43/`, `seed_44/`, aggregated into `aggregated_results.json`, with the full
console transcript in `campaign.log`.

### Headline results

| Algorithm | Seeding score (95% CI) | RR points | vs. Random (margin) | vs. MCTS (margin) | Champion |
|---|---|---|---|---|---|
| AlphaZero | 85.3% [75.0, 95.7] | 3.00 (3/3/3) | +76.7 | +22.7 | 3/3 |
| QPG       | 39.3% [25.0, 53.7] | 1.67 (2/1/2) | +1.3  | −72.7 | 0/3 |
| Deep CFR  | 29.3% [5.9, 52.8]  | 0.67 (1/1/0) | −5.3  | −70.0 | 0/3 |
| NFSP      | 25.3% [0.0, 60.2]  | 0.67 (0/1/1) | −12.7 | −77.3 | 0/3 |

"Margin" is win rate minus loss rate, which is the honest direction indicator in
a game where draws run 11–41% per pairing.

Three findings depend on the evaluation design rather than on the ranking:

1. **Only AlphaZero beat random play.** QPG is at parity and the other two are
   below it. A round-robin alone always yields a champion, so this could only be
   detected by the fixed-reference phase.
2. **Search, not learning, carries most of AlphaZero's margin.** Plain MCTS with
   random rollouts beats every model-free entrant by 70–77 points of margin and
   loses to full AlphaZero by only 22.7.
3. **The model-free trio is not separable at n=3.** Their standings rearrange in
   every seed (seed 43 has all three tied on points), all intervals overlap, and
   no within-tier paired comparison favours one side in all three seeds. The
   thesis reports this as a two-tier result rather than a ranking of four.

Hypothesis verdicts: **H1 supported** (with the search-versus-learning
qualification), **H2 refuted** (Deep CFR was the *least* stable algorithm, not
the most), **H3 half supported and half untested** (no exploitability
computation was implemented), **H4 supported** (QPG plateaus by its first
checkpoint, moving +1.3 points over the whole run).

## Remediation background

The previous `num_turns=10` campaign has been **superseded and discarded**. It
contained a confound that invalidated one of the four subjects of the
comparison: Deep CFR trained on a 5-turn-capped game tree while every other
algorithm trained at 10 turns and *all* evaluation ran at 10 turns. Deep CFR's
tournament standing therefore measured horizon mismatch rather than algorithmic
merit, and the round-robin was additionally played in a single team-slot
orientation.

The replacement campaign fixes the confound at its root by running **every**
algorithm at `num_turns=5`, the longest horizon at which Deep CFR's traversals
fit the compute budget. Nothing is capped per algorithm; there is no
per-algorithm asterisk on any reported number.

## Exact campaign command

```bash
./venv/bin/python open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=multi_seed \
  --results_root=turn_battle_results \
  --seeds=42,43,44 \
  --train_episodes=300 \
  --eval_episodes=50 \
  --eval_every=50 \
  --bracket_size=4 \
  --game_params=num_turns=5 \
  --dcfr_max_turns=0 \
  --round_robin=true \
  --device=auto
```

`--dcfr_max_turns=0` means "use `num_turns`", i.e. no separate Deep CFR horizon.
Launched via `run_campaign.sh`, which sets `PYTHONUNBUFFERED=1` and the
LibTorch library path, refuses to start if another campaign is already writing
the same root, and refuses to overwrite completed results unless `FORCE=1` is
set (so a rerun cannot silently destroy this directory or the report you are
reading). Run it under `setsid`/`nohup` to survive shell teardown.

### Why the horizon is 5 and not 10

A pilot measurement put one Deep CFR iteration at ~42 min on the RTX 3060 Ti at
`num_turns=10`, i.e. ~9 days for a single 300-iteration seed. At `num_turns=5`
the same 300 iterations complete in about an hour. Two options existed:

1. Keep 10 turns and cap only Deep CFR (what the old campaign did) — preserves
   horizon length but invalidates Deep CFR's comparative standing.
2. Run everything at 5 turns — valid comparison, shorter game.

Option 2 was chosen. The cost is external validity (5 turns is a short game),
which the thesis records as its first limitation. The benefit is that internal
validity is intact.

## Run metadata

- **Device:** NVIDIA GeForce RTX 3060 Ti (8 GB)
- **Software:** torch 2.10.0+cu126 (venv); C++ LibTorch AlphaZero subprocess on
  `cuda:0`, launched with `--seed=<campaign seed>`
- **Seeds:** 42, 43, 44, run sequentially (single GPU; parallel seeds would
  contend for memory and couple their timing)
- **Verified GPU usage:** `nvidia-smi` sampled at 74–99% utilisation during both
  the C++ AlphaZero self-play phase and the Deep CFR traversal phase. Deep CFR
  drops to ~10% during traversal because external sampling walks the tree in
  Python; the GPU is used for the advantage/policy network fits, not the walk.
- **Concurrency guard:** an earlier launch attempt had two campaign processes
  writing the same results root, which corrupted the output. `run_campaign.sh`
  now refuses to start if another campaign is already running against the same
  `--results_root`; the completed campaign was launched fresh into a cleared
  directory after that guard was added.

### Stale `turn_battle_results_v2/` paths in the archived artifacts

The campaign ran while this directory was still named
`turn_battle_results_v2/`, and was renamed to `turn_battle_results/`
afterwards. The archived artifacts kept the old string: `campaign.log` writes
`turn_battle_results_v2/seed_N/...` on 23 lines, and each
`seed_N/experiment_config.json` records `flags.results_root` as
`turn_battle_results_v2`. Only the string is stale; the rename moved nothing
inside the directory and changed no recorded number.

Provenance is verified independently of the path. Every per-checkpoint win rate
printed in `campaign.log` matches the corresponding
`seed_N/training_<algorithm>.csv` row exactly — 72 checkpoints across the 12
(seed, algorithm) pairs, with no disagreement in either the episode index or
the win rate. This is the log of the run that produced this dataset; the
directory name it prints is the only misleading part of it.

`run_campaign.sh` now defaults its output root to `turn_battle_results`
(`ROOT="${1:-turn_battle_results}"`), so a fresh reproduction writes to the
location documented above and its artifacts record a path that matches.

## Correctness fixes carried into this campaign

Each item below is a defect that produced **plausible but wrong numbers without
crashing** — the class of bug a comparative study cannot absorb. All are pinned
by tests in `turn_battle_study/turn_battle_study_test.py` (22 tests, all
passing).

1. **Deep CFR evaluation horizon** (`game.py`, `evaluation.py`, `bots.py`).
   `parse_num_turns` now rejects a Deep CFR cap below `num_turns` unless
   `--allow_dcfr_horizon_mismatch` is passed explicitly. Evaluation asserts that
   any Deep CFR solver entering a matchup was built over a game whose
   info-state width matches the matchup's game. The turn-based bot adapter now
   **raises** on an unexpected observation width instead of zero-padding it —
   the previous padding turned a shape error into a silently wrong result.
2. **AlphaZero multi-seed independence** (`az_cpp.py`, C++ `alpha_zero.{h,cc}`,
   `alpha_zero_torch_example.cc`). The run directory is created fresh per seed;
   learner progress is read from the **last** record of `learner.jsonl` rather
   than the file maximum (a stale high-water mark suppressed every mid-training
   checkpoint reload for the rest of a campaign); the checkpoint is always
   reloaded after the subprocess exits; and the C++ binary now accepts `--seed`,
   without which it seeded from the clock and the three "independent" seeds
   differed only in Python-side randomness.
3. **Role-shared NFSP/QPG per-seat bookkeeping** (`role_shared.py`). Two seat
   facades share one learner object, whose per-episode bookkeeping (previous
   timestep, previous action, QPG episode buffer) pairs observations with
   transitions. Each facade now swaps its own bookkeeping in and out around
   every step, so transition pairing is seat-local while weights stay shared.
   Previously seat 2's step was paired against seat 0's observation.
4. **DQN / NFSP checkpoint round-trip** (`open_spiel/python/pytorch/dqn.py`).
   `DQN.load()` read state-dict keys that `DQN.save()` never wrote. A
   save-load-compare test now pins the round trip.
5. **Team-slot bias** (`tournament.py`). Every baseline and round-robin pairing
   is played in both team-slot orientations with half the episodes each and the
   rates averaged, so a uniform slot advantage cancels exactly. Previously a
   documented limitation; now a controlled variable. A test drives the pairwise
   evaluator with a stub in which team 1 always wins 80% and asserts that
   balancing cancels it to exactly 0.5.

## Statistical treatment

`aggregate.py` reports, for every metric: the mean, a **Student-t** interval on
`n-1` degrees of freedom (at n=3 the critical value is 4.30, not 1.96 — the
normal approximation was anticonservative and has been removed), a **percentile
bootstrap** interval, and the **raw per-seed values**. Because seeds form a
paired design, `paired_comparison` also reports each pairwise mean difference
with its interval and the number of seeds favouring each side; the accompanying
paired t-test p-value is descriptive only at this sample size.

## New evaluation phase

A **baseline reference phase** now precedes seeding: every trained algorithm
plays slot-balanced matchups against uniform-random, the hand-crafted heuristic,
and a plain MCTS bot. A round-robin alone only ranks the field relative to
itself, so a uniformly weak field would still yield a clean champion. The three
references anchor the strength scale at three graded absolute points.

## Deep CFR configuration

Flags match `thesis_default.json` (`dcfr_traversals=10`, `dcfr_batch_size=32`,
`dcfr_learning_rate=1e-4`, `dcfr_advantage_steps=10`, `dcfr_policy_steps=10`,
`dcfr_reinitialize_advantage_networks=false`) with **no** horizon cap. The two
departures from common Deep CFR practice — small batch and non-reinitialised
advantage networks — are short-budget adaptations documented in the thesis
(Section 4.4).

## Syncing to the LaTeX thesis

`master's degree docs/thesis/tools/gen_results.py` reads
`aggregated_results.json` plus the per-seed `tournament_results.json` files and
emits:

- `generated/numbers.tex` — every result number as a LaTeX macro, so the
  narrative cannot drift from the data
- `generated/tables.tex` — the main-text result tables
- `generated/figures.tex` — figure floats with captions and labels
- `generated/appendix.tex` — per-seed detail tables
- `figures/*.pdf` — training curves, baseline references, seeding scores,
  champion frequency

```bash
cd "master's degree docs/thesis"
/home/anto/Projects/masters_degree_thesis/venv/bin/python \
  tools/gen_results.py --results_root /path/to/turn_battle_results
./build.sh
```

The generator imports matplotlib, which the system `python3` does not have, so
it must run under the campaign venv (or any interpreter carrying matplotlib).

No result number in the thesis is transcribed by hand. The previous appendix
contained hand-copied numbers from an earlier campaign, which is exactly the
drift this generator removes.
