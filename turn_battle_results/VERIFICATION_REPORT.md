# Thesis Verification Report

**Thesis:** Adaptation and Integration of MARL Algorithms in a Turn-Based Multiagent Combat Environment Using OpenSpiel  
**Author:** Antonio Jiménez Godínez | **Supervisor:** Eduardo Guzmán de los Riscos  
**Game config:** `num_turns=10` (thesis default)

## Status

Previous Phase 10 results at `num_turns=5` were **removed** from the repository during the thesis refactor (legacy algorithms, old training artifacts). Empirical tables in the LaTeX thesis and aggregated JSON/figures must be **regenerated** by running a fresh multi-seed campaign.

## Regenerate canonical results

From the repository root (GPU recommended):

```bash
python open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=multi_seed \
  --results_root=turn_battle_results/production_final \
  --seeds=42,43,44 \
  --train_episodes=300 \
  --eval_episodes=50 \
  --eval_every=50 \
  --game_params=num_turns=10 \
  --dcfr_max_turns=10 \
  --device=auto
```

Outputs land under `turn_battle_results/production_final/seed_*/` (JSON, CSV, figures, checkpoints). Sync figures to the LaTeX thesis with `make sync-figures` in `master's degree docs/thesis/`.

## Deep CFR tuning

See historical tuning notes in git history or re-verify after the 10-turn campaign. Default flags match `thesis_default.json` and `turn_battle_marl_study.py`.
