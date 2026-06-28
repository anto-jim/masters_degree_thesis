# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comparative MARL study on Turn Battle.

Modes: train, evaluate, compare, tournament, multi_seed, aggregate.
"""

from __future__ import annotations

import json
import os
import time

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.examples.turn_battle_study.aggregate import (
    aggregate_multi_seed_results,
    generate_aggregated_latex_report,
)
from open_spiel.python.examples.turn_battle_study.config import (
    BOT_ALGOS,
    DEFAULT_EVAL_EVERY,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_SEEDS,
    DEFAULT_TRAIN_EPISODES,
    THESIS_ALGORITHMS,
    TRAINABLE_ALGOS,
    normalize_algorithm,
    validate_algorithm,
)
from open_spiel.python.examples.turn_battle_study.device import (
    device_label,
    resolve_device,
)
from open_spiel.python.examples.turn_battle_study.evaluation import (
    evaluate_team_matchup,
)
from open_spiel.python.examples.turn_battle_study.game import load_game
from open_spiel.python.examples.turn_battle_study.report import generate_latex_report
from open_spiel.python.examples.turn_battle_study.storage import (
    ensure_output_dir,
    save_comparison_table,
    save_experiment_config,
    save_training_log,
)
from open_spiel.python.examples.turn_battle_study.tournament import run_full_tournament
from open_spiel.python.examples.turn_battle_study.trainers import train_algorithm

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "turn_battle", "OpenSpiel game string.")
flags.DEFINE_string("game_params", "num_turns=10", "Comma-separated game parameters.")
flags.DEFINE_enum(
    "mode", "tournament",
    ["train", "evaluate", "compare", "tournament", "multi_seed", "aggregate"],
    "Experiment mode.")
flags.DEFINE_list(
    "algorithms", THESIS_ALGORITHMS,
    "Algorithms to compare/tournament (thesis default: 4 paradigms).")
flags.DEFINE_string("algorithm", "qpg", "Algorithm for train mode.")
flags.DEFINE_string("team1_algo", "qpg", "Algorithm for team 1 (players 0,1).")
flags.DEFINE_string("team2_algo", "random", "Algorithm for team 2 (players 2,3).")
flags.DEFINE_integer(
    "train_episodes", DEFAULT_TRAIN_EPISODES,
    "Training episodes per algorithm (same count for all algorithms).")
flags.DEFINE_integer(
    "eval_episodes", DEFAULT_EVAL_EPISODES,
    "Evaluation episodes per matchup.")
flags.DEFINE_integer(
    "eval_every", DEFAULT_EVAL_EVERY,
    "Evaluate during training every N episodes.")
flags.DEFINE_integer(
    "bracket_size", 4,
    "Bracket field size (thesis: 4 algorithms).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("output_dir", "turn_battle_results", "Output directory.")
flags.DEFINE_string(
    "results_root", "turn_battle_results",
    "Root directory for multi-seed runs (seed_N subfolders).")
flags.DEFINE_list(
    "seeds", [str(s) for s in DEFAULT_SEEDS],
    "Random seeds for multi_seed mode.")
flags.DEFINE_boolean(
    "save_checkpoints", True,
    "Save trained model checkpoints after each algorithm.")
flags.DEFINE_list(
    "retrain_algorithms", [],
    "When set, only these algorithms are trained; others load from "
    "output_dir/checkpoints/ (requires existing checkpoints).")

# --- MCTS / tree-search flags ---
flags.DEFINE_integer("mcts_simulations", 50, "MCTS simulations per move.")
flags.DEFINE_float("mcts_uct_c", 2.0, "MCTS UCT exploration constant.")
flags.DEFINE_integer("mcts_rollouts", 1, "Random rollouts per MCTS evaluation.")

# --- AlphaZero flags (C++ subprocess via az_cpp.py) ---
flags.DEFINE_integer("az_replay_buffer_size", 4096, "AlphaZero replay buffer.")
flags.DEFINE_integer("az_batch_size", 64, "AlphaZero batch size.")
flags.DEFINE_float("az_learning_rate", 1e-3, "AlphaZero learning rate.")
flags.DEFINE_float("az_temperature", 1.0, "AlphaZero self-play temperature.")
flags.DEFINE_integer("az_temperature_drop", 4, "Greedy after N moves.")

# --- AlphaZero C++ binary flags ---
flags.DEFINE_string(
    "az_cpp_nn_model", "mlp",
    "C++ AlphaZero network type (mlp or resnet; Python eval supports mlp).")
flags.DEFINE_integer("az_cpp_nn_width", 128, "C++ AlphaZero MLP width.")
flags.DEFINE_integer("az_cpp_nn_depth", 2, "C++ AlphaZero MLP depth.")
flags.DEFINE_float("az_cpp_weight_decay", 1e-4, "C++ AlphaZero L2 weight decay.")
flags.DEFINE_integer("az_cpp_replay_reuse", 3, "C++ AlphaZero replay reuse count.")
flags.DEFINE_integer("az_cpp_checkpoint_freq", 100, "C++ AlphaZero checkpoint frequency.")
flags.DEFINE_integer("az_cpp_actors", 2, "C++ AlphaZero self-play actors.")
flags.DEFINE_integer("az_cpp_evaluators", 1, "C++ AlphaZero evaluators.")
flags.DEFINE_integer("az_cpp_eval_levels", 3, "C++ AlphaZero eval MCTS levels.")
flags.DEFINE_integer(
    "az_cpp_evaluation_window", 50,
    "C++ AlphaZero evaluation averaging window.")

# --- Deep CFR flags ---
flags.DEFINE_integer(
    "dcfr_traversals", 10,
    "Deep CFR traversals per player per iteration (balance quality vs GPU time).")
flags.DEFINE_integer(
    "dcfr_batch_size", 32,
    "Deep CFR batch size (must be reachable with few traversals on small games).")
flags.DEFINE_float("dcfr_learning_rate", 1e-4, "Deep CFR learning rate.")
flags.DEFINE_integer("dcfr_advantage_steps", 10, "Deep CFR advantage steps per player.")
flags.DEFINE_integer(
    "dcfr_policy_steps", 10,
    "Deep CFR average-strategy network steps per iteration.")
flags.DEFINE_boolean(
    "dcfr_train_strategy_each_iteration", False,
    "Retrain average policy every iteration; if false, only before eval checkpoints.")
flags.DEFINE_boolean(
    "dcfr_reinitialize_advantage_networks", False,
    "Reset advantage nets each iteration (canonical Deep CFR; off for short budgets).")
flags.DEFINE_integer("dcfr_max_turns", 10, "Deep CFR max turns (match game_params).")

# --- RL mixed-training flags ---
flags.DEFINE_float(
    "train_bot_mix", 0.5,
    "Probability of RL training episodes vs bot opponents (else self-play).")
flags.DEFINE_enum(
    "train_bot_opponent", "mixed", ["random", "heuristic", "mixed"],
    "Bot opponent type for mixed RL training.")
flags.DEFINE_boolean("round_robin", True, "Run round-robin before bracket.")

# --- NFSP flags ---
flags.DEFINE_integer("nfsp_replay_buffer_capacity", 50000, "NFSP replay buffer.")
flags.DEFINE_integer("nfsp_reservoir_buffer_capacity", 100000, "NFSP reservoir.")
flags.DEFINE_float("nfsp_anticipatory_param", 0.1, "NFSP anticipatory param.")
flags.DEFINE_integer("nfsp_batch_size", 64, "NFSP batch size.")
flags.DEFINE_integer("nfsp_min_buffer_size", 200, "NFSP min buffer.")
flags.DEFINE_integer("nfsp_learn_every", 32, "NFSP learn frequency.")
flags.DEFINE_string(
    "device", "auto",
    "PyTorch device for neural training (auto, cpu, cuda, cuda:0, ...).")


def run_compare(rng: np.random.RandomState) -> None:
  """Train every algorithm and evaluate each one against a random-bot baseline.

  Each trainable algorithm listed in ``FLAGS.algorithms`` is trained for
  ``FLAGS.train_episodes`` episodes; bot-only algorithms skip training.  All
  algorithms are then evaluated against a random-bot opponent and the per-algorithm
  win-rate summary is appended to a comparison table written to ``FLAGS.output_dir``.

  Args:
    rng: NumPy RandomState for training and evaluation reproducibility.
  """
  output_dir = ensure_output_dir(FLAGS.output_dir)
  rows = []
  for algo in map(normalize_algorithm, FLAGS.algorithms):
    validate_algorithm(algo)
    row = {"algorithm": algo}
    if algo in TRAINABLE_ALGOS:
      agents, log, _artifact = train_algorithm(
          algo, FLAGS.train_episodes, FLAGS.eval_every, FLAGS.eval_episodes, rng)
      save_training_log(log, output_dir)
      matchup = evaluate_team_matchup(
          algo, "random", FLAGS.eval_episodes, rng, team1_agents=agents)
    elif algo in BOT_ALGOS:
      matchup = evaluate_team_matchup(algo, "random", FLAGS.eval_episodes, rng)
    else:
      raise ValueError(f"Unsupported algorithm '{algo}' in compare mode.")
    row.update(matchup.summary())
    row["opponent"] = "random"
    rows.append(row)
  save_comparison_table(rows, output_dir)


def run_train(rng: np.random.RandomState) -> None:
  """Train a single algorithm and print a quick evaluation against a random bot.

  The algorithm is taken from ``FLAGS.algorithm``.  After training the model is
  evaluated for ``FLAGS.eval_episodes`` games versus a random bot and the summary
  statistics are printed to stdout.  The training log is persisted to
  ``FLAGS.output_dir``.

  Args:
    rng: NumPy RandomState for training and evaluation reproducibility.
  """
  output_dir = ensure_output_dir(FLAGS.output_dir)
  algo = validate_algorithm(FLAGS.algorithm, for_training=True)
  agents, log, _artifact = train_algorithm(
      algo, FLAGS.train_episodes, FLAGS.eval_every, FLAGS.eval_episodes, rng)
  save_training_log(log, output_dir)
  print(evaluate_team_matchup(
      algo, "random", FLAGS.eval_episodes, rng, team1_agents=agents).summary())


def run_evaluate(rng: np.random.RandomState) -> None:
  """Evaluate a head-to-head matchup between two named algorithm teams without training.

  Team algorithms are taken from ``FLAGS.team1_algo`` and ``FLAGS.team2_algo``.
  Results are printed as JSON to stdout and saved to ``FLAGS.output_dir`` as
  ``eval_<team1>_vs_<team2>.json``.

  Args:
    rng: NumPy RandomState for episode-sampling reproducibility.
  """
  output_dir = ensure_output_dir(FLAGS.output_dir)
  team1 = validate_algorithm(FLAGS.team1_algo)
  team2 = validate_algorithm(FLAGS.team2_algo)
  stats = evaluate_team_matchup(
      team1, team2, FLAGS.eval_episodes, rng).summary()
  stats["team1_algo"] = team1
  stats["team2_algo"] = team2
  print(json.dumps(stats, indent=2))
  path = os.path.join(
      output_dir,
      f"eval_{stats['team1_algo']}_vs_{stats['team2_algo']}.json")
  with open(path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)


def run_tournament(rng: np.random.RandomState) -> None:
  """Run a full training-plus-tournament experiment and generate a LaTeX report.

  Saves the experiment configuration, trains all algorithms in ``FLAGS.algorithms``,
  runs the optional round-robin (controlled by ``FLAGS.round_robin``) and the
  single-elimination bracket, then calls ``generate_latex_report`` to produce the
  thesis-ready ``.tex`` file inside ``FLAGS.output_dir``.

  Args:
    rng: NumPy RandomState for training and evaluation reproducibility.
  """
  output_dir = ensure_output_dir(FLAGS.output_dir)
  save_experiment_config(output_dir, seed=FLAGS.seed)
  run_full_tournament(
      FLAGS.algorithms, FLAGS.train_episodes, FLAGS.eval_every,
      FLAGS.eval_episodes, FLAGS.bracket_size, rng, output_dir)
  tex = generate_latex_report(output_dir)
  print(f"LaTeX report: {tex}")


def run_multi_seed(_rng: np.random.RandomState) -> None:
  """Run the full tournament independently for each seed, then aggregate results.

  Each seed listed in ``FLAGS.seeds`` gets its own ``seed_<N>`` subdirectory
  under ``FLAGS.results_root``.  After all per-seed runs complete,
  ``aggregate_multi_seed_results`` merges the individual JSON result files and
  ``generate_aggregated_latex_report`` writes the combined ``.tex`` report.

  Args:
    _rng: Unused; each seed creates its own ``np.random.RandomState`` from the
      seed value so results are fully deterministic per seed.
  """
  root = ensure_output_dir(FLAGS.results_root)
  seeds = [int(s) for s in FLAGS.seeds]
  retrain = getattr(FLAGS, "retrain_algorithms", None) or []
  print(f"Running {len(seeds)} seeds: {seeds} "
        f"(train_episodes={FLAGS.train_episodes})")
  if retrain:
    print(f"Partial retrain: {retrain} (others load from checkpoints)")
  for seed in seeds:
    output_dir = os.path.join(root, f"seed_{seed}")
    ensure_output_dir(output_dir)
    save_experiment_config(output_dir, seed=seed)
    print(f"\n========== Seed {seed} -> {output_dir} ==========")
    run_full_tournament(
        FLAGS.algorithms, FLAGS.train_episodes, FLAGS.eval_every,
        FLAGS.eval_episodes, FLAGS.bracket_size,
        np.random.RandomState(seed), output_dir)
    generate_latex_report(output_dir)
  agg = aggregate_multi_seed_results(root, seeds)
  tex = generate_aggregated_latex_report(root, agg)
  print(f"\nMulti-seed aggregation complete: {root}/aggregated_results.json")
  print(f"Aggregated LaTeX report: {tex}")


def run_aggregate(_rng: np.random.RandomState) -> None:
  """Aggregate results from existing per-seed directories without re-running any training.

  Reads the ``results.json`` files produced by previous ``multi_seed`` runs from
  ``seed_<N>`` subdirectories of ``FLAGS.results_root``, merges them, and
  regenerates the aggregated JSON and LaTeX report.  Useful for re-running
  post-processing after modifying report templates.

  Args:
    _rng: Unused; aggregation is deterministic given the on-disk result files.
  """
  root = ensure_output_dir(FLAGS.results_root)
  seeds = [int(s) for s in FLAGS.seeds]
  agg = aggregate_multi_seed_results(root, seeds)
  tex = generate_aggregated_latex_report(root, agg)
  print(f"Aggregated results: {root}/aggregated_results.json")
  print(f"Aggregated LaTeX report: {tex}")


def main(argv):
  """Entry point: print experiment summary and dispatch to the selected mode handler.

  Loads the game, prints a brief configuration summary (game name, player count,
  training episodes, compute device), initialises the global NumPy RNG, and calls
  the appropriate mode function — one of ``run_train``, ``run_evaluate``,
  ``run_compare``, ``run_tournament``, ``run_multi_seed``, or ``run_aggregate``.

  Args:
    argv: Unused positional arguments absorbed by ``absl.app.run``.
  """
  del argv
  game = load_game()
  print(f"Game: {game.get_type().short_name}, players={game.num_players()}")
  print(f"Train episodes per algorithm: {FLAGS.train_episodes}")
  print(f"Training device: {device_label(resolve_device(FLAGS.device))}")
  rng = np.random.RandomState(FLAGS.seed)
  start = time.time()
  modes = {
      "train": run_train,
      "evaluate": run_evaluate,
      "compare": run_compare,
      "tournament": run_tournament,
      "multi_seed": run_multi_seed,
      "aggregate": run_aggregate,
  }
  modes[FLAGS.mode](rng)
  print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
  app.run(main)
