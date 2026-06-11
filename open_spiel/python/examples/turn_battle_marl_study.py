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

Modes: train, evaluate, compare, tournament (train all + bracket + LaTeX).
"""

from __future__ import annotations

import json
import os
import time

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.examples.turn_battle_study.config import (
    ALL_ALGORITHMS,
    BOT_ALGOS,
    TRAINABLE_ALGOS,
    normalize_algorithm,
)
from open_spiel.python.examples.turn_battle_study.evaluation import (
    evaluate_team_matchup,
)
from open_spiel.python.examples.turn_battle_study.game import load_game
from open_spiel.python.examples.turn_battle_study.report import generate_latex_report
from open_spiel.python.examples.turn_battle_study.storage import (
    ensure_output_dir,
    save_comparison_table,
    save_training_log,
)
from open_spiel.python.examples.turn_battle_study.tournament import run_full_tournament
from open_spiel.python.examples.turn_battle_study.trainers import train_algorithm

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "turn_battle", "OpenSpiel game string.")
flags.DEFINE_string("game_params", "num_turns=15", "Comma-separated game parameters.")
flags.DEFINE_enum(
    "mode", "tournament",
    ["train", "evaluate", "compare", "tournament"],
    "Experiment mode.")
flags.DEFINE_list("algorithms", ALL_ALGORITHMS, "Algorithms to compare/tournament.")
flags.DEFINE_string("algorithm", "qpg", "Algorithm for train mode.")
flags.DEFINE_string("team1_algo", "qpg", "Algorithm for team 1 (players 0,1).")
flags.DEFINE_string("team2_algo", "random", "Algorithm for team 2 (players 2,3).")
flags.DEFINE_integer("train_episodes", 3000, "Training episodes.")
flags.DEFINE_integer("eval_episodes", 100, "Evaluation episodes per matchup.")
flags.DEFINE_integer("eval_every", 500, "Evaluate during training every N episodes.")
flags.DEFINE_integer("bracket_size", 8, "Bracket field size (top-N by seeding).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("output_dir", "turn_battle_results", "Output directory.")
flags.DEFINE_integer("mcts_simulations", 50, "MCTS simulations per move.")
flags.DEFINE_float("mcts_uct_c", 2.0, "MCTS UCT exploration constant.")
flags.DEFINE_integer("mcts_rollouts", 1, "Random rollouts per MCTS evaluation.")
flags.DEFINE_integer("az_replay_buffer_size", 4096, "AlphaZero replay buffer.")
flags.DEFINE_integer("az_batch_size", 64, "AlphaZero batch size.")
flags.DEFINE_float("az_learning_rate", 1e-3, "AlphaZero learning rate.")
flags.DEFINE_float("az_temperature", 1.0, "AlphaZero self-play temperature.")
flags.DEFINE_integer("az_temperature_drop", 4, "Greedy after N moves.")
flags.DEFINE_integer("dcfr_traversals", 3, "Deep CFR traversals per iteration.")
flags.DEFINE_integer("dcfr_batch_size", 128, "Deep CFR batch size.")
flags.DEFINE_float("dcfr_learning_rate", 1e-3, "Deep CFR learning rate.")
flags.DEFINE_integer("dcfr_advantage_steps", 3, "Deep CFR advantage steps.")
flags.DEFINE_integer("dcfr_policy_steps", 25, "Deep CFR policy steps.")
flags.DEFINE_integer("dcfr_max_turns", 15, "Deep CFR max turns (match game_params).")
flags.DEFINE_integer("az_train_episodes", 500, "AlphaZero episodes in tournament.")
flags.DEFINE_integer("dcfr_iterations", 200, "Deep CFR iterations in tournament.")
flags.DEFINE_boolean("round_robin", True, "Run round-robin before bracket.")
flags.DEFINE_integer("nfsp_replay_buffer_capacity", 50000, "NFSP replay buffer.")
flags.DEFINE_integer("nfsp_reservoir_buffer_capacity", 100000, "NFSP reservoir.")
flags.DEFINE_float("nfsp_anticipatory_param", 0.1, "NFSP anticipatory param.")
flags.DEFINE_integer("nfsp_batch_size", 64, "NFSP batch size.")
flags.DEFINE_integer("nfsp_min_buffer_size", 200, "NFSP min buffer.")
flags.DEFINE_integer("nfsp_learn_every", 32, "NFSP learn frequency.")


def run_compare(rng: np.random.RandomState) -> None:
  output_dir = ensure_output_dir(FLAGS.output_dir)
  rows = []
  for algo in map(normalize_algorithm, FLAGS.algorithms):
    row = {"algorithm": algo}
    if algo in TRAINABLE_ALGOS:
      agents, log = train_algorithm(
          algo, FLAGS.train_episodes, FLAGS.eval_every, FLAGS.eval_episodes, rng)
      save_training_log(log, output_dir)
      matchup = evaluate_team_matchup(
          algo, "random", FLAGS.eval_episodes, rng, team1_agents=agents)
    elif algo in BOT_ALGOS:
      matchup = evaluate_team_matchup(algo, "random", FLAGS.eval_episodes, rng)
    else:
      continue
    row.update(matchup.summary())
    row["opponent"] = "random"
    rows.append(row)
  save_comparison_table(rows, output_dir)


def run_train(rng: np.random.RandomState) -> None:
  output_dir = ensure_output_dir(FLAGS.output_dir)
  algo = normalize_algorithm(FLAGS.algorithm)
  agents, log = train_algorithm(
      algo, FLAGS.train_episodes, FLAGS.eval_every, FLAGS.eval_episodes, rng)
  save_training_log(log, output_dir)
  print(evaluate_team_matchup(
      algo, "random", FLAGS.eval_episodes, rng, team1_agents=agents).summary())


def run_evaluate(rng: np.random.RandomState) -> None:
  output_dir = ensure_output_dir(FLAGS.output_dir)
  stats = evaluate_team_matchup(
      FLAGS.team1_algo, FLAGS.team2_algo, FLAGS.eval_episodes, rng).summary()
  stats["team1_algo"] = normalize_algorithm(FLAGS.team1_algo)
  stats["team2_algo"] = normalize_algorithm(FLAGS.team2_algo)
  print(json.dumps(stats, indent=2))
  path = os.path.join(
      output_dir,
      f"eval_{stats['team1_algo']}_vs_{stats['team2_algo']}.json")
  with open(path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)


def run_tournament(rng: np.random.RandomState) -> None:
  output_dir = ensure_output_dir(FLAGS.output_dir)
  run_full_tournament(
      FLAGS.algorithms, FLAGS.train_episodes, FLAGS.eval_every,
      FLAGS.eval_episodes, FLAGS.bracket_size, rng, output_dir)
  tex = generate_latex_report(output_dir)
  print(f"LaTeX report: {tex}")


def main(argv):
  del argv
  game = load_game()
  print(f"Game: {game.get_type().short_name}, players={game.num_players()}")
  rng = np.random.RandomState(FLAGS.seed)
  start = time.time()
  modes = {
      "train": run_train,
      "evaluate": run_evaluate,
      "compare": run_compare,
      "tournament": run_tournament,
  }
  modes[FLAGS.mode](rng)
  print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
  app.run(main)
