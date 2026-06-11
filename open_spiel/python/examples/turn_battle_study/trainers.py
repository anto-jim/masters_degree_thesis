"""Training loops.

Each algorithm uses the correct available implementation. MCTS exists in C++
(open_spiel/algorithms/mcts.cc) and is used for the plain search opponent. The
learning algorithms have no usable C++ Python bindings, so they use the Python
implementations:
  * alphazero -> PyTorch policy-value net guiding OpenSpiel MCTS (az_torch.py).
  * deep_cfr  -> PyTorch ``DeepCFRSolver`` (open_spiel/python/pytorch/deep_cfr.py).
  * nfsp      -> PyTorch ``NFSP`` (open_spiel/python/pytorch/nfsp.py).
  * qpg       -> PyTorch ``PolicyGradient`` (open_spiel/python/pytorch/policy_gradient.py).
"""

from __future__ import annotations

from typing import Tuple

from absl import flags
import numpy as np

from open_spiel.python.examples.turn_battle_study.agents import create_rl_agents
from open_spiel.python.examples.turn_battle_study.az_torch import AlphaZeroTorch
from open_spiel.python.examples.turn_battle_study.bots import (
    DeepCFRPolicyBot,
    bots_to_adapters,
)
from open_spiel.python.examples.turn_battle_study.config import normalize_algorithm
from open_spiel.python.examples.turn_battle_study.device import device_label, resolve_device
from open_spiel.python.examples.turn_battle_study.evaluation import (
    evaluate_team_matchup,
    play_episode_rl,
    play_training_episode_rl,
)
from open_spiel.python.examples.turn_battle_study.game import (
    load_game,
    load_turn_based_game,
    make_rl_environment,
    parse_game_params,
)
from open_spiel.python.examples.turn_battle_study.models import TrainingLog
from open_spiel.python.pytorch import deep_cfr

FLAGS = flags.FLAGS


def _dcfr_turns() -> int:
  params = parse_game_params(FLAGS.game_params)
  return min(int(params.get("num_turns", 5)), FLAGS.dcfr_max_turns)


def _bidirectional_win_rate(
    algo: str, agents, eval_eps: int, rng: np.random.RandomState) -> Tuple[float, float]:
  half = max(1, eval_eps // 2)
  fwd = evaluate_team_matchup(
      algo, "random", half, rng, team1_agents=agents).summary()
  rev = evaluate_team_matchup(
      "random", algo, eval_eps - half, rng, team2_agents=agents).summary()
  win = (fwd["team1_win_rate"] + rev["team2_win_rate"]) / 2
  loss = (fwd["team2_win_rate"] + rev["team1_win_rate"]) / 2
  return win, loss


def _log_checkpoint(
    log: TrainingLog, step: int, win: float, loss: float, label: str) -> None:
  log.episodes.append(step)
  log.team1_win_rate.append(win)
  log.team2_win_rate.append(loss)
  print(f"[train {label}] step {step}: win={win:.3f}")


def _training_device() -> str:
  return str(resolve_device(FLAGS.device))


def train_alphazero(episodes, eval_every, eval_eps, rng):
  print(f"  backend: PyTorch AlphaZero on {device_label(resolve_device(FLAGS.device))}")
  game = load_turn_based_game()
  trainer = AlphaZeroTorch(game, rng, device=FLAGS.device)
  log = TrainingLog(algorithm="alphazero")
  for it in range(1, episodes + 1):
    trainer.train_iteration()
    if it % eval_every == 0:
      agents = bots_to_adapters(trainer.make_bots())
      win, loss = _bidirectional_win_rate("alphazero", agents, eval_eps, rng)
      _log_checkpoint(log, it, win, loss, "alphazero")
  return bots_to_adapters(trainer.make_bots()), log


def _deep_cfr_bots(solver, game, rng):
  return [DeepCFRPolicyBot(p, rng, solver) for p in range(game.num_players())]


def train_deep_cfr(episodes, eval_every, eval_eps, rng):
  print(f"  backend: PyTorch DeepCFRSolver on {device_label(resolve_device(FLAGS.device))}")
  game = load_turn_based_game(num_turns=_dcfr_turns())
  solver = deep_cfr.DeepCFRSolver(
      game,
      policy_network_layers=(128, 128),
      advantage_network_layers=(128, 128),
      num_iterations=max(1, episodes),
      num_traversals=FLAGS.dcfr_traversals,
      learning_rate=FLAGS.dcfr_learning_rate,
      batch_size_advantage=FLAGS.dcfr_batch_size,
      batch_size_strategy=FLAGS.dcfr_batch_size,
      policy_network_train_steps=FLAGS.dcfr_policy_steps,
      advantage_network_train_steps=FLAGS.dcfr_advantage_steps,
      device=_training_device(),
      seed=FLAGS.seed,
  )
  log = TrainingLog(algorithm="deep_cfr")

  for it in range(1, episodes + 1):
    for player in range(solver._num_players):
      for _ in range(FLAGS.dcfr_traversals):
        solver._traverse_game_tree(solver._root_node, player)
      if solver._reinitialize_advantage_networks:
        solver._reinitialize_advantage_network(player)
      solver._learn_advantage_network(player)
    solver._iteration += 1
    if it % eval_every == 0:
      agents = bots_to_adapters(_deep_cfr_bots(solver, game, rng))
      win, loss = _bidirectional_win_rate("deep_cfr", agents, eval_eps, rng)
      _log_checkpoint(log, it, win, loss, "deep_cfr")

  solver._learn_strategy_network()
  agents = bots_to_adapters(_deep_cfr_bots(solver, game, rng))
  if not log.episodes or log.episodes[-1] != episodes:
    win, loss = _bidirectional_win_rate("deep_cfr", agents, eval_eps, rng)
    _log_checkpoint(log, episodes, win, loss, "deep_cfr")
  return agents, log


def _train_rl(algo, episodes, eval_every, eval_eps, rng):
  key = normalize_algorithm(algo)
  print(
      f"  backend: PyTorch {key} on {device_label(resolve_device(FLAGS.device))} "
      f"(bot_mix={FLAGS.train_bot_mix:.0%}, opponent={FLAGS.train_bot_opponent})")
  env = make_rl_environment()
  game = load_game()
  agents = create_rl_agents(algo, env)
  log = TrainingLog(algorithm=key)
  for ep in range(episodes):
    play_training_episode_rl(env, agents, rng, ep, game)
    if ep > 0 and ep % eval_every == 0:
      win, loss = _bidirectional_win_rate(key, agents, eval_eps, rng)
      _log_checkpoint(log, ep, win, loss, key)
  return agents, log


def train_nfsp(episodes, eval_every, eval_eps, rng):
  return _train_rl("nfsp", episodes, eval_every, eval_eps, rng)


def train_qpg(episodes, eval_every, eval_eps, rng):
  return _train_rl("qpg", episodes, eval_every, eval_eps, rng)


def train_algorithm(algo, episodes, eval_every, eval_eps, rng):
  key = normalize_algorithm(algo)
  if key == "alphazero":
    return train_alphazero(episodes, eval_every, eval_eps, rng)
  if key == "deep_cfr":
    return train_deep_cfr(episodes, eval_every, eval_eps, rng)
  if key == "nfsp":
    return train_nfsp(episodes, eval_every, eval_eps, rng)
  if key == "qpg":
    return train_qpg(episodes, eval_every, eval_eps, rng)

  env = make_rl_environment()
  game = load_game()
  agents = create_rl_agents(algo, env)
  log = TrainingLog(algorithm=key)
  for ep in range(episodes):
    play_training_episode_rl(env, agents, rng, ep, game)
    if ep > 0 and ep % eval_every == 0:
      win, loss = _bidirectional_win_rate(key, agents, eval_eps, rng)
      _log_checkpoint(log, ep, win, loss, key)
  return agents, log
