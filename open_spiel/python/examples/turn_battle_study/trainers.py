"""Training loops.

Each algorithm uses the correct available implementation. MCTS exists in C++
(open_spiel/algorithms/mcts.cc) and is used for the plain search opponent. The
learning algorithms have no usable C++ Python bindings, so they use the Python
implementations:
  * alphazero -> C++ LibTorch AlphaZero on the 2-team game view (az_cpp.py).
  * deep_cfr  -> PyTorch ``DeepCFRSolver`` (open_spiel/python/pytorch/deep_cfr.py).
  * nfsp      -> PyTorch ``NFSP`` (open_spiel/python/pytorch/nfsp.py).
  * qpg       -> PyTorch ``PolicyGradient`` (open_spiel/python/pytorch/policy_gradient.py).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from absl import flags
import numpy as np

from open_spiel.python.examples.turn_battle_study.agents import create_rl_agents
from open_spiel.python.examples.turn_battle_study.az_cpp import AlphaZeroCpp
from open_spiel.python.examples.turn_battle_study.bots import (
    DeepCFRPolicyBot,
    bots_to_adapters,
)
from open_spiel.python.examples.turn_battle_study.config import normalize_algorithm
from open_spiel.python.examples.turn_battle_study.device import (
    device_label,
    resolve_cpp_az_devices,
    resolve_device,
)
from open_spiel.python.examples.turn_battle_study.evaluation import (
    evaluate_team_matchup,
    play_episode_rl,
    play_training_episode_rl,
)
from open_spiel.python.examples.turn_battle_study.game import (
    load_game,
    load_turn_based_game,
    make_rl_environment,
    parse_num_turns,
)
from open_spiel.python.examples.turn_battle_study.role_shared import (
    RoleSharedTeam,
    agents_for_matchup,
)
from open_spiel.python.examples.turn_battle_study.models import TrainingLog
from open_spiel.python.pytorch import deep_cfr

FLAGS = flags.FLAGS


def _dcfr_turns() -> int:
  """Return the number of turns for the Deep CFR game, capped at the CFR limit."""
  return parse_num_turns(dcfr_cap=True)


def _fixed_role_win_rate(
    algo: str, agents, eval_eps: int, rng: np.random.RandomState) -> Tuple[float, float]:
  """Evaluate team1 agents against a random opponent and return win rates.

  Args:
    algo: Algorithm label passed to evaluate_team_matchup.
    agents: Trained agents whose team1 view is extracted for evaluation.
    eval_eps: Number of evaluation episodes to run.
    rng: Random state for episode sampling.

  Returns:
    A (team1_win_rate, team2_win_rate) tuple from the matchup summary.
  """
  team1 = agents_for_matchup(agents, for_team1=True)
  stats = evaluate_team_matchup(
      algo, "random", eval_eps, rng, team1_agents=team1).summary()
  return stats["team1_win_rate"], stats["team2_win_rate"]


def _log_checkpoint(
    log: TrainingLog, step: int, win: float, loss: float, label: str) -> None:
  """Append a training checkpoint entry to log and print a progress line.

  Args:
    log: TrainingLog accumulator to update in-place.
    step: Training step (episode or iteration) index.
    win: Team1 win rate at this step.
    loss: Team2 win rate at this step.
    label: Human-readable algorithm label for the progress line.
  """
  log.episodes.append(step)
  log.team1_win_rate.append(win)
  log.team2_win_rate.append(loss)
  print(f"[train {label}] step {step}: win={win:.3f}")


def _training_device() -> str:
  """Resolve and return the training device string from FLAGS."""
  return str(resolve_device(FLAGS.device))


def train_alphazero(episodes, eval_every, eval_eps, rng):
  """Train an AlphaZero agent via the C++ LibTorch backend.

  Args:
    episodes: Total number of training episodes.
    eval_every: Evaluate win rate every this many episodes.
    eval_eps: Number of episodes per evaluation rollout.
    rng: Random state for evaluation sampling.

  Returns:
    A (agents, log, trainer) tuple where agents is the list of bot adapters,
    log is the populated TrainingLog, and trainer is the AlphaZeroCpp instance.
  """
  print(
      "  backend: C++ LibTorch AlphaZero (turn_battle_teams 2-player view) on "
      f"{device_label(resolve_device(FLAGS.device))} "
      f"(train --devices={resolve_cpp_az_devices(FLAGS.device)})"
  )
  trainer = AlphaZeroCpp(rng)
  log = TrainingLog(algorithm="alphazero")

  def _eval_at_step(step: int) -> None:
    agents = bots_to_adapters(trainer.make_bots())
    win, loss = _fixed_role_win_rate("alphazero", agents, eval_eps, rng)
    _log_checkpoint(log, step, win, loss, "alphazero")

  trainer.train(episodes, eval_every, _eval_at_step)
  return bots_to_adapters(trainer.make_bots()), log, trainer


def _deep_cfr_bots(solver, game, rng):
  """Wrap a trained DeepCFRSolver as a list of per-player DeepCFRPolicyBots.

  Args:
    solver: Trained DeepCFRSolver instance.
    game: The game used during solver training.
    rng: Random state forwarded to each bot.

  Returns:
    List of DeepCFRPolicyBot, one per player.
  """
  return [DeepCFRPolicyBot(p, rng, solver) for p in range(game.num_players())]


def build_deep_cfr_solver(rng: np.random.RandomState):
  """Construct the DeepCFRSolver and its game; shared by training and checkpoint loading."""
  del rng  # seed comes from FLAGS.seed
  game = load_turn_based_game(num_turns=_dcfr_turns())
  solver = deep_cfr.DeepCFRSolver(
      game,
      policy_network_layers=(128, 128),
      advantage_network_layers=(128, 128),
      num_iterations=1,
      num_traversals=FLAGS.dcfr_traversals,
      learning_rate=FLAGS.dcfr_learning_rate,
      batch_size_advantage=FLAGS.dcfr_batch_size,
      batch_size_strategy=FLAGS.dcfr_batch_size,
      policy_network_train_steps=FLAGS.dcfr_policy_steps,
      advantage_network_train_steps=FLAGS.dcfr_advantage_steps,
      reinitialize_advantage_networks=FLAGS.dcfr_reinitialize_advantage_networks,
      device=_training_device(),
      seed=FLAGS.seed,
  )
  return game, solver


def train_deep_cfr(episodes, eval_every, eval_eps, rng):
  """Train a Deep CFR solver for the turn-battle game.

  Runs the CFR traversal loop manually so per-iteration callbacks can record
  evaluation checkpoints.

  Args:
    episodes: Number of CFR iterations to run.
    eval_every: Evaluate win rate every this many iterations.
    eval_eps: Number of episodes per evaluation rollout.
    rng: Random state for evaluation sampling.

  Returns:
    A (agents, log, solver) tuple where agents is the list of bot adapters,
    log is the populated TrainingLog, and solver is the DeepCFRSolver instance.
  """
  print(
      f"  backend: PyTorch DeepCFRSolver on {device_label(resolve_device(FLAGS.device))} "
      f"(traversals={FLAGS.dcfr_traversals}, batch={FLAGS.dcfr_batch_size})"
  )
  game, solver = build_deep_cfr_solver(rng)
  log = TrainingLog(algorithm="deep_cfr")

  for it in range(1, episodes + 1):
    for player in range(solver._num_players):
      for _ in range(FLAGS.dcfr_traversals):
        solver._traverse_game_tree(solver._root_node, player)
      if solver._reinitialize_advantage_networks:
        solver._reinitialize_advantage_network(player)
      solver._learn_advantage_network(player)
    solver._iteration += 1
    at_eval = (it % eval_every == 0)
    if FLAGS.dcfr_train_strategy_each_iteration or at_eval:
      solver._learn_strategy_network()
    if at_eval:
      agents = bots_to_adapters(_deep_cfr_bots(solver, game, rng))
      win, loss = _fixed_role_win_rate("deep_cfr", agents, eval_eps, rng)
      _log_checkpoint(log, it, win, loss, "deep_cfr")

  solver._learn_strategy_network()
  agents = bots_to_adapters(_deep_cfr_bots(solver, game, rng))
  if not log.episodes or log.episodes[-1] != episodes:
    win, loss = _fixed_role_win_rate("deep_cfr", agents, eval_eps, rng)
    _log_checkpoint(log, episodes, win, loss, "deep_cfr")
  return agents, log, solver


def _train_rl(algo, episodes, eval_every, eval_eps, rng):
  """Training loop for RL algorithms that use role-shared teams (nfsp, qpg).

  Args:
    algo: Algorithm key (e.g. 'nfsp' or 'qpg').
    episodes: Total number of training episodes.
    eval_every: Evaluate win rate every this many episodes.
    eval_eps: Number of episodes per evaluation rollout.
    rng: Random state for evaluation sampling.

  Returns:
    A (agents, log, None) tuple where agents is the trained RoleSharedTeam and
    log is the populated TrainingLog.
  """
  key = normalize_algorithm(algo)
  role_note = ", role-shared" if key in {"nfsp", "qpg"} else ""
  print(
      f"  backend: PyTorch {key} on {device_label(resolve_device(FLAGS.device))} "
      f"(bot_mix={FLAGS.train_bot_mix:.0%}, opponent={FLAGS.train_bot_opponent}"
      f"{role_note})")
  env = make_rl_environment(include_full_state=FLAGS.train_bot_mix > 0)
  game = load_game()
  agents = create_rl_agents(algo, env)
  log = TrainingLog(algorithm=key)
  train_agents = agents.facades if isinstance(agents, RoleSharedTeam) else agents
  for ep in range(episodes):
    play_training_episode_rl(env, train_agents, rng, ep, game)
    if isinstance(agents, RoleSharedTeam):
      agents.sync_role_weights()
    if ep > 0 and ep % eval_every == 0:
      win, loss = _fixed_role_win_rate(key, agents, eval_eps, rng)
      _log_checkpoint(log, ep, win, loss, key)
  if not log.episodes or log.episodes[-1] != episodes:
    win, loss = _fixed_role_win_rate(key, agents, eval_eps, rng)
    _log_checkpoint(log, episodes, win, loss, key)
  return agents, log, None


def train_algorithm(
    algo, episodes, eval_every, eval_eps, rng
) -> Tuple[Any, TrainingLog, Optional[Any]]:
  """Dispatch training to the correct backend for the given algorithm.

  Args:
    algo: Algorithm name (e.g. 'alphazero', 'deep_cfr', 'nfsp', 'qpg', or any
      other RL algorithm accepted by create_rl_agents).
    episodes: Total training episodes or CFR iterations.
    eval_every: Evaluate win rate every this many steps.
    eval_eps: Number of episodes per evaluation rollout.
    rng: Random state for evaluation sampling.

  Returns:
    A (agents, log, artifact) tuple.  artifact is an AlphaZeroCpp or
    DeepCFRSolver instance for algorithms that need it for checkpointing,
    and None for plain RL algorithms.
  """
  key = normalize_algorithm(algo)
  if key == "alphazero":
    return train_alphazero(episodes, eval_every, eval_eps, rng)
  if key == "deep_cfr":
    return train_deep_cfr(episodes, eval_every, eval_eps, rng)
  if key in {"nfsp", "qpg"}:
    return _train_rl(key, episodes, eval_every, eval_eps, rng)

  env = make_rl_environment(include_full_state=FLAGS.train_bot_mix > 0)
  game = load_game()
  agents = create_rl_agents(algo, env)
  log = TrainingLog(algorithm=key)
  for ep in range(episodes):
    play_training_episode_rl(env, agents, rng, ep, game)
    if ep > 0 and ep % eval_every == 0:
      win, loss = _fixed_role_win_rate(key, agents, eval_eps, rng)
      _log_checkpoint(log, ep, win, loss, key)
  if not log.episodes or log.episodes[-1] != episodes:
    win, loss = _fixed_role_win_rate(key, agents, eval_eps, rng)
    _log_checkpoint(log, episodes, win, loss, key)
  return agents, log, None
