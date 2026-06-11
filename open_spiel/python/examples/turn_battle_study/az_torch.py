"""PyTorch AlphaZero for Turn Battle.

OpenSpiel's C++ AlphaZero (``open_spiel/algorithms/alpha_zero_torch``) is a
LibTorch binary with no Python bindings and is not built here, so AlphaZero is
implemented in Python: a policy-value network guides OpenSpiel's MCTS
(``open_spiel/python/algorithms/mcts.py``) during self-play and evaluation.
"""

from __future__ import annotations

from typing import List, Tuple

from absl import flags
import numpy as np
import torch
from torch import nn

from open_spiel.python.algorithms import mcts
import pyspiel

FLAGS = flags.FLAGS


class AlphaZeroNet(nn.Module):
  """Shared-trunk policy + per-player value network."""

  def __init__(self, input_size: int, num_actions: int, num_players: int,
               hidden: int = 128):
    super().__init__()
    self._trunk = nn.Sequential(
        nn.Linear(input_size, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
    )
    self._policy_head = nn.Linear(hidden, num_actions)
    self._value_head = nn.Sequential(nn.Linear(hidden, num_players), nn.Tanh())

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = self._trunk(x)
    return self._policy_head(z), self._value_head(z)


class TorchAZEvaluator(mcts.Evaluator):
  """MCTS evaluator backed by the AlphaZero network."""

  def __init__(self, net: AlphaZeroNet, value_scale: float):
    self._net = net
    self._value_scale = value_scale

  def _infer(self, state):
    player = state.current_player()
    obs = np.asarray(state.information_state_tensor(player), dtype=np.float32)
    with torch.no_grad():
      logits, value = self._net(torch.from_numpy(obs).unsqueeze(0))
    return player, logits.squeeze(0).numpy(), value.squeeze(0).numpy()

  def evaluate(self, state):
    _, _, value = self._infer(state)
    return np.asarray(value, dtype=np.float64) * self._value_scale

  def prior(self, state):
    if state.is_chance_node():
      return state.chance_outcomes()
    player = state.current_player()
    _, logits, _ = self._infer(state)
    legal = state.legal_actions(player)
    masked = np.full(len(logits), -1e9, dtype=np.float32)
    masked[legal] = logits[legal]
    masked = masked - masked.max()
    exp = np.exp(masked)
    probs = exp / exp.sum()
    return [(a, float(probs[a])) for a in legal]


class AlphaZeroTorch:
  """Self-play AlphaZero trainer using OpenSpiel MCTS for search."""

  def __init__(self, game: pyspiel.Game, rng: np.random.RandomState):
    self._game = game
    self._rng = rng
    self._num_players = game.num_players()
    self._num_actions = game.num_distinct_actions()
    self._value_scale = max(abs(game.max_utility()), abs(game.min_utility()), 1.0)

    dummy = game.new_initial_state()
    while dummy.is_chance_node():
      dummy.apply_action(dummy.chance_outcomes()[0][0])
    self._input_size = len(dummy.information_state_tensor(dummy.current_player()))

    self._net = AlphaZeroNet(
        self._input_size, self._num_actions, self._num_players)
    self._optimizer = torch.optim.Adam(
        self._net.parameters(), lr=FLAGS.az_learning_rate)
    self._evaluator = TorchAZEvaluator(self._net, self._value_scale)
    self._buffer: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

  def _new_bot(self) -> mcts.MCTSBot:
    return mcts.MCTSBot(
        self._game, FLAGS.mcts_uct_c, FLAGS.mcts_simulations, self._evaluator,
        solve=True, random_state=self._rng, dont_return_chance_node=True)

  def _self_play_game(self, bot: mcts.MCTSBot) -> None:
    state = self._game.new_initial_state()
    history = []
    move = 0
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(self._rng.choice(outcomes, p=probs))
        continue
      player = state.current_player()
      root = bot.mcts_search(state)
      counts = np.zeros(self._num_actions, dtype=np.float32)
      for child in root.children:
        counts[child.action] = child.explore_count
      total = counts.sum()
      if total <= 0:
        legal = state.legal_actions(player)
        counts[legal] = 1.0
        total = counts.sum()
      target_policy = counts / total
      obs = np.asarray(
          state.information_state_tensor(player), dtype=np.float32)
      history.append((obs, target_policy))
      if move < FLAGS.az_temperature_drop:
        action = self._rng.choice(self._num_actions, p=target_policy)
      else:
        action = int(np.argmax(target_policy))
      state.apply_action(action)
      move += 1

    returns = np.asarray(state.returns(), dtype=np.float32) / self._value_scale
    for obs, target_policy in history:
      self._buffer.append((obs, target_policy, returns.copy()))
    if len(self._buffer) > FLAGS.az_replay_buffer_size:
      self._buffer = self._buffer[-FLAGS.az_replay_buffer_size:]

  def _optimize(self) -> float:
    if len(self._buffer) < FLAGS.az_batch_size:
      return 0.0
    idx = self._rng.choice(
        len(self._buffer), FLAGS.az_batch_size, replace=False)
    obs = torch.from_numpy(np.stack([self._buffer[i][0] for i in idx]))
    target_pi = torch.from_numpy(np.stack([self._buffer[i][1] for i in idx]))
    target_v = torch.from_numpy(np.stack([self._buffer[i][2] for i in idx]))

    logits, value = self._net(obs)
    log_probs = torch.log_softmax(logits, dim=1)
    policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
    value_loss = nn.functional.mse_loss(value, target_v)
    loss = policy_loss + value_loss

    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()
    return float(loss.item())

  def train_iteration(self) -> float:
    self._self_play_game(self._new_bot())
    return self._optimize()

  def make_bots(self) -> List[pyspiel.Bot]:
    return [self._new_bot() for _ in range(self._num_players)]
