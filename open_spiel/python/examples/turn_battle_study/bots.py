"""Bot implementations and RL adapters."""

from __future__ import annotations

from typing import List, Optional, Sequence

from absl import flags
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.bots import uniform_random
from open_spiel.python.examples.turn_battle_study.config import (
    EVAL_FALLBACK_ALGOS,
    normalize_algorithm,
)
from open_spiel.python.examples.turn_battle_study.game import load_turn_based_game
from open_spiel.python.pytorch import deep_cfr
import pyspiel

FLAGS = flags.FLAGS


class HeuristicBattleBot(pyspiel.Bot):
  DEFENDERS = frozenset({0, 2})

  def __init__(self, player_id: int, rng: np.random.RandomState):
    super().__init__()
    self._player_id = player_id
    self._rng = rng

  def restart_at(self, state):
    del state

  def player_id(self):
    return self._player_id

  def step(self, state):
    player = self._player_id
    legal = state.legal_actions(player)
    if not legal:
      return pyspiel.INVALID_ACTION
    if len(legal) == 1:
      return legal[0]
    if 3 in legal and self._rng.random() < 0.35:
      return 3
    attack_prefs = [0, 1] if player not in self.DEFENDERS else [1, 0]
    for action in attack_prefs:
      if action in legal:
        return action
    if 2 in legal and self._rng.random() < 0.25:
      return 2
    return self._rng.choice(legal)


def _time_step_from_turn_based_state(state: pyspiel.State) -> rl_environment.TimeStep:
  sim = state.simultaneous_game_state()
  n = sim.num_players()
  info_state = [list(sim.information_state_tensor(p)) for p in range(n)]
  legal_actions = [list(sim.legal_actions(p)) for p in range(n)]
  step_type = (
      rl_environment.StepType.LAST if state.is_terminal()
      else rl_environment.StepType.MID)
  return rl_environment.TimeStep(
      observations={
          "info_state": info_state,
          "legal_actions": legal_actions,
          "current_player": state.current_player(),
          "serialized_state": None,
      },
      rewards=None,
      discounts=None,
      step_type=step_type,
  )


class RlAgentTurnBasedBot(pyspiel.Bot):
  """Runs a trained RL agent on the inner simultaneous state of a turn-based game."""

  def __init__(self, agent, player_id: int):
    super().__init__()
    self._agent = agent
    self._player_id = player_id

  def restart_at(self, state):
    del state

  def player_id(self):
    return self._player_id

  def step(self, state):
    if state.is_chance_node() or state.is_terminal():
      return pyspiel.INVALID_ACTION
    ts = _time_step_from_turn_based_state(state)
    return self._agent.step(ts, is_evaluation=True).action


class BotRlAdapter:
  def __init__(self, bot: pyspiel.Bot, player_id: int):
    self._bot = bot
    self._player_id = player_id

  def step(self, time_step, is_evaluation=True):
    del is_evaluation
    from open_spiel.python import rl_agent
    legal = time_step.observations["legal_actions"][self._player_id]
    if time_step.last() or not legal:
      return rl_agent.StepOutput(action=0, probs=[])
    serialized = time_step.observations["serialized_state"]
    if serialized:
      _, state = pyspiel.deserialize_game_and_state(serialized)
      self._bot.restart_at(state)
      action = self._bot.step(state)
    else:
      action = np.random.choice(legal)
    probs = np.zeros(max(legal) + 1)
    probs[action] = 1.0
    return rl_agent.StepOutput(action=action, probs=probs)


def create_pyspiel_bot(
    algorithm: str,
    game: pyspiel.Game,
    player_id: int,
    rng: np.random.RandomState,
    for_mcts: bool = False,
) -> pyspiel.Bot:
  algo = normalize_algorithm(algorithm)
  if algo == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  if algo == "mcts":
    # MCTS is available in C++ (open_spiel/algorithms/mcts.cc); use it.
    mcts_game = load_turn_based_game() if for_mcts else game
    seed = int(rng.randint(0, 2**31 - 1))
    evaluator = pyspiel.RandomRolloutEvaluator(FLAGS.mcts_rollouts, seed)
    return pyspiel.MCTSBot(
        mcts_game, evaluator, FLAGS.mcts_uct_c, FLAGS.mcts_simulations, -1,
        True, int(rng.randint(0, 2**31 - 1)), False,
        pyspiel.ChildSelectionPolicy.PUCT)
  if algo == "heuristic":
    return HeuristicBattleBot(player_id, rng)
  if algo in EVAL_FALLBACK_ALGOS:
    return create_pyspiel_bot(
        EVAL_FALLBACK_ALGOS[algo], game, player_id, rng, for_mcts=for_mcts)
  raise ValueError(f"Algorithm '{algorithm}' is not a pyspiel bot.")


class DeepCFRPolicyBot(pyspiel.Bot):
  """Plays the Deep CFR average strategy network on a turn-based state."""

  def __init__(self, player_id: int, rng: np.random.RandomState,
               solver: deep_cfr.DeepCFRSolver):
    super().__init__()
    self._player_id = player_id
    self._rng = rng
    self._solver = solver

  def restart_at(self, state):
    del state

  def player_id(self):
    return self._player_id

  def step(self, state):
    if state.is_chance_node() or state.is_terminal():
      return pyspiel.INVALID_ACTION
    legal = state.legal_actions(self._player_id)
    if not legal:
      return pyspiel.INVALID_ACTION
    probs = self._solver.action_probabilities(state)
    weights = np.array([probs.get(a, 0.0) for a in legal], dtype=np.float64)
    total = weights.sum()
    if total <= 0:
      return self._rng.choice(legal)
    return self._rng.choice(legal, p=weights / total)


def deep_cfr_bots(
    solver: deep_cfr.DeepCFRSolver,
    rng: np.random.RandomState,
    use_policy_network: bool = True,
) -> List[pyspiel.Bot]:
  # DeepCFRSolver.action_probabilities uses the trained average strategy
  # network; the flag is kept only for call-site API compatibility.
  del use_policy_network
  return [DeepCFRPolicyBot(p, rng, solver) for p in range(solver._num_players)]


def bots_to_adapters(bots: Sequence[pyspiel.Bot]) -> List[BotRlAdapter]:
  return [BotRlAdapter(bot, i) for i, bot in enumerate(bots)]


def trained_bot_game(agents: Optional[Sequence]) -> Optional[pyspiel.Game]:
  if not agents or not hasattr(agents[0], "_bot"):
    return None
  bot = agents[0]._bot
  if isinstance(bot, DeepCFRPolicyBot):
    return bot._solver._game
  return None
