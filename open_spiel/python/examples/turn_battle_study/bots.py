"""Bot implementations and RL adapters."""

from __future__ import annotations

from typing import List, Optional, Sequence

from absl import flags
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.bots import uniform_random
from open_spiel.python.examples.turn_battle_study.config import (
    DEFENDER_SEATS,
    EVAL_FALLBACK_ALGOS,
    normalize_algorithm,
)
from open_spiel.python.examples.turn_battle_study.game import load_turn_based_game
from open_spiel.python.pytorch import deep_cfr
import pyspiel

FLAGS = flags.FLAGS


class HeuristicBattleBot(pyspiel.Bot):
  """Rule-based bot that prefers offensive actions and occasionally heals.

  Defenders bias toward the attacker action (index 1); attackers prefer
  action 0. The special action (index 3) is chosen with 35% probability
  when legal, and heal (action 2) is chosen with 25% probability.
  """

  DEFENDERS = frozenset(DEFENDER_SEATS)

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
  """Build an rl_environment.TimeStep from a turn-based pyspiel state.

  Extracts observations and legal actions from the inner simultaneous
  game state. Rewards and discounts are left as None; the step type is
  LAST when the state is terminal, MID otherwise.

  Args:
    state: A turn-based pyspiel.State wrapping a simultaneous game.

  Returns:
    A TimeStep populated with info_state, legal_actions, current_player,
    and serialized_state (set to None).
  """
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
  """Wraps a pyspiel.Bot to satisfy the RL agent step() interface.

  Deserializes the game state from the time-step observation and
  delegates action selection to the underlying bot.
  """

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
  """Construct a pyspiel.Bot for the given algorithm and player seat.

  Args:
    algorithm: Bot algorithm name (``"random"``, ``"mcts"``,
      ``"heuristic"``, or any key in EVAL_FALLBACK_ALGOS).
    game: The pyspiel.Game the bot will play.
    player_id: Seat index this bot will occupy.
    rng: Random-number generator for stochastic decisions.
    for_mcts: When True, loads the turn-based game for MCTS tree search
      instead of using the supplied game directly.

  Returns:
    A pyspiel.Bot instance configured for the requested algorithm.
  """
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


def bots_to_adapters(bots: Sequence[pyspiel.Bot]) -> List[BotRlAdapter]:
  """Wrap each bot in a BotRlAdapter indexed by its position.

  Args:
    bots: Sequence of pyspiel.Bot instances.

  Returns:
    A list of BotRlAdapter objects, one per bot, with player_id equal
    to the bot's index in the sequence.
  """
  return [BotRlAdapter(bot, i) for i, bot in enumerate(bots)]


def trained_bot_game(agents: Optional[Sequence]) -> Optional[pyspiel.Game]:
  """Return the pyspiel.Game embedded in a DeepCFR bot agent, if any.

  Args:
    agents: A sequence of agent-like objects to inspect.

  Returns:
    The game stored in the first agent's DeepCFRPolicyBot, or None if
    the agents list is empty or the first agent does not wrap a
    DeepCFRPolicyBot.
  """
  if not agents or not hasattr(agents[0], "_bot"):
    return None
  bot = agents[0]._bot
  if isinstance(bot, DeepCFRPolicyBot):
    return bot._solver._game
  return None
