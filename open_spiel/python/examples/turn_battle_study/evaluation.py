"""Match evaluation and episode runners."""

from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence

from absl import flags
import numpy as np

from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.examples.turn_battle_study.bots import (
    BotRlAdapter,
    RlAgentTurnBasedBot,
    create_pyspiel_bot,
    deep_cfr_solver_game,
)
from open_spiel.python.examples.turn_battle_study.config import (
    TEAM1_PLAYERS,
    TEAM2_PLAYERS,
    TRAINABLE_ALGOS,
    TURN_BASED_TRAINED_ALGOS,
    effective_bot_algorithm,
    normalize_algorithm,
)

FLAGS = flags.FLAGS
from open_spiel.python.examples.turn_battle_study.game import (
    load_game,
    load_turn_based_game,
    make_rl_environment,
)
from open_spiel.python.examples.turn_battle_study.role_shared import (
    RoleSeatFacade,
    agents_for_matchup,
)
from open_spiel.python.examples.turn_battle_study.models import MatchResult
from open_spiel.python.pytorch import nfsp
import pyspiel


def _iter_nfsp_agents(agents: Optional[Sequence]) -> List[nfsp.NFSP]:
  """Collect every NFSP instance from a mixed agent sequence.

  Handles both bare ``nfsp.NFSP`` objects and ``RoleSeatFacade`` wrappers
  whose underlying agent is an NFSP instance.

  Args:
    agents: Sequence that may contain RL agents, bot adapters, or None.

  Returns:
    A flat list of every NFSP instance found.
  """
  if not agents:
    return []
  found: List[nfsp.NFSP] = []
  for agent in agents:
    if agent is None:
      continue
    if isinstance(agent, nfsp.NFSP):
      found.append(agent)
    elif isinstance(agent, RoleSeatFacade) and isinstance(
        agent.underlying, nfsp.NFSP):
      found.append(agent.underlying)
  return found


def _uses_rl_stack(algo: str, agents: Optional[Sequence]) -> bool:
  """Return True when *algo* should be evaluated via the shared-state RL stack.

  An algorithm uses the RL stack when it is trainable, not a turn-based-trained
  algorithm, and trained agents have actually been provided.

  Args:
    algo: Normalised algorithm key.
    agents: Agent sequence for the team, or None if not available.

  Returns:
    True if the RL environment episode runner should handle this algorithm.
  """
  return (
      algo in TRAINABLE_ALGOS
      and algo not in TURN_BASED_TRAINED_ALGOS
      and agents is not None
  )


def _uses_turn_based_eval(
    t1: str,
    t2: str,
    team1_agents: Optional[Sequence],
    team2_agents: Optional[Sequence],
) -> bool:
  """Decide whether evaluation should use the turn-based pyspiel bot API.

  Returns True when any of the following hold:
  - Either side was trained with a turn-based algorithm (e.g. AlphaZero).
  - Either side uses MCTS (which requires the bot API).
  - One side uses the RL stack while the other does not (mixed evaluation path).

  Args:
    t1: Normalised algorithm key for team 1.
    t2: Normalised algorithm key for team 2.
    team1_agents: Trained agents for team 1, or None.
    team2_agents: Trained agents for team 2, or None.

  Returns:
    True if the turn-based bot episode runner should be used.
  """
  if t1 in TURN_BASED_TRAINED_ALGOS or t2 in TURN_BASED_TRAINED_ALGOS:
    return True
  if effective_bot_algorithm(t1) == "mcts" or effective_bot_algorithm(t2) == "mcts":
    return True
  if _uses_rl_stack(t1, team1_agents) and _uses_rl_stack(t2, team2_agents):
    return False
  if _uses_rl_stack(t1, team1_agents) or _uses_rl_stack(t2, team2_agents):
    return True
  return False


def _pick_bot_opponent(rng: np.random.RandomState) -> str:
  """Sample a bot opponent algorithm according to the training flag.

  Args:
    rng: Random state used to break ties when the flag is ``"mixed"``.

  Returns:
    One of ``"random"`` or ``"heuristic"`` (or whatever the flag is set to).
  """
  choice = FLAGS.train_bot_opponent
  if choice == "mixed":
    return rng.choice(["random", "heuristic"])
  return choice


def play_training_episode_rl(
    env,
    learning_agents: Sequence,
    rng: np.random.RandomState,
    episode_idx: int,
    game,
) -> List[float]:
  """Self-play, or learning team vs bots with alternating team slots."""
  if rng.random() >= FLAGS.train_bot_mix:
    return play_episode_rl(env, learning_agents, is_evaluation=False)

  bot_algo = _pick_bot_opponent(rng)
  learning_on_team1 = (episode_idx % 2 == 0)
  episode_agents: List[object] = []
  for player in range(env.num_players):
    learns_here = (
        (learning_on_team1 and player in TEAM1_PLAYERS)
        or (not learning_on_team1 and player in TEAM2_PLAYERS))
    if learns_here:
      episode_agents.append(learning_agents[player])
    else:
      episode_agents.append(BotRlAdapter(
          create_pyspiel_bot(bot_algo, game, player, rng), player))
  return play_episode_rl(env, episode_agents, is_evaluation=False)


def play_episode_rl(env, agents, is_evaluation=True):
  """Run one episode through the shared-state RL environment.

  During evaluation, every NFSP agent is placed in average-policy mode via a
  context manager so that their behaviour reflects the learned Nash strategy.

  Args:
    env: OpenSpiel RL environment wrapping the game.
    agents: One agent per player seat; each must implement ``step()``.
    is_evaluation: If True, agents act greedily and terminal ``step()`` is
      skipped (no learning update). Defaults to True.

  Returns:
    A list of terminal rewards, one per player.
  """
  with contextlib.ExitStack() as stack:
    if is_evaluation:
      for agent in _iter_nfsp_agents(agents):
        stack.enter_context(agent.temp_mode_as(nfsp.MODE.AVERAGE_POLICY))
    ts = env.reset()
    while not ts.last():
      actions = [a.step(ts, is_evaluation=is_evaluation).action for a in agents]
      ts = env.step(actions)
    if not is_evaluation:
      for a in agents:
        a.step(ts, is_evaluation=False)
  return list(ts.rewards)


def play_episode_bots(game, bots, rng):
  """Run one episode using the pyspiel bot API.

  Args:
    game: A pyspiel ``Game`` instance from which the initial state is drawn.
    bots: One ``pyspiel.Bot`` per player seat.
    rng: Random state forwarded to ``evaluate_bots`` for stochastic decisions.

  Returns:
    A list of terminal returns, one per player.
  """
  return list(evaluate_bots.evaluate_bots(game.new_initial_state(), bots, rng))


def _player_agent(
    player: int,
    t1_algo: str,
    t2_algo: str,
    t1_agents: Optional[Sequence],
    t2_agents: Optional[Sequence],
    game,
    rng,
) -> object:
  """Return the RL-compatible agent for *player* in a shared-state episode.

  If the player's team uses the RL stack, the pre-trained agent is returned
  directly. Otherwise a ``BotRlAdapter`` wrapping a fresh pyspiel bot is used.

  Args:
    player: Seat index of the player.
    t1_algo: Algorithm key for team 1 (may include ``"mcts:"`` prefix).
    t2_algo: Algorithm key for team 2.
    t1_agents: Trained agents for team 1, or None.
    t2_agents: Trained agents for team 2, or None.
    game: A pyspiel ``Game`` instance used to instantiate bot opponents.
    rng: Random state forwarded to bot constructors.

  Returns:
    An agent object compatible with the ``play_episode_rl`` loop.
  """
  if player in TEAM1_PLAYERS:
    if _uses_rl_stack(normalize_algorithm(t1_algo), t1_agents):
      return t1_agents[player]
    return BotRlAdapter(
        create_pyspiel_bot(t1_algo, game, player, rng, for_mcts=False), player)
  if _uses_rl_stack(normalize_algorithm(t2_algo), t2_agents):
    return t2_agents[player]
  return BotRlAdapter(
      create_pyspiel_bot(t2_algo, game, player, rng, for_mcts=False), player)


def _turn_based_bot_for_player(
    player: int,
    t1: str,
    t2: str,
    team1_algo: str,
    team2_algo: str,
    team1_agents: Optional[Sequence],
    team2_agents: Optional[Sequence],
    bot_game,
    rng,
) -> pyspiel.Bot:
  """Return the pyspiel.Bot for *player* in a turn-based evaluation episode.

  Selection priority per player:
  1. A turn-based-trained algorithm's stored ``._bot`` (e.g. AlphaZero).
  2. An ``RlAgentTurnBasedBot`` wrapping a trained RL agent.
  3. A fresh pyspiel bot constructed for MCTS or untrained bot algorithms.

  Args:
    player: Seat index of the player.
    t1: Normalised algorithm key for team 1.
    t2: Normalised algorithm key for team 2.
    team1_algo: Original (possibly prefixed) algorithm string for team 1.
    team2_algo: Original (possibly prefixed) algorithm string for team 2.
    team1_agents: Trained agents for team 1, or None.
    team2_agents: Trained agents for team 2, or None.
    bot_game: The turn-based pyspiel ``Game`` the bots will play.
    rng: Random state forwarded to bot constructors.

  Returns:
    A ``pyspiel.Bot`` instance for the given player seat.
  """
  if player in TEAM1_PLAYERS:
    if t1 in TURN_BASED_TRAINED_ALGOS and team1_agents:
      return team1_agents[player]._bot
    if _uses_rl_stack(t1, team1_agents):
      return RlAgentTurnBasedBot(team1_agents[player], player)
    return create_pyspiel_bot(
        team1_algo, bot_game, player, rng, for_mcts=True)
  if t2 in TURN_BASED_TRAINED_ALGOS and team2_agents:
    return team2_agents[player]._bot
  if _uses_rl_stack(t2, team2_agents):
    return RlAgentTurnBasedBot(team2_agents[player], player)
  return create_pyspiel_bot(
      team2_algo, bot_game, player, rng, for_mcts=True)


def _assert_deep_cfr_horizon_matches(
    bot_game,
    team1_agents: Optional[Sequence],
    team2_agents: Optional[Sequence],
) -> None:
  """Raise if a Deep CFR solver trained on a different game than *bot_game*.

  Deep CFR's average-strategy network is sized by its training game's
  information-state tensor, so evaluating it on a game with a different
  horizon is both a shape error and a comparability error.

  Args:
    bot_game: The canonical turn-based game used for the matchup.
    team1_agents: Trained agents for team 1, or None.
    team2_agents: Trained agents for team 2, or None.

  Raises:
    ValueError: If a Deep CFR solver's game horizon differs from *bot_game*.
  """
  expected = bot_game.information_state_tensor_size()
  for agents in (team1_agents, team2_agents):
    solver_game = deep_cfr_solver_game(agents)
    if solver_game is None:
      continue
    actual = solver_game.information_state_tensor_size()
    if actual != expected:
      raise ValueError(
          f"Deep CFR was trained on a game with info-state size {actual} but "
          f"the tournament game has size {expected}. Train Deep CFR on the "
          "same num_turns as every other algorithm.")


def evaluate_team_matchup(
    team1_algo, team2_algo, num_episodes, rng,
    team1_agents=None, team2_agents=None) -> MatchResult:
  """Evaluate a head-to-head matchup between two algorithm teams.

  Automatically selects the evaluation path (shared-state RL environment or
  turn-based bot API) based on the algorithm types and available agents.
  NFSP agents are put into average-policy mode for the entire evaluation block.

  Args:
    team1_algo: Algorithm string for team 1 (e.g. ``"nfsp"`` or ``"mcts:200"``).
    team2_algo: Algorithm string for team 2.
    num_episodes: Number of episodes to play.
    rng: Random state for stochastic decisions and bot construction.
    team1_agents: Trained agents for team 1, or None for bot-only algorithms.
    team2_agents: Trained agents for team 2, or None for bot-only algorithms.

  Returns:
    A ``MatchResult`` populated with per-episode outcomes.
  """
  if team1_agents is not None:
    team1_agents = agents_for_matchup(team1_agents, for_team1=True)
  if team2_agents is not None:
    team2_agents = agents_for_matchup(team2_agents, for_team1=False)
  result = MatchResult()
  game = load_game()
  t1, t2 = normalize_algorithm(team1_algo), normalize_algorithm(team2_algo)

  if not _uses_turn_based_eval(t1, t2, team1_agents, team2_agents):
    env = make_rl_environment(include_full_state=True)
    for _ in range(num_episodes):
      agents = [
          _player_agent(p, team1_algo, team2_algo, team1_agents, team2_agents,
                        game, rng)
          for p in range(env.num_players)
      ]
      result.record(play_episode_rl(env, agents))
    return result

  # Every matchup is played on the canonical game. Previously a Deep CFR
  # agent's own (possibly smaller) training game was substituted here, so
  # matches involving Deep CFR silently ran on a different game than the rest
  # of the tournament.
  bot_game = load_turn_based_game()
  _assert_deep_cfr_horizon_matches(bot_game, team1_agents, team2_agents)
  bots = [
      _turn_based_bot_for_player(
          player, t1, t2, team1_algo, team2_algo,
          team1_agents, team2_agents, bot_game, rng)
      for player in range(bot_game.num_players())
  ]

  nfsp_agents = _iter_nfsp_agents(team1_agents) + _iter_nfsp_agents(team2_agents)

  with contextlib.ExitStack() as stack:
    for agent in nfsp_agents:
      stack.enter_context(agent.temp_mode_as(nfsp.MODE.AVERAGE_POLICY))
    for _ in range(num_episodes):
      result.record(play_episode_bots(bot_game, bots, rng))
  return result
