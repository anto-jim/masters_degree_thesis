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
    trained_bot_game,
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
from open_spiel.python.examples.turn_battle_study.models import MatchResult
from open_spiel.python.pytorch import nfsp
import pyspiel


def _uses_rl_stack(algo: str, agents: Optional[Sequence]) -> bool:
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
  with contextlib.ExitStack() as stack:
    if is_evaluation:
      for agent in agents:
        if isinstance(agent, nfsp.NFSP):
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


def evaluate_team_matchup(
    team1_algo, team2_algo, num_episodes, rng,
    team1_agents=None, team2_agents=None) -> MatchResult:
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

  bot_game = (
      trained_bot_game(team1_agents) or trained_bot_game(team2_agents)
      or load_turn_based_game()
  )
  bots = [
      _turn_based_bot_for_player(
          player, t1, t2, team1_algo, team2_algo,
          team1_agents, team2_agents, bot_game, rng)
      for player in range(bot_game.num_players())
  ]

  nfsp_agents = []
  for agents in (team1_agents, team2_agents):
    if agents:
      nfsp_agents.extend(a for a in agents if isinstance(a, nfsp.NFSP))

  with contextlib.ExitStack() as stack:
    for agent in nfsp_agents:
      stack.enter_context(agent.temp_mode_as(nfsp.MODE.AVERAGE_POLICY))
    for _ in range(num_episodes):
      result.record(play_episode_bots(bot_game, bots, rng))
  return result
