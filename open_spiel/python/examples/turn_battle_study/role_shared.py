"""Role-shared RL training (defender / attacker) for 4-seat Turn Battle."""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from open_spiel.python import rl_environment
from open_spiel.python.examples.turn_battle_study.config import (
    ATTACKER_SEATS,
    DEFENDER_SEATS,
    ROLE_SHARED_ALGOS,
    normalize_algorithm,
)

DEFENDER_CANON = 0
ATTACKER_CANON = 1
ALLY = {0: 1, 1: 0, 2: 3, 3: 2}
PUBLIC_SIZE = 9
PRIVATE_SIZE = 2


def role_relative_state_size(num_turns: int) -> int:
  """Role features (10) + joint action history (num_turns * 4)."""
  return 10 + num_turns * 4


def _enemy_seats(seat_id: int) -> Tuple[int, int]:
  if seat_id in (0, 1):
    return 2, 3
  return 0, 1


def role_relative_info_state(
    full_state: Sequence[float], seat_id: int) -> np.ndarray:
  """Map absolute Turn Battle info-state to a role-relative vector."""
  state = np.asarray(full_state, dtype=np.float32)
  ally = ALLY[seat_id]
  edef, eatk = _enemy_seats(seat_id)
  is_def = 1.0 if seat_id in DEFENDER_SEATS else 0.0
  role = np.array(
      [
          state[0],
          state[1 + seat_id],
          state[1 + ally],
          state[1 + edef],
          state[1 + eatk],
          state[5 + seat_id],
          state[5 + ally],
          state[5 + edef],
          state[5 + eatk],
          is_def,
      ],
      dtype=np.float32,
  )
  history = state[PUBLIC_SIZE + PRIVATE_SIZE:]
  return np.concatenate([role, history])


def _remap_timestep(
    time_step: rl_environment.TimeStep,
    seat_id: int,
    canon_id: int,
) -> rl_environment.TimeStep:
  """Route seat observations/rewards to the canonical role learner index."""
  obs = dict(time_step.observations)
  rel = role_relative_info_state(obs["info_state"][seat_id], seat_id)

  info = list(obs["info_state"])
  info[canon_id] = rel.tolist()

  legal = list(obs["legal_actions"])
  legal[canon_id] = list(legal[seat_id])

  rewards = list(time_step.rewards) if time_step.rewards is not None else None
  discounts = (
      list(time_step.discounts) if time_step.discounts is not None else None)
  if rewards is not None:
    rewards[canon_id] = rewards[seat_id]
    discounts[canon_id] = discounts[seat_id]

  new_obs = {**obs, "info_state": info, "legal_actions": legal}
  cp = obs.get("current_player")
  if cp == seat_id:
    new_obs["current_player"] = canon_id
  return time_step._replace(
      observations=new_obs, rewards=rewards, discounts=discounts)


def _sync_agent_weights(src, dst) -> None:
  """Copy trainable weights from src to dst (defender or attacker pair)."""
  from open_spiel.python.pytorch import nfsp
  from open_spiel.python.pytorch import policy_gradient

  if isinstance(src, policy_gradient.PolicyGradient):
    dst._net_torso.load_state_dict(copy.deepcopy(src._net_torso.state_dict()))
    dst._policy_logits_layer.load_state_dict(
        copy.deepcopy(src._policy_logits_layer.state_dict()))
    if hasattr(src, "_baseline_layer"):
      dst._baseline_layer.load_state_dict(
          copy.deepcopy(src._baseline_layer.state_dict()))
    else:
      dst._q_values_layer.load_state_dict(
          copy.deepcopy(src._q_values_layer.state_dict()))
    return

  if isinstance(src, nfsp.NFSP):
    dst._avg_network.load_state_dict(
        copy.deepcopy(src._avg_network.state_dict()))
    dst._rl_agent._q_network.load_state_dict(
        copy.deepcopy(src._rl_agent._q_network.state_dict()))
    dst._rl_agent._target_q_network.load_state_dict(
        copy.deepcopy(src._rl_agent._target_q_network.state_dict()))
    return

  raise TypeError(f"Unsupported agent type for weight sync: {type(src)}")


class RoleSeatFacade:
  """One seat with role-relative observations for its underlying learner."""

  def __init__(self, seat_id: int, role_agent, role: str):
    self._seat_id = seat_id
    self._role_agent = role_agent
    self._role = role
    self.player_id = seat_id

  @property
  def underlying(self):
    return self._role_agent

  def step(self, time_step, is_evaluation=False):
    canon = DEFENDER_CANON if self._role == "defender" else ATTACKER_CANON
    return self._role_agent.step(
        _remap_timestep(time_step, self._seat_id, canon), is_evaluation)

  def __getattr__(self, name):
    return getattr(self._role_agent, name)


class RoleSharedTeam:
  """Four seat facades; defender/attacker weights synced after each episode."""

  def __init__(
      self,
      facades: List[RoleSeatFacade],
      selection: Optional[Tuple[int, int]] = None,
  ):
    self.facades = facades
    self.selection = selection

  @property
  def defender(self):
    return self.facades[0].underlying

  @property
  def attacker(self):
    return self.facades[1].underlying

  def sync_role_weights(self) -> None:
    """Merge learning from both teams into one defender and one attacker net."""
    _sync_agent_weights(self.facades[0].underlying, self.facades[2].underlying)
    _sync_agent_weights(self.facades[1].underlying, self.facades[3].underlying)

  def __len__(self) -> int:
    return 4

  def __getitem__(self, index: int):
    return self.facades[index]

  def lineup_for_matchup(self, for_team1: bool) -> List:
    out: List[Optional[object]] = [None, None, None, None]
    if for_team1:
      out[0] = RoleSeatFacade(0, self.defender, "defender")
      out[1] = RoleSeatFacade(1, self.attacker, "attacker")
    else:
      out[2] = RoleSeatFacade(2, self.defender, "defender")
      out[3] = RoleSeatFacade(3, self.attacker, "attacker")
    return out


def is_role_shared_team(agents) -> bool:
  return isinstance(agents, RoleSharedTeam)


def uses_role_shared_training(algo: str) -> bool:
  return normalize_algorithm(algo) in ROLE_SHARED_ALGOS


def _lineup_with_seats(
    team: RoleSharedTeam,
    defender_seat: int,
    attacker_seat: int,
    for_team1: bool,
) -> List:
  del defender_seat, attacker_seat
  return team.lineup_for_matchup(for_team1=for_team1)


def select_best_role_seats(
    team: RoleSharedTeam,
    algo: str,
    eval_episodes: int,
    rng: np.random.RandomState,
) -> Tuple[int, int, float]:
  """Pick best defender seat {0,2} and attacker seat {1,3} via short eval."""
  from open_spiel.python.examples.turn_battle_study.evaluation import (
      evaluate_team_matchup,
  )
  team.sync_role_weights()
  combos = [(0, 1), (0, 3), (2, 1), (2, 3)]
  per_combo = max(1, eval_episodes // len(combos))
  best_def, best_atk, best_score = 0, 1, -1.0

  for def_seat, atk_seat in combos:
    lineup = _lineup_with_seats(team, def_seat, atk_seat, for_team1=True)
    stats = evaluate_team_matchup(
        algo, "random", per_combo, rng, team1_agents=lineup).summary()
    score = stats["team1_win_rate"]
    if score > best_score:
      best_def, best_atk, best_score = def_seat, atk_seat, score

  team.selection = (best_def, best_atk)
  return best_def, best_atk, best_score


def agents_for_matchup(agents, for_team1: bool) -> List:
  if not is_role_shared_team(agents):
    return agents
  if agents.selection is None:
    return agents.lineup_for_matchup(for_team1=for_team1)
  def_seat, atk_seat = agents.selection
  return _lineup_with_seats(agents, def_seat, atk_seat, for_team1=for_team1)
