"""RL agent factories."""

from __future__ import annotations

from typing import List

from absl import flags

from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.examples.turn_battle_study.config import (
    ROLE_SHARED_ALGOS,
    normalize_algorithm,
)
from open_spiel.python.examples.turn_battle_study.device import resolve_device
from open_spiel.python.examples.turn_battle_study.game import parse_game_params
from open_spiel.python.examples.turn_battle_study.role_shared import (
    DEFENDER_CANON,
    ATTACKER_CANON,
    RoleSeatFacade,
    RoleSharedTeam,
    role_relative_state_size,
)
from open_spiel.python.pytorch import nfsp
from open_spiel.python.pytorch import policy_gradient

FLAGS = flags.FLAGS


def _num_turns() -> int:
  return int(parse_game_params(FLAGS.game_params).get("num_turns", 5))


def create_single_rl_agent(
    algorithm: str,
    env: rl_environment.Environment,
    player_id: int,
    info_state_size: int,
):
  algo = normalize_algorithm(algorithm)
  device = str(resolve_device(FLAGS.device))
  num_actions = env.action_spec()["num_actions"]

  if algo in {"qpg", "a2c", "rpg"}:
    return policy_gradient.PolicyGradient(
        player_id=player_id, info_state_size=info_state_size,
        num_actions=num_actions, loss_str=algo,
        hidden_layers_sizes=[64, 64], batch_size=64, entropy_cost=0.01,
        critic_learning_rate=0.01, pi_learning_rate=0.005,
        num_critic_before_pi=4, optimizer_str="adam",
        device=device)

  if algo == "nfsp":
    dqn_kwargs = {
        "replay_buffer_capacity": FLAGS.nfsp_replay_buffer_capacity,
        "epsilon_decay_duration": FLAGS.train_episodes,
        "epsilon_start": 0.06, "epsilon_end": 0.001, "optimizer_str": "adam",
    }
    return nfsp.NFSP(
        player_id=player_id, state_representation_size=info_state_size,
        num_actions=num_actions, hidden_layers_sizes=[64, 64],
        reservoir_buffer_capacity=FLAGS.nfsp_reservoir_buffer_capacity,
        anticipatory_param=FLAGS.nfsp_anticipatory_param,
        batch_size=FLAGS.nfsp_batch_size,
        min_buffer_size_to_learn=FLAGS.nfsp_min_buffer_size,
        learn_every=FLAGS.nfsp_learn_every,
        device=device,
        **dqn_kwargs)

  raise ValueError(f"Algorithm '{algorithm}' has no single RL agent factory.")


def create_role_shared_rl_agents(
    algorithm: str, env: rl_environment.Environment) -> RoleSharedTeam:
  """Four seat facades with role-relative obs; weights synced each episode."""
  role_size = role_relative_state_size(_num_turns())
  facades = [
      RoleSeatFacade(
          0, create_single_rl_agent(algorithm, env, DEFENDER_CANON, role_size),
          "defender"),
      RoleSeatFacade(
          1, create_single_rl_agent(algorithm, env, ATTACKER_CANON, role_size),
          "attacker"),
      RoleSeatFacade(
          2, create_single_rl_agent(algorithm, env, DEFENDER_CANON, role_size),
          "defender"),
      RoleSeatFacade(
          3, create_single_rl_agent(algorithm, env, ATTACKER_CANON, role_size),
          "attacker"),
  ]
  return RoleSharedTeam(facades)


def create_rl_agents(algorithm: str, env: rl_environment.Environment) -> List:
  algo = normalize_algorithm(algorithm)
  if algo in ROLE_SHARED_ALGOS:
    return create_role_shared_rl_agents(algo, env)

  device = str(resolve_device(FLAGS.device))
  num_actions = env.action_spec()["num_actions"]
  info_state_size = env.observation_spec()["info_state"][0]

  if algo == "q_learning":
    schedule = rl_tools.LinearSchedule(0.3, 0.05, FLAGS.train_episodes)
    return [
        tabular_qlearner.QLearner(
            player_id=p, num_actions=num_actions, step_size=0.2,
            epsilon_schedule=schedule, discount_factor=1.0)
        for p in range(env.num_players)
    ]

  if algo in {"qpg", "a2c", "rpg"}:
    return [
        create_single_rl_agent(algo, env, p, info_state_size)
        for p in range(env.num_players)
    ]

  if algo == "nfsp":
    return [
        create_single_rl_agent(algo, env, p, info_state_size)
        for p in range(env.num_players)
    ]

  raise ValueError(f"Algorithm '{algorithm}' has no RL agent factory.")


def create_legacy_rl_agents(
    algorithm: str, env: rl_environment.Environment) -> List:
  """Four independent per-seat agents (pre-role-sharing checkpoints)."""
  algo = normalize_algorithm(algorithm)
  info_state_size = env.observation_spec()["info_state"][0]
  return [
      create_single_rl_agent(algo, env, p, info_state_size)
      for p in range(env.num_players)
  ]
