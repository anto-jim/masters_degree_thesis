"""RL agent factories."""

from __future__ import annotations

from typing import List

from absl import flags

from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.examples.turn_battle_study.config import normalize_algorithm
from open_spiel.python.pytorch import nfsp
from open_spiel.python.pytorch import policy_gradient

FLAGS = flags.FLAGS


def create_rl_agents(algorithm: str, env: rl_environment.Environment) -> List:
  algo = normalize_algorithm(algorithm)
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
        policy_gradient.PolicyGradient(
            player_id=p, info_state_size=info_state_size,
            num_actions=num_actions, loss_str=algo,
            hidden_layers_sizes=[64, 64], batch_size=64, entropy_cost=0.01,
            critic_learning_rate=0.01, pi_learning_rate=0.005,
            num_critic_before_pi=4, optimizer_str="adam")
        for p in range(env.num_players)
    ]

  if algo == "nfsp":
    dqn_kwargs = {
        "replay_buffer_capacity": FLAGS.nfsp_replay_buffer_capacity,
        "epsilon_decay_duration": FLAGS.train_episodes,
        "epsilon_start": 0.06, "epsilon_end": 0.001, "optimizer_str": "adam",
    }
    return [
        nfsp.NFSP(
            player_id=p, state_representation_size=info_state_size,
            num_actions=num_actions, hidden_layers_sizes=[64, 64],
            reservoir_buffer_capacity=FLAGS.nfsp_reservoir_buffer_capacity,
            anticipatory_param=FLAGS.nfsp_anticipatory_param,
            batch_size=FLAGS.nfsp_batch_size,
            min_buffer_size_to_learn=FLAGS.nfsp_min_buffer_size,
            learn_every=FLAGS.nfsp_learn_every, **dqn_kwargs)
        for p in range(env.num_players)
    ]

  raise ValueError(f"Algorithm '{algorithm}' has no RL agent factory.")
