"""RL agent factories for thesis algorithms (NFSP, QPG)."""

from __future__ import annotations

from absl import flags

from open_spiel.python import rl_environment
from open_spiel.python.examples.turn_battle_study.config import (
    ROLE_SHARED_ALGOS,
    THESIS_ALGORITHMS,
    normalize_algorithm,
)
from open_spiel.python.examples.turn_battle_study.device import resolve_device
from open_spiel.python.examples.turn_battle_study.game import parse_num_turns
from open_spiel.python.examples.turn_battle_study.role_shared import (
    ATTACKER_CANON,
    DEFENDER_CANON,
    RoleSeatFacade,
    RoleSharedTeam,
    role_relative_state_size,
)
from open_spiel.python.pytorch import nfsp
from open_spiel.python.pytorch import policy_gradient

FLAGS = flags.FLAGS


def _num_turns() -> int:
  """Return the number of turns configured via FLAGS."""
  return parse_num_turns()


def create_single_rl_agent(
    algorithm: str,
    env: rl_environment.Environment,
    player_id: int,
    info_state_size: int,
):
  """Instantiate a single RL agent for one player seat (NFSP or QPG only).

  Args:
    algorithm: Algorithm name (``"qpg"`` or ``"nfsp"``).
    env: The RL environment used to query action/observation specs.
    player_id: Seat index this agent will occupy.
    info_state_size: Length of the flattened info-state vector.

  Returns:
    A PolicyGradient or NFSP agent configured for the given algorithm.
  """
  algo = normalize_algorithm(algorithm)
  device = str(resolve_device(FLAGS.device))
  num_actions = env.action_spec()["num_actions"]

  if algo == "qpg":
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
  """Four seat facades sharing one defender and one attacker net per role."""
  role_size = role_relative_state_size(_num_turns())
  defender = create_single_rl_agent(
      algorithm, env, DEFENDER_CANON, role_size)
  attacker = create_single_rl_agent(
      algorithm, env, ATTACKER_CANON, role_size)
  facades = [
      RoleSeatFacade(0, defender, "defender"),
      RoleSeatFacade(1, attacker, "attacker"),
      RoleSeatFacade(2, defender, "defender"),
      RoleSeatFacade(3, attacker, "attacker"),
  ]
  return RoleSharedTeam(facades)


def create_rl_agents(algorithm: str, env: rl_environment.Environment):
  """Create role-shared agents for NFSP or QPG.

  Args:
    algorithm: Algorithm name (must be in ``ROLE_SHARED_ALGOS``).
    env: RL environment used to query action/observation specs.

  Returns:
    A RoleSharedTeam with shared defender and attacker policies.

  Raises:
    ValueError: If *algorithm* is not NFSP or QPG.
  """
  algo = normalize_algorithm(algorithm)
  if algo not in ROLE_SHARED_ALGOS:
    raise ValueError(
        f"Algorithm '{algorithm}' has no RL agent factory; "
        f"thesis algorithms are: {', '.join(THESIS_ALGORITHMS)}.")
  return create_role_shared_rl_agents(algo, env)
