"""Save and load trained agents for tournament reruns."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import numpy as np
import torch

from open_spiel.python.examples.turn_battle_study.agents import (
    create_legacy_rl_agents,
    create_rl_agents,
)
from open_spiel.python.examples.turn_battle_study.az_cpp import AlphaZeroCpp
from open_spiel.python.examples.turn_battle_study.bots import bots_to_adapters
from open_spiel.python.examples.turn_battle_study.config import (
    ROLE_SHARED_ALGOS,
    normalize_algorithm,
)
from open_spiel.python.examples.turn_battle_study.game import make_rl_environment
from open_spiel.python.examples.turn_battle_study.role_shared import (
    RoleSharedTeam,
)
from open_spiel.python.examples.turn_battle_study.trainers import (
    _deep_cfr_bots,
    build_deep_cfr_solver,
)
from open_spiel.python.pytorch import deep_cfr
from open_spiel.python.pytorch import nfsp
from open_spiel.python.pytorch import policy_gradient

_ROLE_SELECTION_FILE = "role_selection.json"
_LEGACY_PLAYER_PREFIX = "player_"


def _save_nfsp(agent: nfsp.NFSP, checkpoint_dir: pathlib.Path) -> None:
  """Persist an NFSP agent's reservoir network and inner RL agent to disk.

  Args:
    agent: The NFSP agent to checkpoint.
    checkpoint_dir: Directory that will receive 'rl_inner/' and 'nfsp.pt'.
  """
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  agent._rl_agent.save(checkpoint_dir / "rl_inner", save_optimiser=True)
  torch.save(
      {
          "iteration": agent._iteration,
          "last_loss_value": agent._last_sl_loss_value,
          "model_state_dict": agent._avg_network.state_dict(),
          "optimizer_state_dict": agent._optimizer.state_dict(),
      },
      checkpoint_dir / "nfsp.pt",
  )


def _load_nfsp(agent: nfsp.NFSP, checkpoint_dir: pathlib.Path) -> None:
  """Restore an NFSP agent's weights from a previously saved checkpoint.

  Args:
    agent: The NFSP agent to restore into (must match the saved architecture).
    checkpoint_dir: Directory containing 'rl_inner/' and 'nfsp.pt'.
  """
  agent._rl_agent.load(checkpoint_dir / "rl_inner", load_optimiser=True)
  data = torch.load(
      checkpoint_dir / "nfsp.pt",
      weights_only=True,
      map_location=agent._device,
  )
  agent._avg_network.load_state_dict(data["model_state_dict"])
  agent._optimizer.load_state_dict(data["optimizer_state_dict"])
  agent._iteration = data["iteration"]
  agent._last_sl_loss_value = data["last_loss_value"]
  agent._last_rl_loss_value = agent._rl_agent.loss


def _save_role_agent(agent, checkpoint_dir: pathlib.Path) -> None:
  """Save a single role agent (NFSP or PolicyGradient) to checkpoint_dir.

  Args:
    agent: NFSP or PolicyGradient agent to persist.
    checkpoint_dir: Target directory for the checkpoint files.

  Raises:
    TypeError: If agent is not a supported type.
  """
  if isinstance(agent, nfsp.NFSP):
    _save_nfsp(agent, checkpoint_dir)
  elif isinstance(agent, policy_gradient.PolicyGradient):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint_dir))
  else:
    raise TypeError(f"Unsupported role agent type: {type(agent)}")


def _load_role_agent(agent, checkpoint_dir: pathlib.Path) -> None:
  """Restore a single role agent (NFSP or PolicyGradient) from checkpoint_dir.

  Args:
    agent: Agent instance to restore into (type must match what was saved).
    checkpoint_dir: Directory containing the previously saved checkpoint.

  Raises:
    TypeError: If agent is not a supported type.
  """
  if isinstance(agent, nfsp.NFSP):
    _load_nfsp(agent, checkpoint_dir)
  elif isinstance(agent, policy_gradient.PolicyGradient):
    agent.restore(str(checkpoint_dir))
  else:
    raise TypeError(f"Unsupported role agent type: {type(agent)}")


def _save_role_shared(agents: RoleSharedTeam, checkpoint_dir: str) -> None:
  """Persist a RoleSharedTeam's defender, attacker, and optional role selection.

  Args:
    agents: The RoleSharedTeam to checkpoint.
    checkpoint_dir: Root directory; 'role_defender/' and 'role_attacker/'
      subdirectories are created automatically.
  """
  root = pathlib.Path(checkpoint_dir)
  root.mkdir(parents=True, exist_ok=True)
  _save_role_agent(agents.defender, root / "role_defender")
  _save_role_agent(agents.attacker, root / "role_attacker")
  if agents.selection is not None:
    def_seat, atk_seat = agents.selection
    with open(root / _ROLE_SELECTION_FILE, "w", encoding="utf-8") as f:
      json.dump(
          {
              "defender_seat": def_seat,
              "attacker_seat": atk_seat,
          },
          f,
          indent=2,
      )


def _load_role_shared(
    algo: str, checkpoint_dir: str, rng: np.random.RandomState
) -> RoleSharedTeam:
  """Reconstruct a RoleSharedTeam from a saved checkpoint directory.

  Args:
    algo: Algorithm key used to create fresh agent instances.
    checkpoint_dir: Root directory previously written by _save_role_shared.
    rng: Random state forwarded to agent creation.

  Returns:
    A RoleSharedTeam with restored weights and selection metadata.
  """
  env = make_rl_environment()
  team = create_rl_agents(algo, env)
  if not isinstance(team, RoleSharedTeam):
    raise TypeError(f"Expected RoleSharedTeam for {algo}, got {type(team)}")
  root = pathlib.Path(checkpoint_dir)
  _load_role_agent(team.defender, root / "role_defender")
  _load_role_agent(team.attacker, root / "role_attacker")
  team.sync_role_weights()
  selection_path = root / _ROLE_SELECTION_FILE
  if selection_path.is_file():
    with open(selection_path, encoding="utf-8") as f:
      data = json.load(f)
    team.selection = (int(data["defender_seat"]), int(data["attacker_seat"]))
  return team


def _save_deep_cfr(solver: deep_cfr.DeepCFRSolver, checkpoint_dir: str) -> None:
  """Save a DeepCFRSolver's network weights and iteration counter to disk.

  Args:
    solver: Trained DeepCFRSolver to persist.
    checkpoint_dir: Target directory; 'model.pt' is written inside it.
  """
  root = pathlib.Path(checkpoint_dir)
  root.mkdir(parents=True, exist_ok=True)
  torch.save(
      {
          "iteration": solver._iteration,
          "policy_network": solver._policy_network.state_dict(),
          "advantage_networks": [
              net.state_dict() for net in solver._advantage_networks
          ],
      },
      root / "model.pt",
  )


def _load_deep_cfr(checkpoint_dir: str, rng: np.random.RandomState):
  """Reconstruct bot adapters from a Deep CFR checkpoint directory.

  Args:
    checkpoint_dir: Directory containing 'model.pt' written by _save_deep_cfr.
    rng: Random state forwarded to bot construction.

  Returns:
    List of bot adapters wrapping a DeepCFRSolver restored from the checkpoint.
  """
  game, solver = build_deep_cfr_solver(rng)
  data = torch.load(
      pathlib.Path(checkpoint_dir) / "model.pt",
      weights_only=True,
      map_location=solver._device,
  )
  solver._iteration = int(data["iteration"])
  solver._policy_network.load_state_dict(data["policy_network"])
  for net, state in zip(solver._advantage_networks, data["advantage_networks"]):
    net.load_state_dict(state)
  return bots_to_adapters(_deep_cfr_bots(solver, game, rng))


def save_trained_checkpoint(
    algo: str,
    agents: Any,
    artifact: Any,
    checkpoint_dir: str,
) -> None:
  """Persist trained weights for later tournament reload."""
  key = normalize_algorithm(algo)
  if key == "alphazero":
    if artifact is None:
      raise ValueError("AlphaZero checkpoint requires AlphaZeroCpp artifact.")
    artifact.save_checkpoint(checkpoint_dir)
    return

  if key == "deep_cfr":
    if artifact is None:
      raise ValueError("Deep CFR checkpoint requires DeepCFRSolver artifact.")
    _save_deep_cfr(artifact, checkpoint_dir)
    return

  if key in ROLE_SHARED_ALGOS:
    if not isinstance(agents, RoleSharedTeam):
      raise TypeError(f"{key} expects RoleSharedTeam agents.")
    _save_role_shared(agents, checkpoint_dir)
    return

  root = pathlib.Path(checkpoint_dir)
  root.mkdir(parents=True, exist_ok=True)
  for player_id, agent in enumerate(agents):
    player_dir = root / f"{_LEGACY_PLAYER_PREFIX}{player_id}"
    if isinstance(agent, nfsp.NFSP):
      _save_nfsp(agent, player_dir)
    elif isinstance(agent, policy_gradient.PolicyGradient):
      _save_role_agent(agent, player_dir)
    else:
      raise TypeError(f"Unsupported legacy agent type: {type(agent)}")


def load_trained_agents(
    algo: str,
    checkpoint_dir: str,
    rng: np.random.RandomState,
):
  """Reconstruct trained agents from ``checkpoint_dir``."""
  key = normalize_algorithm(algo)
  if not os.path.isdir(checkpoint_dir):
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

  if key == "alphazero":
    trainer = AlphaZeroCpp(rng)
    trainer.load_checkpoint(checkpoint_dir)
    return bots_to_adapters(trainer.make_bots())

  if key == "deep_cfr":
    return _load_deep_cfr(checkpoint_dir, rng)

  if key in ROLE_SHARED_ALGOS:
    role_def = pathlib.Path(checkpoint_dir) / "role_defender"
    if role_def.is_dir():
      return _load_role_shared(key, checkpoint_dir, rng)

  root = pathlib.Path(checkpoint_dir)
  legacy_players = sorted(root.glob(f"{_LEGACY_PLAYER_PREFIX}*"))
  if legacy_players:
    env = make_rl_environment()
    agents = create_legacy_rl_agents(key, env)
    for player_dir in legacy_players:
      player_id = int(player_dir.name.split("_")[-1])
      agent = agents[player_id]
      if isinstance(agent, nfsp.NFSP):
        _load_nfsp(agent, player_dir)
      elif isinstance(agent, policy_gradient.PolicyGradient):
        agent.restore(str(player_dir))
      else:
        raise TypeError(f"Unsupported legacy agent type: {type(agent)}")
    return agents

  raise FileNotFoundError(
      f"No supported checkpoint layout under {checkpoint_dir} for {key}.")
