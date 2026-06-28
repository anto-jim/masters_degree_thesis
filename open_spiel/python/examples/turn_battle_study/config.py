"""Algorithm registry and shared constants."""

from __future__ import annotations

TEAM1_PLAYERS = (0, 1)
TEAM2_PLAYERS = (2, 3)
DEFENDER_SEATS = (0, 2)
ATTACKER_SEATS = (1, 3)

TRAINABLE_ALGOS = frozenset({"alphazero", "deep_cfr", "nfsp", "qpg"})
# NFSP/QPG train shared defender + attacker policies (see role_shared.py).
ROLE_SHARED_ALGOS = frozenset({"nfsp", "qpg"})
# Evaluated via turn-based bot play (search/policy bots), not the RL stack.
TURN_BASED_TRAINED_ALGOS = frozenset({"alphazero", "deep_cfr"})
BOT_ALGOS = frozenset({"random", "mcts", "heuristic"})
# Head-to-head eval without checkpoints: map trained algo to a bot stand-in.
EVAL_BOT_ALIASES = {"alphazero": "mcts", "deep_cfr": "heuristic"}

# Master's thesis comparison set (4 paradigms).
THESIS_ALGORITHMS = ["alphazero", "deep_cfr", "nfsp", "qpg"]
SUPPORTED_ALGOS = frozenset(THESIS_ALGORITHMS) | BOT_ALGOS

# Every algorithm uses the same ``train_episodes`` count (see README_experiments.md).
DEFAULT_TRAIN_EPISODES = 300
DEFAULT_EVAL_EPISODES = 50
DEFAULT_EVAL_EVERY = 50
DEFAULT_SEEDS = [42, 43, 44]


def normalize_algorithm(name: str) -> str:
  """Normalize an algorithm name by stripping whitespace and lowercasing."""
  return name.strip().lower()


def validate_algorithm(name: str, *, for_training: bool = False) -> str:
  """Return a normalised algorithm key or raise if it is not supported.

  Args:
    name: Raw algorithm name from CLI or config.
    for_training: When True, require a thesis training backend.

  Raises:
    ValueError: If *name* is not in the allowed set.
  """
  algo = normalize_algorithm(name)
  if for_training:
    if algo not in TRAINABLE_ALGOS:
      raise ValueError(
          f"Unsupported training algorithm '{name}'. "
          f"Thesis trainable: {sorted(TRAINABLE_ALGOS)}.")
    return algo
  if algo not in SUPPORTED_ALGOS:
    raise ValueError(
        f"Unsupported algorithm '{name}'. "
        f"Supported: {sorted(SUPPORTED_ALGOS)}.")
  return algo


def effective_bot_algorithm(name: str) -> str:
  """Return the bot algorithm to use when evaluating *name* head-to-head.

  Turn-based-trained algorithms that cannot drive the RL evaluation stack
  directly are remapped to a compatible bot (e.g. ``alphazero`` → ``mcts``,
  ``deep_cfr`` → ``heuristic``).  All other algorithms are returned unchanged
  after normalisation.

  Args:
    name: Raw algorithm name (may include aliases or mixed case).

  Returns:
    The normalised bot algorithm name suitable for head-to-head evaluation.
  """
  algo = normalize_algorithm(name)
  return EVAL_BOT_ALIASES.get(algo, algo)
