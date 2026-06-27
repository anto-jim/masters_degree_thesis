"""Algorithm registry and shared constants."""

from __future__ import annotations

TEAM1_PLAYERS = (0, 1)
TEAM2_PLAYERS = (2, 3)
DEFENDER_SEATS = (0, 2)
ATTACKER_SEATS = (1, 3)

ALGORITHM_ALIASES: dict[str, str] = {}

TRAINABLE_ALGOS = frozenset({
    "q_learning", "qpg", "a2c", "rpg", "nfsp", "alphazero", "deep_cfr",
})
# NFSP/QPG train shared defender + attacker policies (see role_shared.py).
ROLE_SHARED_ALGOS = frozenset({"nfsp", "qpg"})
# Evaluated via turn-based bot play (search/policy bots), not the RL stack.
TURN_BASED_TRAINED_ALGOS = frozenset({"alphazero", "deep_cfr"})
BOT_ALGOS = frozenset({"random", "mcts", "heuristic"})
EVAL_FALLBACK_ALGOS = {"alphazero": "mcts", "deep_cfr": "heuristic"}

# Master's thesis comparison set (4 paradigms).
THESIS_ALGORITHMS = ["alphazero", "deep_cfr", "nfsp", "qpg"]

# Every algorithm uses the same ``train_episodes`` count (see README_experiments.md).
DEFAULT_TRAIN_EPISODES = 300
DEFAULT_EVAL_EPISODES = 50
DEFAULT_EVAL_EVERY = 50
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def normalize_algorithm(name: str) -> str:
  return ALGORITHM_ALIASES.get(name.strip().lower(), name.strip().lower())


def effective_bot_algorithm(name: str) -> str:
  algo = normalize_algorithm(name)
  return EVAL_FALLBACK_ALGOS.get(algo, algo)
