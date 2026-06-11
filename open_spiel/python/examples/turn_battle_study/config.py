"""Algorithm registry and shared constants."""

from __future__ import annotations

TEAM1_PLAYERS = (0, 1)
TEAM2_PLAYERS = (2, 3)

ALGORITHM_ALIASES: dict[str, str] = {}

TRAINABLE_ALGOS = frozenset({
    "q_learning", "qpg", "a2c", "rpg", "nfsp", "alphazero", "deep_cfr",
})
# Evaluated via turn-based bot play (search/policy bots), not the RL stack.
TURN_BASED_TRAINED_ALGOS = frozenset({"alphazero", "deep_cfr"})
BOT_ALGOS = frozenset({"random", "mcts", "heuristic"})
EVAL_FALLBACK_ALGOS = {"alphazero": "mcts", "deep_cfr": "heuristic"}

ALL_ALGORITHMS = sorted(TRAINABLE_ALGOS | BOT_ALGOS)


def normalize_algorithm(name: str) -> str:
  return ALGORITHM_ALIASES.get(name.strip().lower(), name.strip().lower())


def effective_bot_algorithm(name: str) -> str:
  algo = normalize_algorithm(name)
  return EVAL_FALLBACK_ALGOS.get(algo, algo)
