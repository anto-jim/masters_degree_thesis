"""Game loading helpers."""

from __future__ import annotations

from typing import Dict, Optional

from absl import flags
from open_spiel.python import rl_environment
import pyspiel

FLAGS = flags.FLAGS


def parse_game_params(params_str: str) -> Dict[str, object]:
  """Parse a comma-separated ``key=value`` game-parameter string.

  Values are coerced in order: integer, float, boolean (``true``/``false``),
  and finally left as a raw string.  Entries that do not contain ``=`` are
  silently skipped.

  Args:
    params_str: Parameter string such as ``"num_turns=10"``.

  Returns:
    A dict mapping parameter names to their Python-typed values, or an empty
    dict when *params_str* is falsy.
  """
  if not params_str:
    return {}
  result: Dict[str, object] = {}
  for part in params_str.split(","):
    if "=" not in part:
      continue
    key, value = part.split("=", 1)
    key, value = key.strip(), value.strip()
    try:
      result[key] = int(value)
    except ValueError:
      try:
        result[key] = float(value)
      except ValueError:
        result[key] = value.lower() == "true" if value.lower() in (
            "true", "false") else value
  return result


def parse_num_turns(dcfr_cap: bool = False) -> int:
  """Return num_turns from game_params, optionally capped by dcfr_max_turns."""
  params = parse_game_params(FLAGS.game_params)
  num_turns = int(params.get("num_turns", 10))
  if dcfr_cap:
    return min(num_turns, FLAGS.dcfr_max_turns)
  return num_turns


def inner_game_string() -> str:
  """Return the pyspiel game string for the inner simultaneous game.

  Builds the string from ``FLAGS.game`` and any parameters in
  ``FLAGS.game_params``, e.g. ``"turn_battle(num_turns=10)"``.
  """
  params = parse_game_params(FLAGS.game_params)
  if not params:
    return FLAGS.game
  param_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
  return f"{FLAGS.game}({param_str})"


def load_game() -> pyspiel.Game:
  """Load the inner simultaneous game from ``FLAGS.game`` and ``FLAGS.game_params``."""
  params = parse_game_params(FLAGS.game_params)
  return pyspiel.load_game(FLAGS.game, params) if params else pyspiel.load_game(
      FLAGS.game)


def load_turn_based_game(num_turns: Optional[int] = None) -> pyspiel.Game:
  """Wrap the inner simultaneous game in ``turn_based_simultaneous_game``.

  Args:
    num_turns: Override the ``num_turns`` game parameter.  When ``None`` the
      value from ``FLAGS.game_params`` is used unchanged.

  Returns:
    A ``pyspiel.Game`` instance using the turn-based simultaneous wrapper.
  """
  if num_turns is None:
    return pyspiel.load_game(
        f"turn_based_simultaneous_game(game={inner_game_string()})")
  params = parse_game_params(FLAGS.game_params)
  params["num_turns"] = num_turns
  param_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
  return pyspiel.load_game(
      f"turn_based_simultaneous_game(game={FLAGS.game}({param_str}))")


def make_rl_environment(include_full_state: bool = False) -> rl_environment.Environment:
  """Create an RL environment for the inner simultaneous game.

  Args:
    include_full_state: When ``True``, the full game state is accessible from
      returned time steps (required by some RL agents for auxiliary features).

  Returns:
    An ``rl_environment.Environment`` configured from ``FLAGS``.
  """
  return rl_environment.Environment(
      FLAGS.game,
      include_full_state=include_full_state,
      **parse_game_params(FLAGS.game_params))
