# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Play turn_battle manually via the console (simultaneous-move mode).

Each battle turn, every living player chooses an action; all four are applied
together with ``apply_actions``. Use ``--playerN_type=human`` for seats you
control and ``uniform`` for random opponents.

Example (you play team 1, random team 2):

  PYTHONPATH=.:build/python python3 \\
    open_spiel/python/examples/play_turn_battle_console.py \\
    --player0_type=human --player1_type=human \\
    --player2_type=uniform --player3_type=uniform
"""

from absl import app
from absl import flags

import numpy as np
from open_spiel.python.bots import uniform_random
import pyspiel

_NUM_TURNS = flags.DEFINE_integer(
    "num_turns", 15, "Number of battle turns before draw.")
_SEED = flags.DEFINE_integer("seed", 42, "RNG seed for uniform bots.")
_PLAYER_LABELS = (
    "P0 (Defender, Team 1)",
    "P1 (Attacker, Team 1)",
    "P2 (Defender, Team 2)",
    "P3 (Attacker, Team 2)",
)
_PLAYER0_TYPE = flags.DEFINE_string(
    "player0_type", "human", "P0 bot type: human or uniform.")
_PLAYER1_TYPE = flags.DEFINE_string(
    "player1_type", "human", "P1 bot type: human or uniform.")
_PLAYER2_TYPE = flags.DEFINE_string(
    "player2_type", "uniform", "P2 bot type: human or uniform.")
_PLAYER3_TYPE = flags.DEFINE_string(
    "player3_type", "uniform", "P3 bot type: human or uniform.")

_PLAYER_TYPE_FLAGS = (
    _PLAYER0_TYPE,
    _PLAYER1_TYPE,
    _PLAYER2_TYPE,
    _PLAYER3_TYPE,
)


def _prompt_human_action(state: pyspiel.State, player: int) -> int:
  """Ask the user which action to play for ``player``."""
  legal_actions = state.legal_actions(player)
  if not legal_actions:
    return pyspiel.INVALID_ACTION
  if len(legal_actions) == 1:
    return legal_actions[0]

  action_map = {
      state.action_to_string(player, action): action
      for action in legal_actions
  }

  while True:
    action_str = input(
        f"{_PLAYER_LABELS[player]} — choose action "
        "(empty to list legal actions): "
    )

    if not action_str:
      print("Legal actions:")
      for name, action in sorted(action_map.items()):
        print(f"  {action}: {name}")
      continue

    if action_str in action_map:
      return action_map[action_str]

    try:
      action = int(action_str)
    except ValueError:
      print(f"Could not parse action: {action_str}")
      continue

    if action in legal_actions:
      return action

    print(f"Illegal action: {action_str}")


def _load_bot(bot_type: str, player_id: int, rng: np.random.RandomState):
  if bot_type == "human":
    return None
  if bot_type == "uniform":
    return uniform_random.UniformRandomBot(player_id, rng)
  raise ValueError(
      f"Unknown bot type {bot_type!r} for player {player_id}; "
      "use 'human' or 'uniform'.")


def _choose_action(
    state: pyspiel.State,
    player: int,
    bot_type: str,
    bot: uniform_random.UniformRandomBot | None,
) -> int:
  if bot_type == "human":
    return _prompt_human_action(state, player)
  return bot.step(state)


def play_game(state: pyspiel.State, bot_types: list[str], bots: list) -> None:
  """Play one turn_battle episode via the console."""
  turn = 0
  while not state.is_terminal():
    print(f"\n=== Turn {turn + 1} ===")
    print(state)

    if not state.is_simultaneous_node():
      raise RuntimeError(
          "Expected a simultaneous node in turn_battle, got "
          f"current_player={state.current_player()}."
      )

    actions = []
    for player in range(state.num_players()):
      legal = state.legal_actions(player)
      print(f"\n{_PLAYER_LABELS[player]} legal actions:")
      for action in legal:
        print(f"  {action}: {state.action_to_string(player, action)}")

      action = _choose_action(state, player, bot_types[player], bots[player])
      actions.append(action)
      print(
          f"{_PLAYER_LABELS[player]} chose {action}: "
          f"{state.action_to_string(player, action)}"
      )

    state.apply_actions(actions)
    turn += 1

  print("\n-=- Game over -=-\n")
  print(state)
  print(f"Returns: {state.returns()}")


def main(_):
  bot_types = [flag.value for flag in _PLAYER_TYPE_FLAGS]
  rng = np.random.RandomState(_SEED.value)
  game = pyspiel.load_game(f"turn_battle(num_turns={_NUM_TURNS.value})")
  bots = [
      _load_bot(bot_type, player_id, rng)
      for player_id, bot_type in enumerate(bot_types)
  ]
  state = game.new_initial_state()
  play_game(state, bot_types, bots)


if __name__ == "__main__":
  app.run(main)
