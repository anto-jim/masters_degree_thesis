"""C++ LibTorch AlphaZero for Turn Battle via a 2-team game view.

OpenSpiel's C++ AlphaZero only supports 2-player sequential zero-sum games.
``turn_battle_teams`` collapses the 4-player team battle into two team players:
each AlphaZero player controls both members of its team (P0+P1 vs P2+P3).

Training launches ``alpha_zero_torch_example`` as a subprocess. Evaluation and
tournaments use Python MCTS with a VPNet MLP that loads the C++ checkpoints.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import time
from typing import List, Optional

from absl import flags
import numpy as np
import torch
from torch import nn

from open_spiel.python.algorithms import mcts
from open_spiel.python.examples.turn_battle_study.device import (
    resolve_cpp_az_devices,
    resolve_device,
)
from open_spiel.python.examples.turn_battle_study.game import (
    load_turn_based_game,
    parse_num_turns,
)
import pyspiel

FLAGS = flags.FLAGS

_METADATA_FILE = "metadata.json"
_RUN_DIR_NAME = "az_cpp_run"
_CHECKPOINT_ALIAS = -1


def alphazero_team_game_string(num_turns: Optional[int] = None) -> str:
  """Return the OpenSpiel game string for the 2-team AlphaZero view.

  Args:
    num_turns: Number of battle turns; if ``None`` the value is read from
      ``FLAGS.num_turns`` (defaulting to 10 if the flag is not yet parsed).

  Returns:
    Game string of the form ``"turn_battle_teams(num_turns=N)"``.
  """
  if num_turns is None:
    try:
      num_turns = parse_num_turns()
    except AttributeError:
      num_turns = 10
  return f"turn_battle_teams(num_turns={num_turns})"


def load_alphazero_team_game() -> pyspiel.Game:
  """Load the ``turn_battle_teams`` game used by the C++ AlphaZero trainer."""
  return pyspiel.load_game(alphazero_team_game_string())


def _repo_root() -> pathlib.Path:
  """Return the repository root directory (four levels above this file)."""
  return pathlib.Path(__file__).resolve().parents[4]


def _native_lib_paths() -> List[str]:
  """Directories needed by the C++ AlphaZero binary at runtime."""
  root = _repo_root()
  paths: List[str] = []
  libtorch_lib = root / "open_spiel" / "libtorch" / "libtorch" / "lib"
  if libtorch_lib.is_dir():
    paths.append(str(libtorch_lib))
  local_cuda = root / ".local" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
  if local_cuda.is_dir():
    paths.append(str(local_cuda))
  return paths


def _subprocess_env() -> dict:
  """Copy of ``os.environ`` with LibTorch/CUDA runtime paths prepended."""
  env = os.environ.copy()
  extra = _native_lib_paths()
  if extra:
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(extra + ([prev] if prev else []))
  return env


def find_az_binary() -> pathlib.Path:
  """Locate the compiled ``alpha_zero_torch_example`` binary.

  Searches the two most common CMake output locations under the repository root.

  Returns:
    Path to the executable.

  Raises:
    FileNotFoundError: If no executable is found in any candidate location.
  """
  candidates = [
      _repo_root() / "build" / "examples" / "alpha_zero_torch_example",
      _repo_root() / "build" / "open_spiel" / "examples" /
      "alpha_zero_torch_example",
  ]
  for path in candidates:
    if path.is_file() and os.access(path, os.X_OK):
      return path
  raise FileNotFoundError(
      "C++ AlphaZero binary not found. Build with OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON "
      "and OPEN_SPIEL_BUILD_WITH_LIBNOP=ON, then run "
      "open_spiel/scripts/build_and_run_tests.sh. Expected one of: "
      + ", ".join(str(p) for p in candidates))


def team_state_from_turn_based(
    turn_based_state: pyspiel.State, team_game: pyspiel.Game
) -> pyspiel.State:
  """Replay turn_based history on the isomorphic 2-team sequential game."""
  team_state = team_game.new_initial_state()
  for action in turn_based_state.history():
    if team_state.is_terminal():
      break
    team_state.apply_action(action)
  return team_state


class _MlpBlock(nn.Module):
  def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.relu(self.linear(x))


class _MlpOutputBlock(nn.Module):
  def __init__(self, nn_width: int, num_actions: int):
    super().__init__()
    self.value_linear_1 = nn.Linear(nn_width, nn_width)
    self.value_linear_2 = nn.Linear(nn_width, 1)
    self.policy_linear_1 = nn.Linear(nn_width, nn_width)
    self.policy_linear_2 = nn.Linear(nn_width, num_actions)

  def forward(self, x: torch.Tensor, mask: torch.Tensor):
    value = torch.tanh(self.value_linear_2(torch.relu(self.value_linear_1(x))))
    logits = self.policy_linear_2(torch.relu(self.policy_linear_1(x)))
    logits = torch.where(mask, logits, -(1 << 16) * torch.ones_like(logits))
    return value.squeeze(-1), logits


class VpNetMlp(nn.Module):
  """MLP VPNet with the same module layout as the C++ LibTorch model."""

  def __init__(self, input_size: int, num_actions: int, width: int, depth: int):
    super().__init__()
    self.layers = nn.ModuleList([_MlpBlock(input_size, width)])
    for _ in range(depth):
      self.layers.append(_MlpBlock(width, width))
    self.layers.append(_MlpOutputBlock(width, num_actions))
    self._num_torso = depth + 1

  def forward(self, x: torch.Tensor, mask: torch.Tensor):
    for i in range(self._num_torso):
      x = self.layers[i](x)
    return self.layers[-1](x, mask)


def _parse_model_config(run_dir: pathlib.Path) -> dict:
  """Parse the VPNet architecture config from a C++ AlphaZero run directory.

  Reads ``vpnet.pb`` which stores whitespace-separated integers/floats in the
  order: channels, height, width, num_actions, nn_depth, nn_width, …, nn_model.

  Args:
    run_dir: Path to the directory produced by ``alpha_zero_torch_example``.

  Returns:
    Dict with keys ``"input_size"``, ``"num_actions"``, ``"nn_depth"``,
    ``"nn_width"``, and ``"nn_model"``.

  Raises:
    FileNotFoundError: If ``vpnet.pb`` does not exist.
    ValueError: If ``vpnet.pb`` contains fewer fields than expected.
  """
  config_path = run_dir / "vpnet.pb"
  if not config_path.is_file():
    raise FileNotFoundError(f"Missing VPNet config at {config_path}")
  values = config_path.read_text(encoding="utf-8").strip().split()
  if len(values) < 9:
    raise ValueError(f"Unexpected vpnet.pb format in {config_path}: {values!r}")
  channels, height, width = (int(float(values[0])), int(float(values[1])),
                             int(float(values[2])))
  input_size = channels * max(1, height) * max(1, width)
  return {
      "input_size": input_size,
      "num_actions": int(float(values[3])),
      "nn_depth": int(float(values[4])),
      "nn_width": int(float(values[5])),
      "nn_model": values[8],
  }


class VpNetEvaluator(mcts.Evaluator):
  """MCTS evaluator backed by a C++ VPNet checkpoint."""

  def __init__(self, net: VpNetMlp, device: torch.device):
    self._net = net
    self._device = device

  def _infer(self, state: pyspiel.State):
    obs = np.asarray(state.observation_tensor(), dtype=np.float32)
    player = state.current_player()
    legal = state.legal_actions(player)
    mask = np.zeros(state.get_game().num_distinct_actions(), dtype=bool)
    mask[legal] = True
    with torch.no_grad():
      obs_t = torch.from_numpy(obs).unsqueeze(0).to(self._device)
      mask_t = torch.from_numpy(mask).unsqueeze(0).to(self._device)
      value, logits = self._net(obs_t, mask_t)
    logits_np = logits.squeeze(0).cpu().numpy()
    probs = np.zeros_like(logits_np, dtype=np.float64)
    exp = np.exp(logits_np[legal] - logits_np[legal].max())
    probs[legal] = exp / exp.sum()
    return float(value.item()), probs

  def evaluate(self, state):
    value, _ = self._infer(state)
    return np.asarray([value, -value], dtype=np.float64)

  def prior(self, state):
    if state.is_chance_node():
      return state.chance_outcomes()
    _, probs = self._infer(state)
    player = state.current_player()
    legal = state.legal_actions(player)
    return [(int(a), float(probs[a])) for a in legal]


class TeamAlphaZeroBot(pyspiel.Bot):
  """Plays one seat in the 4-player turn_based game using a shared team policy."""

  def __init__(
      self,
      player_id: int,
      team_id: int,
      team_game: pyspiel.Game,
      turn_based_game: pyspiel.Game,
      mcts_bot: mcts.MCTSBot,
  ):
    super().__init__()
    self._player_id = player_id
    self._team_id = team_id
    self._team_game = team_game
    self._turn_based_game = turn_based_game
    self._mcts_bot = mcts_bot

  def restart_at(self, turn_based_state: pyspiel.State):
    team_state = team_state_from_turn_based(turn_based_state, self._team_game)
    self._mcts_bot.restart_at(team_state)

  def player_id(self):
    return self._player_id

  def step(self, turn_based_state: pyspiel.State):
    if turn_based_state.is_chance_node() or turn_based_state.is_terminal():
      return pyspiel.INVALID_ACTION
    if turn_based_state.current_player() != self._player_id:
      return pyspiel.INVALID_ACTION
    team_state = team_state_from_turn_based(turn_based_state, self._team_game)
    if team_state.current_player() != self._team_id:
      return pyspiel.INVALID_ACTION
    return self._mcts_bot.step(team_state)


class AlphaZeroCpp:
  """C++ AlphaZero trainer/evaluator for the 2-team Turn Battle view."""

  def __init__(self, rng: np.random.RandomState, run_dir: Optional[str] = None):
    self._rng = rng
    self._team_game = load_alphazero_team_game()
    self._turn_based_game = load_turn_based_game()
    self._run_dir = pathlib.Path(run_dir) if run_dir else None
    self._evaluator: Optional[VpNetEvaluator] = None
    self._device = resolve_device(FLAGS.device)

  @property
  def run_dir(self) -> Optional[pathlib.Path]:
    return self._run_dir

  def _ensure_run_dir(self, base_dir: str) -> pathlib.Path:
    run_dir = pathlib.Path(base_dir) / _RUN_DIR_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    self._run_dir = run_dir
    return run_dir

  def _build_train_command(self, run_dir: pathlib.Path, max_steps: int) -> List[str]:
    binary = find_az_binary()
    device = resolve_cpp_az_devices(FLAGS.device)
    cmd = [
        str(binary),
        f"--game={alphazero_team_game_string()}",
        f"--path={run_dir}",
        f"--nn_model={FLAGS.az_cpp_nn_model}",
        f"--nn_width={FLAGS.az_cpp_nn_width}",
        f"--nn_depth={FLAGS.az_cpp_nn_depth}",
        f"--learning_rate={FLAGS.az_learning_rate}",
        f"--weight_decay={FLAGS.az_cpp_weight_decay}",
        f"--train_batch_size={FLAGS.az_batch_size}",
        f"--replay_buffer_size={FLAGS.az_replay_buffer_size}",
        f"--replay_buffer_reuse={FLAGS.az_cpp_replay_reuse}",
        f"--checkpoint_freq={FLAGS.az_cpp_checkpoint_freq}",
        f"--uct_c={FLAGS.mcts_uct_c}",
        f"--max_simulations={FLAGS.mcts_simulations}",
        f"--temperature={FLAGS.az_temperature}",
        f"--temperature_drop={FLAGS.az_temperature_drop}",
        f"--actors={FLAGS.az_cpp_actors}",
        f"--evaluators={FLAGS.az_cpp_evaluators}",
        f"--eval_levels={FLAGS.az_cpp_eval_levels}",
        f"--evaluation_window={FLAGS.az_cpp_evaluation_window}",
        f"--devices={device}",
        f"--max_steps={max_steps}",
    ]
    return cmd

  def _latest_learner_step(self, run_dir: pathlib.Path) -> int:
    learner_path = run_dir / "learner.jsonl"
    if not learner_path.is_file():
      return 0
    last_step = 0
    for line in learner_path.read_text(encoding="utf-8").splitlines():
      if not line.strip():
        continue
      try:
        record = json.loads(line)
        last_step = max(last_step, int(record.get("step", 0)))
      except json.JSONDecodeError:
        continue
    return last_step

  def train(
      self,
      episodes: int,
      eval_every: int,
      eval_callback,
  ) -> None:
    run_dir = self._ensure_run_dir(FLAGS.output_dir)
    cmd = self._build_train_command(run_dir, episodes)
    print(f"  launching C++ AlphaZero: {' '.join(cmd)}")

    log_path = run_dir / "cpp_train.log"
    env = _subprocess_env()
    with open(log_path, "w", encoding="utf-8") as log_fp:
      proc = subprocess.Popen(
          cmd, stdout=log_fp, stderr=subprocess.STDOUT, text=True, env=env)
    last_eval_step = 0
    try:
      while proc.poll() is None:
        step = self._latest_learner_step(run_dir)
        if step > 0 and step >= last_eval_step + eval_every:
          self.load_checkpoint(run_dir)
          eval_callback(step)
          last_eval_step = step
        time.sleep(2.0)
      if proc.returncode != 0:
        output = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        raise RuntimeError(
            f"C++ AlphaZero exited with code {proc.returncode}.\n{output}")
      final_step = self._latest_learner_step(run_dir)
      if final_step > last_eval_step:
        self.load_checkpoint(run_dir)
        eval_callback(final_step)
    finally:
      if proc.poll() is None:
        proc.terminate()

  def _load_vpnet(self, run_dir: pathlib.Path) -> VpNetEvaluator:
    cfg = _parse_model_config(run_dir)
    if cfg["nn_model"] != "mlp":
      raise ValueError(
          f"Only mlp VPNet is supported in Python eval, got {cfg['nn_model']!r}")
    net = VpNetMlp(cfg["input_size"], cfg["num_actions"],
                   cfg["nn_width"], cfg["nn_depth"]).to(self._device)
    ckpt_path = run_dir / f"checkpoint-{_CHECKPOINT_ALIAS}.pt"
    if not ckpt_path.is_file():
      raise FileNotFoundError(f"Missing C++ checkpoint at {ckpt_path}")
    # C++ LibTorch saves TorchScript modules; copy weights into our MLP layout.
    script_module = torch.jit.load(str(ckpt_path), map_location=self._device)
    net.load_state_dict(script_module.state_dict(), strict=True)
    net.eval()
    return VpNetEvaluator(net, self._device)

  def load_checkpoint(self, path: str | pathlib.Path) -> None:
    run_dir = pathlib.Path(path)
    if run_dir.is_file():
      run_dir = run_dir.parent
    self._run_dir = run_dir
    self._evaluator = self._load_vpnet(run_dir)

  def _team_mcts_bot(self) -> mcts.MCTSBot:
    if self._evaluator is None:
      raise RuntimeError("Load a C++ AlphaZero checkpoint before creating bots.")
    return mcts.MCTSBot(
        self._team_game,
        FLAGS.mcts_uct_c,
        FLAGS.mcts_simulations,
        self._evaluator,
        solve=True,
        random_state=self._rng,
        dont_return_chance_node=True,
    )

  def make_bots(self) -> List[pyspiel.Bot]:
    team0_bot = self._team_mcts_bot()
    team1_bot = self._team_mcts_bot()
    return [
        TeamAlphaZeroBot(0, 0, self._team_game, self._turn_based_game, team0_bot),
        TeamAlphaZeroBot(1, 0, self._team_game, self._turn_based_game, team0_bot),
        TeamAlphaZeroBot(2, 1, self._team_game, self._turn_based_game, team1_bot),
        TeamAlphaZeroBot(3, 1, self._team_game, self._turn_based_game, team1_bot),
    ]

  def save_checkpoint(self, checkpoint_dir: str) -> None:
    if self._run_dir is None:
      raise RuntimeError("No C++ AlphaZero run directory to save.")
    dst = pathlib.Path(checkpoint_dir)
    dst.mkdir(parents=True, exist_ok=True)
    if dst.exists() and any(dst.iterdir()):
      for child in dst.iterdir():
        if child.is_dir():
          shutil.rmtree(child)
        else:
          child.unlink()
    for item in self._run_dir.iterdir():
      target = dst / item.name
      if item.is_dir():
        shutil.copytree(item, target, dirs_exist_ok=True)
      else:
        shutil.copy2(item, target)
    meta = {
        "backend": "cpp_alpha_zero",
        "team_game": alphazero_team_game_string(),
        "checkpoint_step": _CHECKPOINT_ALIAS,
    }
    with open(dst / _METADATA_FILE, "w", encoding="utf-8") as f:
      json.dump(meta, f, indent=2)
