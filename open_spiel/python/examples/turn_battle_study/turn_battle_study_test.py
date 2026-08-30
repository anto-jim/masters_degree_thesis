"""Regression tests for the Turn Battle MARL study package.

Each test here pins down a defect that silently corrupted results rather than
crashing, so they are worth keeping even though they are slow relative to a
unit test: an evaluation that quietly runs on the wrong game, or a multi-seed
campaign that quietly reuses one seed's network, produces plausible-looking
numbers that are simply wrong.

Run with either runner:
  PYTHONPATH=. ./venv/bin/python \
    open_spiel/python/examples/turn_battle_study/turn_battle_study_test.py
  PYTHONPATH=. ./venv/bin/python -m pytest \
    open_spiel/python/examples/turn_battle_study/turn_battle_study_test.py -q
"""

from __future__ import annotations

import json
import pathlib
import tempfile

from absl import flags
from absl.testing import absltest
import numpy as np
import torch

from open_spiel.python.examples.turn_battle_study import aggregate
from open_spiel.python.examples.turn_battle_study import az_cpp
from open_spiel.python.examples.turn_battle_study import bots
from open_spiel.python.examples.turn_battle_study import evaluation
from open_spiel.python.examples.turn_battle_study import game as game_mod
from open_spiel.python.examples.turn_battle_study import role_shared
from open_spiel.python.examples.turn_battle_study.seeding import seed_everything
from open_spiel.python.pytorch import dqn
import pyspiel

FLAGS = flags.FLAGS


def _define_missing_flags() -> None:
  """Define and parse the CLI flags the study modules read.

  Called at import rather than from ``__main__`` so the suite behaves the same
  under the absltest runner and under pytest. Only absltest parses the command
  line, and an absl flag that is defined but unparsed raises on every read, so
  the flags are marked parsed here too; absltest reparses them for real
  afterwards.
  """
  defaults = {
      "game": "turn_battle",
      "game_params": "num_turns=5",
      "dcfr_max_turns": 0,
      "seed": 42,
      "mcts_simulations": 10,
      "mcts_uct_c": 2.0,
      "mcts_rollouts": 1,
      "device": "cpu",
      "output_dir": "/tmp/turn_battle_test",
  }
  for name, value in defaults.items():
    if name in FLAGS:
      continue
    if isinstance(value, bool):
      flags.DEFINE_boolean(name, value, "test flag")
    elif isinstance(value, int):
      flags.DEFINE_integer(name, value, "test flag")
    elif isinstance(value, float):
      flags.DEFINE_float(name, value, "test flag")
    else:
      flags.DEFINE_string(name, value, "test flag")
  if "allow_dcfr_horizon_mismatch" not in FLAGS:
    flags.DEFINE_boolean("allow_dcfr_horizon_mismatch", False, "test flag")
  if not FLAGS.is_parsed():
    FLAGS.mark_as_parsed()


_define_missing_flags()


class HorizonConsistencyTest(absltest.TestCase):
  """A Deep CFR horizon cap must not silently shrink the evaluation game."""

  def setUp(self):
    super().setUp()
    FLAGS.game_params = "num_turns=5"
    FLAGS.dcfr_max_turns = 0
    FLAGS.allow_dcfr_horizon_mismatch = False

  def test_uncapped_horizon_matches_game_params(self):
    self.assertEqual(game_mod.parse_num_turns(), 5)
    self.assertEqual(game_mod.parse_num_turns(dcfr_cap=True), 5)

  def test_cap_below_num_turns_is_rejected(self):
    FLAGS.dcfr_max_turns = 3
    with self.assertRaisesRegex(ValueError, "below num_turns"):
      game_mod.parse_num_turns(dcfr_cap=True)

  def test_cap_below_num_turns_allowed_with_explicit_optin(self):
    FLAGS.dcfr_max_turns = 3
    FLAGS.allow_dcfr_horizon_mismatch = True
    self.assertEqual(game_mod.parse_num_turns(dcfr_cap=True), 3)

  def test_cap_above_num_turns_is_ignored(self):
    FLAGS.dcfr_max_turns = 99
    self.assertEqual(game_mod.parse_num_turns(dcfr_cap=True), 5)

  def test_mismatched_info_state_width_raises_instead_of_padding(self):
    """A 3-turn state must not be reshaped into a 5-turn observation."""
    short_game = pyspiel.load_game(
        "turn_based_simultaneous_game(game=turn_battle(num_turns=3))")
    state = short_game.new_initial_state()
    with self.assertRaisesRegex(ValueError, "does not match the canonical"):
      bots._time_step_from_turn_based_state(state)

  def test_deep_cfr_horizon_guard_detects_wrong_game(self):
    class _StubSolver:
      def __init__(self, game):
        self._game = game

    class _Agent:
      def __init__(self, game):
        self._bot = bots.DeepCFRPolicyBot(
            0, np.random.RandomState(0), _StubSolver(game))

    canonical = game_mod.load_turn_based_game(num_turns=5)
    smaller = game_mod.load_turn_based_game(num_turns=3)
    # Same horizon: accepted.
    evaluation._assert_deep_cfr_horizon_matches(
        canonical, [_Agent(canonical)], None)
    with self.assertRaisesRegex(ValueError, "same num_turns"):
      evaluation._assert_deep_cfr_horizon_matches(
          canonical, [_Agent(smaller)], None)


class LearnerStepTrackingTest(absltest.TestCase):
  """AlphaZero progress must be read from the current run, not the file max."""

  def test_latest_step_uses_last_record_not_maximum(self):
    trainer = object.__new__(az_cpp.AlphaZeroCpp)
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = pathlib.Path(tmp)
      # A previous seed left steps 1..300; the current run has reached step 4.
      lines = [json.dumps({"step": s}) for s in (100, 200, 300)]
      lines += [json.dumps({"step": s}) for s in (1, 2, 3, 4)]
      (run_dir / "learner.jsonl").write_text("\n".join(lines) + "\n")
      self.assertEqual(trainer._latest_learner_step(run_dir), 4)

  def test_missing_learner_file_reports_zero(self):
    trainer = object.__new__(az_cpp.AlphaZeroCpp)
    with tempfile.TemporaryDirectory() as tmp:
      self.assertEqual(trainer._latest_learner_step(pathlib.Path(tmp)), 0)

  def test_fresh_run_dir_discards_previous_run(self):
    trainer = object.__new__(az_cpp.AlphaZeroCpp)
    with tempfile.TemporaryDirectory() as tmp:
      stale = pathlib.Path(tmp) / az_cpp._RUN_DIR_NAME
      stale.mkdir()
      (stale / "learner.jsonl").write_text('{"step": 300}\n')
      run_dir = trainer._ensure_run_dir(tmp, fresh=True)
      self.assertFalse((run_dir / "learner.jsonl").exists())

  def test_non_fresh_run_dir_keeps_previous_run(self):
    trainer = object.__new__(az_cpp.AlphaZeroCpp)
    with tempfile.TemporaryDirectory() as tmp:
      existing = pathlib.Path(tmp) / az_cpp._RUN_DIR_NAME
      existing.mkdir()
      (existing / "learner.jsonl").write_text('{"step": 300}\n')
      run_dir = trainer._ensure_run_dir(tmp, fresh=False)
      self.assertTrue((run_dir / "learner.jsonl").exists())


class RoleSharedBookkeepingTest(absltest.TestCase):
  """Two seats sharing one learner must not share transition bookkeeping."""

  def test_each_seat_keeps_its_own_previous_timestep(self):
    class _RecordingAgent:
      """Stands in for NFSP/QPG: records the prev state it was handed."""

      def __init__(self):
        self._prev_time_step = None
        self._prev_action = None
        self._episode_data = []
        self.seen = []

      def step(self, time_step, is_evaluation=False):
        del is_evaluation
        self.seen.append(self._prev_time_step)
        self._prev_time_step = time_step
        self._prev_action = 1
        self._episode_data = self._episode_data + [time_step]

        class _Out:
          action = 0
        return _Out()

    agent = _RecordingAgent()
    slots = [(agent, "_prev_time_step"), (agent, "_prev_action"),
             (agent, "_episode_data")]
    original = role_shared._bookkeeping_slots
    role_shared._bookkeeping_slots = lambda a: slots
    try:
      seat0 = role_shared.RoleSeatFacade(0, agent, "defender")
      seat2 = role_shared.RoleSeatFacade(2, agent, "defender")
      seat0.step(self._time_step(marker=1.0))
      seat2.step(self._time_step(marker=2.0))
      # Neither seat had a previous timestep of its own yet. Without per-seat
      # bookkeeping, seat 2 would have been handed seat 0's state here.
      self.assertIsNone(agent.seen[0])
      self.assertIsNone(agent.seen[1])
      seat0.step(self._time_step(marker=3.0))
      seat2.step(self._time_step(marker=4.0))
      # Each seat's second call must see that seat's own first observation,
      # identified by the marker carried in the role-relative vector.
      self.assertEqual(self._marker_of(agent.seen[2]), 1.0)
      self.assertEqual(self._marker_of(agent.seen[3]), 2.0)
      self.assertLen(seat0._seat_state[2], 2)
      self.assertLen(seat2._seat_state[2], 2)
    finally:
      role_shared._bookkeeping_slots = original

  @staticmethod
  def _marker_of(time_step) -> float:
    return time_step.observations["info_state"][role_shared.DEFENDER_CANON][0]

  @staticmethod
  def _time_step(marker: float):
    from open_spiel.python import rl_environment
    info = [[marker] + [0.0] * 30 for _ in range(4)]
    return rl_environment.TimeStep(
        observations={
            "info_state": info,
            "legal_actions": [[0, 1, 2, 3] for _ in range(4)],
            "current_player": -2,
            "serialized_state": None,
        },
        rewards=[0.0] * 4,
        discounts=[1.0] * 4,
        step_type=rl_environment.StepType.MID,
    )


class DqnCheckpointTest(absltest.TestCase):
  """DQN.load must read the keys DQN.save writes."""

  def test_save_load_round_trip_restores_weights(self):
    agent = dqn.DQN(
        player_id=0, state_representation_size=8, num_actions=4,
        hidden_layers_sizes=[16], replay_buffer_capacity=10, batch_size=2)
    with torch.no_grad():
      for param in agent._q_network.parameters():
        param.add_(1.0)
    expected = [p.detach().clone() for p in agent._q_network.parameters()]

    with tempfile.TemporaryDirectory() as tmp:
      path = pathlib.Path(tmp) / "q_network.pt"
      agent.save(path, save_optimiser=True)
      restored = dqn.DQN(
          player_id=0, state_representation_size=8, num_actions=4,
          hidden_layers_sizes=[16], replay_buffer_capacity=10, batch_size=2)
      restored.load(path, load_optimiser=True)

    for want, got in zip(expected, restored._q_network.parameters()):
      torch.testing.assert_close(want, got)


class SeedingTest(absltest.TestCase):
  """Per-seed runs must actually re-seed the global generators."""

  def test_same_seed_gives_identical_network_init(self):
    seed_everything(7)
    first = torch.nn.Linear(4, 4).weight.detach().clone()
    seed_everything(7)
    second = torch.nn.Linear(4, 4).weight.detach().clone()
    torch.testing.assert_close(first, second)

  def test_different_seeds_give_different_network_init(self):
    seed_everything(7)
    first = torch.nn.Linear(4, 4).weight.detach().clone()
    seed_everything(8)
    second = torch.nn.Linear(4, 4).weight.detach().clone()
    self.assertFalse(torch.allclose(first, second))

  def test_numpy_and_python_streams_are_seeded(self):
    import random
    seed_everything(11)
    a = (np.random.rand(3).tolist(), random.random())
    seed_everything(11)
    b = (np.random.rand(3).tolist(), random.random())
    self.assertEqual(a, b)


class StatisticsTest(absltest.TestCase):
  """Small-sample intervals must not use the normal approximation."""

  def test_three_seed_interval_is_wider_than_normal_approximation(self):
    values = [0.70, 0.72, 0.74]
    summary = aggregate.mean_ci(values)
    normal_margin = 1.96 * np.std(values, ddof=1) / np.sqrt(3)
    t_margin = summary["mean"] - summary["ci_low"]
    self.assertGreater(t_margin, normal_margin)
    self.assertIn("student-t", summary["ci_method"])
    self.assertEqual(summary["n_seeds"], 3)
    self.assertEqual(summary["values"], values)

  def test_rate_interval_is_clipped_to_unit_range(self):
    summary = aggregate.mean_ci([0.02, 0.98, 0.50])
    self.assertGreaterEqual(summary["ci_low"], 0.0)
    self.assertLessEqual(summary["ci_high"], 1.0)

  def test_difference_interval_is_not_clipped(self):
    summary = aggregate.mean_ci([-0.4, -0.5, -0.6], clip=False)
    self.assertLess(summary["ci_high"], 0.0)

  def test_single_seed_reports_no_interval(self):
    summary = aggregate.mean_ci([0.5])
    self.assertEqual(summary["ci_low"], summary["ci_high"])
    self.assertEqual(summary["n_seeds"], 1)

  def test_paired_comparison_counts_per_seed_winners(self):
    result = aggregate.paired_comparison([0.7, 0.8, 0.9], [0.4, 0.5, 0.95])
    self.assertEqual(result["seeds_favouring_a"], 2)
    self.assertEqual(result["seeds_favouring_b"], 1)
    self.assertAlmostEqual(result["mean_difference"], 0.55 / 3.0, places=6)
    self.assertIsNotNone(result["p_value"])

  def test_paired_comparison_handles_zero_variance(self):
    # Exactly representable so the per-seed differences are bitwise identical.
    result = aggregate.paired_comparison([0.5, 0.75], [0.25, 0.5])
    self.assertIsNone(result["p_value"])
    self.assertEqual(result["seeds_favouring_a"], 2)


class SlotBalanceTest(absltest.TestCase):
  """Pairings must be played in both team slots and averaged."""

  def test_pairwise_win_rate_averages_both_orientations(self):
    from open_spiel.python.examples.turn_battle_study import tournament

    calls = []

    def _fake_matchup(t1, t2, episodes, rng, team1_agents=None,
                      team2_agents=None):
      del rng, team1_agents, team2_agents
      calls.append((t1, t2, episodes))

      class _Result:
        def summary(self_inner):
          # Team 1 always wins 80% of the time regardless of who sits there:
          # a pure slot advantage that balancing must cancel to 0.5.
          return {
              "episodes": episodes,
              "team1_win_rate": 0.8,
              "team2_win_rate": 0.2,
              "draw_rate": 0.0,
              "team1_avg_return": 1.0,
              "team2_avg_return": -1.0,
          }
      return _Result()

    original = tournament.evaluate_team_matchup
    tournament.evaluate_team_matchup = _fake_matchup
    try:
      a_rate, b_rate, stats = tournament._pairwise_win_rate(
          "nfsp", "qpg", 50, np.random.RandomState(0), {})
    finally:
      tournament.evaluate_team_matchup = original

    self.assertEqual([(c[0], c[1]) for c in calls],
                     [("nfsp", "qpg"), ("qpg", "nfsp")])
    self.assertAlmostEqual(a_rate, 0.5)
    self.assertAlmostEqual(b_rate, 0.5)
    self.assertEqual(stats["episodes"], 50)
    self.assertTrue(stats["slot_balanced"])


if __name__ == "__main__":
  absltest.main()
