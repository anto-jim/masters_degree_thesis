"""Bracket-style elimination tournament with optional round-robin phase."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

from absl import flags
import numpy as np

from open_spiel.python.examples.turn_battle_study.config import (
    BOT_ALGOS,
    TRAINABLE_ALGOS,
    normalize_algorithm,
)

FLAGS = flags.FLAGS
from open_spiel.python.examples.turn_battle_study.evaluation import evaluate_team_matchup
from open_spiel.python.examples.turn_battle_study.models import BracketMatch
from open_spiel.python.examples.turn_battle_study.storage import save_training_log
from open_spiel.python.examples.turn_battle_study.trainers import train_algorithm


def _pick_winner(algo1: str, algo2: str, stats: Dict[str, float]) -> str:
  if stats["team1_win_rate"] > stats["team2_win_rate"]:
    return algo1
  if stats["team2_win_rate"] > stats["team1_win_rate"]:
    return algo2
  return algo1 if stats["team1_avg_return"] >= stats["team2_avg_return"] else algo2


def _seed_score(
    algo: str,
    agents: object,
    eval_episodes: int,
    rng: np.random.RandomState,
) -> float:
  """Average win rate vs random from both team slots (roles are asymmetric)."""
  half = max(1, eval_episodes // 2)
  fwd = evaluate_team_matchup(
      algo, "random", half, rng, team1_agents=agents).summary()
  rev = evaluate_team_matchup(
      "random", algo, eval_episodes - half, rng, team2_agents=agents).summary()
  return (fwd["team1_win_rate"] + rev["team2_win_rate"]) / 2.0


def _train_budget(algo: str, episodes: int) -> int:
  if algo == "alphazero":
    return min(episodes, getattr(FLAGS, "az_train_episodes", 500))
  if algo == "deep_cfr":
    return min(episodes, getattr(FLAGS, "dcfr_iterations", 200))
  if algo in {"nfsp", "qpg"}:
    return episodes
  return episodes


def train_all(
    algorithms: Sequence[str],
    train_episodes: int,
    eval_every: int,
    eval_episodes: int,
    rng: np.random.RandomState,
    output_dir: str,
) -> Tuple[Dict[str, object], Dict[str, float]]:
  trained: Dict[str, object] = {}
  seeds: Dict[str, float] = {}

  for algo in algorithms:
    key = normalize_algorithm(algo)
    budget = _train_budget(key, train_episodes)
    print(f"=== Training {key} ({budget} episodes) ===")
    if key in TRAINABLE_ALGOS:
      eval_interval = max(1, min(eval_every // 2, max(budget // 4, 25)))
      agents, log = train_algorithm(
          key, budget, eval_interval, eval_episodes, rng)
      save_training_log(log, output_dir)
      trained[key] = agents
      seeds[key] = _seed_score(key, agents, eval_episodes, rng)
    elif key in BOT_ALGOS:
      trained[key] = None
      stats = evaluate_team_matchup(key, "random", eval_episodes, rng).summary()
      seeds[key] = stats["team1_win_rate"]
    else:
      continue
    print(f"  seed score vs random: {seeds[key]:.3f}")
  return trained, seeds


def run_bracket(
    trained: Dict[str, Optional[object]],
    seeds: Dict[str, float],
    bracket_size: int,
    eval_episodes: int,
    rng: np.random.RandomState,
) -> Tuple[str, List[BracketMatch]]:
  eligible = list(seeds.keys())
  ranked = sorted(eligible, key=seeds.get, reverse=True)[:bracket_size]
  target = 1
  while target < len(ranked):
    target <<= 1
  while len(ranked) < target:
    ranked.append(ranked[-1])

  matches: List[BracketMatch] = []
  round_num = 1
  field = ranked[:]

  while len(field) > 1:
    next_round: List[str] = []
    for i in range(0, len(field), 2):
      a, b = field[i], field[i + 1]
      if a == b:
        next_round.append(a)
        continue
      agents_a = trained.get(a)
      agents_b = trained.get(b)
      stats = evaluate_team_matchup(
          a, b, eval_episodes, rng,
          team1_agents=agents_a, team2_agents=agents_b).summary()
      winner = _pick_winner(a, b, stats)
      matches.append(BracketMatch(round_num, a, b, winner, stats))
      next_round.append(winner)
      print(f"R{round_num}: {a} vs {b} -> {winner} "
            f"({stats['team1_win_rate']:.2f}-{stats['team2_win_rate']:.2f})")
    field = next_round
    round_num += 1
  return field[0], matches


def _pairwise_win_rate(
    algo_a: str,
    algo_b: str,
    eval_episodes: int,
    rng: np.random.RandomState,
    trained: Dict[str, Optional[object]],
) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
  """Bidirectional win rates for algo_a vs algo_b (team-slot averaged)."""
  half = max(1, eval_episodes // 2)
  fwd = evaluate_team_matchup(
      algo_a, algo_b, half, rng,
      team1_agents=trained.get(algo_a), team2_agents=trained.get(algo_b)).summary()
  rev = evaluate_team_matchup(
      algo_b, algo_a, eval_episodes - half, rng,
      team1_agents=trained.get(algo_b), team2_agents=trained.get(algo_a)).summary()
  a_rate = (fwd["team1_win_rate"] + rev["team2_win_rate"]) / 2.0
  b_rate = (fwd["team2_win_rate"] + rev["team1_win_rate"]) / 2.0
  return a_rate, b_rate, fwd, rev


def run_round_robin(
    trained: Dict[str, Optional[object]],
    algorithms: Sequence[str],
    eval_episodes: int,
    rng: np.random.RandomState,
) -> Tuple[List[Dict], Dict[str, Dict[str, float]], List[Dict]]:
  algos = [normalize_algorithm(a) for a in algorithms]
  pairs: List[Dict] = []
  matrix: Dict[str, Dict[str, float]] = {a: {b: 0.0 for b in algos} for a in algos}
  points = {a: 0.0 for a in algos}

  for i, a in enumerate(algos):
    matrix[a][a] = 0.5
    for b in algos[i + 1:]:
      a_rate, b_rate, fwd, rev = _pairwise_win_rate(
          a, b, eval_episodes, rng, trained)
      matrix[a][b] = a_rate
      matrix[b][a] = b_rate
      if a_rate > b_rate:
        points[a] += 1.0
      elif b_rate > a_rate:
        points[b] += 1.0
      else:
        points[a] += 0.5
        points[b] += 0.5
      pairs.append({
          "team1": a,
          "team2": b,
          "team1_win_rate": a_rate,
          "team2_win_rate": b_rate,
          "forward": fwd,
          "reverse": rev,
      })
      print(f"RR: {a} vs {b} -> {a_rate:.2f}-{b_rate:.2f}")

  standings = sorted(
      [{"algorithm": a, "round_robin_points": points[a],
        "avg_pairwise_win_rate": (
            sum(matrix[a][b] for b in algos if b != a) / max(len(algos) - 1, 1))}
       for a in algos],
      key=lambda row: (row["round_robin_points"], row["avg_pairwise_win_rate"]),
      reverse=True,
  )
  return pairs, matrix, standings


def run_full_tournament(
    algorithms: Sequence[str],
    train_episodes: int,
    eval_every: int,
    eval_episodes: int,
    bracket_size: int,
    rng: np.random.RandomState,
    output_dir: str,
) -> Dict[str, object]:
  algos = [normalize_algorithm(a) for a in algorithms]
  trained, seeds = train_all(
      algos, train_episodes, eval_every, eval_episodes, rng, output_dir)

  round_robin_pairs: List[Dict] = []
  round_robin_matrix: Dict[str, Dict[str, float]] = {}
  round_robin_standings: List[Dict] = []
  if getattr(FLAGS, "round_robin", True):
    print("=== Round-robin phase ===")
    round_robin_pairs, round_robin_matrix, round_robin_standings = run_round_robin(
        trained, algos, eval_episodes, rng)

  print("=== Bracket phase ===")
  champion, bracket = run_bracket(
      trained, seeds, bracket_size, eval_episodes, rng)

  payload = {
      "algorithms": algos,
      "seeding": seeds,
      "round_robin": {
          "pairs": round_robin_pairs,
          "matrix": round_robin_matrix,
          "standings": round_robin_standings,
      },
      "champion": champion,
      "bracket": [
          {"round": m.round_num, "team1": m.team1_algo, "team2": m.team2_algo,
           "winner": m.winner, **m.stats}
          for m in bracket
      ],
  }
  path = f"{output_dir}/tournament_results.json"
  with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
  print(f"Champion: {champion}")
  print(f"Saved tournament to {path}")
  return payload
