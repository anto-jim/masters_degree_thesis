"""Bracket-style elimination tournament with optional round-robin phase."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
from open_spiel.python.examples.turn_battle_study.checkpoints import (
    load_trained_agents,
    save_trained_checkpoint,
)
from open_spiel.python.examples.turn_battle_study.storage import save_training_log
from open_spiel.python.examples.turn_battle_study.role_shared import (
    RoleSharedTeam,
    select_best_role_seats,
)
from open_spiel.python.examples.turn_battle_study.trainers import (
    _fixed_role_win_rate,
    train_algorithm,
)


def _pick_winner(algo1: str, algo2: str, stats: Dict[str, float]) -> str:
  if stats["team1_win_rate"] > stats["team2_win_rate"]:
    return algo1
  if stats["team2_win_rate"] > stats["team1_win_rate"]:
    return algo2
  return algo1 if stats["team1_avg_return"] >= stats["team2_avg_return"] else algo2


def _finalize_rl_agents(
    key: str,
    agents: object,
    eval_episodes: int,
    rng: np.random.RandomState,
) -> object:
  """Role-shared NFSP/QPG: pick best defender/attacker seats for tournament."""
  if not isinstance(agents, RoleSharedTeam):
    return agents
  def_seat, atk_seat, score = select_best_role_seats(
      agents, key, eval_episodes, rng)
  print(
      f"  role selection: defender seat {def_seat}, attacker seat {atk_seat} "
      f"(eval win={score:.3f})")
  return agents


def _maybe_finalize_rl_agents(
    key: str,
    agents: object,
    eval_episodes: int,
    rng: np.random.RandomState,
) -> object:
  if isinstance(agents, RoleSharedTeam) and agents.selection is None:
    return _finalize_rl_agents(key, agents, eval_episodes, rng)
  return agents


def _retrain_algorithm_set(
    retrain_algorithms: Optional[Sequence[str]],
) -> Optional[Set[str]]:
  if not retrain_algorithms:
    return None
  return {normalize_algorithm(a) for a in retrain_algorithms}


def _load_one_algorithm(
    key: str,
    output_dir: str,
    eval_episodes: int,
    rng: np.random.RandomState,
) -> Tuple[object, float]:
  print(f"=== Loading {key} from checkpoint ===")
  ckpt_dir = f"{output_dir}/checkpoints/{key}"
  if not os.path.isdir(ckpt_dir):
    raise FileNotFoundError(
        f"Checkpoint required for {key} but missing: {ckpt_dir}")
  agents = load_trained_agents(key, ckpt_dir, rng)
  agents = _maybe_finalize_rl_agents(key, agents, eval_episodes, rng)
  seed_score = _fixed_role_win_rate(key, agents, eval_episodes, rng)[0]
  print(f"  seed score vs random: {seed_score:.3f}")
  return agents, seed_score


def _train_one_algorithm(
    key: str,
    train_episodes: int,
    eval_every: int,
    eval_episodes: int,
    rng: np.random.RandomState,
    output_dir: str,
) -> Tuple[object, float]:
  print(f"=== Training {key} ({train_episodes} episodes) ===")
  eval_interval = max(1, min(eval_every, max(train_episodes // 6, 25)))
  agents, log, artifact = train_algorithm(
      key, train_episodes, eval_interval, eval_episodes, rng)
  agents = _finalize_rl_agents(key, agents, eval_episodes, rng)
  save_training_log(log, output_dir)
  if getattr(FLAGS, "save_checkpoints", True):
    ckpt_dir = f"{output_dir}/checkpoints/{key}"
    save_trained_checkpoint(key, agents, artifact, ckpt_dir)
    print(f"  checkpoint saved: {ckpt_dir}")
  seed_score = _fixed_role_win_rate(key, agents, eval_episodes, rng)[0]
  print(f"  seed score vs random: {seed_score:.3f}")
  return agents, seed_score


def _populate_trained_dict(
    algorithms: Sequence[str],
    output_dir: str,
    eval_episodes: int,
    rng: np.random.RandomState,
    should_train,
    train_episodes: int = 0,
    eval_every: int = 0,
) -> Tuple[Dict[str, object], Dict[str, float]]:
  """Shared loop for load_all and train_all: bots handled once, train/load decided by predicate."""
  trained: Dict[str, object] = {}
  seeds: Dict[str, float] = {}
  for algo in algorithms:
    key = normalize_algorithm(algo)
    if key in TRAINABLE_ALGOS:
      if should_train(key):
        agents, seed_score = _train_one_algorithm(
            key, train_episodes, eval_every, eval_episodes, rng, output_dir)
      else:
        agents, seed_score = _load_one_algorithm(
            key, output_dir, eval_episodes, rng)
      trained[key] = agents
      seeds[key] = seed_score
    elif key in BOT_ALGOS:
      trained[key] = None
      stats = evaluate_team_matchup(key, "random", eval_episodes, rng).summary()
      seeds[key] = stats["team1_win_rate"]
      print(f"  seed score vs random: {seeds[key]:.3f}")
    else:
      continue
  return trained, seeds


def load_all(
    algorithms: Sequence[str],
    output_dir: str,
    eval_episodes: int,
    rng: np.random.RandomState,
) -> Tuple[Dict[str, object], Dict[str, float]]:
  """Load trained checkpoints and compute seeding scores (no retraining)."""
  return _populate_trained_dict(
      algorithms, output_dir, eval_episodes, rng, should_train=lambda _: False)


def train_all(
    algorithms: Sequence[str],
    train_episodes: int,
    eval_every: int,
    eval_episodes: int,
    rng: np.random.RandomState,
    output_dir: str,
    retrain_algorithms: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, object], Dict[str, float]]:
  retrain_only = _retrain_algorithm_set(retrain_algorithms)
  should_train = lambda key: retrain_only is None or key in retrain_only
  return _populate_trained_dict(
      algorithms, output_dir, eval_episodes, rng, should_train=should_train,
      train_episodes=train_episodes, eval_every=eval_every)


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
) -> Tuple[float, float, Dict[str, float]]:
  """Win rates with algo_a on team1 and algo_b on team2 (fixed role slots)."""
  stats = evaluate_team_matchup(
      algo_a, algo_b, eval_episodes, rng,
      team1_agents=trained.get(algo_a), team2_agents=trained.get(algo_b)).summary()
  return stats["team1_win_rate"], stats["team2_win_rate"], stats


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
      a_rate, b_rate, stats = _pairwise_win_rate(
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
          "match": stats,
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
  retrain_algorithms = getattr(FLAGS, "retrain_algorithms", None) or None
  if train_episodes <= 0:
    trained, seeds = load_all(algos, output_dir, eval_episodes, rng)
  else:
    trained, seeds = train_all(
        algos, train_episodes, eval_every, eval_episodes, rng, output_dir,
        retrain_algorithms=retrain_algorithms)

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
      "train_episodes": train_episodes,
      "algorithms": algos,
      "seeding": seeds,
      "role_selection": {
          algo: {
              "defender_seat": trained[algo].selection[0],
              "attacker_seat": trained[algo].selection[1],
          }
          for algo in algos
          if isinstance(trained.get(algo), RoleSharedTeam)
          and trained[algo].selection is not None
      },
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
