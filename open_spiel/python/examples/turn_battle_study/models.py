"""Result and logging dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from open_spiel.python.examples.turn_battle_study.config import (
    TEAM1_PLAYERS,
    TEAM2_PLAYERS,
)


@dataclass
class MatchResult:
  team1_wins: int = 0
  team2_wins: int = 0
  draws: int = 0
  team1_return_sum: float = 0.0
  team2_return_sum: float = 0.0
  episodes: int = 0

  def record(self, returns: Sequence[float]) -> None:
    t1 = sum(returns[p] for p in TEAM1_PLAYERS)
    t2 = sum(returns[p] for p in TEAM2_PLAYERS)
    self.team1_return_sum += t1
    self.team2_return_sum += t2
    if t1 > t2:
      self.team1_wins += 1
    elif t2 > t1:
      self.team2_wins += 1
    else:
      self.draws += 1
    self.episodes += 1

  def summary(self) -> Dict[str, float]:
    n = max(self.episodes, 1)
    return {
        "episodes": self.episodes,
        "team1_win_rate": self.team1_wins / n,
        "team2_win_rate": self.team2_wins / n,
        "draw_rate": self.draws / n,
        "team1_avg_return": self.team1_return_sum / n,
        "team2_avg_return": self.team2_return_sum / n,
    }


@dataclass
class TrainingLog:
  algorithm: str
  episodes: List[int] = field(default_factory=list)
  team1_win_rate: List[float] = field(default_factory=list)
  team2_win_rate: List[float] = field(default_factory=list)


@dataclass
class BracketMatch:
  round_num: int
  team1_algo: str
  team2_algo: str
  winner: str
  stats: Dict[str, float]
