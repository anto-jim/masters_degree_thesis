"""Persist experiment outputs."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from absl import flags

from open_spiel.python.examples.turn_battle_study.models import TrainingLog

FLAGS = flags.FLAGS


def ensure_output_dir(path: str) -> str:
  os.makedirs(path, exist_ok=True)
  return path


def save_training_log(log: TrainingLog, output_dir: str) -> None:
  csv_path = os.path.join(output_dir, f"training_{log.algorithm}.csv")
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["episode", "team1_win_rate_vs_random", "random_win_rate"])
    for ep, w1, w2 in zip(log.episodes, log.team1_win_rate, log.team2_win_rate):
      w.writerow([ep, w1, w2])


def save_comparison_table(rows: List[Dict], output_dir: str) -> None:
  if not rows:
    return
  keys = list(rows[0].keys())
  csv_path = os.path.join(output_dir, "algorithm_comparison.csv")
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
  with open(os.path.join(output_dir, "algorithm_comparison.json"), "w",
            encoding="utf-8") as f:
    json.dump(rows, f, indent=2)


def save_experiment_config(output_dir: str, seed: Optional[int] = None) -> str:
  """Snapshot all absl flags for reproducibility."""
  config = {
      "seed": seed if seed is not None else getattr(FLAGS, "seed", None),
      "train_episodes": getattr(FLAGS, "train_episodes", None),
      "flags": {name: FLAGS[name].value for name in FLAGS},
  }
  path = os.path.join(output_dir, "experiment_config.json")
  with open(path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, default=str)
  return path
