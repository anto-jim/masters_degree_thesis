"""Generate LaTeX report and figures from experiment outputs."""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _read_training_curves(output_dir: str) -> Dict[str, Dict[str, List[float]]]:
  curves: Dict[str, Dict[str, List[float]]] = {}
  for path in glob.glob(os.path.join(output_dir, "training_*.csv")):
    algo = os.path.basename(path).replace("training_", "").replace(".csv", "")
    episodes, wins = [], []
    with open(path, encoding="utf-8") as f:
      reader = csv.DictReader(f)
      for row in reader:
        episodes.append(float(row["episode"]))
        wins.append(float(row["team1_win_rate_vs_random"]))
    if episodes:
      curves[algo] = {"episodes": episodes, "win_rate": wins}
  return curves


def _plot_training_curves(curves: Dict[str, Dict[str, List[float]]], fig_dir: str):
  if not curves:
    return
  plt.figure(figsize=(8, 5))
  for algo, data in sorted(curves.items()):
    plt.plot(data["episodes"], data["win_rate"], marker="o", label=algo)
  plt.xlabel("Training progress (episodes / iterations)")
  plt.ylabel("Win rate vs random")
  plt.title("Turn Battle training curves")
  plt.legend(fontsize=8)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(fig_dir, "training_curves.pdf"))
  plt.savefig(os.path.join(fig_dir, "training_curves.png"), dpi=150)
  plt.close()


def _plot_seeding(seeds: Dict[str, float], fig_dir: str):
  if not seeds:
    return
  algos = list(seeds.keys())
  vals = [seeds[a] for a in algos]
  order = np.argsort(vals)[::-1]
  algos = [algos[i] for i in order]
  vals = [vals[i] for i in order]
  plt.figure(figsize=(8, 5))
  plt.bar(algos, vals, color="steelblue")
  plt.xticks(rotation=35, ha="right")
  plt.ylabel("Win rate vs random")
  plt.title("Seeding results before bracket")
  plt.tight_layout()
  plt.savefig(os.path.join(fig_dir, "seeding.pdf"))
  plt.savefig(os.path.join(fig_dir, "seeding.png"), dpi=150)
  plt.close()


def _plot_round_robin(matrix: Dict[str, Dict[str, float]], algos: List[str], fig_dir: str):
  if not matrix or not algos:
    return
  data = np.array([[matrix[a][b] for b in algos] for a in algos])
  plt.figure(figsize=(7, 6))
  im = plt.imshow(data, vmin=0, vmax=1, cmap="RdYlGn")
  plt.colorbar(im, label="Win rate (row vs column)")
  plt.xticks(range(len(algos)), algos, rotation=35, ha="right")
  plt.yticks(range(len(algos)), algos)
  plt.xlabel("Opponent")
  plt.ylabel("Algorithm")
  plt.title("Round-robin pairwise win rates")
  for i in range(len(algos)):
    for j in range(len(algos)):
      if i != j:
        plt.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
  plt.tight_layout()
  plt.savefig(os.path.join(fig_dir, "round_robin.pdf"))
  plt.savefig(os.path.join(fig_dir, "round_robin.png"), dpi=150)
  plt.close()


def _plot_bracket(bracket: List[Dict], champion: str, fig_dir: str):
  if not bracket:
    return
  labels, t1_rates = [], []
  for m in bracket:
    labels.append(f"R{m['round']}: {m['team1']} vs {m['team2']}")
    t1_rates.append(m["team1_win_rate"])
  plt.figure(figsize=(9, max(4, len(labels) * 0.35)))
  y = np.arange(len(labels))
  plt.barh(y, t1_rates, color="seagreen", alpha=0.8)
  plt.yticks(y, labels, fontsize=8)
  plt.xlabel("Team-1 win rate")
  plt.title(f"Bracket matchups (champion: {champion})")
  plt.tight_layout()
  plt.savefig(os.path.join(fig_dir, "bracket.pdf"))
  plt.savefig(os.path.join(fig_dir, "bracket.png"), dpi=150)
  plt.close()


def generate_latex_report(output_dir: str) -> str:
  fig_dir = os.path.join(output_dir, "figures")
  os.makedirs(fig_dir, exist_ok=True)

  curves = _read_training_curves(output_dir)
  tournament_path = os.path.join(output_dir, "tournament_results.json")
  tournament = {}
  if os.path.exists(tournament_path):
    with open(tournament_path, encoding="utf-8") as f:
      tournament = json.load(f)

  seeds = tournament.get("seeding", {})
  bracket = tournament.get("bracket", [])
  champion = tournament.get("champion", "n/a")
  rr = tournament.get("round_robin", {})
  rr_matrix = rr.get("matrix", {})
  rr_standings = rr.get("standings", [])
  algos = tournament.get("algorithms", list(seeds.keys()))

  _plot_training_curves(curves, fig_dir)
  _plot_seeding(seeds, fig_dir)
  _plot_round_robin(rr_matrix, algos, fig_dir)
  _plot_bracket(bracket, champion, fig_dir)

  tex_path = os.path.join(output_dir, "turn_battle_report.tex")
  lines = [
      r"\documentclass[11pt]{article}",
      r"\usepackage[margin=1in]{geometry}",
      r"\usepackage{graphicx}",
      r"\usepackage{booktabs}",
      r"\usepackage{hyperref}",
      r"\title{Turn Battle MARL Study Report}",
      r"\author{OpenSpiel Experiment Pipeline}",
      r"\date{\today}",
      r"\begin{document}",
      r"\maketitle",
      r"\section{Overview}",
      "This report summarizes training, round-robin, and bracket-tournament "
      r"results on the \texttt{turn\_battle} OpenSpiel environment "
      "(4-player team combat, 15 turns).",
      r"\section{Training curves}",
      r"Figure~\ref{fig:training} shows evaluation win rate against a random "
      "baseline during training.",
      r"\begin{figure}[h]",
      r"\centering",
      r"\includegraphics[width=0.9\linewidth]{figures/training_curves.pdf}",
      r"\caption{Training progress per algorithm.}",
      r"\label{fig:training}",
      r"\end{figure}",
      r"\section{Seeding}",
      "Algorithms were ranked by bidirectional win rate vs.\\ random before "
      "the elimination bracket.",
      r"\begin{table}[h]",
      r"\centering",
      r"\begin{tabular}{lc}",
      r"\toprule",
      r"Algorithm & Win rate vs random \\",
      r"\midrule",
  ]
  for algo, score in sorted(seeds.items(), key=lambda x: x[1], reverse=True):
    lines.append(f"{algo} & {score:.3f} \\\\")
  lines += [
      r"\bottomrule",
      r"\end{tabular}",
      r"\caption{Seeding scores (team-slot averaged).}",
      r"\end{table}",
      r"\begin{figure}[h]",
      r"\centering",
      r"\includegraphics[width=0.9\linewidth]{figures/seeding.pdf}",
      r"\caption{Seeding scores.}",
      r"\label{fig:seeding}",
      r"\end{figure}",
  ]

  if rr_standings:
    lines += [
        r"\section{Round-robin}",
        "Every algorithm played every other algorithm. Win rates are averaged "
        "over both team slots per pairing.",
        r"\begin{figure}[h]",
        r"\centering",
        r"\includegraphics[width=0.85\linewidth]{figures/round_robin.pdf}",
        r"\caption{Pairwise win-rate matrix (row vs column).}",
        r"\label{fig:roundrobin}",
        r"\end{figure}",
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Algorithm & RR points & Avg pairwise win rate \\",
        r"\midrule",
    ]
    for row in rr_standings:
      lines.append(
          f"{row['algorithm']} & {row['round_robin_points']:.1f} & "
          f"{row['avg_pairwise_win_rate']:.3f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Round-robin standings.}",
        r"\end{table}",
    ]

  lines += [
      r"\section{Bracket tournament}",
      f"The bracket champion was \\textbf{{{champion}}}.",
      r"\begin{figure}[h]",
      r"\centering",
      r"\includegraphics[width=0.95\linewidth]{figures/bracket.pdf}",
      r"\caption{Head-to-head bracket results.}",
      r"\label{fig:bracket}",
      r"\end{figure}",
      r"\begin{table}[h]",
      r"\centering",
      r"\begin{tabular}{llll}",
      r"\toprule",
      r"Round & Team 1 & Team 2 & Winner \\",
      r"\midrule",
  ]
  for m in bracket:
    lines.append(
        f"R{m['round']} & {m['team1']} & {m['team2']} & {m['winner']} \\\\")
  lines += [
      r"\bottomrule",
      r"\end{tabular}",
      r"\caption{Bracket match log.}",
      r"\end{table}",
      r"\section{Conclusion}",
      f"The bracket winner was \\textbf{{{champion}}}. Round-robin and "
      "bracket phases together provide both exhaustive pairwise data and "
      "a seeded elimination champion.",
      r"\end{document}",
  ]
  with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  return tex_path
