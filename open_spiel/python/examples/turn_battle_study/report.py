"""Generate LaTeX report and figures from experiment outputs."""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _save_figure(fig_dir: str, name: str) -> None:
  """Save the current matplotlib figure as both PDF and PNG into fig_dir.

  Args:
    fig_dir: Directory in which to write the output files.
    name: Base filename (without extension) for the saved figure.
  """
  plt.tight_layout()
  plt.savefig(os.path.join(fig_dir, f"{name}.pdf"))
  plt.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=150)
  plt.close()


def _latex_preamble(title: str, author: str = "") -> List[str]:
  """Return LaTeX document header lines up to and including \\maketitle.

  Args:
    title: Document title inserted into the \\title command.
    author: Optional author string; defaults to an empty author field.

  Returns:
    List of LaTeX source lines forming the document preamble and opening.
  """
  return [
      r"\documentclass[11pt]{article}",
      r"\usepackage[margin=1in]{geometry}",
      r"\usepackage{graphicx}",
      r"\usepackage{booktabs}",
      r"\usepackage{hyperref}",
      rf"\title{{{title}}}",
      rf"\author{{{author}}}",
      r"\date{\today}",
      r"\begin{document}",
      r"\maketitle",
  ]


def _read_training_curves(output_dir: str) -> Dict[str, Dict[str, List[float]]]:
  """Load per-algorithm training-curve CSVs from output_dir.

  Each CSV must contain at least the columns ``episode`` and
  ``team1_win_rate_vs_random``.  The algorithm name is inferred from the
  filename pattern ``training_<algo>.csv``.

  Args:
    output_dir: Directory that contains ``training_*.csv`` files.

  Returns:
    Mapping from algorithm name to a dict with keys ``"episodes"`` and
    ``"win_rate"``, each holding a list of floats.  Algorithms whose CSV
    contains no rows are omitted.
  """
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
  """Plot win-rate-vs-random training curves for all algorithms and save to fig_dir.

  Args:
    curves: Mapping returned by :func:`_read_training_curves`.
    fig_dir: Directory in which the figure is saved as ``training_curves``.
  """
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
  _save_figure(fig_dir, "training_curves")


def _plot_seeding_bar(
    algos: List[str],
    values: List[float],
    fig_dir: str,
    yerr=None,
    name: str = "seeding",
    title: str = "Seeding results before bracket",
) -> None:
  """Plot a descending bar chart of per-algorithm seeding scores.

  Args:
    algos: Algorithm names corresponding to each entry in *values*.
    values: Win-rate (or other scalar) score for each algorithm.
    fig_dir: Directory in which the figure is saved.
    yerr: Optional asymmetric error bars as ``[lower_deltas, upper_deltas]``;
      when provided the y-axis is clamped to ``[0, 1]``.
    name: Base filename (without extension) for the saved figure.
    title: Chart title.
  """
  order = np.argsort(values)[::-1]
  algos_ord = [algos[i] for i in order]
  vals_ord = [values[i] for i in order]
  bar_kwargs: dict = {"color": "steelblue"}
  if yerr is not None:
    bar_kwargs["yerr"] = [[yerr[0][i] for i in order], [yerr[1][i] for i in order]]
    bar_kwargs["capsize"] = 4
  plt.figure(figsize=(8, 5))
  plt.bar(algos_ord, vals_ord, **bar_kwargs)
  plt.xticks(rotation=35, ha="right")
  plt.ylabel("Win rate vs random")
  plt.title(title)
  if yerr is not None:
    plt.ylim(0, 1)
  _save_figure(fig_dir, name)


def _plot_round_robin(matrix: Dict[str, Dict[str, float]], algos: List[str], fig_dir: str):
  """Plot the round-robin pairwise win-rate matrix as a colour-coded heatmap.

  Args:
    matrix: Nested mapping ``matrix[algo_a][algo_b]`` → win rate of *a* vs *b*.
    algos: Ordered list of algorithm names used as row/column labels.
    fig_dir: Directory in which the figure is saved as ``round_robin``.
  """
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
  _save_figure(fig_dir, "round_robin")


def _plot_bracket(bracket: List[Dict], champion: str, fig_dir: str):
  """Plot bracket matchup team-1 win rates as a horizontal bar chart.

  Args:
    bracket: List of match dicts, each containing keys ``round``, ``team1``,
      ``team2``, and ``team1_win_rate``.
    champion: Name of the overall bracket winner used in the chart title.
    fig_dir: Directory in which the figure is saved as ``bracket``.
  """
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
  _save_figure(fig_dir, "bracket")


def generate_latex_report(output_dir: str) -> str:
  """Generate all figures and a self-contained LaTeX report for one experiment run.

  Reads ``tournament_results.json`` and ``training_*.csv`` from *output_dir*,
  produces figures under ``<output_dir>/figures/``, and writes
  ``turn_battle_report.tex`` to *output_dir*.

  Args:
    output_dir: Root directory of a single-seed experiment run.

  Returns:
    Absolute path to the generated ``.tex`` file.
  """
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
  if seeds:
    _plot_seeding_bar(list(seeds.keys()), [seeds[a] for a in seeds], fig_dir)
  _plot_round_robin(rr_matrix, algos, fig_dir)
  _plot_bracket(bracket, champion, fig_dir)

  tex_path = os.path.join(output_dir, "turn_battle_report.tex")
  lines = _latex_preamble(
      "Turn Battle MARL Study Report", "OpenSpiel Experiment Pipeline"
  ) + [
      r"\section{Overview}",
      "This report summarizes training, round-robin, and bracket-tournament "
      r"results on the \texttt{turn\_battle} OpenSpiel environment "
      "(4-player team combat). Every algorithm is trained for the same "
      "number of episodes (\\texttt{train\\_episodes}).",
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
      "Algorithms were ranked by win rate vs.\\ random on team~1 "
      "(defenders on players~0/2, attackers on players~1/3) before "
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
      r"\caption{Seeding scores (fixed role slots).}",
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
        "Every algorithm played every other algorithm on fixed team slots "
        "(team~1 vs team~2; defenders and attackers keep trained roles).",
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
