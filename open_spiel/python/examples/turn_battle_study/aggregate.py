"""Aggregate multi-seed tournament results with confidence intervals."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .report import _latex_preamble, _plot_seeding_bar, _save_figure


_BOOTSTRAP_RESAMPLES = 10000


def mean_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    clip: bool = True,
) -> Dict[str, object]:
  """Summarise per-seed values with a small-sample confidence interval.

  With only a handful of seeds the normal approximation understates the
  interval, so the Student-t critical value for ``n - 1`` degrees of freedom
  is used instead. A percentile bootstrap interval is reported alongside it,
  and the raw per-seed values are carried through so a reader can judge the
  spread directly rather than trusting an interval estimated from very few
  points.

  Args:
    values: One summary statistic per seed.
    confidence: Two-sided confidence level.
    clip: Clip the interval to [0, 1] (appropriate for rates, not for points).

  Returns:
    A dict with the mean, standard deviation, t-interval, bootstrap interval,
    the per-seed values, and the number of seeds.
  """
  vals = [float(v) for v in values]
  if not vals:
    return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0,
            "n_seeds": 0, "values": [], "ci_method": "none"}
  n = len(vals)
  mean = float(np.mean(vals))
  if n < 2:
    return {"mean": mean, "ci_low": mean, "ci_high": mean, "std": 0.0,
            "n_seeds": n, "values": vals, "ci_method": "single-seed"}

  std = float(np.std(vals, ddof=1))
  t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
  margin = t_crit * std / math.sqrt(n)
  low, high = mean - margin, mean + margin

  rng = np.random.default_rng(12345)
  draws = rng.choice(vals, size=(_BOOTSTRAP_RESAMPLES, n), replace=True)
  boot_means = draws.mean(axis=1)
  alpha = (1.0 - confidence) / 2.0
  boot_low = float(np.quantile(boot_means, alpha))
  boot_high = float(np.quantile(boot_means, 1.0 - alpha))

  if clip:
    low, high = max(0.0, low), min(1.0, high)
    boot_low, boot_high = max(0.0, boot_low), min(1.0, boot_high)

  return {
      "mean": mean,
      "std": std,
      "sem": std / math.sqrt(n),
      "ci_low": low,
      "ci_high": high,
      "ci_method": f"student-t (df={n - 1})",
      "t_critical": t_crit,
      "bootstrap_ci_low": boot_low,
      "bootstrap_ci_high": boot_high,
      "n_seeds": n,
      "values": vals,
      "min": float(np.min(vals)),
      "max": float(np.max(vals)),
  }


def paired_comparison(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> Dict[str, object]:
  """Compare two algorithms across the seeds they were both run on.

  Seeds are a paired design: both algorithms saw the same seed, so the
  per-seed difference removes seed-level variance. With a handful of seeds no
  test has real power, so the sign of every per-seed difference is reported
  and the p-value is presented as descriptive rather than confirmatory.

  Args:
    values_a: Per-seed statistic for algorithm A.
    values_b: Per-seed statistic for algorithm B, aligned with *values_a*.

  Returns:
    A dict with the mean difference, its interval, how many seeds favour A,
    and a paired t-test p-value (or None when it cannot be computed).
  """
  a = np.asarray(values_a, dtype=float)
  b = np.asarray(values_b, dtype=float)
  if a.size == 0 or a.size != b.size:
    return {"n_seeds": 0, "mean_difference": None, "p_value": None,
            "seeds_favouring_a": 0}
  diff = a - b
  summary = mean_ci(diff.tolist(), clip=False)
  p_value = None
  if a.size >= 2 and float(np.std(diff, ddof=1)) > 0:
    p_value = float(stats.ttest_rel(a, b).pvalue)
  return {
      "n_seeds": int(a.size),
      "mean_difference": summary["mean"],
      "ci_low": summary["ci_low"],
      "ci_high": summary["ci_high"],
      "seeds_favouring_a": int(np.sum(diff > 0)),
      "seeds_favouring_b": int(np.sum(diff < 0)),
      "p_value": p_value,
      "per_seed_difference": diff.tolist(),
  }


def _load_seed_result(results_root: str, seed: int) -> Dict:
  """Load the tournament results JSON for a single seed.

  Args:
    results_root: Directory that contains per-seed subdirectories named
      ``seed_<n>``.
    seed: Integer seed whose results file should be loaded.

  Returns:
    Parsed JSON dict from ``<results_root>/seed_<seed>/tournament_results.json``.

  Raises:
    FileNotFoundError: If the expected results file does not exist.
  """
  path = os.path.join(results_root, f"seed_{seed}", "tournament_results.json")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Missing tournament results for seed {seed}: {path}")
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def aggregate_multi_seed_results(
    results_root: str,
    seeds: Sequence[int],
) -> Dict[str, object]:
  """Combine per-seed tournaments into mean rates and 95% CIs (normal approx)."""
  per_seed = []
  algos: List[str] = []
  champion_counts: Dict[str, int] = {}
  seeding_values: Dict[str, List[float]] = {}
  rr_points: Dict[str, List[float]] = {}
  rr_avg_win: Dict[str, List[float]] = {}
  pairwise: Dict[str, Dict[str, List[float]]] = {}

  baseline_values: Dict[str, Dict[str, List[float]]] = {}

  train_episodes = None
  num_turns = None
  eval_episodes = None
  for seed in seeds:
    result = _load_seed_result(results_root, seed)
    per_seed.append({"seed": seed, "champion": result.get("champion")})
    if train_episodes is None:
      train_episodes = result.get("train_episodes")
      num_turns = result.get("num_turns")
      eval_episodes = result.get("eval_episodes")
    if not algos:
      algos = list(result.get("algorithms", []))

    for algo, row in result.get("baselines", {}).items():
      for opponent, entry in row.items():
        baseline_values.setdefault(algo, {}).setdefault(
            opponent, []).append(float(entry["win_rate"]))

    champion = result.get("champion")
    if champion:
      champion_counts[champion] = champion_counts.get(champion, 0) + 1

    for algo, score in result.get("seeding", {}).items():
      seeding_values.setdefault(algo, []).append(float(score))

    for row in result.get("round_robin", {}).get("standings", []):
      algo = row["algorithm"]
      rr_points.setdefault(algo, []).append(float(row["round_robin_points"]))
      rr_avg_win.setdefault(algo, []).append(float(row["avg_pairwise_win_rate"]))

    matrix = result.get("round_robin", {}).get("matrix", {})
    for a in algos:
      pairwise.setdefault(a, {})
      for b in algos:
        if a != b and a in matrix and b in matrix[a]:
          pairwise[a].setdefault(b, []).append(float(matrix[a][b]))

  seeding_summary = {algo: mean_ci(vals) for algo, vals in
                     seeding_values.items()}

  baseline_summary: Dict[str, Dict[str, Dict[str, object]]] = {}
  for algo, row in baseline_values.items():
    baseline_summary[algo] = {
        opponent: mean_ci(vals) for opponent, vals in row.items()}

  rr_summary = {}
  for algo in algos:
    pts = rr_points.get(algo, [])
    wins = rr_avg_win.get(algo, [])
    rr_summary[algo] = {
        "round_robin_points": mean_ci(pts, clip=False),
        "pairwise_win_rate": mean_ci(wins),
        # Kept for backwards compatibility with existing report templates.
        "mean_round_robin_points": float(np.mean(pts)) if pts else 0.0,
        "mean_pairwise_win_rate": float(np.mean(wins)) if wins else 0.0,
    }

  # Paired seed-level comparisons between every ordered pair of algorithms,
  # on the seeding score (the one statistic every algorithm has per seed).
  comparisons: Dict[str, Dict[str, object]] = {}
  for a in algos:
    for b in algos:
      if a >= b or a not in seeding_values or b not in seeding_values:
        continue
      if len(seeding_values[a]) != len(seeding_values[b]):
        continue
      comparisons[f"{a}_vs_{b}"] = paired_comparison(
          seeding_values[a], seeding_values[b])

  pairwise_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
  for a in algos:
    pairwise_summary[a] = {}
    for b in algos:
      if a == b:
        continue
      vals = pairwise.get(a, {}).get(b, [])
      if not vals:
        continue
      pairwise_summary[a][b] = {"n_seeds": len(vals), **mean_ci(vals)}

  payload = {
      "train_episodes": train_episodes,
      "num_turns": num_turns,
      "eval_episodes": eval_episodes,
      "n_seeds": len(seeds),
      "seeds": list(seeds),
      "algorithms": algos,
      "champion_counts": champion_counts,
      "champion_frequency": {
          a: champion_counts.get(a, 0) / max(len(seeds), 1) for a in algos
      },
      "seeding": seeding_summary,
      "baselines": baseline_summary,
      "round_robin": rr_summary,
      "pairwise": pairwise_summary,
      "paired_comparisons": comparisons,
      "per_seed": per_seed,
  }

  out_path = os.path.join(results_root, "aggregated_results.json")
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
  _plot_aggregated_figures(results_root, payload)
  return payload


def _plot_aggregated_figures(results_root: str, agg: Dict) -> None:
  """Render and save aggregated seeding and champion-frequency figures.

  Produces ``aggregated_seeding`` (bar chart with 95% CI error bars) and
  ``champion_frequency`` (bar chart) under ``<results_root>/figures/``.

  Args:
    results_root: Root directory of the multi-seed experiment; figures are
      written to its ``figures/`` subdirectory.
    agg: Aggregated payload dict as returned by
      :func:`aggregate_multi_seed_results`.
  """
  fig_dir = os.path.join(results_root, "figures")
  os.makedirs(fig_dir, exist_ok=True)
  algos = agg.get("algorithms", [])
  seeding = agg.get("seeding", {})

  if seeding and algos:
    means = [seeding[a]["mean"] for a in algos]
    lows = [seeding[a]["ci_low"] for a in algos]
    highs = [seeding[a]["ci_high"] for a in algos]
    yerr = [
        [m - lo for m, lo in zip(means, lows)],
        [hi - m for m, hi in zip(means, highs)],
    ]
    _plot_seeding_bar(
        algos,
        means,
        fig_dir,
        yerr=yerr,
        name="aggregated_seeding",
        title="Seeding scores across seeds (mean \u00b1 95% CI)",
    )

  champ = agg.get("champion_frequency", {})
  if champ:
    labels = list(champ.keys())
    vals = [champ[k] for k in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals, color="seagreen")
    plt.ylabel("Bracket champion frequency")
    plt.title(f"Champion across {agg.get('n_seeds', 0)} seeds")
    plt.xticks(rotation=35, ha="right")
    _save_figure(fig_dir, "champion_frequency")


def generate_aggregated_latex_report(
    results_root: str, agg: Dict[str, object]
) -> str:
  """Write a LaTeX thesis report summarising the multi-seed aggregated results.

  Produces tables and figure includes for seeding performance, round-robin
  standings, bracket champion frequency, and per-seed champions, then writes
  ``thesis_aggregated_report.tex`` to *results_root*.

  Args:
    results_root: Root directory of the multi-seed experiment.  Figures are
      expected under its ``figures/`` subdirectory.
    agg: Aggregated payload dict as returned by
      :func:`aggregate_multi_seed_results`.

  Returns:
    Absolute path to the generated ``.tex`` file.
  """
  fig_dir = os.path.join(results_root, "figures")
  tex_path = os.path.join(results_root, "thesis_aggregated_report.tex")
  algos = agg.get("algorithms", [])
  seeding = agg.get("seeding", {})
  rr = agg.get("round_robin", {})
  champ_freq = agg.get("champion_frequency", {})

  lines = _latex_preamble(
      "Turn Battle MARL Thesis --- Aggregated Results", "Multi-seed tournament study"
  ) + [
      r"\section{Experimental setup}",
      f"Each algorithm was trained for {agg.get('train_episodes', 'N')} episodes "
      f"across {agg.get('n_seeds', 0)} independent random seeds.",
      r"\section{Seeding performance}",
      r"\begin{table}[h]",
      r"\centering",
      r"\begin{tabular}{lccc}",
      r"\toprule",
      r"Algorithm & Mean & CI low & CI high \\",
      r"\midrule",
  ]
  for algo in sorted(seeding.keys(),
                     key=lambda a: seeding[a]["mean"], reverse=True):
    s = seeding[algo]
    lines.append(
        f"{algo} & {s['mean']:.3f} & {s['ci_low']:.3f} & "
        f"{s['ci_high']:.3f} \\\\")
  lines += [
      r"\bottomrule",
      r"\end{tabular}",
      r"\caption{Win rate vs random on team~1 with fixed role slots (mean over seeds $\pm$ 95\% CI).}",
      r"\end{table}",
      r"\begin{figure}[h]",
      r"\centering",
      r"\includegraphics[width=0.9\linewidth]{figures/aggregated_seeding.pdf}",
      r"\caption{Seeding scores with 95\% confidence intervals across seeds.}",
      r"\end{figure}",
  ]

  if rr:
    lines += [
        r"\section{Round-robin summary}",
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Algorithm & Mean RR points & Mean pairwise win rate \\",
        r"\midrule",
    ]
    for algo in sorted(rr.keys(),
                       key=lambda a: rr[a]["mean_pairwise_win_rate"],
                       reverse=True):
      row = rr[algo]
      lines.append(
          f"{algo} & {row['mean_round_robin_points']:.2f} & "
          f"{row['mean_pairwise_win_rate']:.3f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Round-robin standings averaged across seeds.}",
        r"\end{table}",
    ]

  lines += [
      r"\section{Bracket champion frequency}",
      r"\begin{table}[h]",
      r"\centering",
      r"\begin{tabular}{lc}",
      r"\toprule",
      r"Algorithm & Champion frequency \\",
      r"\midrule",
  ]
  for algo, freq in sorted(champ_freq.items(), key=lambda x: x[1], reverse=True):
    lines.append(f"{algo} & {freq:.2f} \\\\")
  lines += [
      r"\bottomrule",
      r"\end{tabular}",
      r"\caption{Fraction of seeds where each algorithm won the bracket.}",
      r"\end{table}",
      r"\begin{figure}[h]",
      r"\centering",
      r"\includegraphics[width=0.85\linewidth]{figures/champion_frequency.pdf}",
      r"\caption{Bracket champion frequency across seeds.}",
      r"\end{figure}",
      r"\section{Per-seed champions}",
      r"\begin{itemize}",
  ]
  for row in agg.get("per_seed", []):
    lines.append(
        f"\\item Seed {row['seed']}: champion = \\texttt{{{row['champion']}}}")
  lines += [
      r"\end{itemize}",
      r"\end{document}",
  ]
  with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  return tex_path
