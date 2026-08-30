"""Global RNG seeding for reproducible per-seed runs.

A multi-seed campaign is only a valid sample of independent runs if every
stochastic component is re-seeded per run.  Passing a ``np.random.RandomState``
around covers episode and evaluation sampling, but network initialisation,
dropout, replay-buffer sampling and the Deep CFR solver all draw from the
*global* Python / NumPy / PyTorch generators, which are separate streams.
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np


def seed_everything(seed: int) -> Dict[str, object]:
  """Seed the global Python, NumPy and PyTorch generators.

  Args:
    seed: Base seed for this run.

  Returns:
    A dict describing what was seeded, suitable for writing into a run's
    ``experiment_config.json`` as provenance.
  """
  random.seed(seed)
  np.random.seed(seed)
  info: Dict[str, object] = {"seed": int(seed), "python_random": True,
                             "numpy_global": True}
  try:
    import torch
  except ImportError:
    info["torch"] = False
    return info
  torch.manual_seed(seed)
  info["torch"] = True
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    info["torch_cuda"] = True
  else:
    info["torch_cuda"] = False
  return info
