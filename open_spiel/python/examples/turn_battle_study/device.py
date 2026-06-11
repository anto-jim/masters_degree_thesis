"""PyTorch device selection for training."""

from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
  """Resolve a device string; ``auto`` picks CUDA when available."""
  key = name.strip().lower()
  if key == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(name)


def device_label(device: torch.device) -> str:
  if device.type == "cuda":
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda ({torch.cuda.get_device_name(idx)})"
  return str(device)
