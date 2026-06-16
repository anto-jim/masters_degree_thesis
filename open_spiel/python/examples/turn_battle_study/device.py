"""PyTorch device selection for training."""

from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
  """Resolve a device string; ``auto`` picks CUDA when available."""
  key = name.strip().lower()
  if key == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(name)


def resolve_cpp_az_devices(name: str) -> str:
  """Device string for ``alpha_zero_torch_example`` (--devices flag)."""
  device = resolve_device(name)
  if device.type == "cuda":
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda:{idx}"
  return "cpu"


def device_label(device: torch.device) -> str:
  if device.type == "cuda":
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda ({torch.cuda.get_device_name(idx)})"
  return str(device)
