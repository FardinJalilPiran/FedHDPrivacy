"""Small shared helpers: seeding, device selection, logging, result I/O."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

__all__ = ["set_seed", "resolve_device", "configure_logging", "save_history", "timestamp"]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Turn ``"auto"`` into cuda when available, cpu otherwise."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logging.getLogger(__name__).warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_history(history, output_dir: str | Path, run_name: str | None = None) -> Path:
    """Write a training history to ``output_dir/<run_name>.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = run_name or f"run-{timestamp()}"
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(history.to_dict(), indent=2))
    return path
