from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from configs.loggings import logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    python_state = state.get("python_random")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy_random")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = state.get("torch_random")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)

    cuda_state = state.get("torch_cuda_random")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)

