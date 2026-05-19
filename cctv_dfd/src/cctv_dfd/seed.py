"""Single-entry-point determinism helper.

Call ``set_determinism(seed)`` once at the top of every script. Sets seeds for
``random``, ``numpy``, and ``torch`` (CPU + CUDA) and toggles deterministic
algorithms. Returns the seed so it can be logged.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_determinism(seed: int = 1337, *, warn_only: bool = True) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception:
        torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
