"""Deterministic random seeding across Python, NumPy, and PyTorch."""

import os
import random
from typing import Optional
import numpy as np
import torch


def seed_everything(seed: Optional[int] = 42) -> int:
    """Seed all pseudo-random number generators for reproducible experiments.

    Args:
        seed: Integer seed value. Defaults to 42.

    Returns:
        The integer seed applied.
    """
    if seed is None:
        seed = 42

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
