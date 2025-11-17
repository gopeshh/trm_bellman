import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set RNG seeds across Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

