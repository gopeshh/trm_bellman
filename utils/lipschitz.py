from typing import Tuple, Type

import torch.nn as nn
import torch.nn.utils as nn_utils

try:
    from models.layers import CastedLinear
except ImportError:  # pragma: no cover - fallback during partial imports
    CastedLinear = tuple()  # type: ignore


def _linear_module_types() -> Tuple[Type[nn.Module], ...]:
    base_types: Tuple[Type[nn.Module], ...] = (nn.Linear,)
    if isinstance(CastedLinear, type):
        return base_types + (CastedLinear,)
    return base_types


def apply_spectral_norm_to_trm(inner_model: nn.Module) -> None:
    """
    Apply spectral normalization to linear layers in the inner TRM module.

    This is a coarse control: for now we simply wrap all nn.Linear layers
    in the inner model with spectral_norm. This can be refined later.
    """
    for module in inner_model.modules():
        if isinstance(module, _linear_module_types()):
            nn_utils.spectral_norm(module)


def apply_spectral_norm_to_value_head(value_head: nn.Module) -> None:
    """
    Apply spectral normalization to all linear layers in the value head.
    """
    for module in value_head.modules():
        if isinstance(module, _linear_module_types()):
            nn_utils.spectral_norm(module)
