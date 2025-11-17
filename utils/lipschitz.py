from typing import Tuple, Type

import torch
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

    Spectral norm keeps each linear map at ~1-Lipschitz; we pair this with
    per-layer output scaling to dial in stricter global contraction.
    """
    for module in inner_model.modules():
        if isinstance(module, _linear_module_types()):
            nn_utils.spectral_norm(module)


def apply_spectral_norm_to_value_head(value_head: nn.Module) -> None:
    """
    Apply spectral normalization to all linear layers in the value head.

    Spectral norm stabilizes each layer to ~1-Lipschitz before we apply the
    global contraction scalars.
    """
    for module in value_head.modules():
        if isinstance(module, _linear_module_types()):
            nn_utils.spectral_norm(module)


def enforce_global_contraction(inner_model: nn.Module, target_Lz: float) -> None:
    """
    Compound per-layer output scales so the overall inner latent map contraction
    is coarsely bounded by target_Lz.

    Spectral normalization keeps each layer near 1-Lipschitz; the multiplicative
    output scales produced here push the product of norms below target_Lz.
    """

    if target_Lz <= 0.0:
        return

    linear_modules = [m for m in inner_model.modules() if isinstance(m, _linear_module_types())]
    if not linear_modules:
        return

    alpha = target_Lz ** (1.0 / len(linear_modules))
    for module in linear_modules:
        _scale_linear_output(module, alpha, scale_attr="_inner_lip_scale")


def enforce_global_contraction_on_value_head(value_head: nn.Module, target_Lv: float) -> None:
    """
    Compound per-layer output scales in the value head so that its Lipschitz
    constant w.r.t. z is approximately bounded by target_Lv.
    """

    if target_Lv <= 0.0:
        return

    linear_modules = [m for m in value_head.modules() if isinstance(m, _linear_module_types())]
    if not linear_modules:
        return

    alpha = target_Lv ** (1.0 / len(linear_modules))
    for module in linear_modules:
        _scale_linear_output(module, alpha, scale_attr="_value_lip_scale")


def _scale_linear_output(module: nn.Module, alpha: float, scale_attr: str) -> None:
    """
    Wrap the forward of `module` so its output is multiplied by a learnable
    scalar `module.<scale_attr>`. If already wrapped, multiply the existing
    scale by `alpha`.
    """

    # If already has a scale, just compound it
    if hasattr(module, scale_attr):
        current = getattr(module, scale_attr)
        with torch.no_grad():
            current.mul_(alpha)
        return

    # First-time setup
    device = None
    dtype = torch.float32
    if hasattr(module, "weight") and torch.is_tensor(module.weight):
        device = module.weight.device
        dtype = module.weight.dtype
    scale_tensor = torch.tensor(alpha, dtype=dtype, device=device)
    module.register_buffer(scale_attr, scale_tensor)

    orig_forward = module.forward

    def scaled_forward(x, *args, _orig_forward=orig_forward, _module=module, **kwargs):
        out = _orig_forward(x, *args, **kwargs)
        scale = getattr(_module, scale_attr)
        if torch.is_tensor(scale):
            scale = scale.to(dtype=out.dtype, device=out.device)
        return scale * out

    module.forward = scaled_forward


def estimate_local_Lz(
    inner_model: nn.Module,
    carry,
    context,
    num_samples: int = 4,
    eps: float = 1e-3,
) -> float:
    """
    Estimate ||f(z+δ) - f(z)|| / ||δ|| for the inner latent map using random perturbations.
    """

    if not hasattr(carry, "z_H") or not hasattr(carry, "z_L"):
        raise TypeError("`carry` must expose z_H and z_L tensors.")
    if eps <= 0.0:
        raise ValueError("`eps` must be positive.")

    input_embeddings = context.get("input_embeddings_with_plan", context.get("input_embeddings"))
    seq_info = context.get("seq_info")
    if input_embeddings is None or seq_info is None:
        raise KeyError("context must include `input_embeddings`/`input_embeddings_with_plan` and `seq_info`.")

    norms = []
    with torch.no_grad():
        baseline = inner_model.latent_step(carry, input_embeddings, seq_info)
        for _ in range(max(num_samples, 1)):
            noise_h = torch.randn_like(carry.z_H)
            noise_l = torch.randn_like(carry.z_L)
            noise_norm = torch.sqrt(
                (noise_h.pow(2).sum() + noise_l.pow(2).sum()).clamp_min(1e-12)
            )
            scale = eps / noise_norm
            perturbed = carry.__class__(
                z_H=carry.z_H + noise_h * scale,
                z_L=carry.z_L + noise_l * scale,
            )
            out = inner_model.latent_step(perturbed, input_embeddings, seq_info)
            diff = torch.sqrt(
                (out.z_H - baseline.z_H).pow(2).sum() + (out.z_L - baseline.z_L).pow(2).sum()
            )
            norms.append(diff / eps)

    if not norms:
        return 0.0
    stacked = torch.stack(norms)
    return float(stacked.mean().item())
