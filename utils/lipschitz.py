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
    return_samples: bool = False,
):
    """
    Estimate ||f(z+δ) - f(z)|| / ||δ|| for the inner latent map using random perturbations.

    Set `return_samples=True` to also receive the individual per-sample norm estimates.
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
        if return_samples:
            return 0.0, torch.zeros(0, dtype=torch.float32)
        return 0.0
    stacked = torch.stack(norms)
    mean_value = float(stacked.mean().item())
    if return_samples:
        return mean_value, stacked
    return mean_value


def estimate_Cz(
    model: nn.Module,
    x_batch: dict,
    y_batch: torch.Tensor,
) -> float:
    """
    Estimate C_z = max ||z^(1) - z^(0)|| over a batch.

    This corresponds to Equation 9 in the paper:
        C_z = sup_{(x,y)} ||f_θ(z^(0)(x,y), y, x) - z^(0)(x,y)||

    Args:
        model: TRM model with init_latent and update_latent methods
        x_batch: Batch dict with "inputs" and "puzzle_identifiers"
        y_batch: Plan tensor [B, seq_len]

    Returns:
        Maximum ||z^(1) - z^(0)|| over the batch (scalar float)
    """
    with torch.no_grad():
        z0 = model.init_latent(x_batch, y_batch)
        z1 = model.update_latent(z0, y_batch, x_batch)

        # Compute ||z^(1) - z^(0)|| for each sample in batch
        diff_H = (z1.z_H - z0.z_H).pow(2).sum(dim=(1, 2))  # [B]
        diff_L = (z1.z_L - z0.z_L).pow(2).sum(dim=(1, 2))  # [B]
        diff_norms = torch.sqrt(diff_H + diff_L)  # [B]

        return float(diff_norms.max().item())


def estimate_Lv(
    value_head: nn.Module,
    z_batch: torch.Tensor,
    x_embed_batch: torch.Tensor,
    num_samples: int = 4,
    eps: float = 1e-3,
) -> float:
    """
    Estimate the Lipschitz constant L_V of the value head with respect to z.

    Uses random perturbations to estimate ||V(z+δ, x) - V(z, x)|| / ||δ||.

    Args:
        value_head: Value head module V_ψ(z, x_embed)
        z_batch: Latent tensor [B, z_dim]
        x_embed_batch: Context embedding [B, x_dim]
        num_samples: Number of perturbation samples
        eps: Perturbation magnitude

    Returns:
        Estimated Lipschitz constant (max over samples)
    """
    if eps <= 0.0:
        raise ValueError("`eps` must be positive.")

    ratios = []
    with torch.no_grad():
        baseline = value_head(z_batch, x_embed_batch)  # [B]
        for _ in range(max(num_samples, 1)):
            noise = torch.randn_like(z_batch)
            noise_norm = noise.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            perturbed_z = z_batch + eps * noise / noise_norm
            perturbed_v = value_head(perturbed_z, x_embed_batch)
            diff = (perturbed_v - baseline).abs()  # [B]
            ratios.append(diff / eps)

    if not ratios:
        return 0.0
    stacked = torch.stack(ratios, dim=0)  # [num_samples, B]
    return float(stacked.max().item())


def compute_unrolling_term_proxy(
    hat_Lv: float,
    hat_Lz: float,
    hat_Cz: float,
    n: int,
) -> float:
    """
    Compute the finite-unrolling term proxy from Equation 10 in the paper:

        L_V * L_z^n * C_z / (1 - L_z)

    This bounds the value error due to finite unrolling.

    Args:
        hat_Lv: Estimated Lipschitz constant of value head w.r.t. z
        hat_Lz: Estimated contraction constant of inner map
        hat_Cz: Estimated ||z^(1) - z^(0)|| bound
        n: Number of inner unrolling steps

    Returns:
        Unrolling term proxy (scalar float)
    """
    if hat_Lz >= 1.0:
        # Not contractive; term would be infinite
        return float("inf")
    return hat_Lv * (hat_Lz ** n) * hat_Cz / (1.0 - hat_Lz)
