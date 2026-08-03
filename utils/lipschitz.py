import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

# Threshold for warning about expensive exact baseline computation
# With ~1000 actions and batch size 128, this is O(128K) forward passes per policy update
EXACT_BASELINE_ACTION_THRESHOLD = 1000

def _is_linear_module(module: nn.Module) -> bool:
    """Return whether a module is one of the supported linear layers."""
    return isinstance(module, nn.Linear) or type(module).__name__ == "CastedLinear"


def _require_method(obj: object, name: str) -> Callable[..., Any]:
    method = getattr(obj, name, None)
    if not callable(method):
        raise TypeError(f"{type(obj).__name__} must provide callable `{name}()`.")
    return method


def apply_spectral_norm_to_trm(inner_model: nn.Module) -> None:
    """
    Apply spectral normalization to linear layers in the inner TRM module.

    Spectral norm keeps each linear map at ~1-Lipschitz; we pair this with
    per-layer output scaling to dial in stricter global contraction.
    """
    for module in inner_model.modules():
        if _is_linear_module(module):
            nn_utils.spectral_norm(module)


def apply_spectral_norm_to_value_head(value_head: nn.Module) -> None:
    """
    Apply spectral normalization to all linear layers in the value head.

    Spectral norm stabilizes each layer to ~1-Lipschitz before we apply the
    global contraction scalars.
    """
    for module in value_head.modules():
        if _is_linear_module(module):
            nn_utils.spectral_norm(module)


# =============================================================================
# NEW: Operator-norm clamping (alternative to spectral_norm)
# =============================================================================
#
# These functions provide a more robust alternative to PyTorch's spectral_norm,
# which can be numerically unstable in certain architectures (see diagnose_contraction_components.py).
#
# Key differences:
# - spectral_norm: Reparameterizes weight as W = sigma * W_normalized, updated via hooks
# - opnorm_clamp: Directly rescales weight.data in-place, no hooks or reparameterization
#
# Usage:
#   apply_opnorm_clamp_to_trm(inner, per_layer_max=1.0)  # Clamp each layer to ||W|| <= 1
#   enforce_global_contraction(inner, target_Lz)         # Scale outputs to achieve target_Lz
# =============================================================================


def _power_iteration(weight: torch.Tensor, num_iters: int = 10) -> float:
    """
    Estimate spectral norm of weight matrix via power iteration.

    Args:
        weight: 2D weight tensor [out_features, in_features]
        num_iters: Number of power iteration steps

    Returns:
        Estimated spectral norm (largest singular value)
    """
    # Ensure we work in float32 for numerical stability
    W = weight.detach().float()
    if W.dim() != 2:
        # Reshape to 2D for non-standard weight shapes
        W = W.view(W.size(0), -1)

    out_features, in_features = W.shape

    # Initialize random vector
    v = torch.randn(in_features, device=W.device, dtype=W.dtype)
    v = v / v.norm().clamp(min=1e-12)
    u = torch.zeros(out_features, device=W.device, dtype=W.dtype)

    for _ in range(num_iters):
        # u = W @ v / ||W @ v||
        u = W @ v
        u_norm = u.norm().clamp(min=1e-12)
        u = u / u_norm

        # v = W^T @ u / ||W^T @ u||
        v = W.t() @ u
        v_norm = v.norm().clamp(min=1e-12)
        v = v / v_norm

    # σ = u^T @ W @ v
    sigma = (u @ W @ v).item()
    return max(0.0, sigma)  # Ensure non-negative


def clamp_linear_operator_norm(
    module: nn.Module,
    max_norm: float,
    num_power_iters: int = 10,
) -> float:
    """
    Clamp the operator norm (spectral norm) of a linear layer's weight matrix.

    If ||W|| > max_norm, rescales W in-place: W <- W * (max_norm / ||W||)
    This is a one-shot rescaling, not a reparameterization like spectral_norm.

    Args:
        module: Linear layer (nn.Linear or CastedLinear)
        max_norm: Maximum allowed operator norm
        num_power_iters: Number of power iterations for spectral norm estimation

    Returns:
        The estimated spectral norm AFTER clamping
    """
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return 0.0

    sigma = _power_iteration(weight, num_power_iters)

    if sigma > max_norm and sigma > 1e-8:
        scale = max_norm / sigma
        with torch.no_grad():
            weight.mul_(scale)
        # Recompute sigma after scaling
        sigma = _power_iteration(weight, num_power_iters)

    return sigma


def apply_opnorm_clamp_to_trm(
    inner_model: nn.Module,
    per_layer_max: float = 1.0,
    num_power_iters: int = 10,
    restrict_to_reasoning_layers: bool = True,
) -> Dict[str, float]:
    """
    Apply operator-norm clamping to linear layers in the TRM inner module.

    This is a safer alternative to apply_spectral_norm_to_trm() that avoids
    the numerical instability issues observed with PyTorch's spectral_norm.

    Args:
        inner_model: TRM inner module (TinyRecursiveReasoningModel_ACTV1_Inner)
        per_layer_max: Maximum operator norm per layer (default 1.0)
        num_power_iters: Number of power iterations for spectral norm estimation
        restrict_to_reasoning_layers: If True, only clamp layers in the z->z path
            (L_level), not embedding or output head layers

    Returns:
        Dict mapping layer name to its spectral norm after clamping
    """
    sigma_dict = {}

    for name, module in inner_model.named_modules():
        if not _is_linear_module(module):
            continue

        # Optionally restrict to reasoning (z->z) layers only
        if restrict_to_reasoning_layers:
            # Only clamp layers in L_level (the reasoning stack)
            # Skip: embed_tokens, lm_head, q_head, puzzle_emb
            if not name.startswith("L_level"):
                continue

        sigma = clamp_linear_operator_norm(module, per_layer_max, num_power_iters)
        sigma_dict[name] = sigma

    return sigma_dict


def apply_opnorm_clamp_periodically(
    inner_model: nn.Module,
    per_layer_max: float = 1.0,
    num_power_iters: int = 10,
    restrict_to_reasoning_layers: bool = True,
) -> Dict[str, float]:
    """
    Convenience function to re-apply operator-norm clamping during training.

    Call this periodically (e.g., every N training steps) to enforce the
    per-layer norm constraint. Unlike spectral_norm which maintains the
    constraint via hooks, opnorm_clamp needs periodic re-application.

    This is a no-op wrapper around apply_opnorm_clamp_to_trm for API clarity.

    Returns:
        Dict mapping layer name to its spectral norm after clamping
    """
    return apply_opnorm_clamp_to_trm(
        inner_model, per_layer_max, num_power_iters, restrict_to_reasoning_layers
    )


def enforce_global_contraction(
    inner_model: nn.Module,
    target_Lz: float,
    restrict_to_reasoning_layers: bool = False,
) -> None:
    """
    Compound per-layer output scales so the overall inner latent map contraction
    is coarsely bounded by target_Lz.

    Spectral normalization (or opnorm clamping) keeps each layer near 1-Lipschitz;
    the multiplicative output scales produced here push the product of norms
    below target_Lz.

    Args:
        inner_model: TRM inner module
        target_Lz: Target Lipschitz constant (< 1 for contraction)
        restrict_to_reasoning_layers: If True, only scale z->z path layers (L_level)
    """

    if target_Lz <= 0.0:
        return

    # Collect linear modules, optionally restricting to reasoning layers
    if restrict_to_reasoning_layers:
        linear_modules = []
        for name, m in inner_model.named_modules():
            if _is_linear_module(m) and name.startswith("L_level"):
                linear_modules.append(m)
    else:
        linear_modules = [m for m in inner_model.modules() if _is_linear_module(m)]

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

    linear_modules = [m for m in value_head.modules() if _is_linear_module(m)]
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
        if not isinstance(current, torch.Tensor):
            raise TypeError(f"{scale_attr} must be a tensor buffer.")
        with torch.no_grad():
            current.mul_(alpha)
        return

    # First-time setup
    device = None
    dtype = torch.float32
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        device = weight.device
        dtype = weight.dtype
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
    latent_step = _require_method(inner_model, "latent_step")
    with torch.no_grad():
        baseline = latent_step(carry, input_embeddings, seq_info)
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
            out = latent_step(perturbed, input_embeddings, seq_info)
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
    init_latent = _require_method(model, "init_latent")
    update_latent = _require_method(model, "update_latent")
    with torch.no_grad():
        z0 = init_latent(x_batch, y_batch)
        z1 = update_latent(z0, y_batch, x_batch)

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


# =============================================================================
# NEW: Drift bound monitoring for persistent latents (Section 4.2, Lemma 4.4)
# =============================================================================

def estimate_Cdrift(
    model: nn.Module,
    x_batch: dict,
    y_batch: torch.Tensor,
    z_current,  # TinyRecursiveReasoningModel_ACTV1InnerCarry
    n: int,
) -> float:
    """
    Estimate drift constant C_drift(n) = ||z^(n) - z_init(x, y)|| for persistent latents.
    
    This corresponds to the tracking error bound in Lemma 4.4:
        C_drift(n) = κ_n * E_0 + (κ_n * L_{z*} * Δy_max) / (1 - κ_n)
    
    In practice, we measure the empirical drift from fresh initialization.
    
    Args:
        model: TRM model with init_latent and continue_latent methods
        x_batch: Batch dict with "inputs" and "puzzle_identifiers"
        y_batch: Plan tensor [B, seq_len]
        z_current: Current latent state (carried from previous step)
        n: Number of inner unrolling steps
        
    Returns:
        Maximum ||z^(n)_current - z^(n)_fresh|| over the batch (scalar float)
    """
    unroll_latent = _require_method(model, "unroll_latent")
    continue_latent = _require_method(model, "continue_latent")
    with torch.no_grad():
        # Unroll n steps from fresh init
        z_fresh_n, _ = unroll_latent(x_batch, y_batch, n)
        
        # Unroll n steps from current (persistent) state
        z_current_n, _ = continue_latent(z_current, x_batch, y_batch, n)
        
        # Compute drift: ||z^(n)_current - z^(n)_fresh||
        diff_H = (z_current_n.z_H - z_fresh_n.z_H).pow(2).sum(dim=(1, 2))
        diff_L = (z_current_n.z_L - z_fresh_n.z_L).pow(2).sum(dim=(1, 2))
        drift_norms = torch.sqrt(diff_H + diff_L)
        
        return float(drift_norms.max().item())


def estimate_plan_change(
    y_old: torch.Tensor,
    y_new: torch.Tensor,
) -> float:
    """
    Estimate plan change Δy = ||y_new - y_old|| for two-timescale analysis.
    
    This corresponds to the Δy_max bound in Assumption 4.3:
        ||y_{t+1} - y_t||_E ≤ Δy_max
    
    For discrete tokens (like Sudoku), we use Hamming distance normalized by sequence length.
    For continuous plans, we use L2 distance.
    
    Args:
        y_old: Previous plan tensor [B, seq_len] or [seq_len]
        y_new: New plan tensor [B, seq_len] or [seq_len]
        
    Returns:
        Maximum plan change over the batch (scalar float)
    """
    with torch.no_grad():
        if y_old.dim() == 1:
            y_old = y_old.unsqueeze(0)
        if y_new.dim() == 1:
            y_new = y_new.unsqueeze(0)
            
        # For integer tokens: use Hamming distance (fraction of changed cells)
        if y_old.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            changes = (y_old != y_new).float().sum(dim=-1)  # [B]
            max_change = float(changes.max().item())
        else:
            # For continuous: use L2 distance
            diff = (y_new - y_old).float()
            distances = diff.norm(p=2, dim=-1)  # [B]
            max_change = float(distances.max().item())
            
        return max_change


def compute_theoretical_drift_bound(
    kappa_n: float,
    E_0: float,
    L_zstar: float,
    delta_y_max: float,
) -> float:
    """
    Compute the theoretical drift bound from Lemma 4.4:
    
        C_drift(n) = κ_n * E_0 + (κ_n * L_{z*} * Δy_max) / (1 - κ_n)
    
    Where κ_n = L_z^n is the contraction factor after n steps.
    
    Args:
        kappa_n: L_z^n contraction factor
        E_0: Initial error bound ||z_0 - z*(x, y_0)||
        L_zstar: Lipschitz constant of fixed-point w.r.t. plan
        delta_y_max: Maximum plan change per step
        
    Returns:
        Theoretical drift bound (scalar float)
    """
    if kappa_n >= 1.0:
        return float("inf")
    return kappa_n * E_0 + (kappa_n * L_zstar * delta_y_max) / (1.0 - kappa_n)


# =============================================================================
# NEW: Value of memory analysis (Section 5.4, Corollary 5.4)
# =============================================================================

def compute_value_of_memory_residual(
    model: nn.Module,
    x_batch: dict,
    y_batch: torch.Tensor,
    z_batch,  # Current latent state for persistent mode
    n: int,
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> dict:
    """
    Compute the "value of memory" residual: ||V_memless - T_K V_memless||.
    
    V_memoryless(x, y, z) := U_*(x, y) ignores the latent z entirely.
    The residual captures the cost of approximating a history-dependent
    policy with a memoryless value function (POMDP approximation error).
    
    This implements the interpretation in Remark 5.5:
        The residual measures the combined effect of:
        (i) ignoring memory (latent z)
        (ii) approximation limits of (f_θ, V_ψ) on (x, y) alone
    
    Args:
        model: TRM model with used_value method
        x_batch: Batch dict with "inputs" and "puzzle_identifiers"
        y_batch: Plan tensor [B, seq_len]
        z_batch: Current latent state (for comparison)
        n: Number of inner unrolling steps
        rewards: [B] rewards from transitions
        next_values: [B] V(s') for next states (using persistent z)
        dones: [B] done flags
        gamma: Discount factor
        
    Returns:
        Dictionary with:
        - v_memoryless: Value ignoring latent z (fresh init each time)
        - v_persistent: Value using persistent latent z
        - value_of_memory: ||V_persistent - V_memoryless|| (the "cost of amnesia")
        - bellman_residual_memoryless: ||V_memless - T V_memless||
    """
    used_value = _require_method(model, "used_value")
    with torch.no_grad():
        # V_memoryless: always reinitialize z from (x, y) - ignores history
        v_memoryless, _ = used_value(x_batch, y_batch, n=n, z=None)
        
        # V_persistent: use the carried latent z - includes history
        if z_batch is not None:
            v_persistent, _ = used_value(x_batch, y_batch, n=n, z=z_batch)
        else:
            v_persistent = v_memoryless  # Same if no persistent state
        
        # Value of memory: how much value do we lose by ignoring z?
        value_of_memory = (v_persistent - v_memoryless).abs()
        
        # Bellman residual for memoryless value
        mask = (~dones).float()
        td_target_memoryless = rewards + gamma * next_values * mask
        bellman_residual = (v_memoryless - td_target_memoryless).abs()
        
        return {
            "v_memoryless_mean": float(v_memoryless.mean().item()),
            "v_persistent_mean": float(v_persistent.mean().item()),
            "value_of_memory_mean": float(value_of_memory.mean().item()),
            "value_of_memory_max": float(value_of_memory.max().item()),
            "bellman_residual_memoryless_mean": float(bellman_residual.mean().item()),
            "bellman_residual_memoryless_max": float(bellman_residual.max().item()),
        }


# =============================================================================
# NEW: Exact baseline computation for Theorem 5.9
# =============================================================================

def compute_exact_baseline_summation(
    model: nn.Module,
    x_batch: dict,
    y_batch: torch.Tensor,
    env,  # PlanEditEnv for applying edits
    n: int,
    gamma: float,
    checker_fn,
    action_mask: Optional[torch.Tensor] = None,
    policy_probs: Optional[torch.Tensor] = None,
    successor_latent: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exact baseline E_{a ~ π}[Q̂(s,a)] via summation over ALL discrete actions.
    
    This is the KEY requirement for Theorem 5.9's O(α·ε_A) bound.
    Without exact summation, the bound reverts to naive O(ε_A).
    
    For each action a:
        Q̂(s, a) = r(s, a, s') + γ * V(s')
    
    Where s' = edit(y, a) is deterministic.
    
    Baseline = Σ_a π(a|s) * Q̂(s, a)
    
    This is tractable for discrete action spaces like Sudoku (~800 actions).
    
    IMPORTANT: This function uses the env's compute_transition_reward() helper
    to ensure EXACT consistency with the environment's reward semantics,
    including STOP penalties, terminal rewards, and shaping terms.
    
    LATENT CARRY ORDER:
        In reset-latent mode, leave ``successor_latent`` unset and each successor
        value reinitializes its latent from the successor plan. In persistent
        mode, the recurrent unroll precedes the sampled edit. Pass that post-unroll
        carry as ``successor_latent``; it is then the same input carry for every
        enumerated action successor. This computes the baseline on the augmented
        state rather than a memoryless approximation.
    
    Args:
        model: TRM model with policy_dist and used_value methods
        x_batch: Batch dict with "inputs" and "puzzle_identifiers"  
        y_batch: Plan tensor [B, seq_len]
        env: Environment for applying edits and computing rewards
        n: Number of inner unrolling steps
        gamma: Discount factor
        checker_fn: Checker function for reward computation
        action_mask: Optional [num_actions] or [B, num_actions] mask
        policy_probs: Optional fixed current-policy probabilities [B, A].
            Supplying these keeps the policy fixed when Q is evaluated at a
            different recurrent reference depth.
        successor_latent: Optional post-policy-unroll recurrent carry used as
            the input latent for every nonterminal successor value. Leave unset
            for reset-latent evaluation.
        
    Returns:
        Tuple of:
        - exact_baseline: [B] exact E_{a ~ π}[Q̂(s, a)]
        - q_all: [B, num_actions] Q̂(s, a) for all actions
    """
    if getattr(env, "_enable_undo", False):
        raise NotImplementedError(
            "Exact baseline enumeration does not reconstruct history-dependent UNDO actions."
        )
    
    policy_dist = _require_method(model, "policy_dist")
    used_value = _require_method(model, "used_value")
    if successor_latent is not None and policy_probs is None:
        raise ValueError(
            "Persistent augmented-state baseline enumeration requires fixed "
            "policy_probs computed from the same post-unroll policy state."
        )
    with torch.no_grad():
        batch_size = y_batch.shape[0]
        device = y_batch.device
        
        if policy_probs is None:
            # Without fixed probabilities this is the reset-latent policy path.
            dist, _ = policy_dist(x_batch, y_batch, n=n, action_mask=action_mask)
            probs = dist.probs
        else:
            probs = policy_probs.to(device=device)
            if probs.ndim != 2 or probs.shape[0] != batch_size:
                raise ValueError(
                    "policy_probs must have shape [batch_size, num_actions], "
                    f"got {tuple(probs.shape)}."
                )
        num_actions = probs.shape[-1]
        
        # Warn about expensive computation for large action spaces
        if num_actions > EXACT_BASELINE_ACTION_THRESHOLD:
            warnings.warn(
                f"Exact baseline computation iterates over {num_actions} actions "
                f"(threshold: {EXACT_BASELINE_ACTION_THRESHOLD}). This is O(B × A) forward passes "
                f"and may be slow. Consider disabling exact_baseline_summation for faster training.",
                RuntimeWarning,
                stacklevel=2,
            )
        
        # For each action, compute Q̂(s, a) = r(s, a, s') + γ * V(s')
        q_values = torch.zeros(batch_size, num_actions, device=device)
        
        # Get env parameters
        vocab_size = env.vocab_size
        stop_action_id = env.stop_action_id
        stop_is_terminal = env.is_stop_terminal()
        solved_threshold = getattr(env.config, "solved_threshold", None)
        
        # Optimization: Only iterate over actions that are valid for at least one sample
        # This skips actions that are universally masked (e.g., PAD/Empty tokens)
        if action_mask is not None:
            if action_mask.dim() == 2:
                # [B, A] -> Valid if ANY sample in batch allows it
                valid_actions_mask = action_mask.any(dim=0)
            else:
                # [A]
                valid_actions_mask = action_mask
            
            # Get list of integer indices to iterate over
            action_indices = torch.nonzero(valid_actions_mask, as_tuple=True)[0].tolist()
        else:
            action_indices = range(num_actions)

        for a in action_indices:
            is_stop = (a == stop_action_id)
            
            # Apply edit action to get y' = edit(y, a)
            y_next = _apply_edit_batch(y_batch, a, vocab_size, stop_action_id)
            
            # Compute phi values for reward calculation
            phi_old_batch, phi_new_batch, is_solved_batch = _compute_phi_batch(
                x_batch=x_batch,
                y_old=y_batch,
                y_new=y_next,
                checker_fn=checker_fn,
                solved_threshold=solved_threshold,
                task_type=getattr(env.config, "task_type", None),
                is_solved_fn=getattr(env, "is_plan_solved", None),
            )

            if "remaining_edits" in x_batch:
                budget_terminal = x_batch["remaining_edits"].reshape(-1) <= 1
            else:
                budget_terminal = torch.zeros(
                    batch_size, dtype=torch.bool, device=device
                )
            stop_terminal = torch.full(
                (batch_size,),
                bool(is_stop and stop_is_terminal),
                dtype=torch.bool,
                device=device,
            )
            terminal_batch = is_solved_batch | budget_terminal | stop_terminal
            
            # Compute rewards using the env's canonical reward helper
            # This ensures EXACT consistency with step()
            rewards = _compute_batch_reward_via_env(
                env=env,
                phi_old_batch=phi_old_batch,
                phi_new_batch=phi_new_batch,
                is_stop_action=is_stop,
                is_solved_batch=is_solved_batch,
                is_terminal_batch=terminal_batch,
            )
            
            # Determine if this action is terminal
            # STOP is terminal only if stop_action_mode == "terminal"
            # Otherwise STOP is a no-op and we bootstrap from V(s')
            # Terminal rewards already fold in the absorbing tail and therefore
            # do not bootstrap. Persistent mode uses the post-policy-unroll carry
            # shared by every possible edit successor.
            x_next_batch = dict(x_batch)
            if "remaining_edits" in x_batch:
                x_next_batch["remaining_edits"] = (
                    x_batch["remaining_edits"] - 1
                ).clamp_min(0)
            if successor_latent is None:
                v_next, _ = used_value(x_next_batch, y_next, n=n)
            else:
                v_next, _ = used_value(
                    x_next_batch,
                    y_next,
                    n=n,
                    z=successor_latent,
                )
            q_values[:, a] = rewards + gamma * v_next * (~terminal_batch).to(v_next.dtype)
        
        # Apply action mask if provided (invalid actions get -inf Q-value)
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0).expand(batch_size, -1)
            q_values = q_values.masked_fill(~action_mask, float("-inf"))
            # Renormalize probs for valid actions only
            probs = probs.masked_fill(~action_mask, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Exact baseline: E_{a ~ π}[Q̂(s, a)] = Σ_a π(a|s) * Q̂(s, a)
        # Only sum over valid (non-inf) Q-values
        valid_q = q_values.masked_fill(q_values == float("-inf"), 0.0)
        exact_baseline = (probs * valid_q).sum(dim=-1)  # [B]
        
        return exact_baseline, q_values


def _compute_phi_batch(
    x_batch: dict,
    y_old: torch.Tensor,
    y_new: torch.Tensor,
    checker_fn,
    solved_threshold,
    task_type: Optional[str] = None,
    is_solved_fn=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute checker scores (potentials) for a batch of transitions.
    
    Returns:
        phi_old_batch: [B] Φ(s) = c(x, y_old)
        phi_new_batch: [B] Φ(s') = c(x, y_new)
        is_solved_batch: [B] bool, whether phi_new >= solved_threshold
    """
    batch_size = y_old.shape[0]
    device = y_old.device
    
    phi_old_batch = torch.zeros(batch_size, device=device)
    phi_new_batch = torch.zeros(batch_size, device=device)
    is_solved_batch = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        x_i = {k: v[i:i+1] if torch.is_tensor(v) else v for k, v in x_batch.items()}
        phi_old_batch[i] = checker_fn(x_i, y_old[i])
        phi_new_batch[i] = checker_fn(x_i, y_new[i])
        
        if is_solved_fn is not None:
            is_solved_batch[i] = is_solved_fn(
                y_new[i], checker_score=float(phi_new_batch[i].item())
            )
        elif solved_threshold is not None:
            is_solved_batch[i] = phi_new_batch[i] >= solved_threshold
        elif task_type == "sudoku":
            raise RuntimeError(
                "Sudoku exact-baseline enumeration requires env.is_plan_solved()."
            )
    
    return phi_old_batch, phi_new_batch, is_solved_batch


def _compute_batch_reward_via_env(
    env,  # PlanEditEnv
    phi_old_batch: torch.Tensor,
    phi_new_batch: torch.Tensor,
    is_stop_action: bool,
    is_solved_batch: torch.Tensor,
    is_terminal_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Compute rewards using the env's compute_transition_reward() helper.
    
    This ensures EXACT consistency with the environment's step() rewards.
    """
    batch_size = phi_old_batch.shape[0]
    device = phi_old_batch.device
    rewards = torch.zeros(batch_size, device=device)
    
    for i in range(batch_size):
        is_solved = bool(is_solved_batch[i].item())
        is_terminal = bool(is_terminal_batch[i].item())
        
        rewards[i] = env.compute_transition_reward(
            phi_old=phi_old_batch[i].item(),
            phi_new=phi_new_batch[i].item(),
            is_stop_action=is_stop_action,
            is_terminal=is_terminal,
            is_solved=is_solved,
        )
    
    return rewards


def _apply_edit_batch(
    y_batch: torch.Tensor,
    action: int,
    vocab_size: int,
    stop_action_id: int,
) -> torch.Tensor:
    """Apply a single edit action to a batch of plans."""
    if action == stop_action_id:
        return y_batch.clone()
    
    batch_size = y_batch.shape[0]
    y_next = y_batch.clone()
    
    # Decode action to (position, token)
    pos = action // vocab_size
    tok = action % vocab_size
    
    # Apply edit to flat view
    flat = y_next.view(batch_size, -1)
    if pos < flat.shape[1]:
        flat[:, pos] = tok
    
    return y_next


def compute_exact_advantage(
    q_values: torch.Tensor,
    exact_baseline: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute exact advantage Â(s, a) = Q̂(s, a) - E_{b ~ π}[Q̂(s, b)].
    
    This satisfies the centering property E_{a ~ π}[Â(s, a)] = 0 EXACTLY,
    which is required for the O(α·ε_A) bound in Theorem 5.9.
    
    Args:
        q_values: [B, num_actions] Q̂(s, a) for all actions
        exact_baseline: [B] E_{a ~ π}[Q̂(s, a)]
        actions: [B] actions taken
        
    Returns:
        advantages: [B] exact advantages for the taken actions
    """
    batch_size = actions.shape[0]
    batch_indices = torch.arange(batch_size, device=actions.device)
    q_taken = q_values[batch_indices, actions]  # [B]
    advantages = q_taken - exact_baseline  # [B]
    return advantages
