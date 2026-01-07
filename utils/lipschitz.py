import warnings
from typing import Dict, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

# Threshold for warning about expensive exact baseline computation
# With ~1000 actions and batch size 128, this is O(128K) forward passes per policy update
EXACT_BASELINE_ACTION_THRESHOLD = 1000

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
    with torch.no_grad():
        # Fresh initialization
        z_fresh = model.init_latent(x_batch, y_batch)
        # Unroll n steps from fresh init
        z_fresh_n, _ = model.unroll_latent(x_batch, y_batch, n)
        
        # Unroll n steps from current (persistent) state
        z_current_n, _ = model.continue_latent(z_current, x_batch, y_batch, n)
        
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
    with torch.no_grad():
        # V_memoryless: always reinitialize z from (x, y) - ignores history
        v_memoryless, _ = model.used_value(x_batch, y_batch, n=n, z=None)
        
        # V_persistent: use the carried latent z - includes history
        if z_batch is not None:
            v_persistent, _ = model.used_value(x_batch, y_batch, n=n, z=z_batch)
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
    action_mask: torch.Tensor = None,
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
    
    LATENT MODE ASSUMPTION:
        This implementation assumes the **reset-latent / episodic** setting where
        z is reinitialized from (x, y) at each step. Both the policy distribution
        π(a|s) and the value function V(s') are computed by calling model.policy_dist
        and model.used_value with z=None, which reinitializes the latent.
        
        For **persistent-latent** experiments (episodic_latent=False), the Q-values
        computed here are V_memoryless from Section 5.4. The exact baseline remains
        "memoryless by design" because:
        1. We cannot tractably enumerate all possible latent trajectories z^(0:t).
        2. The "value of memory" analysis (Remark 5.5) treats the mismatch as a 
           bounded residual that vanishes as L_z → 0.
        
        The function compute_value_of_memory_residual() separately estimates this
        gap for monitoring purposes, but the exact baseline algorithm itself does
        not carry latent state.
    
    Args:
        model: TRM model with policy_dist and used_value methods
        x_batch: Batch dict with "inputs" and "puzzle_identifiers"  
        y_batch: Plan tensor [B, seq_len]
        env: Environment for applying edits and computing rewards
        n: Number of inner unrolling steps
        gamma: Discount factor
        checker_fn: Checker function for reward computation
        action_mask: Optional [num_actions] or [B, num_actions] mask
        
    Returns:
        Tuple of:
        - exact_baseline: [B] exact E_{a ~ π}[Q̂(s, a)]
        - q_all: [B, num_actions] Q̂(s, a) for all actions
    """
    # Require reward_shaping for exact baseline (otherwise rewards are sparse and 
    # the exact summation doesn't provide much benefit)
    assert env.config.reward_shaping, (
        "exact_baseline_summation currently requires reward_shaping=True. "
        "With sparse rewards, most Q-values would be zero until terminal."
    )
    
    with torch.no_grad():
        batch_size = y_batch.shape[0]
        device = y_batch.device
        
        # Get policy distribution (always using reset-latent evaluator)
        # NOTE: In persistent-latent mode, this is the "memoryless" approximation
        dist, _ = model.policy_dist(x_batch, y_batch, n=n, action_mask=action_mask)
        probs = dist.probs  # [B, num_actions]
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
            )
            
            # Compute rewards using the env's canonical reward helper
            # This ensures EXACT consistency with step()
            rewards = _compute_batch_reward_via_env(
                env=env,
                phi_old_batch=phi_old_batch,
                phi_new_batch=phi_new_batch,
                is_stop_action=is_stop,
                is_solved_batch=is_solved_batch,
            )
            
            # Determine if this action is terminal
            # STOP is terminal only if stop_action_mode == "terminal"
            # Otherwise STOP is a no-op and we bootstrap from V(s')
            if is_stop and stop_is_terminal:
                # Terminal STOP: Q = r (no bootstrap)
                q_values[:, a] = rewards
            else:
                # Non-terminal: Q = r + γV(s')
                # NOTE: We always use reset-latent critic here (no persistent z passed)
                v_next, _ = model.used_value(x_batch, y_next, n=n)
                q_values[:, a] = rewards + gamma * v_next
        
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
        
        if solved_threshold is not None:
            is_solved_batch[i] = phi_new_batch[i] >= solved_threshold
    
    return phi_old_batch, phi_new_batch, is_solved_batch


def _compute_batch_reward_via_env(
    env,  # PlanEditEnv
    phi_old_batch: torch.Tensor,
    phi_new_batch: torch.Tensor,
    is_stop_action: bool,
    is_solved_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Compute rewards using the env's compute_transition_reward() helper.
    
    This ensures EXACT consistency with the environment's step() rewards.
    """
    batch_size = phi_old_batch.shape[0]
    device = phi_old_batch.device
    rewards = torch.zeros(batch_size, device=device)
    
    # Determine terminal status for each sample
    # For non-STOP actions: terminal if solved
    # For STOP with mode="terminal": always terminal
    # For STOP with mode="noop"/"disabled": never terminal (continues)
    stop_is_terminal = env.is_stop_terminal()
    
    for i in range(batch_size):
        is_solved = is_solved_batch[i].item()
        
        # Terminal if solved, OR if STOP action with terminal mode
        is_terminal = is_solved or (is_stop_action and stop_is_terminal)
        
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
