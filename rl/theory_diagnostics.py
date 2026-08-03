"""Pure summaries for finite-batch theorem-facing diagnostics.

These helpers summarize retained states or trajectories. They do not establish
a supremum over a policy-pair closure and must never be reported as uniform
certificates. Callers remain responsible for reconstructing the exact frozen
policy, recurrent map, augmented state, and ordered held-out data.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch


FINITE_BATCH_SCOPE = "finite-batch diagnostic only"


def _scope(metric: str) -> Dict[str, Any]:
    return {
        "metric": metric,
        "scope": FINITE_BATCH_SCOPE,
        "uniform_certificate": False,
    }


def _nearest_rank(sorted_values: List[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a quantile of an empty sample.")
    rank = max(1, int(math.ceil(probability * len(sorted_values))))
    return sorted_values[min(rank - 1, len(sorted_values) - 1)]


def summarize_finite_batch_metric(
    values: torch.Tensor,
    *,
    metric: str,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    """Summarize a nonnegative retained-batch metric without hiding ``+inf``."""

    if not torch.is_tensor(values):
        raise TypeError("values must be a torch.Tensor.")
    sample = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    raw = [float(value) for value in sample.tolist()]
    if not raw:
        if not allow_empty:
            raise ValueError("Cannot summarize an empty finite-batch metric.")
        return {
            **_scope(metric),
            "count": 0,
            "finite_count": 0,
            "positive_infinite_count": 0,
            "mean": None,
            "minimum": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    if any(math.isnan(value) for value in raw):
        raise ValueError(f"{metric} contains NaN.")
    if any(value < 0.0 or value == -math.inf for value in raw):
        raise ValueError(f"{metric} must be nonnegative.")

    ordered = sorted(raw)
    infinite_count = sum(value == math.inf for value in raw)
    finite = [value for value in raw if math.isfinite(value)]
    mean = math.inf if infinite_count else sum(finite) / len(finite)
    return {
        **_scope(metric),
        "count": len(raw),
        "finite_count": len(finite),
        "positive_infinite_count": infinite_count,
        "mean": mean,
        "minimum": ordered[0],
        "p50": _nearest_rank(ordered, 0.50),
        "p90": _nearest_rank(ordered, 0.90),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "maximum": ordered[-1],
    }


def _expand_action_mask(
    action_mask: Optional[torch.Tensor],
    shape: torch.Size,
) -> torch.Tensor:
    if action_mask is None:
        return torch.ones(shape, dtype=torch.bool)
    mask = action_mask.detach().to(device="cpu", dtype=torch.bool)
    if mask.ndim == 1:
        if mask.shape[0] != shape[1]:
            raise ValueError("One-dimensional action_mask has the wrong action size.")
        mask = mask.unsqueeze(0).expand(shape[0], -1)
    if mask.shape != shape:
        raise ValueError(
            f"action_mask shape {tuple(mask.shape)} does not match {tuple(shape)}."
        )
    if bool((mask.sum(dim=-1) == 0).any().item()):
        raise ValueError("Every retained state must have at least one valid action.")
    return mask


def _prepare_probabilities(
    probabilities: torch.Tensor,
    *,
    action_mask: torch.Tensor,
    label: str,
    tolerance: float,
) -> torch.Tensor:
    if tolerance < 0.0:
        raise ValueError("probability tolerance must be nonnegative.")
    if not torch.is_tensor(probabilities) or probabilities.ndim != 2:
        raise ValueError(f"{label} must have shape [states, actions].")
    probs = probabilities.detach().to(device="cpu", dtype=torch.float64)
    if probs.shape != action_mask.shape:
        raise ValueError(
            f"{label} shape {tuple(probs.shape)} does not match "
            f"action_mask {tuple(action_mask.shape)}."
        )
    if not bool(torch.isfinite(probs).all().item()):
        raise ValueError(f"{label} contains a non-finite probability.")
    if bool((probs < 0).any().item()):
        raise ValueError(f"{label} contains a negative probability.")
    invalid_mass = probs.masked_select(~action_mask)
    if invalid_mass.numel() and float(invalid_mass.max().item()) > tolerance:
        raise ValueError(f"{label} assigns probability to an invalid action.")
    probs = torch.where(action_mask, probs, torch.zeros_like(probs))
    row_sums = probs.sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=tolerance,
        rtol=0.0,
    ):
        raise ValueError(f"{label} rows must sum to one on valid actions.")
    return probs


def summarize_centering_defect(
    policy_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    action_mask: Optional[torch.Tensor] = None,
    probability_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Summarize ``|sum_a pi(a|s) Ahat(s,a)|`` on retained states."""

    if not torch.is_tensor(advantages) or advantages.ndim != 2:
        raise ValueError("advantages must have shape [states, actions].")
    mask = _expand_action_mask(action_mask, advantages.shape)
    probs = _prepare_probabilities(
        policy_probs,
        action_mask=mask,
        label="policy_probs",
        tolerance=probability_tolerance,
    )
    adv = advantages.detach().to(device="cpu", dtype=torch.float64)
    valid_advantages = adv.masked_select(mask)
    if not bool(torch.isfinite(valid_advantages).all().item()):
        raise ValueError("advantages must be finite on every valid action.")
    safe_advantages = torch.where(mask, adv, torch.zeros_like(adv))
    defects = (probs * safe_advantages).sum(dim=-1).abs()
    return {
        **_scope("statewise_centering_defect"),
        "state_count": int(defects.shape[0]),
        "absolute_expected_advantage": summarize_finite_batch_metric(
            defects,
            metric="absolute_expected_advantage",
        ),
    }


def _directional_kl(reference: torch.Tensor, comparison: torch.Tensor) -> torch.Tensor:
    reference_support = reference > 0
    comparison_support = comparison > 0
    missing_support = reference_support & ~comparison_support
    common_support = reference_support & comparison_support
    safe_reference = torch.where(
        common_support, reference, torch.ones_like(reference)
    )
    safe_comparison = torch.where(
        common_support, comparison, torch.ones_like(comparison)
    )
    terms = torch.where(
        common_support,
        reference * (safe_reference.log() - safe_comparison.log()),
        torch.zeros_like(reference),
    )
    # Roundoff can make a mathematically nonnegative finite KL a few ulps below
    # zero. Clamp only that final finite sum; missing support remains +inf below.
    divergence = terms.sum(dim=-1).clamp_min(0.0)
    return divergence.masked_fill(missing_support.any(dim=-1), math.inf)


def summarize_policy_gap(
    reference_probs: torch.Tensor,
    comparison_probs: torch.Tensor,
    *,
    action_mask: Optional[torch.Tensor] = None,
    reference_label: str = "exact_probability_mixture",
    comparison_label: str = "deployed_policy",
    probability_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Summarize TV, both KL directions, and exact support mismatches."""

    if not torch.is_tensor(reference_probs) or reference_probs.ndim != 2:
        raise ValueError("reference_probs must have shape [states, actions].")
    mask = _expand_action_mask(action_mask, reference_probs.shape)
    reference = _prepare_probabilities(
        reference_probs,
        action_mask=mask,
        label="reference_probs",
        tolerance=probability_tolerance,
    )
    comparison = _prepare_probabilities(
        comparison_probs,
        action_mask=mask,
        label="comparison_probs",
        tolerance=probability_tolerance,
    )

    total_variation = 0.5 * (reference - comparison).abs().sum(dim=-1)
    kl_forward = _directional_kl(reference, comparison)
    kl_reverse = _directional_kl(comparison, reference)
    ref_missing = (reference > 0) & (comparison == 0) & mask
    cmp_missing = (comparison > 0) & (reference == 0) & mask
    either = ref_missing | cmp_missing
    valid_action_count = int(mask.sum().item())
    return {
        **_scope("deployment_policy_gap"),
        "reference_policy": reference_label,
        "comparison_policy": comparison_label,
        "state_count": int(reference.shape[0]),
        "total_variation": summarize_finite_batch_metric(
            total_variation,
            metric="total_variation",
        ),
        "kl_reference_to_comparison": summarize_finite_batch_metric(
            kl_forward,
            metric="kl_reference_to_comparison",
        ),
        "kl_comparison_to_reference": summarize_finite_batch_metric(
            kl_reverse,
            metric="kl_comparison_to_reference",
        ),
        "support": {
            **_scope("policy_support_mismatch"),
            "valid_action_count": valid_action_count,
            "reference_not_comparison_action_count": int(ref_missing.sum().item()),
            "comparison_not_reference_action_count": int(cmp_missing.sum().item()),
            "either_direction_action_count": int(either.sum().item()),
            "either_direction_action_rate": (
                float(either.sum().item()) / float(valid_action_count)
            ),
            "reference_not_comparison_state_count": int(
                ref_missing.any(dim=-1).sum().item()
            ),
            "comparison_not_reference_state_count": int(
                cmp_missing.any(dim=-1).sum().item()
            ),
            "either_direction_state_count": int(either.any(dim=-1).sum().item()),
            "either_direction_state_rate": float(
                either.any(dim=-1).to(torch.float64).mean().item()
            ),
        },
    }


def _joint_distance(
    left_h: torch.Tensor,
    left_l: torch.Tensor,
    right_h: torch.Tensor,
    right_l: torch.Tensor,
) -> torch.Tensor:
    delta_h = left_h - right_h
    delta_l = left_l - right_l
    reduce_dims = tuple(range(1, delta_h.ndim))
    return torch.sqrt(
        delta_h.pow(2).sum(dim=reduce_dims)
        + delta_l.pow(2).sum(dim=reduce_dims)
    )


def summarize_depth_path(
    latent_h_by_depth: torch.Tensor,
    latent_l_by_depth: torch.Tensor,
    values_by_depth: torch.Tensor,
    *,
    n_depth: int,
    m_depth: int,
    ratio_denominator_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Summarize recurrent path length and value drift across retained depths."""

    if ratio_denominator_tolerance < 0.0:
        raise ValueError("ratio_denominator_tolerance must be nonnegative.")
    if latent_h_by_depth.shape != latent_l_by_depth.shape:
        raise ValueError("H/L latent-depth tensors must have identical shapes.")
    if latent_h_by_depth.ndim < 3:
        raise ValueError("Latents must have shape [states, depths, ...].")
    state_count, depth_count = latent_h_by_depth.shape[:2]
    if values_by_depth.shape != (state_count, depth_count):
        raise ValueError("values_by_depth must have shape [states, depths].")
    if not (0 <= n_depth < m_depth < depth_count):
        raise ValueError("Require 0 <= n_depth < m_depth < retained depth count.")

    latent_h = latent_h_by_depth.detach().to(device="cpu", dtype=torch.float64)
    latent_l = latent_l_by_depth.detach().to(device="cpu", dtype=torch.float64)
    values = values_by_depth.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(latent_h).all().item()) or not bool(
        torch.isfinite(latent_l).all().item()
    ):
        raise ValueError("Retained latent paths must be finite.")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("Retained depth values must be finite.")

    latent_delta_h = latent_h[:, 1:] - latent_h[:, :-1]
    latent_delta_l = latent_l[:, 1:] - latent_l[:, :-1]
    reduce_dims = tuple(range(2, latent_delta_h.ndim))
    increments = torch.sqrt(
        latent_delta_h.pow(2).sum(dim=reduce_dims)
        + latent_delta_l.pow(2).sum(dim=reduce_dims)
    )
    path_lengths = increments[:, n_depth:m_depth].sum(dim=-1)
    value_drift = (values[:, 1:] - values[:, :-1]).abs()
    depth_discrepancy = (values[:, m_depth] - values[:, n_depth]).abs()

    ratio_denominators = increments[:, :-1]
    ratio_numerators = increments[:, 1:]
    ratio_defined = ratio_denominators > ratio_denominator_tolerance
    ratios = ratio_numerators[ratio_defined] / ratio_denominators[ratio_defined]
    ratio_total = int(ratio_defined.numel())

    per_step: List[Dict[str, Any]] = []
    for depth in range(depth_count - 1):
        per_step.append(
            {
                "from_depth": depth,
                "to_depth": depth + 1,
                "latent_increment": summarize_finite_batch_metric(
                    increments[:, depth],
                    metric="latent_increment",
                ),
                "absolute_value_drift": summarize_finite_batch_metric(
                    value_drift[:, depth],
                    metric="absolute_value_drift",
                ),
            }
        )

    return {
        **_scope("finite_reference_depth_path"),
        "state_count": int(state_count),
        "retained_depth_count": int(depth_count),
        "n_depth": int(n_depth),
        "m_depth": int(m_depth),
        "path_length_n_to_m": summarize_finite_batch_metric(
            path_lengths,
            metric="path_length_n_to_m",
        ),
        "absolute_value_discrepancy_n_to_m": summarize_finite_batch_metric(
            depth_discrepancy,
            metric="absolute_value_discrepancy_n_to_m",
        ),
        "successive_increment_ratio": {
            **summarize_finite_batch_metric(
                ratios,
                metric="successive_increment_ratio",
                allow_empty=True,
            ),
            "denominator_tolerance": ratio_denominator_tolerance,
            "candidate_count": ratio_total,
            "undefined_count": ratio_total - int(ratio_defined.sum().item()),
            "undefined_rate": (
                0.0
                if ratio_total == 0
                else float((~ratio_defined).sum().item()) / float(ratio_total)
            ),
        },
        "per_step": per_step,
    }


def summarize_carry_continuity(
    current_latent_h: torch.Tensor,
    current_latent_l: torch.Tensor,
    successor_latent_h: torch.Tensor,
    successor_latent_l: torch.Tensor,
    next_input_latent_h: torch.Tensor,
    next_input_latent_l: torch.Tensor,
    current_clock: torch.Tensor,
    successor_clock: torch.Tensor,
    next_input_clock: torch.Tensor,
) -> Dict[str, Any]:
    """Summarize carry and clock integrity for retained adjacent record pairs."""

    latent_tensors = (
        current_latent_h,
        current_latent_l,
        successor_latent_h,
        successor_latent_l,
        next_input_latent_h,
        next_input_latent_l,
    )
    if any(tensor.shape != current_latent_h.shape for tensor in latent_tensors):
        raise ValueError("All carry tensors must have identical shapes.")
    if current_latent_h.ndim < 2 or current_latent_h.shape[0] == 0:
        raise ValueError("Carry tensors must have shape [adjacent_pairs, ...].")
    latents = [
        tensor.detach().to(device="cpu", dtype=torch.float64)
        for tensor in latent_tensors
    ]
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in latents):
        raise ValueError("Carry tensors must be finite.")

    clocks = [
        clock.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        for clock in (current_clock, successor_clock, next_input_clock)
    ]
    pair_count = current_latent_h.shape[0]
    if any(clock.shape[0] != pair_count for clock in clocks):
        raise ValueError("Each clock tensor must contain one value per adjacent pair.")
    if not all(bool(torch.isfinite(clock).all().item()) for clock in clocks):
        raise ValueError("Clock tensors must be finite.")
    if any(bool((clock < 0).any().item()) for clock in clocks):
        raise ValueError("Clock values must be nonnegative.")
    if any(bool((clock != clock.round()).any().item()) for clock in clocks):
        raise ValueError("Clock values must be integers.")

    current_h, current_l, successor_h, successor_l, next_h, next_l = latents
    carry_update = _joint_distance(
        successor_h,
        successor_l,
        current_h,
        current_l,
    )
    continuity_error = _joint_distance(
        successor_h,
        successor_l,
        next_h,
        next_l,
    )
    current_h_clock, successor_h_clock, next_h_clock = clocks
    expected_successor_clock = (current_h_clock - 1).clamp_min(0)
    decrement_error = (successor_h_clock - expected_successor_clock).abs()
    link_error = (next_h_clock - successor_h_clock).abs()
    return {
        **_scope("persistent_carry_continuity"),
        "adjacent_pair_count": int(pair_count),
        "recurrent_carry_update_distance": summarize_finite_batch_metric(
            carry_update,
            metric="recurrent_carry_update_distance",
        ),
        "successor_to_next_input_carry_error": summarize_finite_batch_metric(
            continuity_error,
            metric="successor_to_next_input_carry_error",
        ),
        "clock_decrement_error": summarize_finite_batch_metric(
            decrement_error,
            metric="clock_decrement_error",
        ),
        "clock_link_error": summarize_finite_batch_metric(
            link_error,
            metric="clock_link_error",
        ),
        "clock_decrement_violation_count": int((decrement_error != 0).sum().item()),
        "clock_link_violation_count": int((link_error != 0).sum().item()),
    }


def summarize_residual_estimates(
    absolute_residuals: torch.Tensor,
    *,
    estimator: str,
    horizon: int,
    monte_carlo_standard_errors: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Label residual measurements as finite-batch estimates, never certificates."""

    if horizon < 1:
        raise ValueError("horizon must be at least one.")
    if not estimator:
        raise ValueError("estimator must be a nonempty label.")
    result: Dict[str, Any] = {
        **_scope("augmented_bellman_residual_estimate"),
        "estimator": estimator,
        "horizon": int(horizon),
        "absolute_residual": summarize_finite_batch_metric(
            absolute_residuals,
            metric="absolute_residual",
        ),
    }
    if monte_carlo_standard_errors is not None:
        if monte_carlo_standard_errors.shape != absolute_residuals.shape:
            raise ValueError(
                "Monte Carlo standard errors must match the residual tensor shape."
            )
        result["monte_carlo_standard_error"] = summarize_finite_batch_metric(
            monte_carlo_standard_errors,
            metric="monte_carlo_standard_error",
        )
    return result
