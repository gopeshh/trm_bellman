"""Deterministic float64 sanity checks for the UPI-TRM finite-MDP bounds.

This module evaluates finite examples by exact matrix calculation.  It is not
a proof, a learned-model evaluation, or evidence for any uniform theorem
premise.  It deliberately has no checkpoint, Torch, environment, or training
dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt


FloatArray: TypeAlias = npt.NDArray[np.float64]
JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

SUITE_ID: Final[str] = "exact_finite_mdp_sanity_v1"
SCHEMA_VERSION: Final[int] = 1
FLOAT_DTYPE: Final[str] = "float64"
FLOAT_TOLERANCE: Final[float] = 1.0e-10
RANDOM_SEEDS: Final[tuple[int, ...]] = (
    2026081401,
    2026081402,
    2026081403,
    2026081404,
    2026081405,
    2026081406,
    2026081407,
    2026081408,
)

RESULTS_FILENAME: Final[str] = "exact_finite_mdp_results.json"
TABLE_FILENAME: Final[str] = "exact_finite_mdp_table.tsv"
POLICY_PLOT_FILENAME: Final[str] = "policy_improvement_vs_cpi_bound.png"
FINITE_REFERENCE_PLOT_FILENAME: Final[str] = "finite_reference_error_vs_bound.png"
FIRST_MISMATCH_PLOT_FILENAME: Final[str] = "first_mismatch_attainment.png"
OUTPUT_FILENAMES: Final[frozenset[str]] = frozenset(
    {
        RESULTS_FILENAME,
        TABLE_FILENAME,
        POLICY_PLOT_FILENAME,
        FINITE_REFERENCE_PLOT_FILENAME,
        FIRST_MISMATCH_PLOT_FILENAME,
    }
)


class FiniteMDPError(ValueError):
    """Raised when a finite-MDP input violates the registered contract."""


@dataclass(frozen=True)
class FiniteMDP:
    """A finite discounted MDP with expected one-step rewards.

    ``transitions[s, a, next_state]`` is stochastic and ``rewards[s, a]`` is
    the conditional expected reward.  All arrays must be float64.
    """

    transitions: FloatArray
    rewards: FloatArray
    initial: FloatArray
    gamma: float

    def __post_init__(self) -> None:
        if self.transitions.dtype != np.float64:
            raise FiniteMDPError("transitions must have dtype float64")
        if self.rewards.dtype != np.float64:
            raise FiniteMDPError("rewards must have dtype float64")
        if self.initial.dtype != np.float64:
            raise FiniteMDPError("initial must have dtype float64")
        if self.transitions.ndim != 3:
            raise FiniteMDPError("transitions must have shape [state, action, state]")
        state_count, action_count, successor_count = self.transitions.shape
        if state_count == 0 or action_count == 0 or successor_count != state_count:
            raise FiniteMDPError("transitions must have nonempty square state axes")
        if self.rewards.shape != (state_count, action_count):
            raise FiniteMDPError("rewards shape does not match transitions")
        if self.initial.shape != (state_count,):
            raise FiniteMDPError("initial shape does not match transitions")
        if not 0.0 <= self.gamma < 1.0 or not math.isfinite(self.gamma):
            raise FiniteMDPError("gamma must be finite and lie in [0, 1)")
        _require_finite(self.transitions, "transitions")
        _require_finite(self.rewards, "rewards")
        _require_distribution(self.initial, "initial")
        row_sums = np.sum(self.transitions, axis=2)
        if np.any(self.transitions < 0.0) or not np.allclose(
            row_sums, 1.0, rtol=0.0, atol=1.0e-13
        ):
            raise FiniteMDPError("every transition row must be a probability law")

    @property
    def state_count(self) -> int:
        return int(self.transitions.shape[0])

    @property
    def action_count(self) -> int:
        return int(self.transitions.shape[1])


@dataclass(frozen=True)
class SafeStepDecision:
    branch: str
    maximum_alpha: float


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    observed: float
    bound: float
    slack: float
    numerical_violation: float
    passed: bool
    relation: str
    metadata: dict[str, JsonValue]

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "bound": self.bound,
            "case_id": self.case_id,
            "category": self.category,
            "metadata": self.metadata,
            "numerical_violation": self.numerical_violation,
            "observed": self.observed,
            "passed": self.passed,
            "relation": self.relation,
            "slack": self.slack,
        }


@dataclass(frozen=True)
class PlotData:
    policy_alphas: list[float]
    policy_actual_improvements: list[float]
    policy_cpi_lower_bounds: list[float]
    finite_reference_labels: list[str]
    finite_reference_errors: list[float]
    finite_reference_bounds: list[float]
    mismatch_deltas: list[float]
    mismatch_actual: list[float]
    mismatch_bounds: list[float]

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "finite_reference": {
                "bounds": self.finite_reference_bounds,
                "errors": self.finite_reference_errors,
                "labels": self.finite_reference_labels,
            },
            "first_mismatch": {
                "actual": self.mismatch_actual,
                "bounds": self.mismatch_bounds,
                "delta_dep": self.mismatch_deltas,
            },
            "policy_improvement": {
                "actual": self.policy_actual_improvements,
                "alpha": self.policy_alphas,
                "cpi_lower_bounds": self.policy_cpi_lower_bounds,
            },
        }


class CaseCollector:
    def __init__(self, tolerance: float) -> None:
        if tolerance <= 0.0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be positive and finite")
        self.tolerance = tolerance
        self.cases: list[CaseResult] = []

    def add(
        self,
        case_id: str,
        category: str,
        observed: float,
        bound: float,
        *,
        metadata: dict[str, JsonValue] | None = None,
        relation: str = "observed <= bound",
    ) -> None:
        if not math.isfinite(observed) or not math.isfinite(bound):
            raise ValueError(f"{case_id} produced a nonfinite comparison")
        slack = bound - observed
        violation = max(0.0, -slack)
        self.cases.append(
            CaseResult(
                case_id=case_id,
                category=category,
                observed=observed,
                bound=bound,
                slack=slack,
                numerical_violation=violation,
                passed=violation <= self.tolerance,
                relation=relation,
                metadata={} if metadata is None else metadata,
            )
        )

    def equality(
        self,
        case_id: str,
        category: str,
        left: float,
        right: float,
        *,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        self.add(
            case_id,
            category,
            abs(left - right),
            0.0,
            metadata=metadata,
            relation="absolute equality error <= tolerance",
        )

    def assertion(
        self,
        case_id: str,
        category: str,
        condition: bool,
        *,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        self.add(
            case_id,
            category,
            0.0 if condition else 1.0,
            0.0,
            metadata=metadata,
            relation="registered condition is true",
        )


def _require_finite(array: FloatArray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise FiniteMDPError(f"{name} must contain only finite values")


def _require_distribution(array: FloatArray, name: str) -> None:
    _require_finite(array, name)
    if array.ndim != 1 or np.any(array < 0.0):
        raise FiniteMDPError(f"{name} must be a one-dimensional probability law")
    if not math.isclose(float(np.sum(array)), 1.0, rel_tol=0.0, abs_tol=1.0e-13):
        raise FiniteMDPError(f"{name} must sum to one")


def require_policy(mdp: FiniteMDP, policy: FloatArray, name: str) -> None:
    if policy.dtype != np.float64:
        raise FiniteMDPError(f"{name} must have dtype float64")
    if policy.shape != (mdp.state_count, mdp.action_count):
        raise FiniteMDPError(f"{name} shape does not match the MDP")
    _require_finite(policy, name)
    if np.any(policy < 0.0) or not np.allclose(
        np.sum(policy, axis=1), 1.0, rtol=0.0, atol=1.0e-13
    ):
        raise FiniteMDPError(f"every row of {name} must be a probability law")


def policy_transition(mdp: FiniteMDP, policy: FloatArray) -> FloatArray:
    require_policy(mdp, policy, "policy")
    return np.asarray(
        np.einsum("sa,san->sn", policy, mdp.transitions), dtype=np.float64
    )


def policy_reward(mdp: FiniteMDP, policy: FloatArray) -> FloatArray:
    require_policy(mdp, policy, "policy")
    return np.asarray(np.sum(policy * mdp.rewards, axis=1), dtype=np.float64)


def solve_value(mdp: FiniteMDP, policy: FloatArray) -> FloatArray:
    transition = policy_transition(mdp, policy)
    reward = policy_reward(mdp, policy)
    system = np.eye(mdp.state_count, dtype=np.float64) - mdp.gamma * transition
    return np.asarray(np.linalg.solve(system, reward), dtype=np.float64)


def discounted_occupancy(mdp: FiniteMDP, policy: FloatArray) -> FloatArray:
    """Return normalized discounted pre-decision occupancy as a row law."""

    transition = policy_transition(mdp, policy)
    system = np.eye(mdp.state_count, dtype=np.float64) - mdp.gamma * transition.T
    occupancy = np.linalg.solve(system, (1.0 - mdp.gamma) * mdp.initial)
    occupancy = np.asarray(occupancy, dtype=np.float64)
    if np.any(occupancy < -1.0e-12):
        raise ArithmeticError("discounted occupancy has a negative component")
    return occupancy


def q_values(mdp: FiniteMDP, value: FloatArray) -> FloatArray:
    if value.shape != (mdp.state_count,):
        raise FiniteMDPError("value shape does not match the MDP")
    continuation = np.einsum("san,n->sa", mdp.transitions, value)
    return np.asarray(mdp.rewards + mdp.gamma * continuation, dtype=np.float64)


def advantages(mdp: FiniteMDP, policy: FloatArray) -> tuple[FloatArray, FloatArray]:
    value = solve_value(mdp, policy)
    return value, np.asarray(q_values(mdp, value) - value[:, None], dtype=np.float64)


def exact_policy_mixture(
    current: FloatArray, candidate: FloatArray, alpha: float
) -> FloatArray:
    if not 0.0 <= alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and lie in [0, 1]")
    if current.shape != candidate.shape:
        raise FiniteMDPError("policy shapes differ")
    return np.asarray((1.0 - alpha) * current + alpha * candidate, dtype=np.float64)


def total_variation(first: FloatArray, second: FloatArray) -> float:
    if first.shape != second.shape:
        raise ValueError("total-variation arguments have different shapes")
    return 0.5 * float(np.sum(np.abs(first - second)))


def vector_span(vector: FloatArray) -> float:
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("span requires a nonempty vector")
    return float(np.max(vector) - np.min(vector))


def expected_return(mdp: FiniteMDP, policy: FloatArray) -> float:
    return float(mdp.initial @ solve_value(mdp, policy))


def k_step_backup(
    mdp: FiniteMDP, policy: FloatArray, endpoint: FloatArray, k: int
) -> FloatArray:
    if k < 1:
        raise ValueError("k must be positive")
    if endpoint.shape != (mdp.state_count,):
        raise FiniteMDPError("endpoint shape does not match the MDP")
    transition = policy_transition(mdp, policy)
    reward = policy_reward(mdp, policy)
    power = np.eye(mdp.state_count, dtype=np.float64)
    reward_sum = np.zeros(mdp.state_count, dtype=np.float64)
    for step in range(k):
        reward_sum += (mdp.gamma**step) * (power @ reward)
        power = power @ transition
    return np.asarray(
        reward_sum + (mdp.gamma**k) * (power @ endpoint), dtype=np.float64
    )


def target_network_bridge(
    mdp: FiniteMDP,
    policy: FloatArray,
    endpoint: FloatArray,
    target_value: FloatArray,
    k: int,
) -> dict[str, float]:
    if endpoint.shape != (mdp.state_count,) or target_value.shape != endpoint.shape:
        raise FiniteMDPError("target-network values have an invalid shape")
    transition = policy_transition(mdp, policy)
    transition_power = np.linalg.matrix_power(transition, k)
    target_backup = k_step_backup(mdp, policy, target_value, k)
    self_backup = k_step_backup(mdp, policy, endpoint, k)
    target_residual = float(np.max(np.abs(endpoint - target_backup)))
    self_residual = float(np.max(np.abs(endpoint - self_backup)))
    propagated_lag = (mdp.gamma**k) * float(
        np.max(np.abs(transition_power @ (target_value - endpoint)))
    )
    sup_lag = (mdp.gamma**k) * float(np.max(np.abs(target_value - endpoint)))
    return {
        "propagated_bound": target_residual + propagated_lag,
        "propagated_lag": propagated_lag,
        "self_bootstrap_residual": self_residual,
        "sup_bound": target_residual + sup_lag,
        "sup_lag": sup_lag,
        "target_population_residual": target_residual,
    }


def finite_reference_quantities(
    mdp: FiniteMDP,
    policy: FloatArray,
    endpoint_n: FloatArray,
    endpoint_m: FloatArray,
    k: int,
) -> dict[str, float]:
    value = solve_value(mdp, policy)
    discrepancy = float(np.max(np.abs(endpoint_n - endpoint_m)))
    residual = float(
        np.max(np.abs(endpoint_m - k_step_backup(mdp, policy, endpoint_m, k)))
    )
    actual_error = float(np.max(np.abs(endpoint_n - value)))
    bound = discrepancy + residual / (1.0 - mdp.gamma**k)
    return {
        "actual_error": actual_error,
        "bound": bound,
        "endpoint_discrepancy": discrepancy,
        "reference_residual": residual,
    }


def cpi_quantities(
    mdp: FiniteMDP,
    current: FloatArray,
    candidate: FloatArray,
    estimated_advantage: FloatArray,
    alpha: float,
) -> dict[str, float]:
    require_policy(mdp, current, "current policy")
    require_policy(mdp, candidate, "candidate policy")
    if estimated_advantage.shape != (mdp.state_count, mdp.action_count):
        raise FiniteMDPError("estimated advantage shape does not match the MDP")
    value, advantage = advantages(mdp, current)
    occupancy = discounted_occupancy(mdp, current)
    mixture = exact_policy_mixture(current, candidate, alpha)
    candidate_gain = np.sum(candidate * advantage, axis=1)
    centering_defect = np.sum(current * estimated_advantage, axis=1)
    candidate_defect = np.sum(candidate * (estimated_advantage - advantage), axis=1)
    xi_alpha = float(
        occupancy @ ((1.0 - alpha) * centering_defect + alpha * candidate_defect)
    )
    eta_current = float(mdp.initial @ value)
    true_surrogate = eta_current + float(
        occupancy @ np.sum(mixture * advantage, axis=1)
    ) / (1.0 - mdp.gamma)
    estimated_surrogate = eta_current + float(
        occupancy @ np.sum(mixture * estimated_advantage, axis=1)
    ) / (1.0 - mdp.gamma)
    delta_g = vector_span(candidate_gain)
    occupancy_penalty = (
        mdp.gamma
        * alpha
        * alpha
        * delta_g
        / ((1.0 - mdp.gamma) * (1.0 - mdp.gamma + mdp.gamma * alpha))
    )
    return {
        "candidate_gain_mean": float(occupancy @ candidate_gain),
        "delta_g": delta_g,
        "eta_current": eta_current,
        "eta_mixture": expected_return(mdp, mixture),
        "estimated_surrogate": estimated_surrogate,
        "master_lower_bound": estimated_surrogate
        - xi_alpha / (1.0 - mdp.gamma)
        - occupancy_penalty,
        "occupancy_penalty": occupancy_penalty,
        "true_surrogate": true_surrogate,
        "xi_alpha": xi_alpha,
    }


def scalar_cpi_lower_bound(
    estimated_surrogate: float,
    gamma: float,
    alpha: float,
    epsilon_candidate: float,
    epsilon_cpi: float,
) -> float:
    """Return the scalar-error CPI lower bound from the paper."""

    if not 0.0 <= gamma < 1.0 or not math.isfinite(gamma):
        raise ValueError("gamma must be finite and lie in [0, 1)")
    if not 0.0 <= alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and lie in [0, 1]")
    if epsilon_candidate < 0.0 or not math.isfinite(epsilon_candidate):
        raise ValueError("epsilon_candidate must be finite and nonnegative")
    if epsilon_cpi < 0.0 or not math.isfinite(epsilon_cpi):
        raise ValueError("epsilon_cpi must be finite and nonnegative")
    bias_penalty = alpha * epsilon_candidate / (1.0 - gamma)
    occupancy_penalty = (
        2.0
        * epsilon_cpi
        * gamma
        * alpha
        * alpha
        / ((1.0 - gamma) * (1.0 - gamma + gamma * alpha))
    )
    return estimated_surrogate - bias_penalty - occupancy_penalty


def safe_step_decision(
    gamma: float, certified_slope: float, delta_g: float
) -> SafeStepDecision:
    if not 0.0 <= gamma < 1.0 or not math.isfinite(gamma):
        raise ValueError("gamma must be finite and lie in [0, 1)")
    if delta_g < 0.0 or not math.isfinite(delta_g):
        raise ValueError("delta_g must be finite and nonnegative")
    if not math.isfinite(certified_slope):
        raise ValueError("certified_slope must be finite")
    if certified_slope < 0.0:
        return SafeStepDecision("negative_slope_no_positive_step", 0.0)
    if gamma == 0.0:
        return SafeStepDecision("gamma_zero_all_steps", 1.0)
    if certified_slope == 0.0 and delta_g == 0.0:
        return SafeStepDecision("zero_slope_zero_span_all_steps", 1.0)
    if certified_slope == 0.0:
        return SafeStepDecision("zero_slope_positive_span_no_positive_step", 0.0)
    gamma_span = gamma * delta_g
    # Values registered as the exact boundary can differ by one binary64 ULP
    # when the product is recomputed. Treat only that representation error as
    # equality; larger shortfalls remain in the interior branch.
    if certified_slope >= gamma_span - math.ulp(gamma_span):
        return SafeStepDecision("slope_at_least_gamma_span_all_steps", 1.0)
    threshold = certified_slope * (1.0 - gamma) / (gamma * (delta_g - certified_slope))
    return SafeStepDecision("interior_threshold", threshold)


def finite_horizon_values(
    mdp: FiniteMDP, policy: FloatArray, horizon: int, terminal_boundary: float
) -> list[FloatArray]:
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    transition = policy_transition(mdp, policy)
    reward = policy_reward(mdp, policy)
    values = [np.full(mdp.state_count, terminal_boundary, dtype=np.float64)]
    for _ in range(horizon):
        values.append(
            np.asarray(reward + mdp.gamma * (transition @ values[-1]), dtype=np.float64)
        )
    return values


def finite_horizon_reference_quantities(
    mdp: FiniteMDP,
    policy: FloatArray,
    endpoint_n: list[FloatArray],
    endpoint_m: list[FloatArray],
    horizon: int,
    k: int,
    terminal_boundary: float,
) -> list[dict[str, float | int]]:
    if horizon < 0 or k < 1:
        raise ValueError("horizon must be nonnegative and k must be positive")
    if len(endpoint_n) != horizon + 1 or len(endpoint_m) != horizon + 1:
        raise ValueError("finite-horizon endpoint lists have the wrong length")
    true_values = finite_horizon_values(mdp, policy, horizon, terminal_boundary)
    boundary = np.full(mdp.state_count, terminal_boundary, dtype=np.float64)
    if not np.array_equal(endpoint_n[0], boundary) or not np.array_equal(
        endpoint_m[0], boundary
    ):
        raise FiniteMDPError("stage-zero endpoints must equal the common boundary")
    residuals = [0.0]
    for h in range(1, horizon + 1):
        block = min(k, h)
        backup = k_step_backup(mdp, policy, endpoint_m[h - block], block)
        residuals.append(float(np.max(np.abs(endpoint_m[h] - backup))))
    rows: list[dict[str, float | int]] = []
    for h in range(horizon + 1):
        discrepancy = float(np.max(np.abs(endpoint_n[h] - endpoint_m[h])))
        actual = float(np.max(np.abs(endpoint_n[h] - true_values[h])))
        residual_sum = 0.0
        stage = h
        block_index = 0
        while stage > 0:
            residual_sum += (mdp.gamma ** (block_index * k)) * residuals[stage]
            stage = max(0, stage - k)
            block_index += 1
        block_count = 0 if h == 0 else math.ceil(h / k)
        final_block = 0 if h == 0 else h - (block_count - 1) * k
        rows.append(
            {
                "actual_error": actual,
                "bound": discrepancy + residual_sum,
                "endpoint_discrepancy": discrepancy,
                "h": h,
                "j_h": block_count,
                "last_block": final_block,
                "residual_sum": residual_sum,
            }
        )
    return rows


def finite_horizon_cpi_quantities(
    mdp: FiniteMDP,
    current: FloatArray,
    candidate: FloatArray,
    horizon: int,
    alpha: float,
    terminal_boundary: float,
) -> dict[str, float]:
    """Evaluate the exact stagewise finite-horizon CPI span expression."""

    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    require_policy(mdp, current, "current policy")
    require_policy(mdp, candidate, "candidate policy")
    mixture = exact_policy_mixture(current, candidate, alpha)
    current_values = finite_horizon_values(mdp, current, horizon, terminal_boundary)
    mixture_values = finite_horizon_values(mdp, mixture, horizon, terminal_boundary)
    current_transition = policy_transition(mdp, current)
    state_law = np.array(mdp.initial, dtype=np.float64, copy=True)
    estimated_surrogate = float(mdp.initial @ current_values[horizon])
    occupancy_penalty = 0.0
    for time in range(horizon):
        remaining = horizon - time
        action_values = np.asarray(
            mdp.rewards
            + mdp.gamma
            * np.einsum("san,n->sa", mdp.transitions, current_values[remaining - 1]),
            dtype=np.float64,
        )
        stage_advantage = action_values - current_values[remaining][:, None]
        candidate_gain = np.sum(candidate * stage_advantage, axis=1)
        estimated_surrogate += (mdp.gamma**time) * float(
            state_law @ np.sum(mixture * stage_advantage, axis=1)
        )
        occupancy_penalty += (
            alpha
            * (mdp.gamma**time)
            * vector_span(candidate_gain)
            * (1.0 - (1.0 - alpha) ** time)
        )
        state_law = np.asarray(state_law @ current_transition, dtype=np.float64)
    return {
        "actual_return": float(mdp.initial @ mixture_values[horizon]),
        "estimated_surrogate": estimated_surrogate,
        "lower_bound": estimated_surrogate - occupancy_penalty,
        "occupancy_penalty": occupancy_penalty,
    }


def deployment_bound(gamma: float, delta_dep: float, reward_span: float) -> float:
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    if not 0.0 <= delta_dep <= 1.0:
        raise ValueError("delta_dep must lie in [0, 1]")
    if reward_span < 0.0:
        raise ValueError("reward_span must be nonnegative")
    return reward_span * delta_dep / ((1.0 - gamma) * (1.0 - gamma + gamma * delta_dep))


def finite_horizon_deployment_bound(
    gamma: float, delta_dep: float, reward_span: float, horizon: int
) -> float:
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    return reward_span * sum(
        gamma**time * (1.0 - (1.0 - delta_dep) ** (time + 1)) for time in range(horizon)
    )


def first_mismatch_example(
    gamma: float, delta_dep: float, reward_span: float = 1.0
) -> tuple[FiniteMDP, FloatArray, FloatArray]:
    transitions = np.zeros((2, 2, 2), dtype=np.float64)
    transitions[0, 0, 0] = 1.0
    transitions[0, 1, 1] = 1.0
    transitions[1, :, 1] = 1.0
    rewards = np.array(
        [[0.0, reward_span], [reward_span, reward_span]], dtype=np.float64
    )
    mdp = FiniteMDP(
        transitions=transitions,
        rewards=rewards,
        initial=np.array([1.0, 0.0], dtype=np.float64),
        gamma=gamma,
    )
    reference = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    deployed = np.array([[1.0 - delta_dep, delta_dep], [1.0, 0.0]], dtype=np.float64)
    return mdp, reference, deployed


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return np.asarray(exponentials / np.sum(exponentials), dtype=np.float64)


def parameter_and_probability_mixtures(
    old_logits: FloatArray, candidate_logits: FloatArray, alpha: float
) -> tuple[FloatArray, FloatArray]:
    old_policy = _softmax(old_logits)
    candidate_policy = _softmax(candidate_logits)
    probability_mixture = np.asarray(
        (1.0 - alpha) * old_policy + alpha * candidate_policy, dtype=np.float64
    )
    parameter_mixture = _softmax(
        np.asarray(
            (1.0 - alpha) * old_logits + alpha * candidate_logits,
            dtype=np.float64,
        )
    )
    return probability_mixture, parameter_mixture


def _base_mdp(gamma: float = 0.81) -> tuple[FiniteMDP, FloatArray, FloatArray]:
    transitions = np.array(
        [
            [
                [[0.75, 0.20, 0.05]],
                [[0.10, 0.65, 0.25]],
            ],
            [
                [[0.15, 0.70, 0.15]],
                [[0.05, 0.20, 0.75]],
            ],
            [
                [[0.55, 0.15, 0.30]],
                [[0.20, 0.25, 0.55]],
            ],
        ],
        dtype=np.float64,
    ).reshape(3, 2, 3)
    rewards = np.array([[0.15, 0.70], [-0.20, 0.45], [0.55, -0.10]], dtype=np.float64)
    mdp = FiniteMDP(
        transitions=transitions,
        rewards=rewards,
        initial=np.array([0.55, 0.30, 0.15], dtype=np.float64),
        gamma=gamma,
    )
    current = np.array([[0.80, 0.20], [0.65, 0.35], [0.45, 0.55]], dtype=np.float64)
    candidate = np.array([[0.25, 0.75], [0.30, 0.70], [0.75, 0.25]], dtype=np.float64)
    return mdp, current, candidate


def _random_mdp(seed: int, index: int) -> tuple[FiniteMDP, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    state_count = 4
    action_count = 3
    transitions = np.asarray(
        rng.dirichlet(np.ones(state_count), size=(state_count, action_count)),
        dtype=np.float64,
    )
    rewards = np.asarray(
        rng.uniform(-1.0, 1.0, size=(state_count, action_count)), dtype=np.float64
    )
    initial = np.asarray(rng.dirichlet(np.ones(state_count)), dtype=np.float64)
    current = np.asarray(
        rng.dirichlet(np.ones(action_count), size=state_count), dtype=np.float64
    )
    candidate = np.asarray(
        rng.dirichlet(np.ones(action_count), size=state_count), dtype=np.float64
    )
    gamma = 0.51 + 0.07 * (index % 6)
    return FiniteMDP(transitions, rewards, initial, gamma), current, candidate


def _persistent_augmented_example() -> tuple[FiniteMDP, FloatArray, FloatArray]:
    # State index is 2*x + z.  Both policies use this one shared transition map.
    transitions = np.zeros((4, 2, 4), dtype=np.float64)
    rewards = np.zeros((4, 2), dtype=np.float64)
    for x in range(2):
        for z in range(2):
            state = 2 * x + z
            for action in range(2):
                next_x = x ^ action
                next_z = (z + x + action) % 2
                successor = 2 * next_x + next_z
                transitions[state, action, successor] = 1.0
                rewards[state, action] = float(next_x) + 0.2 * float(next_z)
    mdp = FiniteMDP(
        transitions=transitions,
        rewards=rewards,
        initial=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        gamma=0.72,
    )
    current = np.array(
        [[0.85, 0.15], [0.20, 0.80], [0.70, 0.30], [0.25, 0.75]],
        dtype=np.float64,
    )
    candidate = np.array(
        [[0.35, 0.65], [0.75, 0.25], [0.30, 0.70], [0.80, 0.20]],
        dtype=np.float64,
    )
    return mdp, current, candidate


def _record_safe_step_branches(collector: CaseCollector) -> None:
    registered = (
        ("gamma_zero_negative", 0.0, -0.1, 0.8, "negative_slope_no_positive_step", 0.0),
        ("gamma_zero_nonnegative", 0.0, 0.0, 0.8, "gamma_zero_all_steps", 1.0),
        ("negative_slope", 0.8, -0.1, 0.8, "negative_slope_no_positive_step", 0.0),
        (
            "zero_slope_positive_span",
            0.8,
            0.0,
            0.8,
            "zero_slope_positive_span_no_positive_step",
            0.0,
        ),
        (
            "zero_slope_zero_span",
            0.8,
            0.0,
            0.0,
            "zero_slope_zero_span_all_steps",
            1.0,
        ),
        (
            "above_gamma_span",
            0.8,
            0.65,
            0.8,
            "slope_at_least_gamma_span_all_steps",
            1.0,
        ),
        (
            "positive_slope_zero_span",
            0.8,
            0.2,
            0.0,
            "slope_at_least_gamma_span_all_steps",
            1.0,
        ),
        (
            "equal_gamma_span",
            0.8,
            0.8 * 0.8,
            0.8,
            "slope_at_least_gamma_span_all_steps",
            1.0,
        ),
        ("interior", 0.8, 0.2, 0.8, "interior_threshold", 1.0 / 12.0),
    )
    for case_id, gamma, slope, delta_g, expected_branch, expected_alpha in registered:
        decision = safe_step_decision(gamma, slope, delta_g)
        collector.assertion(
            f"safe_step_{case_id}_branch",
            "safe_step",
            decision.branch == expected_branch,
            metadata={
                "actual_branch": decision.branch,
                "delta_g": delta_g,
                "expected_branch": expected_branch,
                "gamma": gamma,
                "m_d": slope,
            },
        )
        collector.equality(
            f"safe_step_{case_id}_threshold",
            "safe_step",
            decision.maximum_alpha,
            expected_alpha,
            metadata={"branch": decision.branch},
        )


def _record_base_cases(
    collector: CaseCollector,
) -> tuple[list[float], list[float], list[float]]:
    mdp, current, candidate = _base_mdp()
    value, advantage = advantages(mdp, current)
    transition = policy_transition(mdp, current)
    reward = policy_reward(mdp, current)
    collector.equality(
        "bellman_fixed_point",
        "bellman",
        float(np.max(np.abs(value - (reward + mdp.gamma * transition @ value)))),
        0.0,
    )
    occupancy = discounted_occupancy(mdp, current)
    collector.equality(
        "discounted_occupancy_normalization",
        "occupancy",
        float(np.sum(occupancy)),
        1.0,
    )
    collector.equality(
        "advantage_current_policy_centering",
        "centering",
        float(np.max(np.abs(np.sum(current * advantage, axis=1)))),
        0.0,
    )

    mdp_gamma_zero, current_zero, _ = _base_mdp(gamma=0.0)
    value_zero = solve_value(mdp_gamma_zero, current_zero)
    collector.equality(
        "gamma_zero_value_equals_immediate_reward",
        "boundary",
        float(np.max(np.abs(value_zero - policy_reward(mdp_gamma_zero, current_zero)))),
        0.0,
    )

    endpoint_m = value + np.array([0.40, -0.25, 0.15], dtype=np.float64)
    endpoint_n = endpoint_m + np.array([-0.10, 0.20, -0.05], dtype=np.float64)
    finite_reference = finite_reference_quantities(
        mdp, current, endpoint_n, endpoint_m, 3
    )
    collector.add(
        "finite_reference_residual_bound",
        "finite_reference",
        finite_reference["actual_error"],
        finite_reference["bound"],
        metadata={
            "endpoint_discrepancy": finite_reference["endpoint_discrepancy"],
            "k": 3,
            "reference_residual": finite_reference["reference_residual"],
        },
    )

    target_value = value + np.array([0.30, -0.15, 0.35], dtype=np.float64)
    bridge = target_network_bridge(mdp, current, endpoint_m, target_value, 3)
    collector.add(
        "target_propagated_lag_bridge",
        "target_network",
        bridge["self_bootstrap_residual"],
        bridge["propagated_bound"],
        metadata={key: number for key, number in bridge.items()},
    )
    collector.add(
        "target_sup_lag_fallback",
        "target_network",
        bridge["propagated_bound"],
        bridge["sup_bound"],
        metadata={key: number for key, number in bridge.items()},
    )

    estimation_error = np.array(
        [[0.10, -0.04], [-0.08, 0.13], [0.06, -0.11]], dtype=np.float64
    )
    estimated_advantage = advantage + estimation_error
    raw_center = np.sum(current * estimated_advantage, axis=1)
    centered_advantage = estimated_advantage - raw_center[:, None]
    collector.equality(
        "exact_statewise_centering",
        "centering",
        float(np.max(np.abs(np.sum(current * centered_advantage, axis=1)))),
        0.0,
    )
    centered_candidate_defect = np.sum(
        candidate * (centered_advantage - advantage), axis=1
    )
    centered_candidate_estimate = np.sum(candidate * centered_advantage, axis=1)
    true_candidate_gain = np.sum(candidate * advantage, axis=1)
    estimated_slope = float(occupancy @ centered_candidate_estimate)
    exact_signed_defect = float(occupancy @ centered_candidate_defect)
    certified_slope = estimated_slope - exact_signed_defect
    collector.equality(
        "safe_step_exact_signed_defect_recovers_true_slope",
        "safe_step",
        certified_slope,
        float(occupancy @ true_candidate_gain),
        metadata={
            "exact_signed_defect": exact_signed_defect,
            "m_d": certified_slope,
            "widehat_g": estimated_slope,
        },
    )

    alphas = [float(alpha) for alpha in np.linspace(0.0, 1.0, 21)]
    actual_improvements: list[float] = []
    lower_bounds: list[float] = []
    for index, alpha in enumerate(alphas):
        mixture = exact_policy_mixture(current, candidate, alpha)
        occupancy_mixture = discounted_occupancy(mdp, mixture)
        occupancy_tv = total_variation(occupancy_mixture, occupancy)
        occupancy_tv_bound = mdp.gamma * alpha / (1.0 - mdp.gamma + mdp.gamma * alpha)
        collector.add(
            f"exact_mixture_occupancy_alpha_{index:02d}",
            "occupancy",
            occupancy_tv,
            occupancy_tv_bound,
            metadata={"alpha": alpha},
        )
        if index in (0, 7, 20):
            l1_distance = float(np.sum(np.abs(occupancy_mixture - occupancy)))
            collector.equality(
                f"l1_equals_two_tv_alpha_{index:02d}",
                "occupancy",
                l1_distance,
                2.0 * occupancy_tv,
                metadata={"alpha": alpha},
            )

        signed = cpi_quantities(mdp, current, candidate, estimated_advantage, alpha)
        collector.equality(
            f"signed_surrogate_identity_alpha_{index:02d}",
            "cpi_signed",
            signed["estimated_surrogate"] - signed["true_surrogate"],
            signed["xi_alpha"] / (1.0 - mdp.gamma),
            metadata={"alpha": alpha},
        )
        collector.add(
            f"cpi_span_lower_bound_alpha_{index:02d}",
            "cpi_span",
            signed["master_lower_bound"],
            signed["eta_mixture"],
            metadata={
                "alpha": alpha,
                "delta_g": signed["delta_g"],
                "orientation": "lower_bound <= actual_return",
            },
            relation="lower bound <= actual return",
        )

        centered = cpi_quantities(mdp, current, candidate, centered_advantage, alpha)
        candidate_defect = np.sum(candidate * (centered_advantage - advantage), axis=1)
        epsilon_candidate = float(np.max(np.abs(candidate_defect)))
        candidate_gain = np.sum(candidate * advantage, axis=1)
        epsilon_cpi = float(np.max(np.abs(candidate_gain)))
        scalar_lower = scalar_cpi_lower_bound(
            centered["estimated_surrogate"],
            mdp.gamma,
            alpha,
            epsilon_candidate,
            epsilon_cpi,
        )
        collector.add(
            f"cpi_scalar_lower_bound_alpha_{index:02d}",
            "cpi_scalar",
            scalar_lower,
            centered["eta_mixture"],
            metadata={
                "alpha": alpha,
                "epsilon_a_candidate": epsilon_candidate,
                "epsilon_cpi": epsilon_cpi,
                "orientation": "lower_bound <= actual_return",
            },
            relation="lower bound <= actual return",
        )
        actual_improvements.append(signed["eta_mixture"] - signed["eta_current"])
        lower_bounds.append(
            alpha * signed["candidate_gain_mean"] / (1.0 - mdp.gamma)
            - signed["occupancy_penalty"]
        )

    candidate_gain = np.sum(candidate * advantage, axis=1)
    mixture_mid = exact_policy_mixture(current, candidate, 0.35)
    occupancy_mid = discounted_occupancy(mdp, mixture_mid)
    expectation_gap = abs(
        float(occupancy_mid @ candidate_gain - occupancy @ candidate_gain)
    )
    tv_mid = total_variation(occupancy_mid, occupancy)
    collector.add(
        "span_times_tv_no_extra_factor_two",
        "cpi_span",
        expectation_gap,
        tv_mid * vector_span(candidate_gain),
        metadata={
            "l1_distance": float(np.sum(np.abs(occupancy_mid - occupancy))),
            "span": vector_span(candidate_gain),
            "tv": tv_mid,
        },
    )

    zero_span_gain = np.zeros(mdp.state_count, dtype=np.float64)
    collector.equality(
        "delta_g_zero",
        "boundary",
        vector_span(zero_span_gain),
        0.0,
    )
    return alphas, actual_improvements, lower_bounds


def _record_finite_horizon_cases(collector: CaseCollector) -> None:
    mdp, current, candidate = _base_mdp(gamma=0.77)
    horizon = 5
    terminal_boundary = 0.35
    true_values = finite_horizon_values(mdp, current, horizon, terminal_boundary)
    direction_m = np.array([0.12, -0.09, 0.05], dtype=np.float64)
    direction_n = np.array([-0.04, 0.07, -0.02], dtype=np.float64)
    endpoint_m = [true_values[0]] + [
        np.asarray(true_values[h] + (0.7**h) * direction_m, dtype=np.float64)
        for h in range(1, horizon + 1)
    ]
    endpoint_n = [true_values[0]] + [
        np.asarray(endpoint_m[h] + (0.6**h) * direction_n, dtype=np.float64)
        for h in range(1, horizon + 1)
    ]
    for k in (3, 7):
        rows = finite_horizon_reference_quantities(
            mdp,
            current,
            endpoint_n,
            endpoint_m,
            horizon,
            k,
            terminal_boundary,
        )
        for row in rows:
            h = int(row["h"])
            if h == 0 or (k == 3 and h in (2, 5)) or (k == 7 and h == 5):
                collector.add(
                    f"finite_horizon_K{k}_h{h}",
                    "finite_horizon",
                    float(row["actual_error"]),
                    float(row["bound"]),
                    metadata={
                        "h": h,
                        "j_h": int(row["j_h"]),
                        "k": k,
                        "last_block": int(row["last_block"]),
                    },
                )
    collector.equality(
        "finite_horizon_H0_deployment",
        "finite_horizon",
        finite_horizon_deployment_bound(0.77, 0.4, 1.2, 0),
        0.0,
    )
    for index, alpha in enumerate((0.0, 0.35, 1.0)):
        cpi = finite_horizon_cpi_quantities(
            mdp,
            current,
            candidate,
            horizon=5,
            alpha=alpha,
            terminal_boundary=terminal_boundary,
        )
        collector.add(
            f"finite_horizon_cpi_alpha_{index}",
            "finite_horizon",
            cpi["lower_bound"],
            cpi["actual_return"],
            metadata={
                "alpha": alpha,
                "orientation": "lower_bound <= actual return",
                "span_penalty": cpi["occupancy_penalty"],
            },
            relation="lower bound <= actual return",
        )
    zero_horizon_cpi = finite_horizon_cpi_quantities(
        mdp,
        current,
        candidate,
        horizon=0,
        alpha=0.35,
        terminal_boundary=terminal_boundary,
    )
    collector.equality(
        "finite_horizon_cpi_H0",
        "finite_horizon",
        zero_horizon_cpi["actual_return"],
        zero_horizon_cpi["lower_bound"],
    )


def _record_first_mismatch_cases(
    collector: CaseCollector,
) -> tuple[list[float], list[float], list[float]]:
    gamma = 0.83
    reward_span = 1.7
    deltas = [float(delta) for delta in np.linspace(0.0, 1.0, 21)]
    actual_values: list[float] = []
    bounds: list[float] = []
    for index, delta_dep in enumerate(deltas):
        mdp, reference, deployed = first_mismatch_example(gamma, delta_dep, reward_span)
        actual = abs(expected_return(mdp, deployed) - expected_return(mdp, reference))
        bound = deployment_bound(gamma, delta_dep, reward_span)
        collector.equality(
            f"first_mismatch_infinite_delta_{index:02d}",
            "deployment",
            actual,
            bound,
            metadata={"delta_dep": delta_dep, "gamma": gamma},
        )
        horizon = 6
        reference_h = finite_horizon_values(mdp, reference, horizon, 0.0)[horizon]
        deployed_h = finite_horizon_values(mdp, deployed, horizon, 0.0)[horizon]
        finite_actual = abs(float(mdp.initial @ (deployed_h - reference_h)))
        finite_bound = finite_horizon_deployment_bound(
            gamma, delta_dep, reward_span, horizon
        )
        collector.equality(
            f"first_mismatch_finite_delta_{index:02d}",
            "deployment",
            finite_actual,
            finite_bound,
            metadata={
                "delta_dep": delta_dep,
                "gamma": gamma,
                "horizon": horizon,
            },
        )
        actual_values.append(actual)
        bounds.append(bound)
    return deltas, actual_values, bounds


def _record_parameter_mixture_case(collector: CaseCollector) -> None:
    old_logits = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    candidate_logits = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    probability, parameter = parameter_and_probability_mixtures(
        old_logits, candidate_logits, 0.5
    )
    distance = total_variation(probability, parameter)
    collector.assertion(
        "parameter_interpolation_is_not_probability_mixture",
        "mixture_semantics",
        distance > 1.0e-6,
        metadata={
            "parameter_interpolation": [float(value) for value in parameter],
            "pointwise_probability_mixture": [float(value) for value in probability],
            "tv": distance,
        },
    )


def _record_persistent_augmented_cases(collector: CaseCollector) -> None:
    mdp, current, candidate = _persistent_augmented_example()
    alpha = 0.4
    mixture = exact_policy_mixture(current, candidate, alpha)
    transition_mixture = policy_transition(mdp, mixture)
    affine_transition = (1.0 - alpha) * policy_transition(
        mdp, current
    ) + alpha * policy_transition(mdp, candidate)
    collector.equality(
        "persistent_shared_map_affine_transition",
        "persistent_augmented_state",
        float(np.max(np.abs(transition_mixture - affine_transition))),
        0.0,
        metadata={
            "state_encoding": "state=2*x+z",
            "transition_map": "next_x=x xor action; next_z=(z+x+action) mod 2",
        },
    )
    value = solve_value(mdp, current)
    collector.equality(
        "persistent_augmented_bellman_fixed_point",
        "persistent_augmented_state",
        float(
            np.max(
                np.abs(
                    value
                    - (
                        policy_reward(mdp, current)
                        + mdp.gamma * policy_transition(mdp, current) @ value
                    )
                )
            )
        ),
        0.0,
    )
    current_occupancy = discounted_occupancy(mdp, current)
    mixture_occupancy = discounted_occupancy(mdp, mixture)
    collector.add(
        "persistent_augmented_occupancy_tv",
        "persistent_augmented_state",
        total_variation(mixture_occupancy, current_occupancy),
        mdp.gamma * alpha / (1.0 - mdp.gamma + mdp.gamma * alpha),
        metadata={"alpha": alpha},
    )


def _record_random_cases(collector: CaseCollector) -> None:
    for index, seed in enumerate(RANDOM_SEEDS):
        mdp, current, candidate = _random_mdp(seed, index)
        rng = np.random.default_rng(seed + 10_000_000)
        value, advantage = advantages(mdp, current)
        alpha = 0.1 + 0.1 * (index % 7)
        mixture = exact_policy_mixture(current, candidate, alpha)
        current_occupancy = discounted_occupancy(mdp, current)
        mixture_occupancy = discounted_occupancy(mdp, mixture)
        collector.add(
            f"random_{seed}_occupancy_tv",
            "random_mdp",
            total_variation(mixture_occupancy, current_occupancy),
            mdp.gamma * alpha / (1.0 - mdp.gamma + mdp.gamma * alpha),
            metadata={"alpha": alpha, "gamma": mdp.gamma, "seed": seed},
        )
        endpoint_m = np.asarray(
            value + rng.normal(0.0, 0.25, mdp.state_count), dtype=np.float64
        )
        endpoint_n = np.asarray(
            endpoint_m + rng.normal(0.0, 0.15, mdp.state_count), dtype=np.float64
        )
        k = 1 + index % 4
        finite_reference = finite_reference_quantities(
            mdp, current, endpoint_n, endpoint_m, k
        )
        collector.add(
            f"random_{seed}_finite_reference",
            "random_mdp",
            finite_reference["actual_error"],
            finite_reference["bound"],
            metadata={"k": k, "seed": seed},
        )
        target_value = np.asarray(
            value + rng.normal(0.0, 0.20, mdp.state_count), dtype=np.float64
        )
        bridge = target_network_bridge(mdp, current, endpoint_m, target_value, k)
        collector.add(
            f"random_{seed}_target_propagated",
            "random_mdp",
            bridge["self_bootstrap_residual"],
            bridge["propagated_bound"],
            metadata={"k": k, "seed": seed},
        )
        estimated_advantage = np.asarray(
            advantage + rng.normal(0.0, 0.12, (mdp.state_count, mdp.action_count)),
            dtype=np.float64,
        )
        cpi = cpi_quantities(mdp, current, candidate, estimated_advantage, alpha)
        collector.equality(
            f"random_{seed}_signed_surrogate",
            "random_mdp",
            cpi["estimated_surrogate"] - cpi["true_surrogate"],
            cpi["xi_alpha"] / (1.0 - mdp.gamma),
            metadata={"alpha": alpha, "seed": seed},
        )
        collector.add(
            f"random_{seed}_cpi_span",
            "random_mdp",
            cpi["master_lower_bound"],
            cpi["eta_mixture"],
            metadata={
                "alpha": alpha,
                "orientation": "lower_bound <= actual_return",
                "seed": seed,
            },
            relation="lower bound <= actual return",
        )


def _finite_reference_plot_data() -> tuple[list[str], list[float], list[float]]:
    mdp, current, _ = _base_mdp(gamma=0.79)
    value = solve_value(mdp, current)
    direction = np.array([0.40, -0.25, 0.18], dtype=np.float64)
    labels: list[str] = []
    errors: list[float] = []
    bounds: list[float] = []
    for k in (1, 2, 4):
        for n, m in ((0, 2), (1, 3), (2, 5), (4, 7)):
            endpoint_n = np.asarray(value + (0.61**n) * direction, dtype=np.float64)
            endpoint_m = np.asarray(value + (0.61**m) * direction, dtype=np.float64)
            quantities = finite_reference_quantities(
                mdp, current, endpoint_n, endpoint_m, k
            )
            labels.append(f"n{n}/m{m}/K{k}")
            errors.append(quantities["actual_error"])
            bounds.append(quantities["bound"])
    return labels, errors, bounds


def build_suite(
    tolerance: float = FLOAT_TOLERANCE,
) -> tuple[dict[str, JsonValue], PlotData]:
    """Evaluate every preregistered finite-MDP case."""

    collector = CaseCollector(tolerance)
    policy_alphas, policy_actual, policy_bounds = _record_base_cases(collector)
    _record_safe_step_branches(collector)
    _record_finite_horizon_cases(collector)
    mismatch_deltas, mismatch_actual, mismatch_bounds = _record_first_mismatch_cases(
        collector
    )
    _record_parameter_mixture_case(collector)
    _record_persistent_augmented_cases(collector)
    _record_random_cases(collector)
    finite_labels, finite_errors, finite_bounds = _finite_reference_plot_data()
    plot_data = PlotData(
        policy_alphas=policy_alphas,
        policy_actual_improvements=policy_actual,
        policy_cpi_lower_bounds=policy_bounds,
        finite_reference_labels=finite_labels,
        finite_reference_errors=finite_errors,
        finite_reference_bounds=finite_bounds,
        mismatch_deltas=mismatch_deltas,
        mismatch_actual=mismatch_actual,
        mismatch_bounds=mismatch_bounds,
    )
    maximum_violation = max(case.numerical_violation for case in collector.cases)
    minimum_slack = min(case.slack for case in collector.cases)
    failed_case_ids = [case.case_id for case in collector.cases if not case.passed]
    report: dict[str, JsonValue] = {
        "cases": [case.as_json() for case in collector.cases],
        "float_dtype": FLOAT_DTYPE,
        "interpretation": (
            "Deterministic finite examples illustrating constants and boundary "
            "cases. These calculations are not a proof, learned-task evidence, "
            "or a uniform theorem certificate."
        ),
        "plot_data": plot_data.as_json(),
        "preregistered_random_seeds": list(RANDOM_SEEDS),
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "summary": {
            "case_count": len(collector.cases),
            "failed_case_ids": failed_case_ids,
            "maximum_numerical_violation": maximum_violation,
            "minimum_raw_slack": minimum_slack,
            "passed": not failed_case_ids,
        },
        "tolerance": tolerance,
    }
    return report, plot_data


def canonical_json_bytes(value: dict[str, JsonValue]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _table_bytes(report: dict[str, JsonValue]) -> bytes:
    cases = report["cases"]
    if not isinstance(cases, list):
        raise TypeError("report cases must be a list")
    lines = ["case_id\tcategory\tobserved\tbound\tslack\tpassed"]
    for item in cases:
        if not isinstance(item, dict):
            raise TypeError("report case must be an object")
        lines.append(
            "\t".join(
                (
                    str(item["case_id"]),
                    str(item["category"]),
                    f"{float(item['observed']):.17g}",
                    f"{float(item['bound']):.17g}",
                    f"{float(item['slack']):.17g}",
                    "true" if item["passed"] else "false",
                )
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_plots(output_dir: Path, data: PlotData) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.size": 9,
            "figure.dpi": 140,
            "savefig.dpi": 140,
        }
    )

    figure, axis = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
    axis.plot(data.policy_alphas, data.policy_actual_improvements, label="actual")
    axis.plot(
        data.policy_alphas,
        data.policy_cpi_lower_bounds,
        linestyle="--",
        label="CPI lower bound",
    )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("exact probability-mixture alpha")
    axis.set_ylabel("return difference")
    axis.set_title("Finite-MDP policy improvement and CPI bound")
    axis.legend()
    figure.savefig(output_dir / POLICY_PLOT_FILENAME)
    plt.close(figure)

    positions = np.arange(len(data.finite_reference_labels))
    figure, axis = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    axis.plot(positions, data.finite_reference_errors, marker="o", label="error")
    axis.plot(positions, data.finite_reference_bounds, marker="x", label="bound")
    axis.set_xticks(positions)
    axis.set_xticklabels(data.finite_reference_labels, rotation=55, ha="right")
    axis.set_ylabel("sup-norm value")
    axis.set_title("Finite-reference error and bound")
    axis.legend()
    figure.savefig(output_dir / FINITE_REFERENCE_PLOT_FILENAME)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
    axis.plot(data.mismatch_deltas, data.mismatch_actual, label="actual")
    axis.plot(
        data.mismatch_deltas,
        data.mismatch_bounds,
        linestyle="--",
        label="first-mismatch bound",
    )
    axis.set_xlabel("deployment TV radius")
    axis.set_ylabel("absolute return difference")
    axis.set_title("Tight two-state first-mismatch example")
    axis.legend()
    figure.savefig(output_dir / FIRST_MISMATCH_PLOT_FILENAME)
    plt.close(figure)

    for filename in (
        POLICY_PLOT_FILENAME,
        FINITE_REFERENCE_PLOT_FILENAME,
        FIRST_MISMATCH_PLOT_FILENAME,
    ):
        with (output_dir / filename).open("rb") as stream:
            os.fsync(stream.fileno())


def write_suite_outputs(output_dir: Path) -> dict[str, JsonValue]:
    """Write the complete registered suite to a new or empty output directory."""

    resolved = output_dir.resolve(strict=False)
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise FileExistsError("output path must be a nonsymlink directory")
        if any(output_dir.iterdir()):
            raise FileExistsError("output directory must be empty")
    else:
        output_dir.mkdir(parents=True, mode=0o700)
    if output_dir.resolve(strict=True) != resolved:
        raise RuntimeError("output directory identity changed during creation")

    report, plot_data = build_suite()
    summary = report["summary"]
    if not isinstance(summary, dict) or summary["passed"] is not True:
        raise ArithmeticError("one or more registered finite-MDP checks failed")
    _write_bytes(output_dir / RESULTS_FILENAME, canonical_json_bytes(report))
    _write_bytes(output_dir / TABLE_FILENAME, _table_bytes(report))
    _write_plots(output_dir, plot_data)
    actual_outputs = frozenset(path.name for path in output_dir.iterdir())
    if actual_outputs != OUTPUT_FILENAMES:
        raise RuntimeError(
            f"unexpected output set: {sorted(actual_outputs ^ OUTPUT_FILENAMES)}"
        )
    artifacts: dict[str, JsonValue] = {}
    for name in sorted(actual_outputs):
        payload = (output_dir / name).read_bytes()
        artifacts[name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "artifacts": artifacts,
        "output_directory": str(output_dir.resolve(strict=True)),
        "suite_id": SUITE_ID,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic float64 finite-MDP sanity checks. This does not "
            "load a checkpoint or execute a learned-model experiment."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for canonical JSON, table, and plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = write_suite_outputs(args.output_dir)
    print(canonical_json_bytes(manifest).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
