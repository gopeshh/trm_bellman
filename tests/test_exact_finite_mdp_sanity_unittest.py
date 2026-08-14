#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.exact_finite_mdp_sanity import (
    FINITE_REFERENCE_PLOT_FILENAME,
    FIRST_MISMATCH_PLOT_FILENAME,
    FLOAT_DTYPE,
    FLOAT_TOLERANCE,
    OUTPUT_FILENAMES,
    POLICY_PLOT_FILENAME,
    RANDOM_SEEDS,
    RESULTS_FILENAME,
    SUITE_ID,
    TABLE_FILENAME,
    FiniteMDP,
    _base_mdp,
    advantages,
    build_suite,
    canonical_json_bytes,
    cpi_quantities,
    deployment_bound,
    discounted_occupancy,
    exact_policy_mixture,
    expected_return,
    finite_horizon_deployment_bound,
    finite_horizon_cpi_quantities,
    finite_horizon_reference_quantities,
    finite_horizon_values,
    finite_reference_quantities,
    first_mismatch_example,
    parameter_and_probability_mixtures,
    policy_reward,
    policy_transition,
    safe_step_decision,
    scalar_cpi_lower_bound,
    solve_value,
    target_network_bridge,
    total_variation,
    vector_span,
    write_suite_outputs,
)


class ExactFiniteMdpSanityTest(unittest.TestCase):
    def test_registered_suite_passes_and_covers_required_cases(self) -> None:
        report, plot_data = build_suite()
        self.assertEqual(report["suite_id"], SUITE_ID)
        self.assertEqual(report["float_dtype"], FLOAT_DTYPE)
        self.assertEqual(report["tolerance"], FLOAT_TOLERANCE)
        self.assertEqual(report["preregistered_random_seeds"], list(RANDOM_SEEDS))
        summary = report["summary"]
        self.assertIsInstance(summary, dict)
        assert isinstance(summary, dict)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["failed_case_ids"], [])
        self.assertGreaterEqual(int(summary["case_count"]), 150)

        cases = report["cases"]
        self.assertIsInstance(cases, list)
        assert isinstance(cases, list)
        case_ids = {str(case["case_id"]) for case in cases if isinstance(case, dict)}
        required = {
            "gamma_zero_value_equals_immediate_reward",
            "finite_reference_residual_bound",
            "target_propagated_lag_bridge",
            "target_sup_lag_fallback",
            "exact_mixture_occupancy_alpha_00",
            "exact_mixture_occupancy_alpha_20",
            "span_times_tv_no_extra_factor_two",
            "signed_surrogate_identity_alpha_07",
            "cpi_span_lower_bound_alpha_07",
            "cpi_scalar_lower_bound_alpha_07",
            "safe_step_equal_gamma_span_branch",
            "safe_step_interior_threshold",
            "finite_horizon_K3_h0",
            "finite_horizon_K3_h2",
            "finite_horizon_K3_h5",
            "finite_horizon_K7_h5",
            "finite_horizon_H0_deployment",
            "finite_horizon_cpi_alpha_1",
            "finite_horizon_cpi_H0",
            "first_mismatch_infinite_delta_00",
            "first_mismatch_infinite_delta_20",
            "parameter_interpolation_is_not_probability_mixture",
            "persistent_shared_map_affine_transition",
            "persistent_augmented_bellman_fixed_point",
        }
        self.assertEqual(required - case_ids, set())
        self.assertEqual(len(plot_data.policy_alphas), 21)
        self.assertEqual(len(plot_data.finite_reference_labels), 12)
        self.assertEqual(len(plot_data.mismatch_deltas), 21)

    def test_float64_linear_solve_and_occupancy(self) -> None:
        mdp, current, _ = _base_mdp()
        value, advantage = advantages(mdp, current)
        occupancy = discounted_occupancy(mdp, current)
        self.assertEqual(value.dtype, np.float64)
        self.assertEqual(advantage.dtype, np.float64)
        self.assertEqual(occupancy.dtype, np.float64)
        np.testing.assert_allclose(
            value,
            policy_reward(mdp, current)
            + mdp.gamma * policy_transition(mdp, current) @ value,
            rtol=0.0,
            atol=FLOAT_TOLERANCE,
        )
        np.testing.assert_allclose(
            np.sum(current * advantage, axis=1),
            0.0,
            rtol=0.0,
            atol=FLOAT_TOLERANCE,
        )
        self.assertAlmostEqual(float(np.sum(occupancy)), 1.0, delta=FLOAT_TOLERANCE)

    @staticmethod
    def _one_state_mdp(*, rewards: tuple[float, ...], gamma: float) -> FiniteMDP:
        action_count = len(rewards)
        return FiniteMDP(
            transitions=np.ones((1, action_count, 1), dtype=np.float64),
            rewards=np.asarray([rewards], dtype=np.float64),
            initial=np.ones(1, dtype=np.float64),
            gamma=gamma,
        )

    def test_hand_oracle_finite_reference_denominator(self) -> None:
        mdp = self._one_state_mdp(rewards=(0.0,), gamma=0.5)
        policy = np.ones((1, 1), dtype=np.float64)
        quantities = finite_reference_quantities(
            mdp,
            policy,
            np.array([5.0], dtype=np.float64),
            np.array([4.0], dtype=np.float64),
            2,
        )
        self.assertEqual(quantities["endpoint_discrepancy"], 1.0)
        self.assertEqual(quantities["reference_residual"], 3.0)
        self.assertEqual(quantities["actual_error"], 5.0)
        self.assertEqual(quantities["bound"], 5.0)

    def test_hand_oracle_target_network_propagated_and_sup_lag(self) -> None:
        mdp = FiniteMDP(
            transitions=np.full((2, 1, 2), 0.5, dtype=np.float64),
            rewards=np.zeros((2, 1), dtype=np.float64),
            initial=np.array([1.0, 0.0], dtype=np.float64),
            gamma=0.5,
        )
        bridge = target_network_bridge(
            mdp,
            np.ones((2, 1), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.array([2.0, 0.0], dtype=np.float64),
            1,
        )
        self.assertEqual(bridge["target_population_residual"], 0.5)
        self.assertEqual(bridge["propagated_lag"], 0.0)
        self.assertEqual(bridge["self_bootstrap_residual"], 0.5)
        self.assertEqual(bridge["propagated_bound"], 0.5)
        self.assertEqual(bridge["sup_lag"], 0.5)
        self.assertEqual(bridge["sup_bound"], 1.0)

    def test_hand_oracle_signed_surrogate_and_zero_span_cpi(self) -> None:
        mdp = self._one_state_mdp(rewards=(0.0, 2.0), gamma=0.5)
        current = np.array([[1.0, 0.0]], dtype=np.float64)
        candidate = np.array([[0.0, 1.0]], dtype=np.float64)
        quantities = cpi_quantities(
            mdp,
            current,
            candidate,
            np.array([[0.0, 3.0]], dtype=np.float64),
            0.25,
        )
        self.assertEqual(quantities["xi_alpha"], 0.25)
        self.assertEqual(quantities["true_surrogate"], 1.0)
        self.assertEqual(quantities["estimated_surrogate"], 1.5)
        self.assertEqual(
            quantities["estimated_surrogate"] - quantities["true_surrogate"],
            quantities["xi_alpha"] / (1.0 - mdp.gamma),
        )
        self.assertEqual(quantities["delta_g"], 0.0)
        self.assertEqual(quantities["occupancy_penalty"], 0.0)
        self.assertEqual(quantities["eta_mixture"], 1.0)
        self.assertEqual(quantities["master_lower_bound"], 1.0)

        self.assertEqual(
            scalar_cpi_lower_bound(
                quantities["estimated_surrogate"],
                mdp.gamma,
                0.25,
                epsilon_candidate=1.0,
                epsilon_cpi=2.0,
            ),
            0.6,
        )

    def test_hand_oracle_nonzero_span_cpi_penalty(self) -> None:
        mdp = FiniteMDP(
            transitions=np.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ],
                dtype=np.float64,
            ),
            rewards=np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
            initial=np.array([1.0, 0.0], dtype=np.float64),
            gamma=0.5,
        )
        current = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        candidate = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        _, exact_advantage = advantages(mdp, current)
        quantities = cpi_quantities(
            mdp,
            current,
            candidate,
            exact_advantage,
            0.25,
        )
        self.assertEqual(quantities["delta_g"], 1.0)
        self.assertEqual(quantities["true_surrogate"], 0.5)
        self.assertEqual(quantities["occupancy_penalty"], 0.1)
        self.assertEqual(quantities["eta_mixture"], 0.4)
        self.assertEqual(quantities["master_lower_bound"], 0.4)

        finite = finite_horizon_cpi_quantities(
            mdp,
            current,
            candidate,
            horizon=2,
            alpha=0.25,
            terminal_boundary=0.0,
        )
        self.assertEqual(finite["estimated_surrogate"], 3.0 / 8.0)
        self.assertEqual(finite["occupancy_penalty"], 1.0 / 32.0)
        self.assertEqual(finite["actual_return"], 11.0 / 32.0)
        self.assertEqual(finite["lower_bound"], 11.0 / 32.0)

    def test_hand_oracle_span_times_tv_has_no_extra_factor_two(self) -> None:
        first = np.array([1.0, 0.0], dtype=np.float64)
        second = np.array([0.0, 1.0], dtype=np.float64)
        values = np.array([0.0, 2.0], dtype=np.float64)
        expectation_gap = abs(float(first @ values - second @ values))
        self.assertEqual(total_variation(first, second), 1.0)
        self.assertEqual(vector_span(values), 2.0)
        self.assertEqual(
            expectation_gap,
            total_variation(first, second) * vector_span(values),
        )

    def test_hand_oracle_multiblock_finite_reference(self) -> None:
        mdp = self._one_state_mdp(rewards=(0.0,), gamma=0.5)
        policy = np.ones((1, 1), dtype=np.float64)
        endpoints = [np.zeros(1, dtype=np.float64)] + [
            np.ones(1, dtype=np.float64) for _ in range(5)
        ]
        row = finite_horizon_reference_quantities(
            mdp,
            policy,
            endpoints,
            endpoints,
            horizon=5,
            k=3,
            terminal_boundary=0.0,
        )[5]
        self.assertEqual(row["actual_error"], 1.0)
        self.assertEqual(row["residual_sum"], 1.0)
        self.assertEqual(row["bound"], 1.0)
        self.assertEqual(row["j_h"], 2)
        self.assertEqual(row["last_block"], 2)

    def test_hand_oracle_finite_horizon_cpi(self) -> None:
        mdp = self._one_state_mdp(rewards=(0.0, 2.0), gamma=0.5)
        quantities = finite_horizon_cpi_quantities(
            mdp,
            np.array([[1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0]], dtype=np.float64),
            horizon=3,
            alpha=0.25,
            terminal_boundary=0.0,
        )
        self.assertEqual(quantities["actual_return"], 0.875)
        self.assertEqual(quantities["estimated_surrogate"], 0.875)
        self.assertEqual(quantities["occupancy_penalty"], 0.0)
        self.assertEqual(quantities["lower_bound"], 0.875)

    def test_exact_mixture_occupancy_boundary_values(self) -> None:
        mdp, current, candidate = _base_mdp()
        current_occupancy = discounted_occupancy(mdp, current)
        for alpha in (0.0, 0.37, 1.0):
            mixture = exact_policy_mixture(current, candidate, alpha)
            mixture_occupancy = discounted_occupancy(mdp, mixture)
            observed = total_variation(mixture_occupancy, current_occupancy)
            bound = mdp.gamma * alpha / (1.0 - mdp.gamma + mdp.gamma * alpha)
            self.assertLessEqual(observed, bound + FLOAT_TOLERANCE)
        np.testing.assert_array_equal(
            exact_policy_mixture(current, candidate, 0.0), current
        )
        np.testing.assert_array_equal(
            exact_policy_mixture(current, candidate, 1.0), candidate
        )

    def test_every_safe_step_branch(self) -> None:
        registered = (
            (0.0, -0.1, 0.8, "negative_slope_no_positive_step", 0.0),
            (0.0, 0.0, 0.8, "gamma_zero_all_steps", 1.0),
            (0.8, -0.1, 0.8, "negative_slope_no_positive_step", 0.0),
            (
                0.8,
                0.0,
                0.8,
                "zero_slope_positive_span_no_positive_step",
                0.0,
            ),
            (0.8, 0.0, 0.0, "zero_slope_zero_span_all_steps", 1.0),
            (0.8, 0.2, 0.0, "slope_at_least_gamma_span_all_steps", 1.0),
            (0.8, 0.8 * 0.8, 0.8, "slope_at_least_gamma_span_all_steps", 1.0),
            (0.8, 0.2, 0.8, "interior_threshold", 1.0 / 12.0),
        )
        for gamma, slope, delta_g, expected_branch, expected_alpha in registered:
            with self.subTest(expected_branch=expected_branch):
                decision = safe_step_decision(gamma, slope, delta_g)
                self.assertEqual(decision.branch, expected_branch)
                self.assertAlmostEqual(
                    decision.maximum_alpha, expected_alpha, delta=FLOAT_TOLERANCE
                )

    def test_first_mismatch_attains_infinite_and_finite_bounds(self) -> None:
        for delta_dep in (0.0, 0.2, 1.0):
            with self.subTest(delta_dep=delta_dep):
                mdp, reference, deployed = first_mismatch_example(
                    gamma=0.83,
                    delta_dep=delta_dep,
                    reward_span=1.7,
                )
                actual = abs(
                    expected_return(mdp, deployed) - expected_return(mdp, reference)
                )
                self.assertAlmostEqual(
                    actual,
                    deployment_bound(0.83, delta_dep, 1.7),
                    delta=FLOAT_TOLERANCE,
                )
                horizon = 6
                reference_value = finite_horizon_values(mdp, reference, horizon, 0.0)[
                    horizon
                ]
                deployed_value = finite_horizon_values(mdp, deployed, horizon, 0.0)[
                    horizon
                ]
                finite_actual = abs(
                    float(mdp.initial @ (deployed_value - reference_value))
                )
                self.assertAlmostEqual(
                    finite_actual,
                    finite_horizon_deployment_bound(0.83, delta_dep, 1.7, horizon),
                    delta=FLOAT_TOLERANCE,
                )

    def test_parameter_interpolation_is_not_probability_mixture(self) -> None:
        probability, parameter = parameter_and_probability_mixtures(
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([2.0, -1.0, 0.5], dtype=np.float64),
            0.5,
        )
        self.assertGreater(total_variation(probability, parameter), 1.0e-6)
        self.assertAlmostEqual(float(np.sum(probability)), 1.0)
        self.assertAlmostEqual(float(np.sum(parameter)), 1.0)

    def test_output_writer_emits_canonical_complete_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            output_dir = Path(temporary_root) / "suite"
            manifest = write_suite_outputs(output_dir)
            self.assertEqual(manifest["suite_id"], SUITE_ID)
            self.assertEqual(
                frozenset(path.name for path in output_dir.iterdir()),
                OUTPUT_FILENAMES,
            )
            results_path = output_dir / RESULTS_FILENAME
            report = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(results_path.read_bytes(), canonical_json_bytes(report))
            self.assertTrue(
                (output_dir / TABLE_FILENAME).read_text().startswith("case_id\t")
            )
            for plot_name in (
                POLICY_PLOT_FILENAME,
                FINITE_REFERENCE_PLOT_FILENAME,
                FIRST_MISMATCH_PLOT_FILENAME,
            ):
                self.assertTrue(
                    (output_dir / plot_name).read_bytes().startswith(b"\x89PNG")
                )

    def test_output_writer_rejects_nonempty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            output_dir = Path(temporary_root) / "suite"
            output_dir.mkdir()
            (output_dir / "unrelated.txt").write_text("do not overwrite")
            with self.assertRaisesRegex(FileExistsError, "must be empty"):
                write_suite_outputs(output_dir)


if __name__ == "__main__":
    unittest.main()
