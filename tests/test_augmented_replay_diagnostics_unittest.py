import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rl.replay import (
    ReplayBuffer,
    ReplayIntegrityError,
    ReplayLatent,
    Transition,
    validate_transition_continuity,
    validate_transition_sequence,
)
from rl.theory_diagnostics import (
    FINITE_BATCH_SCOPE,
    summarize_carry_continuity,
    summarize_centering_defect,
    summarize_depth_path,
    summarize_policy_gap,
    summarize_residual_estimates,
)
from utils.lipschitz import compute_exact_baseline_summation


def _state(clock: int) -> dict:
    return {
        "inputs": torch.tensor([2, 1]),
        "puzzle_identifiers": torch.tensor(0),
        "remaining_edits": torch.tensor(clock),
    }


def _latent(value: float) -> ReplayLatent:
    tensor = torch.tensor([[[value]]])
    return ReplayLatent(z_H=tensor, z_L=-tensor)


def _transition(
    timestep: int,
    *,
    done: bool = False,
    episode_id: int = 7,
) -> Transition:
    clock = 3 - timestep
    return Transition(
        x=_state(clock),
        y=torch.tensor([timestep, 0]),
        action=torch.tensor(0),
        reward=torch.tensor([1.0]),
        x_next=_state(clock - 1),
        y_next=torch.tensor([timestep + 1, 0]),
        done=torch.tensor([done]),
        episode_id=episode_id,
        timestep=timestep,
        latent=_latent(float(timestep)),
        next_latent=_latent(float(timestep + 1)),
    )


class TestAugmentedReplayIntegrity(unittest.TestCase):
    def test_complete_persistent_segment_retains_clock_and_carry(self) -> None:
        transitions = [_transition(0), _transition(1), _transition(2, done=True)]
        validate_transition_sequence(transitions, require_clock=True)

        replay = ReplayBuffer(capacity=8)
        for transition in transitions:
            replay.add(transition)

        segment = replay.contiguous_segment(
            0,
            5,
            require_complete=True,
            require_clock=True,
        )
        self.assertEqual(len(segment), 3)
        self.assertTrue(replay.has_complete_segment(0, 5, require_clock=True))
        self.assertEqual(segment[0].x["remaining_edits"].item(), 3)
        self.assertEqual(segment[1].x["remaining_edits"].item(), 2)
        first_next_latent = segment[0].next_latent
        second_input_latent = segment[1].latent
        self.assertIsNotNone(first_next_latent)
        self.assertIsNotNone(second_input_latent)
        assert first_next_latent is not None
        assert second_input_latent is not None
        torch.testing.assert_close(
            first_next_latent.z_H,
            second_input_latent.z_H,
        )

    def test_nonterminal_short_segment_is_not_complete(self) -> None:
        replay = ReplayBuffer(capacity=8)
        replay.add(_transition(0))
        replay.add(_transition(1))
        self.assertFalse(replay.has_complete_segment(0, 3, require_clock=True))
        with self.assertRaisesRegex(ReplayIntegrityError, "fixed horizon"):
            replay.contiguous_segment(
                0,
                3,
                require_complete=True,
                require_clock=True,
            )

    def test_timestep_state_clock_and_latent_gaps_fail_closed(self) -> None:
        current = _transition(0)

        bad_timestep = _transition(1)
        bad_timestep.timestep = 2
        with self.assertRaisesRegex(ReplayIntegrityError, "timesteps"):
            validate_transition_continuity(current, bad_timestep, require_clock=True)

        bad_state = _transition(1)
        bad_state.x["inputs"] = torch.tensor([9, 9])
        with self.assertRaisesRegex(ReplayIntegrityError, "x_next"):
            validate_transition_continuity(current, bad_state, require_clock=True)

        bad_clock = _transition(1)
        bad_clock.x_next["remaining_edits"] = torch.tensor(0)
        with self.assertRaisesRegex(ReplayIntegrityError, "decrement"):
            validate_transition_continuity(current, bad_clock, require_clock=True)

        bad_latent = _transition(1)
        bad_latent.latent = _latent(99.0)
        with self.assertRaisesRegex(ReplayIntegrityError, "next_latent"):
            validate_transition_continuity(current, bad_latent, require_clock=True)

    def test_terminal_record_cannot_have_a_successor(self) -> None:
        terminal = _transition(0, done=True)
        successor = _transition(1)
        with self.assertRaisesRegex(ReplayIntegrityError, "terminal"):
            validate_transition_continuity(terminal, successor, require_clock=True)

        replay = ReplayBuffer(capacity=4)
        replay.add(terminal)
        replay.add(successor)
        self.assertFalse(replay.has_complete_segment(1, 1, require_clock=True))


class _PersistentValueModel:
    def __init__(self, expected_latent: object) -> None:
        self.expected_latent = expected_latent
        self.seen_latents = []

    def used_value(self, x, y, n, z=None):
        self.seen_latents.append(z)
        value = 7.0 if z is self.expected_latent else -11.0
        return torch.full((y.shape[0],), value), z

    def policy_dist(self, x, y, n, action_mask=None):
        probs = action_mask.to(torch.float32)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.distributions.Categorical(probs=probs), None


class TestPersistentExactBaseline(unittest.TestCase):
    def test_every_successor_value_uses_post_unroll_latent(self) -> None:
        successor_latent = object()
        model = _PersistentValueModel(successor_latent)
        env = SimpleNamespace(
            config=SimpleNamespace(
                reward_shaping=True,
                solved_threshold=None,
                task_type="dummy",
            ),
            vocab_size=2,
            stop_action_id=2,
            _enable_undo=False,
            _stop_mode="disabled",
            is_stop_terminal=lambda: False,
            is_plan_solved=None,
        )
        x_batch = {
            "inputs": torch.zeros(2, 1, dtype=torch.long),
            "puzzle_identifiers": torch.arange(2),
            "remaining_edits": torch.full((2,), 2, dtype=torch.long),
        }
        y_batch = torch.zeros(2, 1, dtype=torch.long)
        action_mask = torch.tensor([[True, True, False], [True, True, False]])
        policy_probs = torch.tensor([[0.25, 0.75, 0.0], [0.5, 0.5, 0.0]])
        zeros = torch.zeros(2)

        with patch(
            "utils.lipschitz._apply_edit_batch",
            return_value=y_batch,
        ), patch(
            "utils.lipschitz._compute_phi_batch",
            return_value=(zeros, zeros, torch.zeros(2, dtype=torch.bool)),
        ), patch(
            "utils.lipschitz._compute_batch_reward_via_env",
            return_value=torch.ones(2),
        ):
            baseline, q_all = compute_exact_baseline_summation(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                env=env,
                n=3,
                gamma=0.5,
                checker_fn=lambda x, y: 0.0,
                action_mask=action_mask,
                policy_probs=policy_probs,
                successor_latent=successor_latent,
            )

        self.assertEqual(model.seen_latents, [successor_latent, successor_latent])
        torch.testing.assert_close(q_all[:, :2], torch.full((2, 2), 4.5))
        torch.testing.assert_close(baseline, torch.full((2,), 4.5))


class TestFiniteBatchTheoryDiagnostics(unittest.TestCase):
    def assertFiniteBatchScope(self, result: dict) -> None:
        self.assertEqual(result["scope"], FINITE_BATCH_SCOPE)
        self.assertFalse(result["uniform_certificate"])

    def test_centering_masks_invalid_infinite_entries(self) -> None:
        probs = torch.tensor([[0.25, 0.75, 0.0]])
        advantages = torch.tensor([[3.0, -1.0, -math.inf]])
        mask = torch.tensor([[True, True, False]])
        result = summarize_centering_defect(probs, advantages, action_mask=mask)
        self.assertFiniteBatchScope(result)
        self.assertEqual(
            result["absolute_expected_advantage"]["maximum"],
            0.0,
        )

    def test_policy_gap_preserves_infinite_kl_and_support_mismatch(self) -> None:
        # The third action has p=q=0 and must contribute exactly zero, not NaN.
        reference = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        comparison = torch.tensor([[0.0, 1.0, 0.0], [0.25, 0.75, 0.0]])
        result = summarize_policy_gap(reference, comparison)
        self.assertFiniteBatchScope(result)
        self.assertEqual(
            result["kl_reference_to_comparison"]["positive_infinite_count"],
            1,
        )
        self.assertEqual(
            result["kl_comparison_to_reference"]["positive_infinite_count"],
            1,
        )
        self.assertEqual(result["support"]["either_direction_state_count"], 1)
        self.assertEqual(result["total_variation"]["maximum"], 1.0)

    def test_policy_gap_renormalizes_rows_after_masking_tolerated_mass(self) -> None:
        reference = torch.tensor(
            [[0.5999997, 0.3999998, 0.0000005]], dtype=torch.float64
        )
        comparison = torch.tensor(
            [[0.6000003, 0.4000002, 0.0]], dtype=torch.float64
        )
        mask = torch.tensor([[True, True, False]])

        result = summarize_policy_gap(
            reference,
            comparison,
            action_mask=mask,
            probability_tolerance=1e-6,
        )

        self.assertLess(result["total_variation"]["maximum"], 1e-14)
        self.assertLess(
            result["kl_reference_to_comparison"]["maximum"], 1e-14
        )
        self.assertLess(
            result["kl_comparison_to_reference"]["maximum"], 1e-14
        )

    def test_depth_path_reports_undefined_zero_increment_ratios(self) -> None:
        latent_h = torch.tensor(
            [
                [[[[0.0]]], [[[1.0]]], [[[3.0]]], [[[6.0]]]],
                [[[[0.0]]], [[[0.0]]], [[[0.0]]], [[[0.0]]]],
            ]
        ).reshape(2, 4, 1, 1)
        latent_l = torch.zeros_like(latent_h)
        values = torch.tensor([[0.0, 1.0, 3.0, 6.0], [0.0, 0.0, 0.0, 0.0]])
        result = summarize_depth_path(
            latent_h,
            latent_l,
            values,
            n_depth=1,
            m_depth=3,
        )
        self.assertFiniteBatchScope(result)
        self.assertEqual(result["path_length_n_to_m"]["maximum"], 5.0)
        self.assertEqual(
            result["absolute_value_discrepancy_n_to_m"]["maximum"],
            5.0,
        )
        self.assertEqual(result["successive_increment_ratio"]["count"], 2)
        self.assertEqual(result["successive_increment_ratio"]["undefined_count"], 2)

    def test_carry_and_clock_continuity_are_separate(self) -> None:
        current_h = torch.tensor([[0.0], [1.0]])
        current_l = torch.zeros_like(current_h)
        successor_h = torch.tensor([[1.0], [3.0]])
        successor_l = torch.zeros_like(successor_h)
        result = summarize_carry_continuity(
            current_h,
            current_l,
            successor_h,
            successor_l,
            successor_h.clone(),
            successor_l.clone(),
            torch.tensor([3, 2]),
            torch.tensor([2, 1]),
            torch.tensor([2, 0]),
        )
        self.assertFiniteBatchScope(result)
        self.assertEqual(
            result["successor_to_next_input_carry_error"]["maximum"],
            0.0,
        )
        self.assertEqual(result["clock_decrement_violation_count"], 0)
        self.assertEqual(result["clock_link_violation_count"], 1)

    def test_residual_summary_cannot_be_mistaken_for_certificate(self) -> None:
        result = summarize_residual_estimates(
            torch.tensor([0.1, 0.4]),
            estimator="monte_carlo_k_step",
            horizon=5,
            monte_carlo_standard_errors=torch.tensor([0.01, 0.02]),
        )
        self.assertFiniteBatchScope(result)
        self.assertEqual(result["horizon"], 5)
        self.assertAlmostEqual(result["absolute_residual"]["maximum"], 0.4)


if __name__ == "__main__":
    unittest.main()
