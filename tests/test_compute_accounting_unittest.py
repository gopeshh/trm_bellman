"""Focused tests for shared confirmatory compute accounting."""

import unittest

from utils.compute_accounting import (
    ModelComputeCounters,
    aggregate_model_compute,
    capture_model_compute_state,
    restore_model_compute_state,
)


class _InstrumentedModel:
    def __init__(self, **counters):
        self.counters = ModelComputeCounters(**counters)

    def compute_counter_snapshot(self):
        return self.counters.snapshot()

    def restore_compute_counters(self, counters):
        self.counters.restore(counters)


class TestComputeAccounting(unittest.TestCase):
    def test_aliases_are_aggregated_once_but_distinct_modules_are_counted(self):
        shared = _InstrumentedModel(
            policy_api_calls=2,
            policy_state_evaluations=4,
            action_logits_evaluated=20,
        )
        candidate = _InstrumentedModel(
            policy_api_calls=1,
            policy_state_evaluations=2,
            action_logits_evaluated=10,
        )
        aggregate, role_groups, uninstrumented = aggregate_model_compute(
            {
                "model": shared,
                "policy_model_old": shared,
                "policy_model_candidate": candidate,
            }
        )
        self.assertEqual(aggregate["policy_api_calls"], 3)
        self.assertEqual(aggregate["policy_state_evaluations"], 6)
        self.assertEqual(aggregate["action_logits_evaluated"], 30)
        self.assertEqual(
            role_groups,
            [["model", "policy_model_old"], ["policy_model_candidate"]],
        )
        self.assertEqual(uninstrumented, [])

    def test_unique_role_groups_round_trip_without_alias_double_restore(self):
        original = _InstrumentedModel(
            recurrent_latent_update_calls=3,
            recurrent_latent_state_updates=12,
        )
        candidate = _InstrumentedModel(action_logits_evaluated=17)
        state = capture_model_compute_state(
            {"model": original, "policy_model_old": original, "candidate": candidate}
        )

        restored = _InstrumentedModel()
        restored_candidate = _InstrumentedModel()
        restore_model_compute_state(
            {
                "model": restored,
                "policy_model_old": restored,
                "candidate": restored_candidate,
            },
            state,
        )
        self.assertEqual(
            restored.compute_counter_snapshot(),
            original.compute_counter_snapshot(),
        )
        self.assertEqual(
            restored_candidate.compute_counter_snapshot(),
            candidate.compute_counter_snapshot(),
        )


if __name__ == "__main__":
    unittest.main()
