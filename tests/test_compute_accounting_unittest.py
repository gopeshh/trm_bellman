"""Focused tests for shared confirmatory compute accounting."""

import copy
import unittest

from utils.compute_accounting import (
    COMPUTE_SNAPSHOT_SCHEMA_VERSION,
    MODEL_COUNTER_DEFINITIONS,
    OPTIMIZER_STEP_FIELDS,
    ModelComputeCounters,
    aggregate_model_compute,
    capture_model_compute_state,
    restore_model_compute_state,
    validate_compute_snapshot,
    zero_model_counters,
)


class _InstrumentedModel:
    def __init__(self, **counters):
        self.counters = ModelComputeCounters(**counters)

    def compute_counter_snapshot(self):
        return self.counters.snapshot()

    def restore_compute_counters(self, counters):
        self.counters.restore(counters)


class TestComputeAccounting(unittest.TestCase):
    @staticmethod
    def _snapshot():
        zero = zero_model_counters()
        return {
            "compute_schema_version": COMPUTE_SNAPSHOT_SCHEMA_VERSION,
            "model_work": {
                "total": dict(zero),
                "training": dict(zero),
                "evaluation": dict(zero),
                "counter_definitions": dict(MODEL_COUNTER_DEFINITIONS),
                "role_groups": [],
                "uninstrumented_roles": [],
            },
            "progress": {
                "environment_interactions": 0,
                "outer_updates": 0,
                "optimizer_steps_total": 0,
                "optimizer_steps_by_kind": {
                    field: 0 for field in OPTIMIZER_STEP_FIELDS
                },
            },
            "wall_time_seconds": {"training": 0.0, "evaluation": 0.0},
            "peak_memory_bytes": {
                "cuda_allocated": None,
                "cuda_reserved": None,
                "process_rss": 0,
            },
        }

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

    def test_compute_schema_requires_exact_non_bool_integer(self):
        self.assertEqual(
            validate_compute_snapshot(self._snapshot())["compute_schema_version"],
            COMPUTE_SNAPSHOT_SCHEMA_VERSION,
        )
        for invalid in (True, float(COMPUTE_SNAPSHOT_SCHEMA_VERSION), "2"):
            with self.subTest(invalid=invalid):
                snapshot = copy.deepcopy(self._snapshot())
                snapshot["compute_schema_version"] = invalid
                with self.assertRaisesRegex(ValueError, "snapshot schema"):
                    validate_compute_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
