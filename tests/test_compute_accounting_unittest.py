"""Focused tests for shared confirmatory compute accounting."""

import copy
import unittest

from utils.compute_accounting import (
    AUTHENTICATED_COMPUTE_ACCOUNTING_SCHEMA_NAME,
    COMPUTE_SNAPSHOT_SCHEMA_VERSION,
    MODEL_COUNTER_DEFINITIONS,
    OPTIMIZER_STEP_FIELDS,
    ModelComputeCounters,
    aggregate_model_compute,
    build_audit_compute_accounting,
    build_training_compute_accounting,
    capture_model_compute_state,
    execution_device_identity,
    restore_model_compute_state,
    validate_authenticated_compute_accounting,
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

    def test_training_compute_contract_binds_every_registered_cpu_measurement(self):
        snapshot = self._snapshot()
        snapshot["model_work"]["training"].update(
            {
                "recurrent_latent_state_updates": 48,
                "action_logits_evaluated": 96,
                "action_values_evaluated": 12,
                "value_api_calls": 7,
            }
        )
        snapshot["model_work"]["total"] = dict(snapshot["model_work"]["training"])
        snapshot["progress"].update(
            {
                "environment_interactions": 32,
                "outer_updates": 2,
                "optimizer_steps_total": 4,
            }
        )
        snapshot["progress"]["optimizer_steps_by_kind"]["value"] = 2
        snapshot["progress"]["optimizer_steps_by_kind"]["policy"] = 2
        snapshot["wall_time_seconds"] = {"training": 1.25, "evaluation": 0.5}
        snapshot["peak_memory_bytes"]["process_rss"] = 4096

        accounting = build_training_compute_accounting(
            snapshot,
            device="cpu",
            checkpoint_seconds=0.25,
            cuda_utilization={
                "device_type": "cpu",
                "sampling_interval_seconds": None,
                "samples": [],
            },
        )

        self.assertEqual(
            accounting["schema_name"],
            AUTHENTICATED_COMPUTE_ACCOUNTING_SCHEMA_NAME,
        )
        self.assertEqual(accounting["environment_interactions"]["value"], 32)
        self.assertEqual(accounting["recurrent_latent_state_updates"]["value"], 48)
        self.assertEqual(accounting["action_logit_evaluations"]["value"], 96)
        self.assertEqual(accounting["action_value_evaluations"]["value"], 12)
        self.assertEqual(accounting["value_head_calls"]["value"], 7)
        self.assertEqual(accounting["checkpoint_seconds"]["value"], 0.25)
        self.assertEqual(
            accounting["audit_seconds"],
            {
                "status": "unavailable",
                "reason": "owned_by_authenticated_audit_role",
            },
        )
        self.assertEqual(
            accounting["cuda_utilization_samples"]["status"], "unavailable"
        )
        self.assertEqual(
            accounting["execution_device_identity"]["value"]["canonical"],
            "cpu",
        )

    def test_source_snapshot_substitution_is_rejected(self):
        snapshot = self._snapshot()
        accounting = build_training_compute_accounting(
            snapshot,
            device="cpu",
            checkpoint_seconds=0.0,
            cuda_utilization={
                "device_type": "cpu",
                "sampling_interval_seconds": None,
                "samples": [],
            },
        )
        substituted = copy.deepcopy(snapshot)
        substituted["progress"]["environment_interactions"] = 1
        with self.assertRaisesRegex(ValueError, "source snapshot digest differs"):
            validate_authenticated_compute_accounting(
                accounting,
                source_compute_snapshot=substituted,
            )

    def test_training_role_cannot_claim_audit_time(self):
        snapshot = self._snapshot()
        accounting = build_training_compute_accounting(
            snapshot,
            device="cpu",
            checkpoint_seconds=0.0,
            cuda_utilization={
                "device_type": "cpu",
                "sampling_interval_seconds": None,
                "samples": [],
            },
        )
        accounting["audit_seconds"] = {"status": "available", "value": 1.0}
        with self.assertRaisesRegex(ValueError, "cannot claim audit_seconds"):
            validate_authenticated_compute_accounting(
                accounting,
                source_compute_snapshot=snapshot,
            )

    def test_audit_contract_owns_only_audit_time_and_process_memory(self):
        accounting = build_audit_compute_accounting(
            audit_seconds=2.5,
            process_rss_bytes=8192,
        )
        self.assertEqual(accounting["owner_role"], "audit")
        self.assertEqual(accounting["audit_seconds"]["value"], 2.5)
        self.assertEqual(accounting["process_peak_rss_bytes"]["value"], 8192)
        self.assertEqual(
            accounting["wall_clock_training_seconds"],
            {
                "status": "unavailable",
                "reason": "not_owned_by_authenticated_audit_role",
            },
        )

    def test_cpu_execution_identity_is_canonical(self):
        identity = execution_device_identity("cpu")
        self.assertEqual(identity["canonical"], "cpu")
        self.assertEqual(identity["device_type"], "cpu")
        self.assertIsNone(identity["device_index"])


if __name__ == "__main__":
    unittest.main()
