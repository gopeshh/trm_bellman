#!/usr/bin/env fbpython
"""Focused tests for the fresh policy-improvement v2 registration."""

from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

from scripts.policy_improvement_populations import (
    PolicyImprovementPopulationError,
    canonical_json_bytes as population_json_bytes,
    derive_population_score,
    load_registered_populations,
    load_strict_json as load_population_json,
    materialize_v2_populations,
    population_binding_sha256,
    population_for_id,
    validate_v2_populations,
)
from scripts.policy_improvement_registry import (
    load_flat_registered_yaml as load_v1_flat_registered_yaml,
)
from scripts.policy_improvement_schema import (
    PolicyImprovementSchemaError,
    validate_protocol as validate_v1_protocol,
)
from scripts.policy_improvement_v2_registry import (
    EXPECTED_PHASE_COUNTS,
    generate_v2_registry,
    load_v2_base_configs,
    validate_v2_registry_document,
)
from scripts.policy_improvement_v2_schema import (
    IMMUTABLE_DATASET_V1,
    PolicyImprovementV2SchemaError,
    canonical_json_bytes,
    load_strict_json,
    sha256_json,
    validate_v2_protocol,
    validate_v2_result,
)


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs/policy_improvement_v2"
PROTOCOL = V2 / "protocol.json"
POPULATIONS = V2 / "populations.json"
REGISTRY = V2 / "registry.json"
THEORY_AMENDMENT = V2 / "amendments/theory_bridge_v2.json"
V1_PROTOCOL = ROOT / "configs/policy_improvement_v1/protocol.json"


def _digest(namespace: str, index: int) -> str:
    return hashlib.sha256(f"{namespace}:{index}".encode("ascii")).hexdigest()


class PolicyImprovementV2RegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_strict_json(PROTOCOL)
        self.populations = load_population_json(POPULATIONS)
        self.registry = load_strict_json(REGISTRY)
        self.configs = load_v2_base_configs(self.protocol, ROOT)

    def test_canonical_documents_validate_and_regenerate(self) -> None:
        checked_protocol = validate_v2_protocol(self.protocol)
        checked_populations = validate_v2_populations(self.populations)
        checked_registry = validate_v2_registry_document(
            self.registry,
            checked_protocol,
            checked_populations,
            base_configs=self.configs,
        )
        regenerated = generate_v2_registry(
            checked_protocol,
            checked_populations,
            base_configs=self.configs,
        )
        self.assertEqual(
            PROTOCOL.read_bytes(), canonical_json_bytes(checked_protocol) + b"\n"
        )
        self.assertEqual(
            POPULATIONS.read_bytes(),
            population_json_bytes(checked_populations) + b"\n",
        )
        self.assertEqual(
            REGISTRY.read_bytes(), canonical_json_bytes(regenerated) + b"\n"
        )
        self.assertEqual(checked_registry, regenerated)

    def test_v1_and_v2_protocol_schemas_are_not_interchangeable(self) -> None:
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_v1_protocol(self.protocol)
        with self.assertRaises(PolicyImprovementV2SchemaError):
            validate_v2_protocol(load_strict_json(V1_PROTOCOL))

    def test_v2_reuses_the_exact_immutable_v1_dataset_registration(self) -> None:
        v1 = load_strict_json(V1_PROTOCOL)
        self.assertEqual(self.protocol["dataset"], v1["dataset"])
        self.assertEqual(self.protocol["dataset"], IMMUTABLE_DATASET_V1)
        self.assertEqual(
            self.populations["dataset_name"], self.protocol["dataset"]["name"]
        )
        for split in ("train", "validation"):
            self.assertEqual(
                self.populations["split_ordered_record_sha256"][split],
                self.protocol["dataset"]["splits"][split]["ordered_record_sha256"][
                    "value"
                ],
            )

    def test_stage0_is_the_registered_train_only_population(self) -> None:
        stage0_grid = self.protocol["grid"]["stage0"]
        self.assertEqual(stage0_grid["upi_optimizer_batch_size"], 1)
        self.assertEqual(stage0_grid["upi_rollout_episodes_per_step"], 1)
        self.assertEqual(stage0_grid["ppo_rollout_environment_interactions"], 16)
        stage0 = population_for_id(self.populations, "stage0_smoke")
        self.assertEqual(stage0["split"], "train")
        self.assertEqual(stage0["count"], 8)
        self.assertEqual(stage0["indices"], [749, 910, 352, 318, 877, 605, 148, 280])
        self.assertEqual(
            stage0["ordered_record_sha256"],
            "522a60c5f0b4276c5a66743005ff430800ad0085e020cd4a7403a0f303456033",
        )
        rows = [row for row in self.registry["rows"] if row["phase"] == "stage0_smoke"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {row["method_id"] for row in rows},
            {
                "fixed_base_exact_persistent",
                "fixed_base_exact_episodic",
                "legacy_parameter_interpolation",
                "matched_ppo",
            },
        )
        for row in rows:
            self.assertEqual(row["evaluation_split"], "train")
            self.assertEqual(row["evaluation_population"], "stage0_smoke")
            self.assertFalse(row["scientific_selection"])
            self.assertFalse(row["paper_evidence_eligible"])
            self.assertEqual(row["checkpoint_environment_interactions"], [16, 32])

    def test_validation_partition_is_disjoint_exhaustive_and_ordered(self) -> None:
        select = population_for_id(self.populations, "validation_select")
        bridge = population_for_id(self.populations, "validation_bridge")
        self.assertEqual(len(select["indices"]), 128)
        self.assertEqual(len(bridge["indices"]), 128)
        self.assertFalse(set(select["indices"]) & set(bridge["indices"]))
        self.assertEqual(
            set(select["indices"]) | set(bridge["indices"]), set(range(256))
        )
        pairs = list(
            zip(
                select["selection_scores"] + bridge["selection_scores"],
                select["indices"] + bridge["indices"],
            )
        )
        self.assertEqual(pairs, sorted(pairs))
        self.assertEqual(
            select["ordered_record_sha256"],
            "a140a2240a8b8a89c405216751b1f2b9a332b9ee4265f377987e607ec2d4a19a",
        )
        self.assertEqual(
            bridge["ordered_record_sha256"],
            "ba724676b172248219918afb7073a07f08adc9b74f57c3395bfda609f0a78a42",
        )

    def test_partition_overlap_and_reordering_fail_closed(self) -> None:
        overlap = copy.deepcopy(self.populations)
        overlap["populations"]["validation_bridge"]["indices"][0] = overlap[
            "populations"
        ]["validation_select"]["indices"][0]
        overlap["populations"]["validation_bridge"]["binding_sha256"] = (
            population_binding_sha256(overlap["populations"]["validation_bridge"])
        )
        with self.assertRaises(PolicyImprovementPopulationError):
            validate_v2_populations(overlap)

        reordered = copy.deepcopy(self.populations)
        population = reordered["populations"]["validation_select"]
        for field in (
            "indices",
            "record_sha256s",
            "input_sha256s",
            "selection_scores",
        ):
            population[field][0], population[field][1] = (
                population[field][1],
                population[field][0],
            )
        population["binding_sha256"] = population_binding_sha256(population)
        with self.assertRaises(PolicyImprovementPopulationError):
            validate_v2_populations(reordered)

    def test_authenticated_manifest_parity_rejects_mixed_identity(self) -> None:
        train_records = [_digest("train-record", index) for index in range(1024)]
        train_inputs = [_digest("train-input", index) for index in range(1024)]
        validation_records = [
            _digest("validation-record", index) for index in range(256)
        ]
        validation_inputs = [_digest("validation-input", index) for index in range(256)]
        document = materialize_v2_populations(
            train_record_sha256s=train_records,
            train_input_sha256s=train_inputs,
            validation_record_sha256s=validation_records,
            validation_input_sha256s=validation_inputs,
        )
        mixed = copy.deepcopy(document)
        population = mixed["populations"]["validation_bridge"]
        population["input_sha256s"][0] = _digest("wrong-input", 0)
        from scripts.policy_improvement_populations import ordered_sha256

        population["ordered_input_sha256"] = ordered_sha256(population["input_sha256s"])
        population["binding_sha256"] = population_binding_sha256(population)
        with self.assertRaises(PolicyImprovementPopulationError):
            validate_v2_populations(
                mixed,
                train_record_sha256s=train_records,
                train_input_sha256s=train_inputs,
                validation_record_sha256s=validation_records,
                validation_input_sha256s=validation_inputs,
            )

    def test_population_loader_authenticates_protocol_path_and_digest(self) -> None:
        self.assertEqual(
            load_registered_populations(self.protocol, ROOT), self.populations
        )
        changed = copy.deepcopy(self.protocol)
        changed["population_registry"]["sha256"] = "0" * 64
        with self.assertRaises(PolicyImprovementPopulationError):
            load_registered_populations(changed, ROOT)

    def test_registry_has_only_the_prespecified_139_rows(self) -> None:
        self.assertEqual(self.registry["counts"]["total_rows"], 139)
        self.assertEqual(self.registry["counts"]["smoke_rows"], 4)
        self.assertEqual(self.registry["counts"]["full_rows"], 135)
        for phase, count in EXPECTED_PHASE_COUNTS.items():
            self.assertEqual(self.registry["counts"][phase], count)
            self.assertEqual(
                sum(row["phase"] == phase for row in self.registry["rows"]),
                count,
            )

    def test_stage1_screen_contains_exact_methods_only(self) -> None:
        rows = [row for row in self.registry["rows"] if row["phase"] == "stage1_screen"]
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            {row["method_id"] for row in rows},
            {"fixed_base_exact_persistent", "fixed_base_exact_episodic"},
        )
        self.assertEqual({row["n"] for row in rows}, {2, 4})
        self.assertEqual({row["K"] for row in rows}, {1, 5})
        self.assertEqual({row["alpha"] for row in rows}, {0.1})
        self.assertEqual(
            {row["seed"] for row in rows},
            {784831257, 2087907586, 4056782312},
        )
        self.assertTrue(all(row["scientific_selection"] for row in rows))

    def test_baselines_are_six_separate_nonselection_templates(self) -> None:
        rows = [
            row
            for row in self.registry["rows"]
            if row["phase"] == "stage1_baseline_readiness"
        ]
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["method_id"] for row in rows},
            {"legacy_parameter_interpolation", "matched_ppo"},
        )
        for row in rows:
            self.assertEqual(row["row_kind"], "selection_template")
            self.assertIsNone(row["K"])
            self.assertIsNone(row["alpha"])
            self.assertFalse(row["scientific_selection"])

    def test_stage1_is_blocked_by_unavailable_base_policy(self) -> None:
        artifact = self.protocol["base_policy_artifact"]
        self.assertEqual(artifact["status"], "unavailable")
        self.assertFalse(artifact["stage1_execution_allowed"])
        self.assertEqual(
            artifact["reason"],
            "no_authenticated_train_only_base_policy_artifact_supplied_for_v2",
        )

    def test_v2_configs_preserve_v1_canonical_values(self) -> None:
        for method in self.protocol["methods"]:
            v2 = self.configs[method["id"]]
            v1 = load_v1_flat_registered_yaml(
                ROOT / "configs/policy_improvement_v1" / f"{method['id']}.yaml"
            )
            self.assertEqual(v2, v1)

    def test_result_envelope_binds_v2_identities(self) -> None:
        row = self.registry["rows"][0]
        result = {
            "schema_name": "policy_improvement_result_v2",
            "schema_version": 1,
            "protocol_id": self.protocol["protocol_id"],
            "protocol_schema_name": self.protocol["schema_name"],
            "protocol_schema_version": self.protocol["schema_version"],
            "protocol_sha256": sha256_json(self.protocol),
            "population_registry_schema_name": self.populations["schema_name"],
            "population_registry_schema_version": self.populations["schema_version"],
            "population_registry_sha256": self.protocol["population_registry"][
                "sha256"
            ],
            "registry_schema_name": self.registry["schema_name"],
            "registry_schema_version": self.registry["registry_schema_version"],
            "registry_sha256": sha256_json(self.registry),
            "registry_row_schema_name": row["schema_name"],
            "registry_row_schema_version": row["schema_version"],
            "registry_row_sha256": sha256_json(row),
            "run_id": row["run_id"],
            "phase": row["phase"],
            "method_id": row["method_id"],
            "status": "complete",
            "evaluation_split": row["evaluation_split"],
            "evaluation_population_id": row["evaluation_population"],
            "evaluation_population_binding_sha256": self.populations["populations"][
                "stage0_smoke"
            ]["binding_sha256"],
            "evaluation_population_ordered_record_sha256": self.populations[
                "populations"
            ]["stage0_smoke"]["ordered_record_sha256"],
            "evaluation_population_ordered_input_sha256": self.populations[
                "populations"
            ]["stage0_smoke"]["ordered_input_sha256"],
            "evaluation_record_count": 8,
            "validation_data_opened": False,
            "test_data_opened": False,
            "scientific_selection": False,
            "paper_evidence_eligible": False,
            "payload": {},
        }
        self.assertEqual(validate_v2_result(result), result)
        result["schema_name"] = "policy_improvement_v1"
        with self.assertRaises(PolicyImprovementV2SchemaError):
            validate_v2_result(result)

    def test_population_score_is_namespace_bound(self) -> None:
        record = self.populations["populations"]["stage0_smoke"]["record_sha256s"][0]
        self.assertNotEqual(
            derive_population_score(
                "upi-trm-policy-improvement-v2-stage0-smoke:", record
            ),
            derive_population_score(
                "upi-trm-policy-improvement-v2-validation-partition:", record
            ),
        )

    def test_theory_amendment_is_preoutcome_and_nonselection(self) -> None:
        amendment = load_strict_json(THEORY_AMENDMENT)
        self.assertEqual(
            amendment["schema_name"],
            "policy_improvement_theory_bridge_amendment_v2",
        )
        self.assertEqual(amendment["schema_version"], 1)
        self.assertEqual(amendment["protocol_id"], self.protocol["protocol_id"])
        self.assertEqual(amendment["protocol_sha256"], sha256_json(self.protocol))
        self.assertEqual(
            amendment["population_registry_sha256"],
            self.protocol["population_registry"]["sha256"],
        )
        self.assertEqual(
            amendment["source_registry_sha256"], sha256_json(self.registry)
        )
        self.assertFalse(amendment["outcome_evidence_inspected"])
        self.assertFalse(amendment["test_data_opened"])
        self.assertFalse(amendment["analysis"]["scientific_selection"])
        self.assertEqual(
            amendment["evaluator_contract"]["population_id"],
            "validation_bridge",
        )
        self.assertEqual(amendment["reference_depths"]["primary_m"], 8)
        self.assertEqual(amendment["reference_depths"]["optional_exploratory_m"], [16])
        self.assertEqual(amendment["bellman_estimators"]["K1"]["rollout_count"], 0)
        self.assertEqual(amendment["bellman_estimators"]["K5"]["rollout_count"], 64)
        self.assertEqual(
            amendment["predictive_return_estimator"]["maximum_environment_steps"],
            16,
        )
        self.assertEqual(
            amendment["predictive_return_estimator"]["terminal_requirement"],
            "must_terminate_without_bootstrap",
        )
        self.assertEqual(
            amendment["centering_contract"][
                "training_estimator_parity_absolute_tolerance"
            ],
            1e-6,
        )
        self.assertIn("training_estimator_parity_max_abs_error", amendment["metrics"])
        self.assertEqual(
            amendment["deployment_contract"][
                "exact_mixture_identity_tv_absolute_tolerance"
            ],
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
