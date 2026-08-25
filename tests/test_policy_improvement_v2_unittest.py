#!/usr/bin/env fbpython
"""Focused tests for the fresh policy-improvement v2 registration."""

from __future__ import annotations

import ast
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.policy_improvement_analysis import _registered_secondary_contrasts
from scripts.policy_improvement_populations import (
    canonical_json_bytes as population_json_bytes,
    derive_population_score,
    load_registered_populations,
    load_strict_json as load_population_json,
    materialize_v2_populations,
    ordered_sha256,
    PolicyImprovementPopulationError,
    population_binding_sha256,
    population_for_id,
    validate_v2_populations,
)
from scripts.policy_improvement_registry import (
    generate_registry,
    load_flat_registered_yaml as load_v1_flat_registered_yaml,
    main as registry_main,
)
from scripts.policy_improvement_schema import (
    amendment_history_sha256,
    PolicyImprovementSchemaError,
    runtime_authorization_sha256,
    validate_protocol,
    validate_runtime_authorization,
)
from scripts.policy_improvement_smoke_plan import render_smoke_plan
import scripts.policy_improvement_test_open as test_open_module
from scripts.policy_improvement_test_open_cli import main as test_open_main
from scripts.policy_improvement_v2_registry import (
    EXPECTED_PHASE_COUNTS,
    generate_v2_registry,
    load_v2_base_configs,
    validate_v2_registry_document,
)
from scripts.policy_improvement_v2_schema import (
    bind_v2_result_to_registration,
    bind_v2_result_to_row,
    canonical_json_bytes,
    IMMUTABLE_DATASET_V1,
    load_strict_json,
    PolicyImprovementV2SchemaError,
    sha256_json,
    validate_base_policy_amendment,
    validate_compute_freeze_v2,
    validate_stage1_configuration_selection,
    validate_v2_amendment_history,
    validate_v2_protocol,
    validate_v2_registry_row,
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


def _buck_targets() -> dict[str, dict[str, set[str]]]:
    tree = ast.parse((ROOT / "BUCK").read_text(encoding="utf-8"))
    targets: dict[str, dict[str, set[str]]] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.func.id not in {
            "python_binary",
            "python_library",
        }:
            continue
        keywords = {item.arg: item.value for item in call.keywords if item.arg}
        name_value = keywords.get("name")
        if not isinstance(name_value, ast.Constant) or not isinstance(
            name_value.value, str
        ):
            continue
        deps: set[str] = set()
        deps_value = keywords.get("deps")
        if isinstance(deps_value, ast.List):
            deps = {
                item.value.removeprefix(":")
                for item in deps_value.elts
                if isinstance(item, ast.Constant)
                and isinstance(item.value, str)
                and item.value.startswith(":")
            }
        resources: set[str] = set()
        resource_value = keywords.get("resources")
        if (
            isinstance(resource_value, ast.Call)
            and isinstance(resource_value.func, ast.Name)
            and resource_value.func.id == "glob"
            and resource_value.args
            and isinstance(resource_value.args[0], ast.List)
        ):
            resources = {
                item.value
                for item in resource_value.args[0].elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
        targets[name_value.value] = {"deps": deps, "resources": resources}
    return targets


def _local_dependency_closure(
    targets: dict[str, dict[str, set[str]]], root: str
) -> set[str]:
    closure: set[str] = set()
    pending = [root]
    while pending:
        target = pending.pop()
        if target in closure:
            continue
        closure.add(target)
        pending.extend(targets.get(target, {}).get("deps", set()) - closure)
    return closure


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

    def test_production_buck_targets_package_the_complete_v2_closure(self) -> None:
        targets = _buck_targets()
        v2_libraries = {
            "policy_improvement_populations",
            "policy_improvement_v2_registry",
            "policy_improvement_v2_schema",
        }
        runtime_targets = {
            "upi_trm_train",
            "policy_improvement_full",
            "policy_improvement_theory_bridge",
            "policy_improvement_audit",
            "policy_improvement_analysis",
        }
        for target in runtime_targets:
            with self.subTest(target=target):
                self.assertLessEqual(
                    v2_libraries,
                    _local_dependency_closure(targets, target),
                )
                self.assertLessEqual(
                    {
                        "configs/policy_improvement_v2/*.json",
                        "configs/policy_improvement_v2/*.yaml",
                        "configs/policy_improvement_v2/amendments/*.json",
                    },
                    targets[target]["resources"],
                )

    def test_v1_and_v2_protocol_schemas_are_not_interchangeable(self) -> None:
        self.assertEqual(validate_protocol(self.protocol), self.protocol)
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

    def test_stage0_training_complement_is_registered(self) -> None:
        """Stage 0 trains on the registered complement of its evaluation pool."""

        stage0 = population_for_id(self.populations, "stage0_smoke")
        training = population_for_id(self.populations, "train_minus_stage0_smoke")
        self.assertEqual(training["split"], "train")
        self.assertEqual(training["count"], 1016)
        self.assertEqual(
            training["selection_algorithm"],
            "sha256_namespace_record_sha256_lexicographic_complement_v1",
        )
        self.assertEqual(
            training["selection_namespace"], stage0["selection_namespace"]
        )
        # Deterministic complement: index ordered, disjoint, and exhaustive.
        self.assertEqual(training["indices"], sorted(training["indices"]))
        self.assertFalse(set(training["indices"]) & set(stage0["indices"]))
        self.assertEqual(
            set(training["indices"]) | set(stage0["indices"]), set(range(1024))
        )
        self.assertLess(
            max(zip(stage0["selection_scores"], stage0["indices"])),
            min(zip(training["selection_scores"], training["indices"])),
        )
        for field in ("record_sha256s", "input_sha256s", "selection_scores"):
            self.assertEqual(len(training[field]), 1016)
        self.assertEqual(
            training["binding_sha256"], population_binding_sha256(training)
        )
        # Reassembling both halves by index reproduces the registered split.
        by_index = dict(
            zip(
                list(stage0["indices"]) + list(training["indices"]),
                list(stage0["record_sha256s"]) + list(training["record_sha256s"]),
            )
        )
        self.assertEqual(
            self.populations["split_ordered_record_sha256"]["train"],
            ordered_sha256([by_index[index] for index in range(1024)]),
        )
        # The protocol, registry, and plan all bind it explicitly.
        self.assertEqual(
            self.protocol["training_populations"],
            {
                "train_minus_stage0_smoke": {
                    "population_id": "train_minus_stage0_smoke",
                    "split": "train",
                    "count": 1016,
                    "complement_of": "stage0_smoke",
                    "used_by_phase": "stage0_smoke",
                }
            },
        )
        for row in self.registry["rows"]:
            expected = (
                "train_minus_stage0_smoke"
                if row["phase"] == "stage0_smoke"
                else None
            )
            self.assertEqual(row["training_population"], expected, row["run_id"])

    def _corrupted_complement(self, mutate) -> dict:
        document = copy.deepcopy(self.populations)
        population = document["populations"]["train_minus_stage0_smoke"]
        mutate(document, population)
        population["binding_sha256"] = population_binding_sha256(population)
        return document

    def test_training_complement_overlap_fails_closed(self) -> None:
        def mutate(document, population):
            stage0 = document["populations"]["stage0_smoke"]
            population["indices"][0] = stage0["indices"][0]
            population["record_sha256s"][0] = stage0["record_sha256s"][0]
            population["input_sha256s"][0] = stage0["input_sha256s"][0]
            population["selection_scores"][0] = stage0["selection_scores"][0]
            population["ordered_record_sha256"] = ordered_sha256(
                population["record_sha256s"]
            )
            population["ordered_input_sha256"] = ordered_sha256(
                population["input_sha256s"]
            )

        with self.assertRaises(PolicyImprovementPopulationError):
            validate_v2_populations(self._corrupted_complement(mutate))

    def test_training_complement_reordering_fails_closed(self) -> None:
        def mutate(_document, population):
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
            population["ordered_record_sha256"] = ordered_sha256(
                population["record_sha256s"]
            )
            population["ordered_input_sha256"] = ordered_sha256(
                population["input_sha256s"]
            )

        with self.assertRaisesRegex(
            PolicyImprovementPopulationError, "registered index order"
        ):
            validate_v2_populations(self._corrupted_complement(mutate))

    def test_training_complement_missing_record_fails_closed(self) -> None:
        def mutate(_document, population):
            for field in (
                "indices",
                "record_sha256s",
                "input_sha256s",
                "selection_scores",
            ):
                population[field].pop()
            population["count"] = 1015
            population["ordered_record_sha256"] = ordered_sha256(
                population["record_sha256s"]
            )
            population["ordered_input_sha256"] = ordered_sha256(
                population["input_sha256s"]
            )

        with self.assertRaises(PolicyImprovementPopulationError):
            validate_v2_populations(self._corrupted_complement(mutate))

    def test_training_complement_extra_record_fails_closed(self) -> None:
        def mutate(_document, population):
            # A duplicated train record keeps the registered count but makes
            # the complement cover one index twice and another not at all.
            for field in (
                "indices",
                "record_sha256s",
                "input_sha256s",
                "selection_scores",
            ):
                population[field][1] = population[field][0]
            population["ordered_record_sha256"] = ordered_sha256(
                population["record_sha256s"]
            )
            population["ordered_input_sha256"] = ordered_sha256(
                population["input_sha256s"]
            )

        with self.assertRaises(PolicyImprovementPopulationError):
            validate_v2_populations(self._corrupted_complement(mutate))

    def test_training_population_outside_stage0_fails_closed(self) -> None:
        """The complement is bound to Stage 0 and may not be reused elsewhere."""

        stage0_row = next(
            row for row in self.registry["rows"] if row["phase"] == "stage0_smoke"
        )
        other_row = next(
            row for row in self.registry["rows"] if row["phase"] != "stage0_smoke"
        )
        validate_v2_registry_row(copy.deepcopy(stage0_row))

        borrowed = copy.deepcopy(other_row)
        borrowed["training_population"] = "train_minus_stage0_smoke"
        with self.assertRaisesRegex(
            PolicyImprovementV2SchemaError, "training-population binding differs"
        ):
            validate_v2_registry_row(borrowed)

        unbound = copy.deepcopy(stage0_row)
        unbound["training_population"] = None
        with self.assertRaisesRegex(
            PolicyImprovementV2SchemaError, "training-population binding differs"
        ):
            validate_v2_registry_row(unbound)

        unregistered = copy.deepcopy(stage0_row)
        unregistered["training_population"] = "validation_bridge"
        with self.assertRaises(PolicyImprovementV2SchemaError):
            validate_v2_registry_row(unregistered)

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

    def test_generic_registry_cli_regenerates_v2_with_bound_populations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "registry.json"
            self.assertEqual(
                registry_main(
                    [
                        "--protocol",
                        str(PROTOCOL),
                        "--project-root",
                        str(ROOT),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            self.assertEqual(output.read_bytes(), REGISTRY.read_bytes())

    def test_test_open_record_binds_the_supplied_v2_population_document(self) -> None:
        history = [
            {"created_at_utc": f"2026-08-18T00:00:0{index}Z"} for index in range(4)
        ]
        authorization = {"protocol_sha256": sha256_json(self.protocol)}
        with (
            mock.patch.object(
                test_open_module,
                "validate_amendment_history",
                return_value=history,
            ),
            mock.patch.object(
                test_open_module,
                "validate_registry_document",
                return_value=self.registry,
            ) as validate_registry,
            mock.patch.object(
                test_open_module,
                "validate_runtime_authorization",
                return_value=authorization,
            ),
            mock.patch.object(
                test_open_module,
                "runtime_authorization_sha256",
                return_value="f" * 64,
            ),
        ):
            test_open_module.expected_test_open_record(
                protocol=self.protocol,
                registry=self.registry,
                amendment_history=history,
                runtime_authorization=authorization,
                opened_at_utc="2026-08-20T00:00:00Z",
                base_configs=self.configs,
                populations_value=self.populations,
            )
        validate_registry.assert_called_once_with(
            self.registry,
            self.protocol,
            history,
            base_configs=self.configs,
            populations_value=self.populations,
        )

    def test_test_open_cli_loads_v2_populations_before_any_publication(self) -> None:
        class StopAfterRegistryAuthentication(Exception):
            pass

        history = [
            {"created_at_utc": f"2026-08-18T00:00:0{index}Z"} for index in range(4)
        ]

        def stop_after_registry(*_args: object, **kwargs: object) -> dict[str, object]:
            self.assertEqual(kwargs["populations_value"], self.populations)
            raise StopAfterRegistryAuthentication

        with tempfile.TemporaryDirectory() as directory:
            evidence_root = Path(directory) / "evidence"
            evidence_root.mkdir(mode=0o700)
            arguments = [
                "--protocol",
                str(PROTOCOL),
                "--registry",
                str(REGISTRY),
                "--amendment",
                str(THEORY_AMENDMENT),
                "--project-root",
                str(ROOT),
                "--dataset-root",
                str(Path(directory) / "dataset"),
                "--evidence-root",
                str(evidence_root),
                "--amendment-evidence",
                "stage0_smoke=/not/read.json",
                "--audit-runtime-sha256",
                "1" * 64,
                "--audit-runtime-profile-sha256",
                "2" * 64,
                "--audit-source-git-commit",
                "3" * 40,
                "--launcher-sha256",
                "4" * 64,
                "--producer-git-commit",
                "5" * 40,
                "--producer-source-manifest-sha256",
                "6" * 64,
                "--runtime-authorization-json",
                "{}",
                "--runtime-authorization-sha256",
                "7" * 64,
            ]
            with (
                mock.patch(
                    "scripts.policy_improvement_test_open_cli.validate_amendment_history",
                    return_value=history,
                ),
                mock.patch(
                    "scripts.policy_improvement_test_open_cli.validate_registry_document",
                    side_effect=stop_after_registry,
                ),
                self.assertRaises(StopAfterRegistryAuthentication),
            ):
                test_open_main(arguments, checkpoint_validator=lambda _: {})
            self.assertFalse((evidence_root / "TEST_OPEN.json").exists())

    def test_v2_analysis_runs_only_registered_secondary_contrasts(self) -> None:
        self.assertEqual(
            _registered_secondary_contrasts(self.protocol["statistics"]),
            (),
        )
        v1 = validate_protocol(load_strict_json(V1_PROTOCOL))
        self.assertEqual(
            _registered_secondary_contrasts(v1["statistics"]),
            (
                (
                    "fixed_base_exact_episodic-minus-matched_ppo",
                    "fixed_base_exact_episodic",
                ),
                (
                    "legacy_parameter_interpolation-minus-matched_ppo",
                    "legacy_parameter_interpolation",
                ),
            ),
        )

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

    def test_future_stage1_selection_is_vselect_only_and_fully_bound(self) -> None:
        selected: list[dict[str, object]] = []
        for method_id, n, horizon in (
            ("fixed_base_exact_persistent", 2, 1),
            ("fixed_base_exact_episodic", 4, 5),
        ):
            selected.append(
                {
                    "method_id": method_id,
                    "n": n,
                    "K": horizon,
                    "source_results": [
                        {
                            "run_id": (f"s1-{method_id}-n{n}-k{horizon}-s{seed}"),
                            "seed": seed,
                            "result_sha256": _digest(
                                f"{method_id}-{n}-{horizon}", seed
                            ),
                        }
                        for seed in (784831257, 2087907586, 4056782312)
                    ],
                }
            )
        selection = {
            "schema_name": "policy_improvement_stage1_configuration_selection_v2",
            "schema_version": 1,
            "amendment_id": "stage1-configuration-selection-v2",
            "created_at_utc": "2026-08-20T12:00:00Z",
            "protocol_id": self.protocol["protocol_id"],
            "protocol_schema_name": self.protocol["schema_name"],
            "protocol_schema_version": self.protocol["schema_version"],
            "protocol_sha256": sha256_json(self.protocol),
            "population_registry_sha256": self.protocol["population_registry"][
                "sha256"
            ],
            "source_registry_schema_name": self.registry["schema_name"],
            "source_registry_schema_version": self.registry["registry_schema_version"],
            "source_registry_sha256": sha256_json(self.registry),
            "prior_amendment_history_sha256": _digest("prior", 1),
            "compute_freeze_sha256": _digest("compute", 1),
            "base_policy_artifact_sha256": _digest("base", 1),
            "runtime_authorization_sha256": _digest("authorization", 1),
            "selection_checkpoint_environment_interactions": 10000,
            "selection_population_id": "validation_select",
            "selection_population_binding_sha256": self.populations["populations"][
                "validation_select"
            ]["binding_sha256"],
            "bridge_population_id": "validation_bridge",
            "bridge_population_binding_sha256": self.populations["populations"][
                "validation_bridge"
            ]["binding_sha256"],
            "outcome_evidence_inspected": True,
            "inspected_population_ids": ["validation_select"],
            "bridge_not_inspected": True,
            "test_data_opened": False,
            "selection_rule": (
                "mean_seed_level_interaction_matched_solve_rate_then_"
                "n_ascending_then_K_ascending_v2"
            ),
            "source_audit_sha256": _digest("audit", 1),
            "selected_configurations": selected,
        }
        self.assertEqual(validate_stage1_configuration_selection(selection), selection)
        opened_bridge = copy.deepcopy(selection)
        opened_bridge["bridge_not_inspected"] = False
        with self.assertRaisesRegex(
            PolicyImprovementV2SchemaError,
            "contract differs",
        ):
            validate_stage1_configuration_selection(opened_bridge)
        wrong_source = copy.deepcopy(selection)
        wrong_source["selected_configurations"][0]["source_results"][0][
            "run_id"
        ] = "another-run"
        with self.assertRaisesRegex(
            PolicyImprovementV2SchemaError,
            "source rows differ",
        ):
            validate_stage1_configuration_selection(wrong_source)

    def test_future_base_policy_amendment_is_train_only_and_registry_stable(
        self,
    ) -> None:
        theory = load_strict_json(THEORY_AMENDMENT)
        amendment = {
            "schema_name": "policy_improvement_base_policy_amendment_v2",
            "schema_version": 1,
            "amendment_id": "base-policy-v2",
            "created_at_utc": "2026-08-20T12:00:00Z",
            "protocol_id": self.protocol["protocol_id"],
            "protocol_schema_name": self.protocol["schema_name"],
            "protocol_schema_version": self.protocol["schema_version"],
            "protocol_sha256": sha256_json(self.protocol),
            "population_registry_sha256": self.protocol["population_registry"][
                "sha256"
            ],
            "source_registry_schema_name": self.registry["schema_name"],
            "source_registry_schema_version": self.registry["registry_schema_version"],
            "source_registry_sha256": sha256_json(self.registry),
            "prior_amendment_history_sha256": amendment_history_sha256([theory]),
            "runtime_authorization_sha256": _digest("authorization", 2),
            "validation_data_inspected": False,
            "test_data_opened": False,
            "base_policy_artifact": {
                "status": "available",
                "initialization_kind": "train_only_pretrained",
                "architecture_sha256": sha256_json(self.protocol["architecture"]),
                "model_state_sha256": _digest("model", 2),
                "producer_git_commit": "1" * 40,
                "producer_source_manifest_sha256": _digest("source", 2),
                "training_dataset_manifest_sha256": self.protocol["dataset"][
                    "manifest_sha256"
                ]["value"],
                "training_split": "train",
                "training_split_ordered_record_sha256": self.protocol["dataset"][
                    "splits"
                ]["train"]["ordered_record_sha256"]["value"],
                "training_procedure_sha256": _digest("procedure", 2),
                "checkpoint_sha256": _digest("checkpoint", 2),
                "checkpoint_size_bytes": 4096,
                "shared_across_persistent_and_episodic": True,
                "not_selected_by_validation_or_test": True,
            },
        }
        self.assertEqual(validate_base_policy_amendment(amendment), amendment)
        active = generate_registry(
            self.protocol,
            [theory, amendment],
            base_configs=self.configs,
            populations_value=self.populations,
        )
        self.assertEqual(active, self.registry)
        compute = {
            "schema_name": "policy_improvement_compute_freeze_v2",
            "schema_version": 1,
            "amendment_id": "post-smoke-compute-freeze-v2",
            "created_at_utc": "2026-08-20T12:01:00Z",
            "protocol_id": self.protocol["protocol_id"],
            "protocol_schema_name": self.protocol["schema_name"],
            "protocol_schema_version": self.protocol["schema_version"],
            "protocol_sha256": sha256_json(self.protocol),
            "population_registry_sha256": self.protocol["population_registry"][
                "sha256"
            ],
            "source_registry_schema_name": self.registry["schema_name"],
            "source_registry_schema_version": self.registry["registry_schema_version"],
            "source_registry_sha256": sha256_json(self.registry),
            "prior_amendment_history_sha256": amendment_history_sha256(
                [theory, amendment]
            ),
            "base_policy_artifact_sha256": sha256_json(
                amendment["base_policy_artifact"]
            ),
            "runtime_authorization_sha256": amendment["runtime_authorization_sha256"],
            "test_data_opened": False,
            "evidence": {
                "phase": "stage0_smoke",
                "audit_report_sha256": _digest("stage0 audit", 2),
                "result_set_sha256": _digest("stage0 results", 2),
                "per_instance_set_sha256": _digest("stage0 instances", 2),
                "expected_rows": 4,
                "complete_rows": 4,
                "failed_rows": 0,
            },
            "common_compute_targets": {
                "unit": "recurrent_map_applications",
                "pilot": 1000,
                "confirmatory": 2000,
                "ablation": 2000,
                "maximum_relative_mismatch": 0.05,
            },
        }
        self.assertEqual(validate_compute_freeze_v2(compute), compute)
        self.assertEqual(
            validate_v2_amendment_history(
                [theory, amendment, compute],
                protocol=self.protocol,
                registry=self.registry,
                populations=self.populations,
            ),
            [theory, amendment, compute],
        )
        v1_compute = copy.deepcopy(compute)
        v1_compute["schema_name"] = "policy_improvement_compute_freeze_v1"
        with self.assertRaisesRegex(
            PolicyImprovementV2SchemaError,
            "amendment 2 is invalid",
        ):
            validate_v2_amendment_history(
                [theory, amendment, v1_compute],
                protocol=self.protocol,
                registry=self.registry,
                populations=self.populations,
            )
        leaked = copy.deepcopy(amendment)
        leaked["validation_data_inspected"] = True
        with self.assertRaisesRegex(
            PolicyImprovementV2SchemaError,
            "train-only",
        ):
            validate_base_policy_amendment(leaked)
        wrong_training_data = copy.deepcopy(amendment)
        wrong_training_data["base_policy_artifact"][
            "training_split_ordered_record_sha256"
        ] = _digest("another train split", 2)
        with self.assertRaisesRegex(
            PolicyImprovementSchemaError,
            "amendment history is invalid",
        ):
            generate_registry(
                self.protocol,
                [theory, wrong_training_data],
                base_configs=self.configs,
                populations_value=self.populations,
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
        self.stage0_result = copy.deepcopy(result)
        result["schema_name"] = "policy_improvement_v1"
        with self.assertRaises(PolicyImprovementV2SchemaError):
            validate_v2_result(result)

    def _stage0_result(self) -> dict:
        """Build the same validated Stage 0 envelope the test above checks."""

        row = next(
            item for item in self.registry["rows"] if item["phase"] == "stage0_smoke"
        )
        population = self.populations["populations"]["stage0_smoke"]
        return {
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
            "evaluation_population_binding_sha256": population["binding_sha256"],
            "evaluation_population_ordered_record_sha256": population[
                "ordered_record_sha256"
            ],
            "evaluation_population_ordered_input_sha256": population[
                "ordered_input_sha256"
            ],
            "evaluation_record_count": 8,
            "validation_data_opened": False,
            "test_data_opened": False,
            "scientific_selection": False,
            "paper_evidence_eligible": False,
            "payload": {},
        }, row

    def test_split_isolation_attestations_are_derived_not_declared(self) -> None:
        """A Stage 0 result cannot claim it opened validation or test content."""

        base, _row = self._stage0_result()
        self.assertEqual(validate_v2_result(copy.deepcopy(base)), base)

        for field in ("validation_data_opened", "test_data_opened"):
            with self.subTest(field=field):
                hostile = copy.deepcopy(base)
                hostile[field] = True
                with self.assertRaisesRegex(
                    PolicyImprovementV2SchemaError, "must hold exactly"
                ):
                    validate_v2_result(hostile)

        # And a validation-split result may not deny opening validation.
        # (Stage 0 is train-only, so flip the split coherently.)
        denied = copy.deepcopy(base)
        denied["evaluation_split"] = "validation"
        denied["evaluation_population_id"] = "validation_select"
        denied["validation_data_opened"] = False
        with self.assertRaisesRegex(
            PolicyImprovementV2SchemaError, "must hold exactly"
        ):
            validate_v2_result(denied)

    def test_stage0_cannot_claim_selection_or_paper_eligibility(self) -> None:
        base, _row = self._stage0_result()
        for field in ("scientific_selection", "paper_evidence_eligible"):
            with self.subTest(field=field):
                hostile = copy.deepcopy(base)
                hostile[field] = True
                with self.assertRaisesRegex(
                    PolicyImprovementV2SchemaError, "selection or paper eligibility"
                ):
                    validate_v2_result(hostile)

    def test_v1_and_v2_results_cannot_cross_schemas(self) -> None:
        """The generic router dispatches on the exact schema name only."""

        from scripts.policy_improvement_schema import (
            PolicyImprovementSchemaError as V1Error,
            validate_result as validate_router,
        )

        base, _row = self._stage0_result()
        # The router accepts a genuine v2 envelope by dispatch.
        self.assertEqual(validate_router(copy.deepcopy(base)), base)

        # A v2 envelope wearing the v1 name must not be accepted by either.
        spoofed = copy.deepcopy(base)
        spoofed["schema_name"] = "policy_improvement_v1"
        with self.assertRaises((V1Error, PolicyImprovementV2SchemaError)):
            validate_router(spoofed)
        with self.assertRaises(PolicyImprovementV2SchemaError):
            validate_v2_result(spoofed)

        # A v1-shaped envelope must never satisfy the v2 validator.
        v1_shaped = {"schema_name": "policy_improvement_v1", "schema_version": 5}
        with self.assertRaises(PolicyImprovementV2SchemaError):
            validate_v2_result(v1_shaped)

        # A v1 envelope wearing the v2 name must not be accepted either.
        v1_as_v2 = dict(v1_shaped)
        v1_as_v2["schema_name"] = "policy_improvement_result_v2"
        with self.assertRaises((V1Error, PolicyImprovementV2SchemaError)):
            validate_router(v1_as_v2)

    def test_result_row_cross_binding_rejects_mixed_identities(self) -> None:
        base, row = self._stage0_result()
        validated = validate_v2_result(copy.deepcopy(base))
        bind_v2_result_to_row(validated, row)
        bind_v2_result_to_registration(
            validated,
            row,
            self.protocol,
            self.registry,
            self.populations,
        )

        for field, value in (
            ("run_id", "s0-not-this-run"),
            (
                "method_id",
                (
                    "matched_ppo"
                    if row["method_id"] != "matched_ppo"
                    else "fixed_base_exact_episodic"
                ),
            ),
            ("phase", "stage1_screen"),
        ):
            with self.subTest(field=field):
                mixed = dict(row)
                mixed[field] = value
                with self.assertRaises(PolicyImprovementV2SchemaError):
                    bind_v2_result_to_row(validated, mixed)

        mixed_population = dict(row)
        mixed_population["evaluation_population"] = "validation_select"
        with self.assertRaises(PolicyImprovementV2SchemaError):
            bind_v2_result_to_row(validated, mixed_population)

        for field, value in (
            ("protocol_sha256", "0" * 64),
            ("population_registry_sha256", "1" * 64),
            ("registry_sha256", "2" * 64),
            ("registry_row_sha256", "3" * 64),
            ("evaluation_population_binding_sha256", "4" * 64),
            ("evaluation_population_ordered_record_sha256", "5" * 64),
            ("evaluation_population_ordered_input_sha256", "6" * 64),
            ("evaluation_record_count", 7),
        ):
            with self.subTest(registration_field=field):
                hostile = copy.deepcopy(validated)
                hostile[field] = value
                validate_v2_result(hostile)
                with self.assertRaises(PolicyImprovementV2SchemaError):
                    bind_v2_result_to_registration(
                        hostile,
                        row,
                        self.protocol,
                        self.registry,
                        self.populations,
                    )

    def test_runtime_authorization_v3_binds_six_distinct_roles(self) -> None:
        amendment = load_strict_json(THEORY_AMENDMENT)
        role_names = (
            "policy-improvement-training",
            "policy-improvement-evaluation",
            "policy-improvement-audit",
            "policy-improvement-analysis",
            "policy-improvement-full",
            "policy-improvement-theory-bridge",
        )
        authorization = {
            "schema_name": "policy_improvement_runtime_authorization_v3",
            "schema_version": 3,
            "authorization_id": "policy-improvement-v2-current-head",
            "created_at_utc": "2026-08-18T12:00:00Z",
            "protocol_sha256": sha256_json(self.protocol),
            "protocol": {
                "schema_name": self.protocol["schema_name"],
                "schema_version": self.protocol["schema_version"],
                "protocol_id": self.protocol["protocol_id"],
                "sha256": sha256_json(self.protocol),
            },
            "registry": {
                "schema_name": self.registry["schema_name"],
                "schema_version": self.registry["registry_schema_version"],
                "sha256": sha256_json(self.registry),
            },
            "amendments": [
                {
                    "schema_name": amendment["schema_name"],
                    "schema_version": amendment["schema_version"],
                    "amendment_id": amendment["amendment_id"],
                    "sha256": sha256_json(amendment),
                }
            ],
            "producer_git_commit": "a" * 40,
            "producer_source_manifest_sha256": "b" * 64,
            "launcher_sha256": "c" * 64,
            "roles": [
                {
                    "role": role,
                    "source_git_commit": "a" * 40,
                    "runtime_sha256": f"{index + 1:x}" * 64,
                    "runtime_profile_sha256": f"{index + 7:x}" * 64,
                    "selected_source_manifest_sha256": f"{index + 7:x}" * 64,
                }
                for index, role in enumerate(role_names)
            ],
        }
        self.assertEqual(validate_runtime_authorization(authorization), authorization)
        self.assertNotEqual(
            authorization["roles"][0]["runtime_sha256"],
            authorization["roles"][4]["runtime_sha256"],
        )
        self.assertNotEqual(
            authorization["roles"][1]["runtime_sha256"],
            authorization["roles"][5]["runtime_sha256"],
        )

        swapped = copy.deepcopy(authorization)
        swapped["roles"][0], swapped["roles"][4] = (
            swapped["roles"][4],
            swapped["roles"][0],
        )
        with self.assertRaises(PolicyImprovementSchemaError):
            validate_runtime_authorization(swapped)

    def test_v2_smoke_plan_is_train_only(self) -> None:
        amendment = load_strict_json(THEORY_AMENDMENT)
        role_names = (
            "policy-improvement-training",
            "policy-improvement-evaluation",
            "policy-improvement-audit",
            "policy-improvement-analysis",
            "policy-improvement-full",
            "policy-improvement-theory-bridge",
        )
        authorization = {
            "schema_name": "policy_improvement_runtime_authorization_v3",
            "schema_version": 3,
            "authorization_id": "policy-improvement-v2-stage0-plan",
            "created_at_utc": "2026-08-19T12:00:00Z",
            "protocol_sha256": sha256_json(self.protocol),
            "protocol": {
                "schema_name": self.protocol["schema_name"],
                "schema_version": self.protocol["schema_version"],
                "protocol_id": self.protocol["protocol_id"],
                "sha256": sha256_json(self.protocol),
            },
            "registry": {
                "schema_name": self.registry["schema_name"],
                "schema_version": self.registry["registry_schema_version"],
                "sha256": sha256_json(self.registry),
            },
            "amendments": [
                {
                    "schema_name": amendment["schema_name"],
                    "schema_version": amendment["schema_version"],
                    "amendment_id": amendment["amendment_id"],
                    "sha256": sha256_json(amendment),
                }
            ],
            "producer_git_commit": "a" * 40,
            "producer_source_manifest_sha256": "b" * 64,
            "launcher_sha256": "c" * 64,
            "roles": [
                {
                    "role": role,
                    "source_git_commit": "a" * 40,
                    "runtime_sha256": (
                        "1" * 64 if index < 2 else f"{index + 1:x}" * 64
                    ),
                    "runtime_profile_sha256": (
                        "b" * 64 if index < 2 else f"{index + 7:x}" * 64
                    ),
                    "selected_source_manifest_sha256": (
                        "b" * 64 if index < 2 else f"{index + 7:x}" * 64
                    ),
                }
                for index, role in enumerate(role_names)
            ],
        }
        train_manifest = self.protocol["dataset"]["splits"]["train"]["manifest_sha256"][
            "value"
        ]
        keyword_arguments = {
            "protocol_path": str(PROTOCOL),
            "launcher_path": "/artifacts/phase4_runtime_launcher",
            "training_runtime_path": "/artifacts/upi_trm_train.par",
            "training_runtime_sha256": "1" * 64,
            "source_project_root": str(ROOT),
            "expected_source_git_commit": "a" * 40,
            "dataset_root": "/evidence/data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1",
            "train_manifest_sha256": train_manifest,
            "validation_manifest_sha256": None,
            "evidence_root": "/evidence",
            "runtime_authorization_value": authorization,
            "runtime_authorization_path": "/evidence/runtime-authorization-v3.json",
            "expected_runtime_authorization_sha256": runtime_authorization_sha256(
                authorization
            ),
        }
        plan = render_smoke_plan(self.protocol, **keyword_arguments)
        self.assertEqual(plan["schema_name"], "policy_improvement_smoke_plan_v2")
        self.assertEqual(len(plan["rows"]), 4)
        for row in plan["rows"]:
            self.assertEqual(row["evaluation_split"], "train")
            self.assertEqual(row["evaluation_population_id"], "stage0_smoke")
            for command in row["commands"].values():
                self.assertNotIn("--validation-manifest-sha256", command)
                self.assertNotIn("--test-manifest-sha256", command)

        with self.assertRaisesRegex(
            PolicyImprovementSchemaError,
            "must not accept a validation manifest",
        ):
            render_smoke_plan(
                self.protocol,
                **{
                    **keyword_arguments,
                    "validation_manifest_sha256": self.protocol["dataset"]["splits"][
                        "validation"
                    ]["manifest_sha256"]["value"],
                },
            )

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
