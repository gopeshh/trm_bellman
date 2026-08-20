#!/usr/bin/env fbpython
"""Focused tests for retained failed-attempt authorization in the audit."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.policy_improvement_audit import (
    _audited_stage0_theory_request,
    _authenticate_authorization_producer_source,
    _git_object_bytes,
    _historical_failed_attempt_manifest_sha256s,
    _load_historical_runtime_authorizations,
    _producer_role_name,
    _require_result_namespace,
    _validated_runtime_authorization_map,
)
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    PolicyImprovementSchemaError,
    runtime_authorization_sha256,
)


def _authorization(protocol_sha256: str, *, marker: int = 1) -> dict[str, object]:
    roles = []
    for index, role in enumerate(
        (
            "policy-improvement-training",
            "policy-improvement-evaluation",
            "policy-improvement-audit",
            "policy-improvement-analysis",
        ),
        start=marker,
    ):
        roles.append(
            {
                "role": role,
                "source_git_commit": (
                    format(marker, "x") * 40
                    if role == "policy-improvement-training"
                    else format(index + 4, "x") * 40
                ),
                "runtime_sha256": format(index, "x") * 64,
                "runtime_profile_sha256": format(index + 4, "x") * 64,
                "selected_source_manifest_sha256": format(index + 4, "x") * 64,
            }
        )
    return {
        "schema_name": "policy_improvement_runtime_authorization_v1",
        "schema_version": 1,
        "authorization_id": f"authorization-{marker}",
        "created_at_utc": "2026-08-14T12:00:00Z",
        "protocol_sha256": protocol_sha256,
        "producer_git_commit": format(marker, "x") * 40,
        "producer_source_manifest_sha256": format(marker + 8, "x") * 64,
        "launcher_sha256": format(marker + 9, "x") * 64,
        "roles": roles,
    }


def _historical_attempt(*, marker: str = "a") -> dict[str, object]:
    return {
        "attempt_id": marker * 32,
        "segment": "prepare",
        "failure_phase": "training",
        "runtime_authorization_sha256": marker * 64,
        "generation_manifest_sha256": chr(ord(marker) + 1) * 64,
        "result_sha256": chr(ord(marker) + 2) * 64,
    }


class HistoricalAuthorizationLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol_sha256 = hashlib.sha256(b"protocol").hexdigest()
        self.authorization = _authorization(self.protocol_sha256)
        self.digest = runtime_authorization_sha256(self.authorization)

    def test_loads_exact_canonical_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authorization.json"
            path.write_bytes(canonical_json_bytes(self.authorization))
            loaded = _load_historical_runtime_authorizations(
                [f"{path}={self.digest}"],
                protocol_sha256=self.protocol_sha256,
            )
        self.assertEqual(list(loaded), [self.digest])
        self.assertEqual(
            canonical_json_bytes(loaded[self.digest]),
            canonical_json_bytes(self.authorization),
        )

    def test_v3_audit_binds_non_smoke_results_to_the_full_role(self) -> None:
        authorization = {
            "schema_name": "policy_improvement_runtime_authorization_v3"
        }
        self.assertEqual(
            _producer_role_name(authorization, {"tier": "pilot"}),
            "policy-improvement-full",
        )
        self.assertEqual(
            _producer_role_name(authorization, {"tier": "smoke"}),
            "policy-improvement-training",
        )

    def test_green_audit_builds_an_executable_stage0_theory_request(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_root = root / "configs/policy_improvement_v2"
        protocol = json.loads((config_root / "protocol.json").read_text())
        registry = json.loads((config_root / "registry.json").read_text())
        populations = json.loads((config_root / "populations.json").read_text())
        amendment = json.loads(
            (config_root / "amendments/theory_bridge_v2.json").read_text()
        )
        row = next(
            item
            for item in registry["rows"]
            if item["phase"] == "stage0_smoke"
            and item["method_id"] == "fixed_base_exact_persistent"
        )
        digest = lambda value: hashlib.sha256(value.encode("ascii")).hexdigest()
        roles = []
        for index, role in enumerate(
            (
                "policy-improvement-training",
                "policy-improvement-evaluation",
                "policy-improvement-audit",
                "policy-improvement-analysis",
                "policy-improvement-full",
                "policy-improvement-theory-bridge",
            ),
            start=1,
        ):
            roles.append(
                {
                    "role": role,
                    "source_git_commit": "a" * 40,
                    "runtime_sha256": f"{index:x}" * 64,
                    "runtime_profile_sha256": f"{index + 6:x}" * 64,
                    "selected_source_manifest_sha256": f"{index + 6:x}" * 64,
                }
            )
        authorization = {
            "schema_name": "policy_improvement_runtime_authorization_v3",
            "schema_version": 3,
            "authorization_id": "stage0-theory-request-test",
            "created_at_utc": "2026-08-20T00:00:00Z",
            "protocol_sha256": hashlib.sha256(
                canonical_json_bytes(protocol)
            ).hexdigest(),
            "protocol": {
                "schema_name": protocol["schema_name"],
                "schema_version": protocol["schema_version"],
                "protocol_id": protocol["protocol_id"],
                "sha256": hashlib.sha256(
                    canonical_json_bytes(protocol)
                ).hexdigest(),
            },
            "registry": {
                "schema_name": registry["schema_name"],
                "schema_version": registry["registry_schema_version"],
                "sha256": hashlib.sha256(
                    canonical_json_bytes(registry)
                ).hexdigest(),
            },
            "amendments": [
                {
                    "schema_name": amendment["schema_name"],
                    "schema_version": amendment["schema_version"],
                    "amendment_id": amendment["amendment_id"],
                    "sha256": hashlib.sha256(
                        canonical_json_bytes(amendment)
                    ).hexdigest(),
                }
            ],
            "producer_git_commit": "a" * 40,
            "producer_source_manifest_sha256": "e" * 64,
            "launcher_sha256": "f" * 64,
            "roles": roles,
        }
        model_identity = {
            "model_sha256": digest("model"),
            "model_config_sha256": digest("model-config"),
            "current_policy_sha256": digest("current"),
            "candidate_policy_sha256": digest("candidate"),
            "deployed_policy_sha256": digest("deployed"),
            "recurrent_transition_sha256": digest("recurrent"),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"authenticated checkpoint")
            built = _audited_stage0_theory_request(
                protocol=protocol,
                registry=registry,
                row=row,
                population=populations["populations"]["stage0_smoke"],
                result_authorization=authorization,
                evaluator_authorization=authorization,
                theory_amendment=amendment,
                semantic_validation={
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "theory_model_identity": model_identity,
                },
                checkpoint_path=checkpoint,
                checkpoint_size_bytes=checkpoint.stat().st_size,
                base_config={"gamma": 0.99, "advantage_clip": 10.0},
            )
        request = built["request"]
        self.assertEqual(request["identity"]["model"], model_identity)
        self.assertEqual(request["identity"]["checkpoint"]["size_bytes"], 24)
        self.assertEqual(
            built["request_sha256"],
            hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
        )

    def test_audit_rejects_both_result_namespace_crossings(self) -> None:
        v1_protocol = {"schema_name": "policy_improvement_v1"}
        v2_protocol = {"schema_name": "policy_improvement_protocol_v2"}
        v1_result = {"schema_name": "policy_improvement_v1"}
        v2_result = {"schema_name": "policy_improvement_result_v2"}
        _require_result_namespace(v1_protocol, [v1_result])
        _require_result_namespace(v2_protocol, [v2_result])
        for protocol, result in (
            (v1_protocol, v2_result),
            (v2_protocol, v1_result),
        ):
            with self.assertRaisesRegex(
                PolicyImprovementSchemaError, "protocol namespace"
            ):
                _require_result_namespace(protocol, [result])

    def test_accepts_ascii_formatting_but_rejects_wrong_duplicate_and_foreign_documents(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.json"
            canonical_path.write_bytes(canonical_json_bytes(self.authorization))
            newline_path = root / "newline.json"
            newline_path.write_bytes(canonical_json_bytes(self.authorization) + b"\n")
            loaded = _load_historical_runtime_authorizations(
                [f"{newline_path}={self.digest}"],
                protocol_sha256=self.protocol_sha256,
            )
            self.assertEqual(list(loaded), [self.digest])
            foreign = _authorization("f" * 64, marker=2)
            foreign_path = root / "foreign.json"
            foreign_path.write_bytes(canonical_json_bytes(foreign))
            cases = (
                [f"{canonical_path}={'f' * 64}"],
                [f"{canonical_path}={self.digest}", f"{canonical_path}={self.digest}"],
                [
                    f"{foreign_path}={runtime_authorization_sha256(foreign)}",
                ],
                ["relative.json=" + self.digest],
                [str(canonical_path)],
            )
            for specifications in cases:
                with self.subTest(specifications=specifications):
                    with self.assertRaises(PolicyImprovementSchemaError):
                        _load_historical_runtime_authorizations(
                            specifications,
                            protocol_sha256=self.protocol_sha256,
                        )

    def test_map_includes_current_and_rejects_mismatched_key(self) -> None:
        historical = _authorization(self.protocol_sha256, marker=2)
        historical_digest = runtime_authorization_sha256(historical)
        with mock.patch(
            "scripts.policy_improvement_audit._authenticate_authorization_producer_source"
        ):
            checked = _validated_runtime_authorization_map(
                current_authorization=self.authorization,
                historical_runtime_authorizations={historical_digest: historical},
                protocol_sha256=self.protocol_sha256,
                producer_source_authenticator=lambda _: None,
            )
        self.assertEqual(set(checked), {self.digest, historical_digest})
        with mock.patch(
            "scripts.policy_improvement_audit._authenticate_authorization_producer_source"
        ):
            with self.assertRaisesRegex(
                PolicyImprovementSchemaError,
                "digest differs",
            ):
                _validated_runtime_authorization_map(
                    current_authorization=self.authorization,
                    historical_runtime_authorizations={"f" * 64: historical},
                    protocol_sha256=self.protocol_sha256,
                    producer_source_authenticator=lambda _: None,
                )

    def test_producer_source_requires_exact_resolved_commit(self) -> None:
        with mock.patch(
            "scripts.policy_improvement_audit._git_object_bytes",
            return_value=b"f" * 40 + b"\n",
        ):
            with self.assertRaisesRegex(
                PolicyImprovementSchemaError,
                "does not resolve exactly",
            ):
                _authenticate_authorization_producer_source(
                    self.authorization,
                    project_root=Path.cwd(),
                )

    def test_git_object_reads_ignore_replacement_refs_and_git_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*arguments: str) -> bytes:
                return subprocess.run(
                    ["git", *arguments],
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ).stdout

            git("init", "--quiet")
            git("config", "user.name", "Policy Improvement Test")
            git("config", "user.email", "policy-improvement-test@example.invalid")
            marker = root / "marker.txt"
            marker.write_bytes(b"authorized producer\n")
            git("add", "marker.txt")
            git("commit", "--quiet", "-m", "authorized producer")
            authorized_commit = git("rev-parse", "HEAD").decode("ascii").strip()

            marker.write_bytes(b"replacement producer\n")
            git("commit", "--quiet", "-am", "replacement producer")
            replacement_commit = git("rev-parse", "HEAD").decode("ascii").strip()
            git("replace", authorized_commit, replacement_commit)

            self.assertEqual(
                git("cat-file", "blob", f"{authorized_commit}:marker.txt"),
                b"replacement producer\n",
            )
            hostile_environment = {
                "GIT_NO_REPLACE_OBJECTS": "0",
                "GIT_REPLACE_REF_BASE": "refs/replace/",
                "GIT_OPTIONAL_LOCKS": "1",
            }
            with mock.patch.dict(os.environ, hostile_environment):
                self.assertEqual(
                    _git_object_bytes(
                        root,
                        ["cat-file", "blob", f"{authorized_commit}:marker.txt"],
                        label="replacement-ref regression",
                    ),
                    b"authorized producer\n",
                )


class HistoricalAttemptBindingTest(unittest.TestCase):
    def test_returns_manifest_inventory_and_accepts_empty_history(self) -> None:
        first = _historical_attempt(marker="a")
        second = _historical_attempt(marker="d")
        self.assertEqual(
            _historical_failed_attempt_manifest_sha256s(
                {"historical_failed_attempts": [first, second]}
            ),
            ["b" * 64, "e" * 64],
        )
        self.assertEqual(
            _historical_failed_attempt_manifest_sha256s(
                {"historical_failed_attempts": []}
            ),
            [],
        )

    def test_rejects_missing_malformed_and_duplicate_history(self) -> None:
        malformed = _historical_attempt(marker="a")
        malformed["generation_manifest_sha256"] = "not-a-digest"
        duplicate = _historical_attempt(marker="a")
        duplicate["attempt_id"] = "d" * 32
        for identity in (
            {},
            {"historical_failed_attempts": "not-a-list"},
            {"historical_failed_attempts": [malformed]},
            {
                "historical_failed_attempts": [
                    _historical_attempt(marker="a"),
                    duplicate,
                ]
            },
        ):
            with self.subTest(identity=identity):
                with self.assertRaises(PolicyImprovementSchemaError):
                    _historical_failed_attempt_manifest_sha256s(identity)

    def test_rejects_unregistered_attempt_fields(self) -> None:
        attempt = copy.deepcopy(_historical_attempt())
        attempt["untrusted"] = True
        with self.assertRaisesRegex(PolicyImprovementSchemaError, "fields differ"):
            _historical_failed_attempt_manifest_sha256s(
                {"historical_failed_attempts": [attempt]}
            )


if __name__ == "__main__":
    unittest.main()
