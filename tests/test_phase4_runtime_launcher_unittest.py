#!/usr/bin/env fbpython
"""Regression tests for the pre-import Phase 4 runtime boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from zipfile import ZipFile

import phase4_runtime_launcher
from confirmatory_runtime_launcher import (
    ConfirmatoryRuntimeError,
    validate_archive_layout,
    validate_runtime_archive,
)
from phase4_runtime_launcher import (
    _canonical_policy_protocol_sha256,
    _load_policy_consumer_runtime_authorization,
    _load_policy_runtime_authorization,
    _normalize_child_args,
    _training_archive_validator,
    POLICY_CONSUMER_LAUNCHER_SHA256_ENV,
    POLICY_DATASET_BUILDER_LAUNCHER_SHA256_ENV,
    POLICY_DATASET_BUILDER_PURPOSE,
    POLICY_IMPROVEMENT_ANALYSIS_PURPOSE,
    POLICY_IMPROVEMENT_AUDIT_PURPOSE,
    POLICY_IMPROVEMENT_FULL_LAUNCHER_SHA256_ENV,
    POLICY_IMPROVEMENT_FULL_PURPOSE,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_LAUNCHER_SHA256_ENV,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
    POLICY_PRODUCER_GIT_COMMIT_ENV,
    POLICY_PRODUCER_SOURCE_MANIFEST_SHA256_ENV,
    POLICY_PROTOCOL_SHA256_ENV,
    POLICY_SMOKE_LAUNCHER_SHA256_ENV,
    POLICY_SMOKE_PURPOSE,
    POLICY_SMOKE_RUNTIME_AUTHORIZATION_ENV,
    POLICY_SMOKE_RUNTIME_AUTHORIZATION_SHA256_ENV,
    POLICY_SMOKE_RUNTIME_PROFILE_SHA256_ENV,
    POLICY_SMOKE_SELECTED_SOURCE_MANIFEST_SHA256_ENV,
)
from phase4_runtime_profile import (
    assert_phase4_archive_matches_profile,
    authorize_phase4_source_profile,
    AuthorizedPhase4Profile,
    PHASE4_EVALUATOR_SOURCE_PROFILE,
    PHASE4_PROFILE_ENTRYPOINTS,
    PHASE4_ROOT_SOURCES,
    PHASE4_SHARED_SOURCES,
    Phase4RuntimeProfileError,
    POLICY_DATASET_BUILDER_PROFILE_PATHS,
    POLICY_DATASET_BUILDER_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_ANALYSIS_PROFILE_PATHS,
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_AUDIT_PROFILE_PATHS,
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_FULL_PROFILE_PATHS,
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_PROFILE_PATHS,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
)
from runtime_archive_preflight import (
    _validate_policy_runtime_authorization,
    POLICY_FULL_ROLE,
    POLICY_THEORY_BRIDGE_ROLE,
    preflight_runtime,
)


def _run_git(root: Path, *arguments: str) -> str:
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_AUTHOR_NAME": "Phase4 Test",
            "GIT_AUTHOR_EMAIL": "phase4@example.invalid",
            "GIT_COMMITTER_NAME": "Phase4 Test",
            "GIT_COMMITTER_EMAIL": "phase4@example.invalid",
        }
    )
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    return completed.stdout.strip()


def _profile_sources() -> dict[str, bytes]:
    paths = {
        *PHASE4_ROOT_SOURCES,
        *PHASE4_SHARED_SOURCES,
        *PHASE4_PROFILE_ENTRYPOINTS[PHASE4_EVALUATOR_SOURCE_PROFILE],
        "dataset/__init__.py",
        "dataset/common.py",
        "models/model.py",
        "rl/trainer.py",
        "utils/identity.py",
    }
    return {
        relative_path: f"# {relative_path}\n".encode("ascii")
        for relative_path in sorted(paths)
    }


_POLICY_TRAINING_TEST_PATHS = (
    "confirmatory_runtime_launcher.py",
    "phase4_runtime_profile.py",
    "policy_improvement_checkpoint_allowlist.py",
    "policy_improvement_smoke_checkpoint.py",
    "policy_improvement_smoke_runtime.py",
    "puzzle_dataset.py",
    "runtime_archive_preflight.py",
    "scripts/policy_improvement_registry.py",
    "scripts/policy_improvement_schema.py",
    "upi_trm_train.py",
    "configs/iclr_confirmatory/cell.yaml",
    "dataset/source.py",
    "evaluators/source.py",
    "models/source.py",
    "rl/source.py",
    "utils/source.py",
)

_POLICY_IMPROVEMENT_V2_RUNTIME_PATHS = frozenset(
    {
        "configs/policy_improvement_v2/amendments/theory_bridge_v2.json",
        "configs/policy_improvement_v2/fixed_base_exact_episodic.yaml",
        "configs/policy_improvement_v2/fixed_base_exact_persistent.yaml",
        "configs/policy_improvement_v2/legacy_parameter_interpolation.yaml",
        "configs/policy_improvement_v2/matched_ppo.yaml",
        "configs/policy_improvement_v2/populations.json",
        "configs/policy_improvement_v2/protocol.json",
        "configs/policy_improvement_v2/registry.json",
        "scripts/policy_improvement_populations.py",
        "scripts/policy_improvement_v2_registry.py",
        "scripts/policy_improvement_v2_schema.py",
    }
)


def _policy_consumer_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({*paths, *_POLICY_TRAINING_TEST_PATHS}))


def _authorized(
    sources: dict[str, bytes],
    profile: str = PHASE4_EVALUATOR_SOURCE_PROFILE,
) -> AuthorizedPhase4Profile:
    return AuthorizedPhase4Profile(
        git_commit="a" * 40,
        source_manifest_sha256="b" * 64,
        profile=profile,
        sources={
            relative_path: hashlib.sha256(payload).hexdigest()
            for relative_path, payload in sources.items()
        },
    )


def _policy_authorization_v2() -> dict[str, object]:
    role_values = {
        "policy-improvement-training": ("a" * 40, "d" * 64, "b" * 64),
        "policy-improvement-evaluation": ("e" * 40, "9" * 64, "f" * 64),
        "policy-improvement-audit": ("e" * 40, "7" * 64, "6" * 64),
        "policy-improvement-analysis": ("e" * 40, "8" * 64, "5" * 64),
        "policy-improvement-full": ("a" * 40, "d" * 64, "b" * 64),
        "policy-improvement-theory-bridge": ("e" * 40, "9" * 64, "f" * 64),
    }
    return {
        "schema_name": "policy_improvement_runtime_authorization_v2",
        "schema_version": 2,
        "authorization_id": "strict-launcher-test-v2",
        "created_at_utc": "2026-08-16T12:00:00Z",
        "protocol_sha256": "1" * 64,
        "producer_git_commit": "a" * 40,
        "producer_source_manifest_sha256": "c" * 64,
        "launcher_sha256": "2" * 64,
        "roles": [
            {
                "role": role,
                "source_git_commit": role_values[role][0],
                "runtime_sha256": role_values[role][1],
                "runtime_profile_sha256": role_values[role][2],
                "selected_source_manifest_sha256": role_values[role][2],
            }
            for role in (
                "policy-improvement-training",
                "policy-improvement-evaluation",
                "policy-improvement-audit",
                "policy-improvement-analysis",
                "policy-improvement-full",
                "policy-improvement-theory-bridge",
            )
        ],
    }


class Phase4RuntimeLauncherTest(unittest.TestCase):
    def test_policy_smoke_authorization_is_strict_and_stable(self) -> None:
        authorization = {
            "schema_name": "policy_improvement_runtime_authorization_v1",
            "schema_version": 1,
            "authorization_id": "strict-launcher-test-v1",
            "created_at_utc": "2026-08-14T12:00:00Z",
            "protocol_sha256": "f" * 64,
            "producer_git_commit": "a" * 40,
            "producer_source_manifest_sha256": "c" * 64,
            "launcher_sha256": "e" * 64,
            "roles": [
                {
                    "role": role,
                    "source_git_commit": "a" * 40,
                    "runtime_sha256": "d" * 64,
                    "runtime_profile_sha256": "c" * 64,
                    "selected_source_manifest_sha256": "c" * 64,
                }
                for role in (
                    "policy-improvement-training",
                    "policy-improvement-evaluation",
                    "policy-improvement-audit",
                    "policy-improvement-analysis",
                )
            ],
        }
        payload = json.dumps(
            authorization,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "authorization.json"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            canonical, profile, selected = _load_policy_runtime_authorization(
                str(path),
                digest,
                runtime_sha256="d" * 64,
                source_git_commit="a" * 40,
                source_manifest_sha256="c" * 64,
                launcher_sha256="e" * 64,
                protocol_sha256="f" * 64,
            )
            self.assertEqual(json.loads(canonical), authorization)
            self.assertEqual(profile, "c" * 64)
            self.assertEqual(selected, "c" * 64)

            with self.assertRaisesRegex(ConfirmatoryRuntimeError, "protocol"):
                _load_policy_runtime_authorization(
                    str(path),
                    digest,
                    runtime_sha256="d" * 64,
                    source_git_commit="a" * 40,
                    source_manifest_sha256="c" * 64,
                    launcher_sha256="e" * 64,
                    protocol_sha256="0" * 64,
                )

            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "digest differs",
            ):
                _load_policy_runtime_authorization(
                    str(path),
                    "0" * 64,
                    runtime_sha256="d" * 64,
                    source_git_commit="a" * 40,
                    source_manifest_sha256="c" * 64,
                    launcher_sha256="e" * 64,
                    protocol_sha256="f" * 64,
                )

            alias = root / "authorization-alias.json"
            alias.symlink_to(path)
            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "opened safely",
            ):
                _load_policy_runtime_authorization(
                    str(alias),
                    digest,
                    runtime_sha256="d" * 64,
                    source_git_commit="a" * 40,
                    source_manifest_sha256="c" * 64,
                    launcher_sha256="e" * 64,
                    protocol_sha256="f" * 64,
                )

            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schema_name":"a","schema_name":"b"}',
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "duplicate JSON key",
            ):
                _load_policy_runtime_authorization(
                    str(duplicate),
                    hashlib.sha256(duplicate.read_bytes()).hexdigest(),
                    runtime_sha256="d" * 64,
                    source_git_commit="a" * 40,
                    source_manifest_sha256="c" * 64,
                    launcher_sha256="e" * 64,
                    protocol_sha256="f" * 64,
                )

    def test_v2_full_and_theory_authorization_bind_semantic_roles(self) -> None:
        authorization = _policy_authorization_v2()
        payload = json.dumps(
            authorization,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authorization-v2.json"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            for purpose, runtime_sha256, source_commit, profile in (
                (
                    POLICY_IMPROVEMENT_FULL_PURPOSE,
                    "d" * 64,
                    "a" * 40,
                    "b" * 64,
                ),
                (
                    POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
                    "9" * 64,
                    "e" * 40,
                    "f" * 64,
                ),
            ):
                with self.subTest(purpose=purpose):
                    canonical = _load_policy_consumer_runtime_authorization(
                        str(path),
                        digest,
                        purpose=purpose,
                        runtime_sha256=runtime_sha256,
                        source_git_commit=source_commit,
                        source_manifest_sha256=profile,
                        launcher_sha256="2" * 64,
                        producer_git_commit="a" * 40,
                        producer_source_manifest_sha256="c" * 64,
                        protocol_sha256="1" * 64,
                    )
                    self.assertEqual(json.loads(canonical), authorization)

            with self.assertRaisesRegex(ConfirmatoryRuntimeError, "protocol"):
                _load_policy_consumer_runtime_authorization(
                    str(path),
                    digest,
                    purpose=POLICY_IMPROVEMENT_FULL_PURPOSE,
                    runtime_sha256="d" * 64,
                    source_git_commit="a" * 40,
                    source_manifest_sha256="b" * 64,
                    launcher_sha256="2" * 64,
                    producer_git_commit="a" * 40,
                    producer_source_manifest_sha256="c" * 64,
                    protocol_sha256="0" * 64,
                )

            mixed = json.loads(payload)
            mixed["roles"][4]["runtime_sha256"] = "0" * 64
            mixed_payload = json.dumps(
                mixed,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            mixed_path = Path(directory) / "mixed.json"
            mixed_path.write_bytes(mixed_payload)
            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "semantic and launcher roles differ",
            ):
                _load_policy_consumer_runtime_authorization(
                    str(mixed_path),
                    hashlib.sha256(mixed_payload).hexdigest(),
                    purpose=POLICY_IMPROVEMENT_FULL_PURPOSE,
                    runtime_sha256="0" * 64,
                    source_git_commit="a" * 40,
                    source_manifest_sha256="b" * 64,
                    launcher_sha256="2" * 64,
                    producer_git_commit="a" * 40,
                    producer_source_manifest_sha256="c" * 64,
                    protocol_sha256="1" * 64,
                )

            legacy = copy.deepcopy(authorization)
            legacy["schema_name"] = "policy_improvement_runtime_authorization_v1"
            legacy["schema_version"] = 1
            legacy["roles"] = legacy["roles"][:4]
            legacy_payload = json.dumps(
                legacy,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            legacy_path = Path(directory) / "legacy.json"
            legacy_path.write_bytes(legacy_payload)
            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "require runtime authorization v2",
            ):
                _load_policy_consumer_runtime_authorization(
                    str(legacy_path),
                    hashlib.sha256(legacy_payload).hexdigest(),
                    purpose=POLICY_IMPROVEMENT_FULL_PURPOSE,
                    runtime_sha256="d" * 64,
                    source_git_commit="a" * 40,
                    source_manifest_sha256="b" * 64,
                    launcher_sha256="2" * 64,
                    producer_git_commit="a" * 40,
                    producer_source_manifest_sha256="c" * 64,
                    protocol_sha256="1" * 64,
                )

    def test_policy_protocol_digest_is_canonical_and_source_bound(self) -> None:
        protocol = {"protocol_id": "launcher-test", "schema_version": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "configs/policy_improvement_v1/protocol.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(protocol, indent=2, sort_keys=False) + "\n",
                encoding="ascii",
            )
            self.assertEqual(
                _canonical_policy_protocol_sha256(str(root)),
                hashlib.sha256(
                    json.dumps(
                        protocol,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                ).hexdigest(),
            )

    def test_v2_preflight_rejects_mixed_full_and_theory_roles(self) -> None:
        authorization = _policy_authorization_v2()
        _validate_policy_runtime_authorization(
            authorization,
            phase4_role=POLICY_FULL_ROLE,
            runtime_sha256="d" * 64,
            source_git_commit="a" * 40,
            source_manifest_sha256="b" * 64,
            protocol_sha256="1" * 64,
        )
        _validate_policy_runtime_authorization(
            authorization,
            phase4_role=POLICY_THEORY_BRIDGE_ROLE,
            runtime_sha256="9" * 64,
            source_git_commit="e" * 40,
            source_manifest_sha256="f" * 64,
            protocol_sha256="1" * 64,
        )
        mixed = copy.deepcopy(authorization)
        mixed["roles"][5]["selected_source_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "authorization is invalid"):
            _validate_policy_runtime_authorization(
                mixed,
                phase4_role=POLICY_THEORY_BRIDGE_ROLE,
                runtime_sha256="9" * 64,
                source_git_commit="e" * 40,
                source_manifest_sha256="f" * 64,
                protocol_sha256="1" * 64,
            )
        with self.assertRaisesRegex(RuntimeError, "authorization is invalid"):
            _validate_policy_runtime_authorization(
                authorization,
                phase4_role=POLICY_FULL_ROLE,
                runtime_sha256="d" * 64,
                source_git_commit="a" * 40,
                source_manifest_sha256="b" * 64,
                protocol_sha256="0" * 64,
            )

    def test_dataset_builder_launcher_owns_entrypoint_and_identity(self) -> None:
        self.assertEqual(
            _normalize_child_args(
                POLICY_DATASET_BUILDER_PURPOSE,
                "/repo",
                [
                    "--",
                    "build",
                    "--owner-root",
                    "/data",
                    "--output-root",
                    "/data/frozen",
                ],
            ),
            [
                "--policy-dataset-builder-entrypoint",
                "build",
                "--owner-root",
                "/data",
                "--output-root",
                "/data/frozen",
            ],
        )
        self.assertEqual(
            _normalize_child_args(
                POLICY_DATASET_BUILDER_PURPOSE,
                "/repo",
                [
                    "verify",
                    "--owner-root",
                    "/data",
                    "--root",
                    "/data/frozen",
                ],
            ),
            [
                "--policy-dataset-builder-entrypoint",
                "verify",
                "--owner-root",
                "/data",
                "--root",
                "/data/frozen",
            ],
        )
        for child_args in (
            [
                "build",
                "--owner-root",
                "/data",
                "--output-root",
                "/data/frozen",
                "--source-root",
                "/repo",
            ],
            ["build", "--expected-source-commit", "a" * 40],
            [
                "build",
                "--owner-root",
                "/data",
                "--output-root",
                "--source-project-root",
            ],
            [
                "build",
                "--owner-root",
                "relative",
                "--output-root",
                "/data/frozen",
            ],
            ["--policy-dataset-builder-entrypoint"],
            [
                "verify",
                "--owner-root",
                "/data",
                "--output-root",
                "/data/frozen",
            ],
        ):
            with (
                self.subTest(child_args=child_args),
                self.assertRaises(ConfirmatoryRuntimeError),
            ):
                _normalize_child_args(
                    POLICY_DATASET_BUILDER_PURPOSE,
                    "/repo",
                    child_args,
                )

    def test_policy_smoke_launcher_owns_entrypoint_and_source_flags(self) -> None:
        child_args = _normalize_child_args(
            POLICY_SMOKE_PURPOSE,
            "/repo",
            [
                "--",
                "--policy-improvement-protocol",
                "/repo/configs/policy_improvement_v1/protocol.json",
                "--policy-improvement-row-id",
                "s0-matched-ppo-s1",
                "--policy-improvement-smoke-segment",
                "prepare",
                "--dataset-root",
                "/data/frozen",
                "--train-manifest-sha256",
                "a" * 64,
                "--validation-manifest-sha256",
                "b" * 64,
                "--evidence-root",
                "/evidence",
            ],
        )
        self.assertEqual(
            child_args[:3],
            [
                "--policy-improvement-smoke-entrypoint",
                "--source-project-root",
                "/repo",
            ],
        )
        for protected in (
            "--confirmatory",
            "--policy-improvement-smoke-entrypoint",
            "--source-project-root=/other",
            "--resume-checkpoint=/tmp/forged.pt",
            "--config=/tmp/unregistered.yaml",
        ):
            with (
                self.subTest(protected=protected),
                self.assertRaises(ConfirmatoryRuntimeError),
            ):
                _normalize_child_args(
                    POLICY_SMOKE_PURPOSE,
                    "/repo",
                    [protected],
                )
        self.assertEqual(
            _normalize_child_args(POLICY_SMOKE_PURPOSE, "/repo", ["--help"]),
            [
                "--policy-improvement-smoke-entrypoint",
                "--source-project-root",
                "/repo",
                "--help",
            ],
        )

    def test_policy_consumers_own_runtime_and_source_identity_flags(self) -> None:
        for purpose in (
            POLICY_IMPROVEMENT_AUDIT_PURPOSE,
            POLICY_IMPROVEMENT_ANALYSIS_PURPOSE,
        ):
            with self.subTest(purpose=purpose):
                normalized = _normalize_child_args(
                    purpose,
                    "/repo",
                    [
                        "--",
                        "--protocol",
                        "/evidence/protocol.json",
                    ],
                )
                self.assertEqual(
                    normalized[:3],
                    [
                        "--policy-improvement-consumer-entrypoint",
                        "--protocol",
                        "/evidence/protocol.json",
                    ],
                )
                self.assertEqual(
                    _normalize_child_args(purpose, "/repo", ["--help"]),
                    ["--policy-improvement-consumer-entrypoint", "--help"],
                )
                for protected in (
                    "--policy-improvement-consumer-entrypoint",
                    "--launcher-sha256=" + "a" * 64,
                    "--launcher-sha=" + "a" * 64,
                    "--producer-git-commit=" + "a" * 40,
                    "--producer-source-manifest-sha256=" + "a" * 64,
                    "--audit-runtime-sha256=" + "a" * 64,
                    "--audit-source-git-commit=" + "a" * 40,
                    "--analysis-runtime-profile-sha256=" + "a" * 64,
                    "--runtime-authorization=/forged.json",
                    "--runtime-authorization-json={}",
                    "--runtime-authorization-sha256=" + "a" * 64,
                    "--source-project-root=/other",
                ):
                    with (
                        self.subTest(
                            purpose=purpose,
                            protected=protected,
                        ),
                        self.assertRaisesRegex(
                            ConfirmatoryRuntimeError,
                            "launcher-owned",
                        ),
                    ):
                        _normalize_child_args(
                            purpose,
                            "/repo",
                            [protected],
                        )

    def test_full_and_theory_launchers_own_registered_inputs(self) -> None:
        theory_amendment = (
            "/producer/configs/policy_improvement_v1/amendments/"
            "theory_bridge_v1.json"
        )
        full = _normalize_child_args(
            POLICY_IMPROVEMENT_FULL_PURPOSE,
            "/producer",
            [
                "--evidence-root",
                "/evidence",
                "--dataset-root",
                "/dataset",
                "--row-id",
                "stage1-row",
                "--amendment",
                "/evidence/compute-freeze.json",
                "--print-contract",
            ],
            runtime_authorization_sha256="4" * 64,
        )
        self.assertEqual(
            full,
            [
                "--project-root",
                "/producer",
                "--protocol",
                "/producer/configs/policy_improvement_v1/protocol.json",
                "--registry",
                "/producer/configs/policy_improvement_v1/registry.json",
                "--amendment",
                theory_amendment,
                "--amendment",
                "/evidence/compute-freeze.json",
                "--evidence-root",
                "/evidence",
                "--dataset-root",
                "/dataset",
                "--row-id",
                "stage1-row",
                "--runtime-authorization-sha256",
                "4" * 64,
                "--print-contract",
            ],
        )

        theory = _normalize_child_args(
            POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
            "/evaluator",
            [
                "--request=/evidence/request.json",
                "--checkpoint",
                "/evidence/checkpoint.pt",
                "--evidence-root",
                "/evidence",
                "--dataset-root",
                "/dataset",
                "--row-id",
                "stage1-row",
                "--amendment",
                "/evidence/compute-freeze.json",
            ],
            policy_project_root="/producer",
        )
        self.assertEqual(
            theory,
            [
                "--policy-improvement-theory-bridge-entrypoint",
                "--request",
                "/evidence/request.json",
                "--theory-amendment",
                theory_amendment,
                "--amendment",
                theory_amendment,
                "--amendment",
                "/evidence/compute-freeze.json",
                "--checkpoint",
                "/evidence/checkpoint.pt",
                "--project-root",
                "/producer",
                "--protocol",
                "/producer/configs/policy_improvement_v1/protocol.json",
                "--registry",
                "/producer/configs/policy_improvement_v1/registry.json",
                "--evidence-root",
                "/evidence",
                "--dataset-root",
                "/dataset",
                "--row-id",
                "stage1-row",
            ],
        )

        for purpose, option in (
            (POLICY_IMPROVEMENT_FULL_PURPOSE, "--project-root=/forged"),
            (POLICY_IMPROVEMENT_FULL_PURPOSE, "--protocol=/forged"),
            (
                POLICY_IMPROVEMENT_FULL_PURPOSE,
                "--runtime-authorization-sha256=" + "0" * 64,
            ),
            (
                POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
                "--policy-improvement-theory-bridge-entrypoint",
            ),
            (
                POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
                "--theory-amendment=/forged",
            ),
        ):
            with (
                self.subTest(purpose=purpose, option=option),
                self.assertRaisesRegex(
                    ConfirmatoryRuntimeError,
                    "protected or unsupported",
                ),
            ):
                _normalize_child_args(
                    purpose,
                    "/evaluator",
                    [option],
                    policy_project_root="/producer",
                    runtime_authorization_sha256="4" * 64,
                )
        with self.assertRaisesRegex(ConfirmatoryRuntimeError, "must be absolute"):
            _normalize_child_args(
                POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
                "/evaluator",
                [
                    "--request",
                    "relative.json",
                    "--checkpoint",
                    "/evidence/checkpoint.pt",
                    "--evidence-root",
                    "/evidence",
                    "--dataset-root",
                    "/dataset",
                    "--row-id",
                    "stage1-row",
                ],
                policy_project_root="/producer",
            )

    def test_training_launcher_owns_publication_and_producer_flags(self) -> None:
        child_args = _normalize_child_args(
            "phase4-training",
            "/repo",
            ["--", "--run-id", "phase4_2x2_norm_ablation.nc_nv.seed41"],
        )
        self.assertEqual(
            child_args[:3],
            ["--phase4-publication", "--producer-repo-root", "/repo"],
        )
        with self.assertRaisesRegex(
            ConfirmatoryRuntimeError,
            "owns its publication and producer flags",
        ):
            _normalize_child_args(
                "phase4-training",
                "/repo",
                ["--producer-repo-root", "/other"],
            )
        with self.assertRaisesRegex(
            ConfirmatoryRuntimeError,
            "owns its publication and producer flags",
        ):
            _normalize_child_args(
                "phase4-training",
                "/repo",
                ["--producer-repo-root=/other"],
            )
        with self.assertRaisesRegex(
            ConfirmatoryRuntimeError,
            "cannot use confirmatory mode",
        ):
            _normalize_child_args(
                "phase4-evaluator",
                "/repo",
                ["--confirmatory"],
            )

    def test_training_archive_requires_the_authorized_manifest_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "training.par"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "configs/iclr_confirmatory/producer_source_manifest.json",
                    b"different",
                )
            validator = _training_archive_validator(b"authorized")
            with (
                ZipFile(archive_path) as archive,
                mock.patch.object(
                    phase4_runtime_launcher,
                    "validate_confirmatory_archive_sources",
                ),
                self.assertRaisesRegex(
                    ConfirmatoryRuntimeError,
                    "differs from the authorized checkout",
                ),
            ):
                validator(archive)

    def test_policy_consumer_requires_distinct_complete_producer_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = str(Path(directory).resolve())
            base = [
                "--runtime-archive",
                "/runtime.par",
                "--expected-runtime-sha256",
                "d" * 64,
                "--source-project-root",
                root,
                "--expected-source-git-commit",
                "a" * 40,
                "--",
                "--help",
            ]
            self.assertEqual(
                phase4_runtime_launcher.main(
                    ["--purpose", POLICY_IMPROVEMENT_AUDIT_PURPOSE, *base]
                ),
                2,
            )
            self.assertEqual(
                phase4_runtime_launcher.main(
                    [
                        "--purpose",
                        "phase4-audit",
                        "--producer-source-project-root",
                        root,
                        "--expected-producer-git-commit",
                        "a" * 40,
                        *base,
                    ]
                ),
                2,
            )

    def test_main_dispatches_every_role_with_bound_attestation(self) -> None:
        runtime = object()
        consumer_authorized = SimpleNamespace(
            git_commit="a" * 40,
            source_manifest_sha256="b" * 64,
        )
        training_authorized = SimpleNamespace(
            git_commit="a" * 40,
            source_manifest_sha256="c" * 64,
            manifest_bytes=b"manifest",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            producer_root = root / "producer"
            producer_root.mkdir()
            authorization = {
                "schema_name": "policy_improvement_runtime_authorization_v1",
                "schema_version": 1,
                "authorization_id": "launcher-test-v1",
                "created_at_utc": "2026-08-14T12:00:00Z",
                "protocol_sha256": "f" * 64,
                "producer_git_commit": "a" * 40,
                "producer_source_manifest_sha256": "c" * 64,
                "launcher_sha256": "e" * 64,
                "roles": [
                    {
                        "role": role,
                        "source_git_commit": "a" * 40,
                        "runtime_sha256": "d" * 64,
                        "runtime_profile_sha256": (
                            "c" * 64
                            if role
                            in {
                                "policy-improvement-training",
                                "policy-improvement-evaluation",
                            }
                            else "b" * 64
                        ),
                        "selected_source_manifest_sha256": (
                            "c" * 64
                            if role
                            in {
                                "policy-improvement-training",
                                "policy-improvement-evaluation",
                            }
                            else "b" * 64
                        ),
                    }
                    for role in (
                        "policy-improvement-training",
                        "policy-improvement-evaluation",
                        "policy-improvement-audit",
                        "policy-improvement-analysis",
                    )
                ],
            }
            authorization_payload = json.dumps(
                authorization,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            authorization_path = root / "runtime-authorization.json"
            authorization_path.write_bytes(authorization_payload)
            authorization_sha256 = hashlib.sha256(authorization_payload).hexdigest()
            for purpose, expected_role in (
                ("phase4-training", "training"),
                (POLICY_SMOKE_PURPOSE, POLICY_SMOKE_PURPOSE),
                (
                    POLICY_DATASET_BUILDER_PURPOSE,
                    POLICY_DATASET_BUILDER_SOURCE_PROFILE,
                ),
                (
                    POLICY_IMPROVEMENT_AUDIT_PURPOSE,
                    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
                ),
                (
                    POLICY_IMPROVEMENT_ANALYSIS_PURPOSE,
                    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
                ),
                ("phase4-evaluator", "evaluator"),
                ("phase4-audit", "audit"),
                ("phase4-figure", "figure"),
            ):
                with (
                    self.subTest(purpose=purpose),
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "authorize_phase4_training_source",
                        return_value=training_authorized,
                    ),
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "authorize_phase4_source_profile",
                        return_value=consumer_authorized,
                    ),
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "validate_runtime_archive",
                        return_value=runtime,
                    ),
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "launch_verified_runtime",
                        return_value=0,
                    ) as launch,
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "_launcher_artifact_sha256",
                        return_value="e" * 64,
                    ),
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "_canonical_policy_protocol_sha256",
                        return_value="f" * 64,
                    ),
                ):
                    launcher_arguments = [
                        "--purpose",
                        purpose,
                        "--runtime-archive",
                        "/runtime.par",
                        "--expected-runtime-sha256",
                        "d" * 64,
                        "--source-project-root",
                        str(root),
                        "--expected-source-git-commit",
                        "a" * 40,
                    ]
                    if purpose in {
                        POLICY_IMPROVEMENT_AUDIT_PURPOSE,
                        POLICY_IMPROVEMENT_ANALYSIS_PURPOSE,
                    }:
                        launcher_arguments.extend(
                            [
                                "--producer-source-project-root",
                                str(producer_root),
                                "--expected-producer-git-commit",
                                "a" * 40,
                            ]
                        )
                    if purpose == POLICY_SMOKE_PURPOSE or purpose in {
                        POLICY_IMPROVEMENT_AUDIT_PURPOSE,
                        POLICY_IMPROVEMENT_ANALYSIS_PURPOSE,
                    }:
                        launcher_arguments.extend(
                            [
                                "--runtime-authorization",
                                str(authorization_path),
                                "--expected-runtime-authorization-sha256",
                                authorization_sha256,
                            ]
                        )
                    launcher_arguments.extend(["--", "--help"])
                    self.assertEqual(
                        phase4_runtime_launcher.main(launcher_arguments),
                        0,
                    )
                    attestation = launch.call_args.kwargs["attestation_environment"]
                    self.assertEqual(
                        attestation[phase4_runtime_launcher.PHASE4_RUNTIME_ROLE_ENV],
                        expected_role,
                    )
                    self.assertEqual(
                        attestation[phase4_runtime_launcher.PHASE4_SOURCE_COMMIT_ENV],
                        "a" * 40,
                    )
                    if purpose == "phase4-training":
                        child_args = launch.call_args.args[1]
                        self.assertEqual(
                            child_args[:3],
                            [
                                "--phase4-publication",
                                "--producer-repo-root",
                                str(root),
                            ],
                        )
                    if purpose == POLICY_SMOKE_PURPOSE:
                        self.assertEqual(
                            attestation[POLICY_SMOKE_RUNTIME_AUTHORIZATION_SHA256_ENV],
                            authorization_sha256,
                        )
                        self.assertEqual(
                            attestation[POLICY_SMOKE_RUNTIME_PROFILE_SHA256_ENV],
                            "c" * 64,
                        )
                        self.assertEqual(
                            attestation[
                                POLICY_SMOKE_SELECTED_SOURCE_MANIFEST_SHA256_ENV
                            ],
                            "c" * 64,
                        )
                        self.assertEqual(
                            json.loads(
                                attestation[POLICY_SMOKE_RUNTIME_AUTHORIZATION_ENV]
                            ),
                            authorization,
                        )
                        self.assertEqual(
                            attestation[POLICY_SMOKE_LAUNCHER_SHA256_ENV],
                            "e" * 64,
                        )
                        self.assertEqual(
                            attestation[POLICY_PROTOCOL_SHA256_ENV],
                            "f" * 64,
                        )
                        child_args = launch.call_args.args[1]
                        self.assertEqual(
                            child_args[:3],
                            [
                                "--policy-improvement-smoke-entrypoint",
                                "--source-project-root",
                                str(root),
                            ],
                        )
                    if purpose == POLICY_DATASET_BUILDER_PURPOSE:
                        self.assertEqual(
                            attestation[POLICY_DATASET_BUILDER_LAUNCHER_SHA256_ENV],
                            "e" * 64,
                        )
                        self.assertEqual(
                            launch.call_args.args[1],
                            [
                                "--policy-dataset-builder-entrypoint",
                                "--help",
                            ],
                        )
                    if purpose in {
                        POLICY_IMPROVEMENT_AUDIT_PURPOSE,
                        POLICY_IMPROVEMENT_ANALYSIS_PURPOSE,
                    }:
                        self.assertEqual(
                            attestation[POLICY_CONSUMER_LAUNCHER_SHA256_ENV],
                            "e" * 64,
                        )
                        self.assertEqual(
                            attestation[POLICY_PRODUCER_SOURCE_MANIFEST_SHA256_ENV],
                            "c" * 64,
                        )
                        self.assertEqual(
                            attestation[POLICY_PRODUCER_GIT_COMMIT_ENV],
                            "a" * 40,
                        )
                        self.assertEqual(
                            attestation[
                                phase4_runtime_launcher.POLICY_CONSUMER_RUNTIME_AUTHORIZATION_SHA256_ENV
                            ],
                            authorization_sha256,
                        )
                        self.assertEqual(
                            json.loads(
                                attestation[
                                    phase4_runtime_launcher.POLICY_CONSUMER_RUNTIME_AUTHORIZATION_ENV
                                ]
                            ),
                            authorization,
                        )
                        self.assertEqual(
                            launch.call_args.args[1],
                            [
                                "--policy-improvement-consumer-entrypoint",
                                "--help",
                            ],
                        )

    def test_main_dispatches_v2_full_and_theory_roles(self) -> None:
        runtime = object()
        authorization = _policy_authorization_v2()
        authorization_payload = json.dumps(
            authorization,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            producer_root = root / "producer"
            evaluator_root = root / "evaluator"
            producer_root.mkdir()
            evaluator_root.mkdir()
            authorization_path = root / "runtime-authorization-v2.json"
            authorization_path.write_bytes(authorization_payload)
            authorization_sha256 = hashlib.sha256(authorization_payload).hexdigest()
            producer = SimpleNamespace(
                git_commit="a" * 40,
                source_manifest_sha256="c" * 64,
                manifest_bytes=b"manifest",
            )
            for purpose, source_root, source_commit, runtime_sha256, profile in (
                (
                    POLICY_IMPROVEMENT_FULL_PURPOSE,
                    producer_root,
                    "a" * 40,
                    "d" * 64,
                    "b" * 64,
                ),
                (
                    POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE,
                    evaluator_root,
                    "e" * 40,
                    "9" * 64,
                    "f" * 64,
                ),
            ):
                consumer = SimpleNamespace(
                    git_commit=source_commit,
                    source_manifest_sha256=profile,
                )
                with (
                    self.subTest(purpose=purpose),
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "authorize_phase4_training_source",
                        return_value=producer,
                    ) as authorize_producer,
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "authorize_phase4_source_profile",
                        return_value=consumer,
                    ),
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "validate_runtime_archive",
                        return_value=runtime,
                    ),
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "launch_verified_runtime",
                        return_value=0,
                    ) as launch,
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "_launcher_artifact_sha256",
                        return_value="2" * 64,
                    ),
                    mock.patch.object(
                        phase4_runtime_launcher,
                        "_canonical_policy_protocol_sha256",
                        return_value="1" * 64,
                    ) as protocol_digest,
                ):
                    launcher_arguments = [
                        "--purpose",
                        purpose,
                        "--runtime-archive",
                        "/runtime.par",
                        "--expected-runtime-sha256",
                        runtime_sha256,
                        "--source-project-root",
                        str(source_root),
                        "--expected-source-git-commit",
                        source_commit,
                        "--runtime-authorization",
                        str(authorization_path),
                        "--expected-runtime-authorization-sha256",
                        authorization_sha256,
                    ]
                    if purpose == POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE:
                        launcher_arguments.extend(
                            [
                                "--producer-source-project-root",
                                str(producer_root),
                                "--expected-producer-git-commit",
                                "a" * 40,
                            ]
                        )
                    runtime_arguments = [
                        "--evidence-root",
                        "/evidence",
                        "--dataset-root",
                        "/dataset",
                        "--row-id",
                        "stage1-row",
                    ]
                    if purpose == POLICY_IMPROVEMENT_THEORY_BRIDGE_PURPOSE:
                        runtime_arguments.extend(
                            [
                                "--request",
                                "/evidence/request.json",
                                "--checkpoint",
                                "/evidence/checkpoint.pt",
                            ]
                        )
                    launcher_arguments.extend(["--", *runtime_arguments])
                    self.assertEqual(
                        phase4_runtime_launcher.main(launcher_arguments),
                        0,
                    )

                    protocol_digest.assert_called_once_with(str(producer_root))
                    authorize_producer.assert_called_once_with(
                        str(producer_root),
                        "a" * 40,
                    )
                    attestation = launch.call_args.kwargs["attestation_environment"]
                    self.assertEqual(
                        attestation[phase4_runtime_launcher.PHASE4_RUNTIME_ROLE_ENV],
                        purpose,
                    )
                    self.assertEqual(
                        attestation[POLICY_SMOKE_RUNTIME_AUTHORIZATION_SHA256_ENV],
                        authorization_sha256,
                    )
                    self.assertEqual(
                        json.loads(attestation[POLICY_SMOKE_RUNTIME_AUTHORIZATION_ENV]),
                        authorization,
                    )
                    self.assertEqual(
                        attestation[POLICY_PROTOCOL_SHA256_ENV],
                        "1" * 64,
                    )
                    launcher_env = (
                        POLICY_IMPROVEMENT_FULL_LAUNCHER_SHA256_ENV
                        if purpose == POLICY_IMPROVEMENT_FULL_PURPOSE
                        else POLICY_IMPROVEMENT_THEORY_BRIDGE_LAUNCHER_SHA256_ENV
                    )
                    self.assertEqual(attestation[launcher_env], "2" * 64)
                    child_args = launch.call_args.args[1]
                    self.assertIn(
                        str(
                            producer_root / "configs/policy_improvement_v1/amendments/"
                            "theory_bridge_v1.json"
                        ),
                        child_args,
                    )
                    if purpose == POLICY_IMPROVEMENT_FULL_PURPOSE:
                        self.assertIn(
                            "--runtime-authorization-sha256",
                            child_args,
                        )
                    else:
                        self.assertEqual(
                            child_args[0],
                            "--policy-improvement-theory-bridge-entrypoint",
                        )

    def test_direct_phase4_entrypoint_has_no_attestation(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "verified packaged-runtime launcher",
        ):
            preflight_runtime(
                module_file=__file__,
                expected_module_name="phase4_runtime_entrypoint.py",
                environ={},
                attestation_required=True,
                allowed_phase4_roles=frozenset({"evaluator"}),
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "verified packaged-runtime launcher",
        ):
            preflight_runtime(
                module_file=__file__,
                expected_module_name="policy_improvement_consumer_entrypoint.py",
                environ={},
                attestation_required=True,
                allowed_phase4_roles=frozenset(
                    {
                        POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
                        POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
                    }
                ),
            )

    def test_archive_profile_requires_exact_source_bytes_and_no_bytecode(
        self,
    ) -> None:
        sources = _profile_sources()
        authorized = _authorized(sources)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_archive = root / "valid.par"
            with ZipFile(valid_archive, "w") as archive:
                for relative_path, payload in sources.items():
                    archive.writestr(relative_path, payload)
            with ZipFile(valid_archive, "r") as archive:
                validate_archive_layout(archive)
                assert_phase4_archive_matches_profile(archive, authorized)

            for tampered_source in ("models/model.py", "dataset/common.py"):
                with self.subTest(tampered_source=tampered_source):
                    tampered_archive = root / (
                        tampered_source.replace("/", "_") + ".par"
                    )
                    with ZipFile(tampered_archive, "w") as archive:
                        for relative_path, payload in sources.items():
                            archive.writestr(
                                relative_path,
                                (
                                    b"# different\n"
                                    if relative_path == tampered_source
                                    else payload
                                ),
                            )
                    tampered_digest = hashlib.sha256(
                        tampered_archive.read_bytes()
                    ).hexdigest()

                    def validate_tampered(archive: ZipFile) -> None:
                        validate_archive_layout(archive)
                        try:
                            assert_phase4_archive_matches_profile(archive, authorized)
                        except Phase4RuntimeProfileError as exc:
                            raise ConfirmatoryRuntimeError(str(exc)) from exc

                    with self.assertRaisesRegex(
                        ConfirmatoryRuntimeError,
                        "differs from the authorized checkout",
                    ):
                        validate_runtime_archive(
                            tampered_archive,
                            tampered_digest,
                            archive_validator=validate_tampered,
                        )

            for bytecode_path in ("models/model.pyc", "dataset/common.pyc"):
                with self.subTest(bytecode_path=bytecode_path):
                    bytecode_archive = root / (bytecode_path.replace("/", "_") + ".par")
                    with ZipFile(bytecode_archive, "w") as archive:
                        for relative_path, payload in sources.items():
                            archive.writestr(relative_path, payload)
                        archive.writestr(bytecode_path, b"bytecode")
                    with (
                        ZipFile(bytecode_archive, "r") as archive,
                        self.assertRaisesRegex(
                            Phase4RuntimeProfileError,
                            "bytecode",
                        ),
                    ):
                        assert_phase4_archive_matches_profile(archive, authorized)

    def test_dataset_builder_profile_rejects_extra_or_tampered_source(self) -> None:
        sources = {
            relative_path: f"# {relative_path}\n".encode("ascii")
            for relative_path in POLICY_DATASET_BUILDER_PROFILE_PATHS
        }
        authorized = _authorized(
            sources,
            profile=POLICY_DATASET_BUILDER_SOURCE_PROFILE,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.par"
            with ZipFile(valid, "w") as archive:
                for relative_path, payload in sources.items():
                    archive.writestr(relative_path, payload)
            with ZipFile(valid) as archive:
                assert_phase4_archive_matches_profile(archive, authorized)

            for name, payload in (
                ("dataset/unregistered.py", b"pass\n"),
                (
                    "configs/policy_improvement_v1/unregistered.json",
                    b"{}\n",
                ),
            ):
                archive_path = root / name.replace("/", "_")
                with ZipFile(archive_path, "w") as archive:
                    for relative_path, expected_payload in sources.items():
                        archive.writestr(relative_path, expected_payload)
                    archive.writestr(name, payload)
                with (
                    ZipFile(archive_path) as archive,
                    self.assertRaises(Phase4RuntimeProfileError),
                ):
                    assert_phase4_archive_matches_profile(archive, authorized)

            tampered = root / "tampered.par"
            with ZipFile(tampered, "w") as archive:
                for relative_path, payload in sources.items():
                    archive.writestr(
                        relative_path,
                        (
                            b"tampered\n"
                            if relative_path
                            == "dataset/build_policy_improvement_4x4.py"
                            else payload
                        ),
                    )
            with (
                ZipFile(tampered) as archive,
                self.assertRaises(Phase4RuntimeProfileError),
            ):
                assert_phase4_archive_matches_profile(archive, authorized)

    def test_dataset_builder_profile_authorizes_exact_clean_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative_path in POLICY_DATASET_BUILDER_PROFILE_PATHS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative_path}\n", encoding="ascii")
            _run_git(root, "init", "-q")
            _run_git(root, "add", ".")
            _run_git(root, "commit", "-q", "-m", "dataset profile")
            commit = _run_git(root, "rev-parse", "HEAD")
            identity = authorize_phase4_source_profile(
                root,
                commit,
                POLICY_DATASET_BUILDER_SOURCE_PROFILE,
            )
            self.assertEqual(identity.git_commit, commit)
            self.assertEqual(
                set(identity.sources),
                set(POLICY_DATASET_BUILDER_PROFILE_PATHS),
            )
            (root / "dataset/build_policy_improvement_4x4.py").write_text(
                "tampered\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                Phase4RuntimeProfileError,
                "authorized clean commit",
            ):
                authorize_phase4_source_profile(
                    root,
                    commit,
                    POLICY_DATASET_BUILDER_SOURCE_PROFILE,
                )

    def test_policy_consumer_profiles_reject_extra_tampered_and_bytecode(
        self,
    ) -> None:
        profiles = (
            (
                POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
                _policy_consumer_paths(POLICY_IMPROVEMENT_AUDIT_PROFILE_PATHS),
                "scripts/policy_improvement_audit.py",
            ),
            (
                POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
                _policy_consumer_paths(POLICY_IMPROVEMENT_ANALYSIS_PROFILE_PATHS),
                "scripts/policy_improvement_analysis.py",
            ),
            (
                POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
                _policy_consumer_paths(POLICY_IMPROVEMENT_FULL_PROFILE_PATHS),
                "policy_improvement_full_entrypoint.py",
            ),
            (
                POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
                _policy_consumer_paths(POLICY_IMPROVEMENT_THEORY_BRIDGE_PROFILE_PATHS),
                "scripts/policy_improvement_theory_bridge.py",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile, paths, tampered_name in profiles:
                sources = {
                    relative_path: f"# {relative_path}\n".encode("ascii")
                    for relative_path in paths
                }
                authorized = _authorized(sources, profile=profile)
                with self.subTest(profile=profile, case="valid"):
                    valid = root / f"{profile}.par"
                    with ZipFile(valid, "w") as archive:
                        for relative_path, payload in sources.items():
                            archive.writestr(relative_path, payload)
                with ZipFile(valid) as archive:
                    assert_phase4_archive_matches_profile(
                        archive,
                        authorized,
                    )
                for omitted_name in (
                    "scripts/policy_improvement_v2_schema.py",
                    "configs/policy_improvement_v2/protocol.json",
                    "configs/policy_improvement_v2/amendments/theory_bridge_v2.json",
                ):
                    with self.subTest(
                        profile=profile,
                        case="omitted-v2",
                        omitted_name=omitted_name,
                    ):
                        omitted = root / (
                            f"{profile}-{omitted_name.replace('/', '_')}.par"
                        )
                        with ZipFile(omitted, "w") as archive:
                            for relative_path, payload in sources.items():
                                if relative_path != omitted_name:
                                    archive.writestr(relative_path, payload)
                        with (
                            ZipFile(omitted) as archive,
                            self.assertRaises(Phase4RuntimeProfileError),
                        ):
                            assert_phase4_archive_matches_profile(
                                archive,
                                authorized,
                            )
                for case, extra_name in (
                    ("extra-source", "scripts/policy_improvement_unregistered.py"),
                    (
                        "extra-config",
                        "configs/policy_improvement_v1/unregistered.json",
                    ),
                    ("bytecode", "scripts/policy_improvement_schema.pyc"),
                ):
                    with self.subTest(profile=profile, case=case):
                        archive_path = root / f"{profile}-{case}.par"
                        with ZipFile(archive_path, "w") as archive:
                            for relative_path, payload in sources.items():
                                archive.writestr(relative_path, payload)
                            archive.writestr(extra_name, b"extra")
                        with (
                            ZipFile(archive_path) as archive,
                            self.assertRaises(Phase4RuntimeProfileError),
                        ):
                            assert_phase4_archive_matches_profile(
                                archive,
                                authorized,
                            )
                with self.subTest(profile=profile, case="tampered"):
                    tampered = root / f"{profile}-tampered.par"
                    with ZipFile(tampered, "w") as archive:
                        for relative_path, payload in sources.items():
                            archive.writestr(
                                relative_path,
                                (
                                    b"tampered\n"
                                    if relative_path == tampered_name
                                    else payload
                                ),
                            )
                    with (
                        ZipFile(tampered) as archive,
                        self.assertRaises(Phase4RuntimeProfileError),
                    ):
                        assert_phase4_archive_matches_profile(
                            archive,
                            authorized,
                        )

    def test_every_policy_runtime_profile_contains_the_v2_closure(self) -> None:
        for profile, paths in (
            (
                POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
                POLICY_IMPROVEMENT_AUDIT_PROFILE_PATHS,
            ),
            (
                POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
                POLICY_IMPROVEMENT_ANALYSIS_PROFILE_PATHS,
            ),
            (
                POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
                POLICY_IMPROVEMENT_FULL_PROFILE_PATHS,
            ),
            (
                POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
                POLICY_IMPROVEMENT_THEORY_BRIDGE_PROFILE_PATHS,
            ),
        ):
            with self.subTest(profile=profile):
                self.assertLessEqual(
                    _POLICY_IMPROVEMENT_V2_RUNTIME_PATHS,
                    set(paths),
                )

    def test_policy_consumer_profiles_authorize_exact_clean_tree(self) -> None:
        for profile, paths in (
            (
                POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
                _policy_consumer_paths(POLICY_IMPROVEMENT_AUDIT_PROFILE_PATHS),
            ),
            (
                POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
                _policy_consumer_paths(POLICY_IMPROVEMENT_ANALYSIS_PROFILE_PATHS),
            ),
            (
                POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
                _policy_consumer_paths(POLICY_IMPROVEMENT_FULL_PROFILE_PATHS),
            ),
            (
                POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
                _policy_consumer_paths(POLICY_IMPROVEMENT_THEORY_BRIDGE_PROFILE_PATHS),
            ),
        ):
            with (
                self.subTest(profile=profile),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                for relative_path in paths:
                    path = root / relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"# {relative_path}\n", encoding="ascii")
                _run_git(root, "init", "-q")
                _run_git(root, "add", ".")
                _run_git(root, "commit", "-q", "-m", "policy profile")
                commit = _run_git(root, "rev-parse", "HEAD")
                identity = authorize_phase4_source_profile(
                    root,
                    commit,
                    profile,
                )
                self.assertEqual(identity.git_commit, commit)
                self.assertEqual(set(identity.sources), set(paths))
                (root / "scripts/policy_improvement_schema.py").write_text(
                    "tampered\n",
                    encoding="ascii",
                )
                with self.assertRaisesRegex(
                    Phase4RuntimeProfileError,
                    "authorized clean commit",
                ):
                    authorize_phase4_source_profile(root, commit, profile)

    def test_source_profile_requires_exact_clean_commit_and_safe_index(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative_path, payload in _profile_sources().items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            _run_git(root, "init", "-q")
            _run_git(root, "add", ".")
            _run_git(root, "commit", "-q", "-m", "profile")
            commit = _run_git(root, "rev-parse", "HEAD")

            identity = authorize_phase4_source_profile(
                root,
                commit,
                PHASE4_EVALUATOR_SOURCE_PROFILE,
            )
            self.assertEqual(identity.git_commit, commit)
            self.assertEqual(len(identity.source_manifest_sha256), 64)

            with self.assertRaisesRegex(
                Phase4RuntimeProfileError,
                "authorized clean commit",
            ):
                authorize_phase4_source_profile(
                    root,
                    "f" * 40,
                    PHASE4_EVALUATOR_SOURCE_PROFILE,
                )

            _run_git(root, "update-index", "--skip-worktree", "models/model.py")
            with self.assertRaisesRegex(
                Phase4RuntimeProfileError,
                "unsafe index state",
            ):
                authorize_phase4_source_profile(
                    root,
                    commit,
                    PHASE4_EVALUATOR_SOURCE_PROFILE,
                )

    def test_source_profile_ignores_ambient_git_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative_path, payload in _profile_sources().items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            _run_git(root, "init", "-q")
            _run_git(root, "add", ".")
            _run_git(root, "commit", "-q", "-m", "profile")
            commit = _run_git(root, "rev-parse", "HEAD")

            with mock.patch.dict(
                os.environ,
                {
                    "GIT_DIR": "/tmp/hostile-git-dir",
                    "GIT_NO_REPLACE_OBJECTS": "0",
                    "GIT_REPLACE_REF_BASE": "refs/hostile/",
                    "GIT_WORK_TREE": "/tmp/hostile-work-tree",
                },
            ):
                identity = authorize_phase4_source_profile(
                    root,
                    commit,
                    PHASE4_EVALUATOR_SOURCE_PROFILE,
                )
            self.assertEqual(identity.git_commit, commit)

    def test_source_profile_rejects_tree_hidden_by_replacement_ref(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = _profile_sources()
            for relative_path, payload in sources.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            _run_git(root, "init", "-q")
            _run_git(root, "add", ".")
            _run_git(root, "commit", "-q", "-m", "authorized profile")
            authorized_commit = _run_git(root, "rev-parse", "HEAD")

            replacement_payload = b"# unauthorized replacement\n"
            (root / "models/model.py").write_bytes(replacement_payload)
            _run_git(root, "commit", "-q", "-am", "replacement profile")
            replacement_commit = _run_git(root, "rev-parse", "HEAD")
            _run_git(root, "checkout", "-q", "--detach", authorized_commit)
            _run_git(root, "replace", authorized_commit, replacement_commit)

            self.assertEqual(
                _run_git(root, "show", "HEAD:models/model.py"),
                replacement_payload.decode("ascii").strip(),
            )
            identity = authorize_phase4_source_profile(
                root,
                authorized_commit,
                PHASE4_EVALUATOR_SOURCE_PROFILE,
            )
            self.assertEqual(
                identity.sources["models/model.py"],
                hashlib.sha256(sources["models/model.py"]).hexdigest(),
            )

            _run_git(root, "read-tree", "--reset", "-u", replacement_commit)
            self.assertEqual(
                _run_git(
                    root,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ),
                "",
            )
            with self.assertRaisesRegex(
                Phase4RuntimeProfileError,
                "authorized clean commit",
            ):
                authorize_phase4_source_profile(
                    root,
                    authorized_commit,
                    PHASE4_EVALUATOR_SOURCE_PROFILE,
                )


if __name__ == "__main__":
    unittest.main()
