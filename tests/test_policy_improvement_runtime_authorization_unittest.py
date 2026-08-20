#!/usr/bin/env fbpython
"""Tests for the policy-improvement runtime authorization v3 generator."""

# pyre-strict

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

from phase4_runtime_profile import (
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
    AuthorizedPhase4Profile,
    AuthorizedTrainingSource,
)
from scripts import policy_improvement_runtime_authorization as authorization
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    validate_runtime_authorization,
)


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def _source_authorization() -> authorization.SourceAuthorization:
    profile_names = (
        POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
        POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
        POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
        POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
    )
    return authorization.SourceAuthorization(
        training=AuthorizedTrainingSource(
            git_commit=COMMIT,
            source_manifest_sha256="1" * 64,
            manifest_bytes=b"{}\n",
        ),
        profiles={
            profile: AuthorizedPhase4Profile(
                git_commit=COMMIT,
                source_manifest_sha256=f"{index + 2:x}" * 64,
                profile=profile,
                sources={},
            )
            for index, profile in enumerate(profile_names)
        },
    )


def _artifact(
    name: str,
    marker: int,
    *,
    inode: int | None = None,
) -> authorization.ArtifactIdentity:
    return authorization.ArtifactIdentity(
        path=Path(f"/artifacts/{name}.par"),
        sha256=f"{marker:x}" * 64,
        device=1,
        inode=marker if inode is None else inode,
    )


def _artifact_authorization() -> authorization.ArtifactAuthorization:
    return authorization.ArtifactAuthorization(
        launcher=_artifact("launcher", 7),
        runtimes={
            "training": _artifact("training", 8),
            "full": _artifact("full", 9),
            "theory": _artifact("theory", 10),
            "audit": _artifact("audit", 11),
            "analysis": _artifact("analysis", 12),
        },
    )


class RuntimeAuthorizationGeneratorTest(unittest.TestCase):
    def test_registration_and_six_role_document_are_deterministic(self) -> None:
        registration = authorization._load_policy_registration(ROOT)
        sources = _source_authorization()
        artifacts = _artifact_authorization()
        first = authorization.build_runtime_authorization(
            authorization_id="policy-improvement-v2-test",
            created_at_utc="2026-08-19T12:00:00Z",
            expected_git_commit=COMMIT,
            registration=registration,
            sources=sources,
            artifacts=artifacts,
        )
        second = authorization.build_runtime_authorization(
            authorization_id="policy-improvement-v2-test",
            created_at_utc="2026-08-19T12:00:00Z",
            expected_git_commit=COMMIT,
            registration=registration,
            sources=sources,
            artifacts=artifacts,
        )
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(validate_runtime_authorization(first), first)
        roles = {item["role"]: item for item in first["roles"]}
        self.assertEqual(
            roles["policy-improvement-training"]["runtime_sha256"],
            roles["policy-improvement-evaluation"]["runtime_sha256"],
        )
        self.assertEqual(
            roles["policy-improvement-training"]["runtime_profile_sha256"],
            roles["policy-improvement-evaluation"]["runtime_profile_sha256"],
        )
        self.assertNotEqual(
            roles["policy-improvement-full"]["runtime_sha256"],
            roles["policy-improvement-theory-bridge"]["runtime_sha256"],
        )
        self.assertEqual(len({item["runtime_sha256"] for item in first["roles"]}), 5)

    def test_generate_publishes_one_canonical_private_file_without_replace(
        self,
    ) -> None:
        sources = _source_authorization()
        artifacts = _artifact_authorization()
        with tempfile.TemporaryDirectory() as directory:
            owner = Path(directory) / "owner"
            owner.mkdir(mode=0o700)
            with (
                mock.patch.object(
                    authorization,
                    "require_clean_exact_git_commit",
                    return_value=ROOT,
                ),
                mock.patch.object(
                    authorization,
                    "_authenticate_sources",
                    side_effect=[sources, sources],
                ),
                mock.patch.object(
                    authorization,
                    "_authenticate_artifacts",
                    return_value=artifacts,
                ),
            ):
                published = authorization.generate_runtime_authorization(
                    project_root=str(ROOT),
                    expected_git_commit=COMMIT,
                    authorization_id="policy-improvement-v2-test",
                    created_at_utc="2026-08-19T12:00:00Z",
                    launcher_path="/artifacts/launcher",
                    runtime_paths={
                        name: f"/artifacts/{name}.par"
                        for name in authorization._RUNTIME_NAMES
                    },
                    owner_directory=str(owner),
                    output_name="runtime-authorization-v3.json",
                )
            payload = published.path.read_bytes()
            self.assertEqual(payload, canonical_json_bytes(published.document))
            self.assertEqual(
                stat.S_IMODE(published.path.stat().st_mode),
                0o600,
            )
            self.assertEqual(published.path.stat().st_nlink, 1)
            self.assertEqual(
                validate_runtime_authorization(json.loads(payload)), published.document
            )
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in owner.iterdir())
            )

            with self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "already exists",
            ):
                authorization._publish_new_private_file(
                    owner_directory=str(owner),
                    output_name=published.path.name,
                    project_root=ROOT,
                    payload=payload,
                )

    def test_clean_exact_git_commit_rejects_wrong_head_dirty_and_untracked(
        self,
    ) -> None:
        environment = {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            subprocess.run(
                ["git", "init", "-q", str(root)], check=True, env=environment
            )
            (root / "tracked.txt").write_text("first\n", encoding="ascii")
            subprocess.run(
                ["git", "-C", str(root), "add", "tracked.txt"],
                check=True,
                env=environment,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=Authorization Test",
                    "-c",
                    "user.email=authorization@example.invalid",
                    "commit",
                    "-qm",
                    "initial",
                ],
                check=True,
                env=environment,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                env=environment,
                text=True,
            ).strip()
            self.assertEqual(
                authorization.require_clean_exact_git_commit(str(root), commit),
                root,
            )
            with self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "expected commit",
            ):
                authorization.require_clean_exact_git_commit(str(root), "0" * 40)
            (root / "tracked.txt").write_text("changed\n", encoding="ascii")
            with self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "not clean",
            ):
                authorization.require_clean_exact_git_commit(str(root), commit)
            (root / "tracked.txt").write_text("first\n", encoding="ascii")
            (root / "untracked.txt").write_text("new\n", encoding="ascii")
            with self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "not clean",
            ):
                authorization.require_clean_exact_git_commit(str(root), commit)

    def test_registration_rejects_noncanonical_or_mixed_identity_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "configs/policy_improvement_v2"
            destination.parent.mkdir(parents=True)
            shutil.copytree(ROOT / "configs/policy_improvement_v2", destination)
            protocol_path = destination / "protocol.json"
            protocol_path.write_text(
                '{"schema_name":"first","schema_name":"second"}\n',
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "not strict JSON",
            ):
                authorization._load_policy_registration(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "configs/policy_improvement_v2"
            destination.parent.mkdir(parents=True)
            shutil.copytree(ROOT / "configs/policy_improvement_v2", destination)
            amendment_path = destination / "amendments/theory_bridge_v2.json"
            amendment = json.loads(amendment_path.read_text(encoding="ascii"))
            amendment["unexpected"] = False
            amendment_path.write_bytes(canonical_json_bytes(amendment) + b"\n")
            with self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "field inventory differs",
            ):
                authorization._load_policy_registration(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "configs/policy_improvement_v2"
            destination.parent.mkdir(parents=True)
            shutil.copytree(ROOT / "configs/policy_improvement_v2", destination)
            amendment_path = destination / "amendments/theory_bridge_v2.json"
            amendment = json.loads(amendment_path.read_text(encoding="ascii"))
            amendment["protocol_sha256"] = "0" * 64
            amendment_path.write_bytes(canonical_json_bytes(amendment) + b"\n")
            with self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "identity or pre-outcome attestations differ",
            ):
                authorization._load_policy_registration(root)

    def test_artifact_authentication_rejects_aliases_and_symlinks(self) -> None:
        sources = _source_authorization()
        duplicate = _artifact("duplicate", 8, inode=99)
        identities = [
            _artifact("launcher", 7),
            duplicate,
            _artifact("full", 9, inode=99),
            _artifact("theory", 10),
            _artifact("audit", 11),
            _artifact("analysis", 12),
        ]
        with (
            mock.patch.object(
                authorization,
                "_authenticate_artifact",
                side_effect=identities,
            ),
            self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "distinct files",
            ),
        ):
            authorization._authenticate_artifacts(
                launcher_path="/launcher",
                runtime_paths={
                    name: f"/{name}" for name in authorization._RUNTIME_NAMES
                },
                sources=sources,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifact = root / "artifact"
            artifact.write_bytes(b"artifact")
            alias = root / "alias"
            alias.symlink_to(artifact)
            with self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "non-symlinked",
            ):
                authorization._authenticate_artifact(str(alias), label="Artifact")

    def test_artifact_hash_and_archive_validation_share_one_stable_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "runtime.par"
            with ZipFile(path, "w") as archive:
                archive.writestr("identity.txt", b"authenticated")
            observed: list[bytes] = []

            def validate(archive: ZipFile) -> None:
                observed.append(archive.read("identity.txt"))

            identity = authorization._authenticate_artifact(
                str(path),
                label="Runtime",
                archive_validator=validate,
            )
            self.assertEqual(observed, [b"authenticated"])
            self.assertEqual(identity.path, path)
            self.assertEqual(
                identity.sha256,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_private_publication_rejects_repo_output_and_wrong_directory_mode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inside = root / "owner"
            inside.mkdir(mode=0o700)
            with self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "inside the source repository",
            ):
                authorization._publish_new_private_file(
                    owner_directory=str(inside),
                    output_name="authorization.json",
                    project_root=root,
                    payload=b"{}",
                )

        with tempfile.TemporaryDirectory() as project_directory, tempfile.TemporaryDirectory() as owner_directory:
            project_root = Path(project_directory).resolve()
            owner = Path(owner_directory).resolve()
            owner.chmod(0o755)
            with self.assertRaisesRegex(
                authorization.RuntimeAuthorizationGenerationError,
                "mode 0700",
            ):
                authorization._publish_new_private_file(
                    owner_directory=str(owner),
                    output_name="authorization.json",
                    project_root=project_root,
                    payload=b"{}",
                )

    def test_builder_rejects_invalid_v3_identity(self) -> None:
        registration = authorization._load_policy_registration(ROOT)
        invalid = copy.deepcopy(registration)
        object.__setattr__(invalid, "protocol_sha256", "0" * 64)
        with self.assertRaisesRegex(
            authorization.RuntimeAuthorizationGenerationError,
            "registration digests differ",
        ):
            authorization.build_runtime_authorization(
                authorization_id="policy-improvement-v2-test",
                created_at_utc="2026-08-19T12:00:00Z",
                expected_git_commit=COMMIT,
                registration=invalid,
                sources=_source_authorization(),
                artifacts=_artifact_authorization(),
            )


if __name__ == "__main__":
    unittest.main()
