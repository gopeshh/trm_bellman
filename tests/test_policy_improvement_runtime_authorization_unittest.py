#!/usr/bin/env fbpython
"""Tests for the policy-improvement runtime authorization v3 generator."""

# pyre-strict

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

from phase4_runtime_profile import (
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_LAUNCHER_PROFILE_PATHS,
    POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE,
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
        POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE,
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

    def test_artifact_authentication_rejects_a_non_par_launcher(self) -> None:
        sources = _source_authorization()
        runtime_identities = iter(
            _artifact(name, marker)
            for marker, name in enumerate(authorization._RUNTIME_NAMES, start=8)
        )
        authenticate_artifact = authorization._authenticate_artifact

        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory).resolve() / "launcher"
            launcher.write_bytes(b"#!/bin/sh\nexec /bin/true\n")

            def authenticate(
                path: str,
                *,
                label: str,
                archive_validator: Callable[[ZipFile], None] | None = None,
            ) -> authorization.ArtifactIdentity:
                if label == "Launcher":
                    return authenticate_artifact(
                        path,
                        label=label,
                        archive_validator=archive_validator,
                    )
                return next(runtime_identities)

            with (
                mock.patch.object(
                    authorization,
                    "_authenticate_artifact",
                    side_effect=authenticate,
                ),
                self.assertRaisesRegex(
                    authorization.RuntimeAuthorizationGenerationError,
                    "Launcher is not a valid ZIP-based PAR",
                ),
            ):
                authorization._authenticate_artifacts(
                    launcher_path=str(launcher),
                    runtime_paths={
                        name: f"/{name}.par" for name in authorization._RUNTIME_NAMES
                    },
                    sources=sources,
                )

    def test_launcher_archive_binds_current_source_profile_and_entrypoint(
        self,
    ) -> None:
        launcher_source = b"def main():\n    return 0\n"
        phase4_launcher_source = b"def main():\n    return 0\n"
        runtime_profile_source = b"PROFILES = {}\n"
        producer_manifest = b'{"source_manifest_schema_version":3,"sources":{}}\n'
        authorized = AuthorizedPhase4Profile(
            git_commit=COMMIT,
            source_manifest_sha256=hashlib.sha256(b"launcher profile").hexdigest(),
            profile=POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE,
            sources={
                "confirmatory_runtime_launcher.py": hashlib.sha256(
                    launcher_source
                ).hexdigest(),
                "phase4_runtime_launcher.py": hashlib.sha256(
                    phase4_launcher_source
                ).hexdigest(),
                "phase4_runtime_profile.py": hashlib.sha256(
                    runtime_profile_source
                ).hexdigest(),
                "configs/iclr_confirmatory/producer_source_manifest.json": (
                    hashlib.sha256(producer_manifest).hexdigest()
                ),
            },
        )
        support = {
            "__main__.py": b"# bootstrap\n",
            "__main__.pyc": b"compiled bootstrap",
            "__par__/bootstrap.py": b"# par bootstrap\n",
            "sitecustomize.py": b"# site customization\n",
        }
        support_digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    name: hashlib.sha256(payload).hexdigest()
                    for name, payload in support.items()
                }
            )
        ).hexdigest()
        native_support = {
            "runtime/bin/phase4_runtime_launcher#native-main#platform-runtime#python#py_version_3_12": b"native main",
            "runtime/lib/__python_generated_allocator_preload": b"allocator",
        }
        native_support_digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    name: hashlib.sha256(payload).hexdigest()
                    for name, payload in native_support.items()
                }
            )
        ).hexdigest()
        startup_variables = (
            'VARS = {"label": "fbsource//test:phase4_runtime_launcher '
            '(cfg:opt-linux-x86_64-fbcode-platform010-clang21-no-san#'
            '0123456789abcdef)", "name": "phase4_runtime_launcher"}\n'
        )
        startup_body = (
            "STARTUP_FUNCTIONS=['''static_extension_finder:_initialize''',]\n"
        )
        startup_loader = (startup_variables + startup_body).encode("ascii")
        startup_digest = hashlib.sha256(startup_body.encode("ascii")).hexdigest()
        self.assertEqual(
            set(authorized.sources),
            set(POLICY_IMPROVEMENT_LAUNCHER_PROFILE_PATHS),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def write_launcher(
                name: str,
                *,
                confirmatory_source: bytes = launcher_source,
                phase4_source: bytes = phase4_launcher_source,
                manifest: bytes = producer_manifest,
                main_module: str = "phase4_runtime_launcher",
                extra_members: Mapping[str, bytes] | None = None,
                support_overrides: Mapping[str, bytes] | None = None,
                native_overrides: Mapping[str, bytes] | None = None,
                duplicate_member: str | None = None,
                buildstamp_override: bytes | None = None,
            ) -> Path:
                path = root / name
                fbmake = {
                    "build_mode": "opt",
                    "build_rule": "fbsource//test:phase4_runtime_launcher",
                    "build_rule_type": "python_binary",
                    "build_tool": "buck2",
                    "link_strategy": "native",
                    "main_function": None,
                    "main_module": main_module,
                    "par_style": "fastzip",
                    "platform": "platform010",
                    "rule_type_is_unit_test": 0,
                }
                executable_manifest = {
                    "buck_labels": [],
                    "env": {},
                    "fbmake": fbmake,
                    "library_versions": [],
                    "python_features": [],
                    "startup_functions": dict(
                        authorization._LAUNCHER_STARTUP_FUNCTIONS
                    ),
                }
                modules = [
                    "__par__.bootstrap",
                    "confirmatory_runtime_launcher",
                    "phase4_runtime_launcher",
                    "phase4_runtime_profile",
                ]
                python_manifest = "\n".join(
                    (
                        f"fbmake = {fbmake!r}",
                        "buck_labels = []",
                        "env = {}",
                        "python_features = []",
                        "startup_functions = "
                        f"{authorization._LAUNCHER_STARTUP_FUNCTIONS!r}",
                        "library_versions = []",
                        f"modules = {modules!r}",
                        f"origins = {tuple('test' for _ in modules)!r}",
                        "",
                    )
                ).encode("ascii")
                selected_support = {**support, **(support_overrides or {})}
                selected_native = {
                    **native_support,
                    **(native_overrides or {}),
                }
                with ZipFile(path, "w") as archive:
                    for support_name, payload in selected_support.items():
                        archive.writestr(support_name, payload)
                    archive.writestr(
                        "confirmatory_runtime_launcher.py",
                        confirmatory_source,
                    )
                    archive.writestr(
                        "phase4_runtime_launcher.py",
                        phase4_source,
                    )
                    archive.writestr(
                        "phase4_runtime_profile.py",
                        runtime_profile_source,
                    )
                    archive.writestr(
                        "configs/iclr_confirmatory/producer_source_manifest.json",
                        manifest,
                    )
                    archive.writestr(
                        "__manifest__.json",
                        canonical_json_bytes(executable_manifest),
                    )
                    archive.writestr("__manifest__.py", python_manifest)
                    archive.writestr(
                        "__par__/__startup_function_loader__.py",
                        startup_loader,
                    )
                    for native_name, payload in selected_native.items():
                        archive.writestr(native_name, payload)
                    for extra_name, payload in (extra_members or {}).items():
                        archive.writestr(extra_name, payload)
                    if duplicate_member is not None:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            archive.writestr(
                                duplicate_member,
                                selected_support[duplicate_member],
                            )
                buildstamp = (
                    buildstamp_override
                    if buildstamp_override is not None
                    else hashlib.md5(path.read_bytes()).hexdigest().encode("ascii")
                )
                with ZipFile(path, "a") as archive:
                    archive.writestr("BUILDSTAMP", buildstamp)
                return path

            with (
                mock.patch.object(
                    authorization,
                    "_LAUNCHER_PINNED_SUPPORT_MEMBERS",
                    tuple(support),
                ),
                mock.patch.object(
                    authorization,
                    "_LAUNCHER_PINNED_SUPPORT_MANIFEST_SHA256",
                    support_digest,
                ),
                mock.patch.object(
                    authorization,
                    "_LAUNCHER_STARTUP_LOADER_NORMALIZED_SHA256",
                    startup_digest,
                ),
                mock.patch.object(
                    authorization,
                    "_LAUNCHER_ARCHIVE_PREFIX_SIZE",
                    0,
                ),
                mock.patch.object(
                    authorization,
                    "_LAUNCHER_ARCHIVE_PREFIX_SHA256",
                    hashlib.sha256(b"").hexdigest(),
                ),
                mock.patch.object(
                    authorization,
                    "_LAUNCHER_NATIVE_SUPPORT_MEMBERS",
                    tuple(native_support),
                ),
                mock.patch.object(
                    authorization,
                    "_LAUNCHER_NATIVE_SUPPORT_MANIFEST_SHA256",
                    native_support_digest,
                ),
            ):
                valid = write_launcher("valid.par")
                identity = authorization._authenticate_artifact(
                    str(valid),
                    label="Launcher",
                    archive_validator=authorization._launcher_archive_validator(
                        authorized
                    ),
                )
                self.assertEqual(
                    identity.sha256,
                    hashlib.sha256(valid.read_bytes()).hexdigest(),
                )
                prefixed = root / "prefixed.par"
                prefixed.write_bytes(b"#!/bin/sh\nexit 0\n" + valid.read_bytes())

                for name, path in (
                    ("modified executable prefix", prefixed),
                    (
                        "modified BUILDSTAMP",
                        write_launcher(
                            "modified-buildstamp.par",
                            buildstamp_override=b"0" * 32,
                        ),
                    ),
                    (
                        "stale phase4 launcher",
                        write_launcher("stale-phase4.par", phase4_source=b"old\n"),
                    ),
                    (
                        "stale confirmatory launcher",
                        write_launcher(
                            "stale-confirmatory.par",
                            confirmatory_source=b"old\n",
                        ),
                    ),
                    (
                        "stale producer manifest",
                        write_launcher("stale-manifest.par", manifest=b"{}\n"),
                    ),
                    (
                        "different entrypoint",
                        write_launcher("wrong-main.par", main_module="attacker"),
                    ),
                    (
                        "stdlib shadow",
                        write_launcher(
                            "hashlib-shadow.par",
                            extra_members={"hashlib.py": b"raise SystemExit\n"},
                        ),
                    ),
                    (
                        "subprocess shadow",
                        write_launcher(
                            "subprocess-shadow.par",
                            extra_members={"subprocess.py": b"raise SystemExit\n"},
                        ),
                    ),
                    (
                        "versioned shared-library shadow",
                        write_launcher(
                            "versioned-libpython.par",
                            extra_members={
                                "runtime/lib/libpython3.12.so.1.0": b"malicious\n"
                            },
                        ),
                    ),
                    (
                        "modified sitecustomize",
                        write_launcher(
                            "sitecustomize.par",
                            support_overrides={"sitecustomize.py": b"malicious\n"},
                        ),
                    ),
                    (
                        "modified bootstrap",
                        write_launcher(
                            "bootstrap.par",
                            support_overrides={
                                "__par__/bootstrap.py": b"malicious\n"
                            },
                        ),
                    ),
                    (
                        "modified native main",
                        write_launcher(
                            "native-main.par",
                            native_overrides={
                                "runtime/bin/phase4_runtime_launcher#native-main#platform-runtime#python#py_version_3_12": b"malicious\n"
                            },
                        ),
                    ),
                    (
                        "duplicate sitecustomize",
                        write_launcher(
                            "duplicate-sitecustomize.par",
                            duplicate_member="sitecustomize.py",
                        ),
                    ),
                ):
                    with (
                        self.subTest(case=name),
                        self.assertRaises(
                            authorization.RuntimeAuthorizationGenerationError
                        ),
                    ):
                        authorization._authenticate_artifact(
                            str(path),
                            label="Launcher",
                            archive_validator=authorization._launcher_archive_validator(
                                authorized
                            ),
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
