#!/usr/bin/env fbpython
"""Regression tests for the pre-import Phase 4 runtime boundary."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from confirmatory_runtime_launcher import (
    ConfirmatoryRuntimeError,
    validate_archive_layout,
    validate_runtime_archive,
)
from phase4_runtime_launcher import _normalize_child_args
from phase4_runtime_profile import (
    PHASE4_EVALUATOR_SOURCE_PROFILE,
    PHASE4_PROFILE_ENTRYPOINTS,
    PHASE4_ROOT_SOURCES,
    PHASE4_SHARED_SOURCES,
    AuthorizedPhase4Profile,
    Phase4RuntimeProfileError,
    assert_phase4_archive_matches_profile,
    authorize_phase4_source_profile,
)
from runtime_archive_preflight import preflight_runtime


def _run_git(root: Path, *arguments: str) -> str:
    environment = dict(os.environ)
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
        "models/model.py",
        "rl/trainer.py",
        "utils/identity.py",
    }
    return {
        relative_path: f"# {relative_path}\n".encode("ascii")
        for relative_path in sorted(paths)
    }


def _authorized(sources: dict[str, bytes]) -> AuthorizedPhase4Profile:
    return AuthorizedPhase4Profile(
        git_commit="a" * 40,
        source_manifest_sha256="b" * 64,
        profile=PHASE4_EVALUATOR_SOURCE_PROFILE,
        sources={
            relative_path: hashlib.sha256(payload).hexdigest()
            for relative_path, payload in sources.items()
        },
    )


class Phase4RuntimeLauncherTest(unittest.TestCase):
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
            "cannot use confirmatory mode",
        ):
            _normalize_child_args(
                "phase4-evaluator",
                "/repo",
                ["--confirmatory"],
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

            tampered_archive = root / "tampered.par"
            with ZipFile(tampered_archive, "w") as archive:
                for relative_path, payload in sources.items():
                    archive.writestr(
                        relative_path,
                        b"# different\n"
                        if relative_path == "models/model.py"
                        else payload,
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

            bytecode_archive = root / "bytecode.par"
            with ZipFile(bytecode_archive, "w") as archive:
                for relative_path, payload in sources.items():
                    archive.writestr(relative_path, payload)
                archive.writestr("models/model.pyc", b"bytecode")
            with ZipFile(bytecode_archive, "r") as archive, self.assertRaisesRegex(
                Phase4RuntimeProfileError,
                "bytecode",
            ):
                assert_phase4_archive_matches_profile(archive, authorized)

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


if __name__ == "__main__":
    unittest.main()
