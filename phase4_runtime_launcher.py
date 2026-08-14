#!/usr/bin/env fbpython
"""Authenticate and execute one Phase 4 publication PAR before imports."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from confirmatory_runtime_launcher import (
    ConfirmatoryRuntimeError,
    launch_verified_runtime,
    validate_archive_layout,
    validate_confirmatory_archive_sources,
    validate_runtime_archive,
)
from phase4_runtime_profile import (
    PHASE4_AUDIT_SOURCE_PROFILE,
    PHASE4_EVALUATOR_SOURCE_PROFILE,
    PHASE4_FIGURE_SOURCE_PROFILE,
    PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH,
    Phase4RuntimeProfileError,
    assert_phase4_archive_matches_profile,
    authorize_phase4_source_profile,
    authorize_phase4_training_source,
)


PHASE4_RUNTIME_ROLE_ENV = "UPI_TRM_PHASE4_RUNTIME_ROLE"
PHASE4_SOURCE_COMMIT_ENV = "UPI_TRM_PHASE4_SOURCE_COMMIT"
PHASE4_SOURCE_MANIFEST_SHA256_ENV = (
    "UPI_TRM_PHASE4_SOURCE_MANIFEST_SHA256"
)

_PURPOSE_TO_PROFILE = {
    "phase4-evaluator": PHASE4_EVALUATOR_SOURCE_PROFILE,
    "phase4-audit": PHASE4_AUDIT_SOURCE_PROFILE,
    "phase4-figure": PHASE4_FIGURE_SOURCE_PROFILE,
}


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate and execute a Phase 4 publication PAR."
    )
    parser.add_argument(
        "--purpose",
        required=True,
        choices=["phase4-training", *_PURPOSE_TO_PROFILE],
    )
    parser.add_argument("--runtime-archive", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--source-project-root", required=True)
    parser.add_argument("--expected-source-git-commit", required=True)
    parser.add_argument("runtime_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def _training_archive_validator(
    expected_manifest_bytes: bytes,
):
    def validate(archive: ZipFile) -> None:
        try:
            validate_confirmatory_archive_sources(archive)
            if archive.read(PRODUCER_SOURCE_MANIFEST_RELATIVE_PATH) != (
                expected_manifest_bytes
            ):
                raise ConfirmatoryRuntimeError(
                    "Training runtime manifest differs from the authorized checkout."
                )
        except (BadZipFile, KeyError, OSError, RuntimeError) as exc:
            if isinstance(exc, ConfirmatoryRuntimeError):
                raise
            raise ConfirmatoryRuntimeError(
                "Training runtime source profile cannot be authenticated."
            ) from exc

    return validate


def _consumer_archive_validator(authorized_profile):
    def validate(archive: ZipFile) -> None:
        try:
            validate_archive_layout(archive)
            assert_phase4_archive_matches_profile(archive, authorized_profile)
        except (BadZipFile, OSError, RuntimeError) as exc:
            if isinstance(exc, ConfirmatoryRuntimeError):
                raise
            raise ConfirmatoryRuntimeError(
                "Phase 4 runtime source profile cannot be authenticated."
            ) from exc

    return validate


def _normalize_child_args(
    purpose: str,
    source_project_root: str,
    runtime_args: Sequence[str],
) -> list[str]:
    arguments = list(runtime_args)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]

    def contains_option(name: str) -> bool:
        return any(
            argument == name or argument.startswith(f"{name}=")
            for argument in arguments
        )

    if contains_option("--confirmatory"):
        raise ConfirmatoryRuntimeError(
            "Phase 4 publication runtimes cannot use confirmatory mode."
        )
    if purpose != "phase4-training":
        if contains_option("--phase4-publication"):
            raise ConfirmatoryRuntimeError(
                "Phase 4 consumer arguments contain a training-only flag."
            )
        return arguments
    if contains_option("--phase4-publication") or contains_option(
        "--producer-repo-root"
    ):
        raise ConfirmatoryRuntimeError(
            "Phase 4 training launcher owns its publication and producer flags."
        )
    return [
        "--phase4-publication",
        "--producer-repo-root",
        source_project_root,
        *arguments,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        source_root = str(
            Path(arguments.source_project_root).resolve(strict=True)
        )
        child_args = _normalize_child_args(
            arguments.purpose,
            source_root,
            arguments.runtime_args,
        )
        if arguments.purpose == "phase4-training":
            authorized = authorize_phase4_training_source(
                source_root,
                arguments.expected_source_git_commit,
            )
            role = "training"
            archive_validator = _training_archive_validator(
                authorized.manifest_bytes
            )
        else:
            profile = _PURPOSE_TO_PROFILE[arguments.purpose]
            authorized = authorize_phase4_source_profile(
                source_root,
                arguments.expected_source_git_commit,
                profile,
            )
            role = profile
            archive_validator = _consumer_archive_validator(authorized)
        runtime = validate_runtime_archive(
            arguments.runtime_archive,
            arguments.expected_runtime_sha256,
            archive_validator=archive_validator,
        )
        return_code = launch_verified_runtime(
            runtime,
            child_args,
            required_argument=None,
            attestation_environment={
                PHASE4_RUNTIME_ROLE_ENV: role,
                PHASE4_SOURCE_COMMIT_ENV: authorized.git_commit,
                PHASE4_SOURCE_MANIFEST_SHA256_ENV: (
                    authorized.source_manifest_sha256
                ),
            },
        )
    except (
        ConfirmatoryRuntimeError,
        OSError,
        Phase4RuntimeProfileError,
    ) as exc:
        print(f"Phase 4 runtime rejected: {exc}", file=sys.stderr)
        return 2
    return 128 - return_code if return_code < 0 else return_code


if __name__ == "__main__":
    raise SystemExit(main())
