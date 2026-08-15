#!/usr/bin/env fbpython
"""Pre-import entrypoint for authenticated policy evidence consumers."""

from __future__ import annotations

import importlib
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable, cast

from runtime_archive_preflight import preflight_runtime


POLICY_IMPROVEMENT_AUDIT_ROLE = "policy-improvement-audit"
POLICY_IMPROVEMENT_ANALYSIS_ROLE = "policy-improvement-analysis"
POLICY_CONSUMER_LAUNCHER_SHA256_ENV = "UPI_TRM_POLICY_CONSUMER_LAUNCHER_SHA256"
POLICY_PRODUCER_SOURCE_MANIFEST_SHA256_ENV = (
    "UPI_TRM_POLICY_PRODUCER_SOURCE_MANIFEST_SHA256"
)
POLICY_PRODUCER_GIT_COMMIT_ENV = "UPI_TRM_POLICY_PRODUCER_GIT_COMMIT"
POLICY_CONSUMER_RUNTIME_AUTHORIZATION_ENV = (
    "UPI_TRM_POLICY_CONSUMER_RUNTIME_AUTHORIZATION"
)
POLICY_CONSUMER_RUNTIME_AUTHORIZATION_SHA256_ENV = (
    "UPI_TRM_POLICY_CONSUMER_RUNTIME_AUTHORIZATION_SHA256"
)

_PREFLIGHT = preflight_runtime(
    module_file=__file__,
    expected_module_name="policy_improvement_consumer_entrypoint.py",
    environ=os.environ,
    attestation_required=True,
    allowed_phase4_roles=frozenset(
        {
            POLICY_IMPROVEMENT_AUDIT_ROLE,
            POLICY_IMPROVEMENT_ANALYSIS_ROLE,
        }
    ),
)
assert _PREFLIGHT is not None
assert _PREFLIGHT.phase4_role is not None
assert _PREFLIGHT.source_git_commit is not None
assert _PREFLIGHT.source_manifest_sha256 is not None

_LAUNCHER_SHA256 = os.environ.pop(
    POLICY_CONSUMER_LAUNCHER_SHA256_ENV,
    None,
)
_PRODUCER_SOURCE_MANIFEST_SHA256 = os.environ.pop(
    POLICY_PRODUCER_SOURCE_MANIFEST_SHA256_ENV,
    None,
)
_PRODUCER_GIT_COMMIT = os.environ.pop(
    POLICY_PRODUCER_GIT_COMMIT_ENV,
    None,
)
_RUNTIME_AUTHORIZATION_JSON = os.environ.pop(
    POLICY_CONSUMER_RUNTIME_AUTHORIZATION_ENV,
    None,
)
_RUNTIME_AUTHORIZATION_SHA256 = os.environ.pop(
    POLICY_CONSUMER_RUNTIME_AUTHORIZATION_SHA256_ENV,
    None,
)
if (
    not isinstance(_LAUNCHER_SHA256, str)
    or re.fullmatch(r"[0-9a-f]{64}", _LAUNCHER_SHA256) is None
    or not isinstance(_PRODUCER_SOURCE_MANIFEST_SHA256, str)
    or re.fullmatch(r"[0-9a-f]{64}", _PRODUCER_SOURCE_MANIFEST_SHA256) is None
    or not isinstance(_PRODUCER_GIT_COMMIT, str)
    or re.fullmatch(r"[0-9a-f]{40}", _PRODUCER_GIT_COMMIT) is None
    or not isinstance(_RUNTIME_AUTHORIZATION_JSON, str)
    or not _RUNTIME_AUTHORIZATION_JSON
    or not isinstance(_RUNTIME_AUTHORIZATION_SHA256, str)
    or re.fullmatch(r"[0-9a-f]{64}", _RUNTIME_AUTHORIZATION_SHA256) is None
    or hashlib.sha256(_RUNTIME_AUTHORIZATION_JSON.encode("ascii")).hexdigest()
    != _RUNTIME_AUTHORIZATION_SHA256
):
    raise RuntimeError("Policy consumer launcher attestation is invalid.")

# No policy module is imported until the complete sealed runtime and launcher
# attestations have been consumed. The private cache also excludes ambient
# timestamp-valid bytecode.
_RUNTIME_BYTECODE_CACHE = tempfile.TemporaryDirectory(
    prefix="upi_trm_policy_consumer_bytecode."
)
sys.pycache_prefix = str(Path(_RUNTIME_BYTECODE_CACHE.name).resolve())

_ROLE_MODULES = {
    POLICY_IMPROVEMENT_AUDIT_ROLE: (
        "scripts.policy_improvement_audit",
        "audit",
    ),
    POLICY_IMPROVEMENT_ANALYSIS_ROLE: (
        "scripts.policy_improvement_analysis",
        "analysis",
    ),
}
_publish_test_open = (
    _PREFLIGHT.phase4_role == POLICY_IMPROVEMENT_AUDIT_ROLE
    and "--publish-test-open" in sys.argv[1:]
)
if _publish_test_open:
    _module_name, _identity_prefix = (
        "scripts.policy_improvement_test_open_cli",
        "audit",
    )
else:
    _module_name, _identity_prefix = _ROLE_MODULES[_PREFLIGHT.phase4_role]
_module = importlib.import_module(_module_name)
_main = cast(Callable[..., int], getattr(_module, "main"))
_checkpoint_validator_module = importlib.import_module(
    "policy_improvement_checkpoint_validator"
)
_checkpoint_validator = cast(
    Callable[[dict[str, object]], dict[str, object]],
    getattr(_checkpoint_validator_module, "validate_checkpoint"),
)


def main() -> int:
    arguments = list(sys.argv[1:])
    if not arguments or arguments.pop(0) != "--policy-improvement-consumer-entrypoint":
        raise RuntimeError(
            "Policy consumer execution requires its authenticated entrypoint flag."
        )
    if _publish_test_open:
        if not arguments or arguments.pop(0) != "--publish-test-open":
            raise RuntimeError("Test-opening operation dispatch changed.")
    protected = [
        f"--{_identity_prefix}-runtime-sha256",
        _PREFLIGHT.runtime_sha256,
        f"--{_identity_prefix}-runtime-profile-sha256",
        _PREFLIGHT.source_manifest_sha256,
        f"--{_identity_prefix}-source-git-commit",
        _PREFLIGHT.source_git_commit,
        "--launcher-sha256",
        _LAUNCHER_SHA256,
        "--producer-git-commit",
        _PRODUCER_GIT_COMMIT,
        "--producer-source-manifest-sha256",
        _PRODUCER_SOURCE_MANIFEST_SHA256,
        "--runtime-authorization-json",
        _RUNTIME_AUTHORIZATION_JSON,
        "--runtime-authorization-sha256",
        _RUNTIME_AUTHORIZATION_SHA256,
    ]
    return _main(
        [*protected, *arguments],
        checkpoint_validator=_checkpoint_validator,
    )


if __name__ == "__main__":
    raise SystemExit(main())
