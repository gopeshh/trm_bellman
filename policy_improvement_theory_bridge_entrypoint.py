#!/usr/bin/env fbpython
"""Pre-import entrypoint for the authenticated theory-bridge evaluator."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable, cast

from runtime_archive_preflight import preflight_runtime


POLICY_THEORY_BRIDGE_ROLE = "policy-improvement-theory-bridge"
POLICY_THEORY_BRIDGE_LAUNCHER_SHA256_ENV = (
    "UPI_TRM_POLICY_THEORY_BRIDGE_LAUNCHER_SHA256"
)

_PREFLIGHT = preflight_runtime(
    module_file=__file__,
    expected_module_name="policy_improvement_theory_bridge_entrypoint.py",
    environ=os.environ,
    attestation_required=True,
    allowed_phase4_roles=frozenset({POLICY_THEORY_BRIDGE_ROLE}),
)
assert _PREFLIGHT is not None
assert _PREFLIGHT.phase4_role == POLICY_THEORY_BRIDGE_ROLE
assert _PREFLIGHT.source_git_commit is not None
assert _PREFLIGHT.source_manifest_sha256 is not None

_LAUNCHER_SHA256 = os.environ.pop(
    POLICY_THEORY_BRIDGE_LAUNCHER_SHA256_ENV,
    None,
)
_AUTHORIZATION_JSON = _PREFLIGHT.policy_runtime_authorization_json
_AUTHORIZATION_SHA256 = _PREFLIGHT.policy_runtime_authorization_sha256
_RUNTIME_PROFILE_SHA256 = _PREFLIGHT.policy_runtime_profile_sha256
_SELECTED_SOURCE_MANIFEST_SHA256 = _PREFLIGHT.policy_selected_source_manifest_sha256
_PROTOCOL_SHA256 = _PREFLIGHT.policy_protocol_sha256
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWER_COMMIT = re.compile(r"^[0-9a-f]{40}$")
if (
    not isinstance(_LAUNCHER_SHA256, str)
    or _LOWER_SHA256.fullmatch(_LAUNCHER_SHA256) is None
    or not isinstance(_AUTHORIZATION_JSON, str)
    or not isinstance(_AUTHORIZATION_SHA256, str)
    or _LOWER_SHA256.fullmatch(_AUTHORIZATION_SHA256) is None
    or not isinstance(_RUNTIME_PROFILE_SHA256, str)
    or _LOWER_SHA256.fullmatch(_RUNTIME_PROFILE_SHA256) is None
    or not isinstance(_SELECTED_SOURCE_MANIFEST_SHA256, str)
    or _LOWER_SHA256.fullmatch(_SELECTED_SOURCE_MANIFEST_SHA256) is None
    or _RUNTIME_PROFILE_SHA256 != _SELECTED_SOURCE_MANIFEST_SHA256
    or _PREFLIGHT.source_manifest_sha256 != _SELECTED_SOURCE_MANIFEST_SHA256
    or not isinstance(_PROTOCOL_SHA256, str)
    or _LOWER_SHA256.fullmatch(_PROTOCOL_SHA256) is None
):
    raise RuntimeError("Theory-bridge launcher attestation is invalid.")
try:
    _authorization_bytes = _AUTHORIZATION_JSON.encode("ascii")
    _authorization = json.loads(_AUTHORIZATION_JSON)
    _canonical_authorization = json.dumps(
        _authorization,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
except (UnicodeEncodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
    raise RuntimeError("Theory-bridge runtime authorization is invalid.") from exc
if (
    _authorization_bytes != _canonical_authorization
    or hashlib.sha256(_authorization_bytes).hexdigest() != _AUTHORIZATION_SHA256
    or not isinstance(_authorization, dict)
    or _authorization.get("launcher_sha256") != _LAUNCHER_SHA256
    or _authorization.get("protocol_sha256") != _PROTOCOL_SHA256
):
    raise RuntimeError("Theory-bridge runtime authorization is invalid.")
_roles = _authorization.get("roles")
if not isinstance(_roles, list):
    raise RuntimeError("Theory-bridge runtime authorization has no role list.")
_matching_roles = [
    role
    for role in _roles
    if isinstance(role, dict) and role.get("role") == POLICY_THEORY_BRIDGE_ROLE
]
if len(_matching_roles) != 1:
    raise RuntimeError(
        "Theory-bridge runtime authorization role is missing or repeated."
    )
_ROLE = _matching_roles[0]
if (
    set(_ROLE)
    != {
        "role",
        "source_git_commit",
        "runtime_sha256",
        "runtime_profile_sha256",
        "selected_source_manifest_sha256",
    }
    or not isinstance(_ROLE["source_git_commit"], str)
    or _LOWER_COMMIT.fullmatch(_ROLE["source_git_commit"]) is None
    or _ROLE["source_git_commit"] != _PREFLIGHT.source_git_commit
    or _ROLE["runtime_sha256"] != _PREFLIGHT.runtime_sha256
    or _ROLE["runtime_profile_sha256"] != _RUNTIME_PROFILE_SHA256
    or _ROLE["selected_source_manifest_sha256"] != _SELECTED_SOURCE_MANIFEST_SHA256
):
    raise RuntimeError("Theory-bridge runtime authorization role differs from runtime.")

# No model, dataset, checkpoint, or evaluator behavior module is imported until
# the sealed runtime and external authorization have both been consumed.
_RUNTIME_BYTECODE_CACHE = tempfile.TemporaryDirectory(
    prefix="upi_trm_policy_theory_bridge_bytecode."
)
sys.pycache_prefix = str(Path(_RUNTIME_BYTECODE_CACHE.name).resolve())

_core = importlib.import_module("scripts.policy_improvement_theory_bridge")
_backend_module = importlib.import_module("scripts.policy_improvement_theory_backend")
_main = cast(Callable[..., int], getattr(_core, "main"))
_backend_factory = cast(
    Callable[..., object],
    getattr(_backend_module, "create_theory_bridge_backend"),
)
_ATTESTATION = {
    "source_git_commit": _ROLE["source_git_commit"],
    "source_manifest_sha256": _ROLE["selected_source_manifest_sha256"],
    "runtime_sha256": _ROLE["runtime_sha256"],
    "runtime_profile_sha256": _ROLE["runtime_profile_sha256"],
    "runtime_authorization_sha256": _AUTHORIZATION_SHA256,
    "launcher_sha256": _LAUNCHER_SHA256,
}


def main() -> int:
    arguments = list(sys.argv[1:])
    if (
        not arguments
        or arguments.pop(0) != "--policy-improvement-theory-bridge-entrypoint"
    ):
        raise RuntimeError(
            "Theory-bridge execution requires its authenticated entrypoint flag."
        )
    return _main(
        ["--policy-improvement-theory-bridge-entrypoint", *arguments],
        backend_factory=_backend_factory,
        evaluator_attestation=_ATTESTATION,
        runtime_authorization=_authorization,
    )


if __name__ == "__main__":
    raise SystemExit(main())
