#!/usr/bin/env fbpython
"""Pre-import entrypoint for the authenticated full learned runtime."""

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


_FULL_RUNTIME_ROLE = "policy-improvement-full"
_FULL_LAUNCHER_SHA256_ENV = "UPI_TRM_POLICY_FULL_LAUNCHER_SHA256"
_THROUGHPUT_ENTRYPOINT = "--policy-improvement-throughput-entrypoint"
_EXP1B_ENTRYPOINT = "--policy-improvement-exp1b-entrypoint"
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_PREFLIGHT = preflight_runtime(
    module_file=__file__,
    expected_module_name="policy_improvement_full_entrypoint.py",
    environ=os.environ,
    attestation_required=True,
    allowed_phase4_roles=frozenset({_FULL_RUNTIME_ROLE}),
)
if _PREFLIGHT is None:  # pragma: no cover - attestation is mandatory.
    raise RuntimeError("Full learned runtime has no packaged-runtime preflight.")
launcher_sha256 = os.environ.pop(_FULL_LAUNCHER_SHA256_ENV, None)
if (
    not isinstance(launcher_sha256, str)
    or _LOWER_SHA256.fullmatch(launcher_sha256) is None
):
    raise RuntimeError("Full learned runtime launcher identity is invalid.")
required_attestation = (
    _PREFLIGHT.phase4_role,
    _PREFLIGHT.runtime_sha256,
    _PREFLIGHT.source_git_commit,
    _PREFLIGHT.source_manifest_sha256,
    _PREFLIGHT.policy_runtime_authorization_json,
    _PREFLIGHT.policy_runtime_authorization_sha256,
    _PREFLIGHT.policy_runtime_profile_sha256,
    _PREFLIGHT.policy_selected_source_manifest_sha256,
    _PREFLIGHT.policy_protocol_sha256,
)
if any(value is None for value in required_attestation):
    raise RuntimeError("Full learned runtime authorization is incomplete.")
authorization_json = _PREFLIGHT.policy_runtime_authorization_json
if not isinstance(authorization_json, str):
    raise RuntimeError("Full learned runtime authorization is incomplete.")
try:
    authorization_bytes = authorization_json.encode("ascii")
    authorization = json.loads(authorization_json)
    canonical_authorization = json.dumps(
        authorization,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
except (UnicodeEncodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
    raise RuntimeError("Full learned runtime authorization is invalid.") from exc
if (
    canonical_authorization != authorization_bytes
    or hashlib.sha256(authorization_bytes).hexdigest()
    != _PREFLIGHT.policy_runtime_authorization_sha256
    or not isinstance(authorization, dict)
    or authorization.get("launcher_sha256") != launcher_sha256
    or authorization.get("protocol_sha256") != _PREFLIGHT.policy_protocol_sha256
):
    raise RuntimeError("Full learned runtime authorization identity differs.")
roles = authorization.get("roles")
full_roles = (
    [
        role
        for role in roles
        if isinstance(role, dict) and role.get("role") == _FULL_RUNTIME_ROLE
    ]
    if isinstance(roles, list)
    else []
)
if len(full_roles) != 1:
    raise RuntimeError("Full learned runtime authorization lacks one full role.")
full_role = full_roles[0]
if (
    full_role
    != {
        "role": _FULL_RUNTIME_ROLE,
        "source_git_commit": _PREFLIGHT.source_git_commit,
        "runtime_sha256": _PREFLIGHT.runtime_sha256,
        "runtime_profile_sha256": _PREFLIGHT.policy_runtime_profile_sha256,
        "selected_source_manifest_sha256": (
            _PREFLIGHT.policy_selected_source_manifest_sha256
        ),
    }
    or _PREFLIGHT.source_manifest_sha256 != full_role["runtime_profile_sha256"]
):
    raise RuntimeError("Full learned runtime role identity differs from preflight.")
producer_source_manifest_sha256 = authorization.get("producer_source_manifest_sha256")
if (
    not isinstance(producer_source_manifest_sha256, str)
    or _LOWER_SHA256.fullmatch(producer_source_manifest_sha256) is None
):
    raise RuntimeError("Full learned runtime producer identity is invalid.")

# Compile all behavior modules only after the immutable PAR and authorization
# have been authenticated.  The process-unique cache prevents ambient bytecode
# from replacing manifest-covered source.
_RUNTIME_BYTECODE_CACHE = tempfile.TemporaryDirectory(
    prefix="upi_trm_policy_full_runtime_bytecode."
)
sys.pycache_prefix = str(Path(_RUNTIME_BYTECODE_CACHE.name).resolve())

throughput_requested = sys.argv[1:2] == [_THROUGHPUT_ENTRYPOINT]
if _THROUGHPUT_ENTRYPOINT in sys.argv[2:]:
    raise RuntimeError("Throughput entrypoint marker is not launcher-owned.")
exp1b_requested = sys.argv[1:2] == [_EXP1B_ENTRYPOINT]
if _EXP1B_ENTRYPOINT in sys.argv[2:]:
    raise RuntimeError("Experiment 1B entrypoint marker is not launcher-owned.")
if throughput_requested and exp1b_requested:  # pragma: no cover - argv is a list
    raise RuntimeError("Exactly one launcher-owned entrypoint marker is allowed.")
backend_module = importlib.import_module("policy_improvement_full_backend")
if exp1b_requested:
    _runtime_module_name = "scripts.policy_improvement_exp1b_runtime"
elif throughput_requested:
    _runtime_module_name = "scripts.policy_improvement_throughput"
else:
    _runtime_module_name = "scripts.policy_improvement_full_runtime"
runtime_module = importlib.import_module(_runtime_module_name)
training_module = importlib.import_module("upi_trm_train")
backend_class = getattr(backend_module, "SealedFullRunBackend")
backend = backend_class(
    runtime_identity={
        "role": _PREFLIGHT.phase4_role,
        "runtime_sha256": _PREFLIGHT.runtime_sha256,
        "source_git_commit": _PREFLIGHT.source_git_commit,
        "source_manifest_sha256": _PREFLIGHT.source_manifest_sha256,
        "producer_source_manifest_sha256": producer_source_manifest_sha256,
        "runtime_profile_sha256": _PREFLIGHT.policy_runtime_profile_sha256,
        "selected_source_manifest_sha256": (
            _PREFLIGHT.policy_selected_source_manifest_sha256
        ),
        "runtime_authorization_sha256": (
            _PREFLIGHT.policy_runtime_authorization_sha256
        ),
        "launcher_sha256": launcher_sha256,
    },
    training_module=training_module,
)
main = cast(Callable[..., int], getattr(runtime_module, "main"))


_EXP1B_ATTESTATION = {
    "source_git_commit": _PREFLIGHT.source_git_commit,
    "runtime_sha256": _PREFLIGHT.runtime_sha256,
    "launcher_sha256": launcher_sha256,
    "runtime_authorization_sha256": _PREFLIGHT.policy_runtime_authorization_sha256,
}


if __name__ == "__main__":
    # Every launcher-owned marker is stripped before the handler sees argv.
    marked = throughput_requested or exp1b_requested
    runtime_arguments = sys.argv[2:] if marked else sys.argv[1:]
    if exp1b_requested:
        raise SystemExit(
            main(
                runtime_arguments,
                backend=backend,
                runtime_attestation=_EXP1B_ATTESTATION,
            )
        )
    raise SystemExit(main(runtime_arguments, backend=backend))
