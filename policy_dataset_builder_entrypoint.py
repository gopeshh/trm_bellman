#!/usr/bin/env fbpython
"""Pre-import entrypoint for the authenticated policy dataset builder."""

from __future__ import annotations

import importlib
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable, Mapping, cast

from runtime_archive_preflight import preflight_runtime


POLICY_DATASET_BUILDER_ROLE = "policy-dataset-builder"
POLICY_DATASET_BUILDER_LAUNCHER_SHA256_ENV = (
    "UPI_TRM_POLICY_DATASET_BUILDER_LAUNCHER_SHA256"
)

_PREFLIGHT = preflight_runtime(
    module_file=__file__,
    expected_module_name="policy_dataset_builder_entrypoint.py",
    environ=os.environ,
    attestation_required=True,
    allowed_phase4_roles=frozenset({POLICY_DATASET_BUILDER_ROLE}),
)
assert _PREFLIGHT is not None
if (
    _PREFLIGHT.phase4_role != POLICY_DATASET_BUILDER_ROLE
    or _PREFLIGHT.source_git_commit is None
    or _PREFLIGHT.source_manifest_sha256 is None
):  # pragma: no cover - preflight rejects this state.
    raise RuntimeError("Dataset-builder runtime attestation is incomplete.")

_LAUNCHER_SHA256 = os.environ.pop(
    POLICY_DATASET_BUILDER_LAUNCHER_SHA256_ENV,
    None,
)
if (
    not isinstance(_LAUNCHER_SHA256, str)
    or re.fullmatch(r"[0-9a-f]{64}", _LAUNCHER_SHA256) is None
):
    raise RuntimeError("Dataset-builder launcher attestation is invalid.")

# Compile behavior imports only after the complete sealed runtime has passed
# preflight. A private cache prevents ambient timestamp-valid bytecode reuse.
_RUNTIME_BYTECODE_CACHE = tempfile.TemporaryDirectory(
    prefix="upi_trm_policy_dataset_builder_bytecode."
)
sys.pycache_prefix = str(Path(_RUNTIME_BYTECODE_CACHE.name).resolve())

_module = importlib.import_module("dataset.build_policy_improvement_4x4")
_main = cast(Callable[..., int], getattr(_module, "main"))
_ATTESTATION: Mapping[str, str] = {
    "git_commit": _PREFLIGHT.source_git_commit,
    "launcher_sha256": _LAUNCHER_SHA256,
    "runtime_sha256": _PREFLIGHT.runtime_sha256,
    "source_manifest_sha256": _PREFLIGHT.source_manifest_sha256,
}


def main() -> int:
    return _main(runtime_attestation=_ATTESTATION)


if __name__ == "__main__":
    raise SystemExit(main())
