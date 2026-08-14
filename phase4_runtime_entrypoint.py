#!/usr/bin/env fbpython
"""Pre-import entrypoint for authenticated Phase 4 publication tools."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, cast

from runtime_archive_preflight import preflight_runtime


_PHASE4_TOOL_ROLES = frozenset({"evaluator", "audit", "figure"})
_PREFLIGHT = preflight_runtime(
    module_file=__file__,
    expected_module_name="phase4_runtime_entrypoint.py",
    environ=os.environ,
    attestation_required=True,
    allowed_phase4_roles=_PHASE4_TOOL_ROLES,
)
assert _PREFLIGHT is not None
assert _PREFLIGHT.phase4_role is not None

# Compile imports only after the sealed runtime has been authenticated. A
# process-unique cache also prevents ambient timestamp-valid bytecode reuse.
_RUNTIME_BYTECODE_CACHE = tempfile.TemporaryDirectory(
    prefix="upi_trm_phase4_runtime_bytecode."
)
sys.pycache_prefix = str(Path(_RUNTIME_BYTECODE_CACHE.name).resolve())

_ROLE_MODULES = {
    "evaluator": "scripts.eval_phase4_2x2_norm_ablation",
    "audit": "scripts.audit_phase4_paper_ready",
    "figure": "scripts.make_paper_figures_phase4",
}
_module_name = _ROLE_MODULES.get(_PREFLIGHT.phase4_role)
if _module_name is None:  # pragma: no cover - preflight rejects other roles.
    raise RuntimeError("Unsupported authenticated Phase 4 role.")
_module = importlib.import_module(_module_name)
main = cast(Callable[..., int], getattr(_module, "main"))


if __name__ == "__main__":
    raise SystemExit(
        main(
            runtime_attestation={
                "runtime_sha256": _PREFLIGHT.runtime_sha256,
                "role": _PREFLIGHT.phase4_role,
                "source_git_commit": _PREFLIGHT.source_git_commit,
                "source_manifest_sha256": (
                    _PREFLIGHT.source_manifest_sha256
                ),
            }
        )
    )
