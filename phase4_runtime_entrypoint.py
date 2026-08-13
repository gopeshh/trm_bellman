#!/usr/bin/env fbpython
"""Pre-import entrypoint for authenticated Phase 4 publication tools."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

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

if _PREFLIGHT.phase4_role == "evaluator":
    from scripts.eval_phase4_2x2_norm_ablation import main
elif _PREFLIGHT.phase4_role == "audit":
    from scripts.audit_phase4_paper_ready import main
elif _PREFLIGHT.phase4_role == "figure":
    from scripts.make_paper_figures_phase4 import main
else:  # pragma: no cover - preflight rejects every other role.
    raise RuntimeError("Unsupported authenticated Phase 4 role.")


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
