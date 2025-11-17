#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ZIPNAME="$SCRIPT_DIR/upi_trm_artifact.zip"
rm -f "$ZIPNAME"

zip -r "$ZIPNAME" \
    upi_trm_train.py \
    models/ \
    rl/ \
    evaluators/ \
    utils/ \
    configs/ \
    scripts/ \
    tests/ \
    README.md \
    LICENSE \
    VERSION \
    artifact/README_ARTIFACT.md \
    artifact/requirements-artifact.txt \
    artifact/run_all_tests.sh \
    artifact/run_all_ablations.sh \
    artifact/zip.sh \
    -x "*/__pycache__/*" "*/.pytest_cache/*" "*.pt" "*.pth" "artifact/upi_trm_artifact.zip"
