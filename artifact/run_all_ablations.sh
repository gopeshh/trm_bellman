#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

./scripts/run_ablation.sh configs/ablations/upi_trm_K1.yaml
./scripts/run_ablation.sh configs/ablations/upi_trm_K3.yaml
./scripts/run_ablation.sh configs/ablations/upi_trm_unroll2.yaml
