#!/bin/bash
# Real Stage B, chain run5. $1=seed_position
set -u
ART=/data/repos/fbsource/buck-out/v2/art/fbcode
RT=$ART/c59829718afa7235/buiksat_trm
OWNER=/home/buiksat/upi-trm-owner-20260911
export RUN_UPITRM_FULL_EXPERIMENTS=1
$ART/52588bd379d6be4f/buiksat_trm/__phase4_runtime_launcher__/phase4_runtime_launcher.par \
  --purpose policy-improvement-exp1b-bridge \
  --runtime-archive $RT/__policy_improvement_theory_bridge__/policy_improvement_theory_bridge.par \
  --expected-runtime-sha256 77dabda68715e206ba15f676fd4360ef061ab815eafbeb12de0d1bcc849ad177 \
  --source-project-root /home/buiksat/trm_bellman_evaluator \
  --expected-source-git-commit 2a14eb656bd983c37b60e92852083524348511d8 \
  --producer-source-project-root /home/buiksat/trm_bellman \
  --expected-producer-git-commit 2a14eb656bd983c37b60e92852083524348511d8 \
  --runtime-authorization $OWNER/exp1b_runtime_authorization_20260913_run5.json \
  --expected-runtime-authorization-sha256 203621ee0c5eb3d3b042b6d447f2832c4f37c8c99d8ed0f628480f1b30326ab4 \
  -- \
  --execution-admission $OWNER/exp1b_execution_admission_20260913_run5.json \
  --evidence-root /home/buiksat/upi-trm-owner-20260911/evidence4 --evidence-generation gen-20260913-run5 \
  --seed-position "$1"
rc=$?
echo "STAGE B POSITION $1 EXIT=$rc"
exit $rc
