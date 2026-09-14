#!/bin/bash
# Real Stage A, chain run5. $1=seed_position $2=run_id $3=seed $4=gpu
set -u
ART=/data/repos/fbsource/buck-out/v2/art/fbcode
RT=$ART/c59829718afa7235/buiksat_trm
OWNER=/home/buiksat/upi-trm-owner-20260911
export CUDA_VISIBLE_DEVICES=$4
export RUN_UPITRM_FULL_EXPERIMENTS=1
$ART/52588bd379d6be4f/buiksat_trm/__phase4_runtime_launcher__/phase4_runtime_launcher.par \
  --purpose policy-improvement-exp1b-training \
  --runtime-archive $RT/__policy_improvement_full__/policy_improvement_full.par \
  --expected-runtime-sha256 2a4d3ca7bf44262cb7629526d983031432c9134fa438109a649314339d4ee700 \
  --source-project-root /home/buiksat/trm_bellman \
  --expected-source-git-commit 2a14eb656bd983c37b60e92852083524348511d8 \
  --runtime-authorization $OWNER/exp1b_runtime_authorization_20260913_run5.json \
  --expected-runtime-authorization-sha256 203621ee0c5eb3d3b042b6d447f2832c4f37c8c99d8ed0f628480f1b30326ab4 \
  -- \
  --base-policy-artifact $OWNER/base-policy-artifact/base_policy.pt \
  --base-policy-amendment $OWNER/base_policy_amendment_exp0_20260913_run5.json \
  --execution-admission $OWNER/exp1b_execution_admission_20260913_run5.json \
  --evidence-root /home/buiksat/upi-trm-owner-20260911/evidence4 --evidence-generation gen-20260913-run5 \
  --row-id "$2" --seed "$3"
rc=$?
echo "SEED $1 EXIT=$rc"
exit $rc
