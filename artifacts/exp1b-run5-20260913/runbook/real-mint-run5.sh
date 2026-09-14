#!/bin/bash
# Mint the real attestation chain run5 against the ported real PARs.
# Run after real-port-run5.sh reports BUCK_OPT_EXIT=0.
set -euo pipefail

REAL=/home/buiksat/trm_bellman
OWNER=/home/buiksat/upi-trm-owner-20260911
ART=/data/repos/fbsource/buck-out/v2/art/fbcode
RT=$ART/c59829718afa7235/buiksat_trm
LAUNCHER=$ART/52588bd379d6be4f/buiksat_trm/__phase4_runtime_launcher__/phase4_runtime_launcher.par

COMMIT=$(cat "$OWNER/.run5_commit")
GEN=gen-20260913-run5

# The port verified the tree and wrote .run5_commit last, but the PARs were built
# from the fbsource cell copy taken at that moment. If the repo moves in between,
# the authorization describes source that is not in the PAR and nothing downstream
# catches it: the launcher recomputes the manifest from the same moved tree.
[ "$(git -C "$REAL" rev-parse HEAD)" = "$COMMIT" ] || { echo "producer moved since the port: $(git -C "$REAL" rev-parse HEAD) != $COMMIT" >&2; exit 1; }
[ -z "$(git -C "$REAL" status --porcelain)" ] || { echo "producer tree is dirty" >&2; exit 1; }

# Fail closed on a partial mint, but name every document so the cleanup is one step.
for f in exp1b_runtime_authorization_20260913_run5.json \
         base_policy_amendment_exp0_20260913_run5.json \
         exp1b_execution_admission_20260913_run5.json; do
  [ -e "$OWNER/$f" ] && { echo "run5 document already published: $f" >&2
                          echo "remove the whole run5 document set deliberately before re-minting" >&2
                          exit 1; }
done
STAMP_A=2026-09-13T23:50:00Z
STAMP_B=2026-09-13T23:51:00Z
STAMP_C=2026-09-13T23:52:00Z

echo "=== mint the real runtime authorization (commit $COMMIT) ==="
cd "$REAL"
# The minter imports confirmatory_runtime_launcher from the repo root.
PYTHONPATH="$REAL":"$REAL"/scripts \
python3 scripts/policy_improvement_runtime_authorization.py \
  --project-root "$REAL" \
  --expected-git-commit "$COMMIT" \
  --authorization-id upi-trm-exp1b-runtime-authorization-20260913-run5 \
  --created-at-utc "$STAMP_A" \
  --launcher "$LAUNCHER" \
  --training-runtime "$RT/__upi_trm_train__/upi_trm_train.par" \
  --full-runtime "$RT/__policy_improvement_full__/policy_improvement_full.par" \
  --theory-runtime "$RT/__policy_improvement_theory_bridge__/policy_improvement_theory_bridge.par" \
  --audit-runtime "$RT/__policy_improvement_audit__/policy_improvement_audit.par" \
  --analysis-runtime "$RT/__policy_improvement_analysis__/policy_improvement_analysis.par" \
  --owner-directory "$OWNER" \
  --output-name exp1b_runtime_authorization_20260913_run5.json

echo "=== derive the base-policy amendment and the execution admission ==="
python3 - "$OWNER" "$REAL" "$LAUNCHER" "$COMMIT" "$STAMP_B" "$STAMP_C" "$GEN" <<'PY'
import hashlib, json, os, sys

owner, real, launcher_path, commit, stamp_b, stamp_c, gen = sys.argv[1:8]
sys.path.insert(0, os.path.join(real, "scripts"))
from policy_improvement_v2_schema import canonical_json_bytes


def digest_of(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load(name):
    with open(os.path.join(owner, name)) as handle:
        return json.load(handle)


def publish(name, document):
    payload = canonical_json_bytes(document)
    path = os.path.join(owner, name)
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(handle, "wb") as out:
        out.write(payload)
    return hashlib.sha256(payload).hexdigest()


auth = load("exp1b_runtime_authorization_20260913_run5.json")
auth_sha = digest_of(auth)
assert auth["producer_git_commit"] == commit, auth["producer_git_commit"]

runtimes = {role["role"]: role["runtime_sha256"] for role in auth["roles"]}
full_rt = runtimes["policy-improvement-full"]
bridge_rt = runtimes["policy-improvement-theory-bridge"]
launcher_sha = auth["launcher_sha256"]
assert launcher_sha == hashlib.sha256(open(launcher_path, "rb").read()).hexdigest()

bp = load("base_policy_amendment_exp0_20260913_run4.json")
bp["amendment_id"] = "base-policy-exp0-20260913-run5"
bp["created_at_utc"] = stamp_b
bp["runtime_authorization_sha256"] = auth_sha
bp_sha = publish("base_policy_amendment_exp0_20260913_run5.json", bp)

reduced_path = os.path.join(
    real, "configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json"
)
with open(reduced_path) as handle:
    reduced = json.load(handle)
reduced_sha = digest_of(reduced)
assert "centering_contract" in reduced, "the real amendment is missing centering_contract"
assert (
    reduced["centering_contract"]["training_estimator_parity_relative_tolerance"]
    == 4.76837158203125e-07
)

adm = load("exp1b_execution_admission_20260913_run4.json")
adm["amendment_id"] = "exp1b-execution-admission-20260913-run5"
adm["created_at_utc"] = stamp_c
adm["prior_amendment_sha256"] = reduced_sha
adm["base_policy_artifact"]["value"]["amendment_sha256"] = bp_sha
for role, runtime_sha in (
    ("policy-improvement-full", full_rt),
    ("policy-improvement-theory-bridge", bridge_rt),
):
    slot = adm["runtime_authorization"]["value"][role]
    slot["authorization_sha256"] = auth_sha
    slot["launcher_sha256"] = launcher_sha
    slot["runtime_sha256"] = runtime_sha
    slot["source_git_commit"] = commit
adm_sha = publish("exp1b_execution_admission_20260913_run5.json", adm)

chain = {
    "commit": commit,
    "authorization_sha256": auth_sha,
    "base_policy_amendment_sha256": bp_sha,
    "reduced_study_amendment_sha256": reduced_sha,
    "admission_sha256": adm_sha,
    "full_runtime_sha256": full_rt,
    "bridge_runtime_sha256": bridge_rt,
    "launcher_sha256": launcher_sha,
    "evidence_root": os.path.join(owner, "evidence4"),
    "evidence_generation": gen,
}
with open(os.path.join(owner, "chain_run5.json"), "w") as out:
    json.dump(chain, out, indent=1, sort_keys=True)
    out.write("\n")
print(json.dumps(chain, indent=1, sort_keys=True))
PY

# The runtime refuses to create these: existence, ownership and mode of the
# evidence tree are the owner's boundary, not the run's. All four are demanded by
# open_evidence_generation via _require_owned_directory, and payloads/ is required
# before Stage A even though only Stage B writes into it. base_policy/ and the
# per-seed checkpoint subdirectories are runtime-created; leave those alone.
mkdir -m 0700 -p "$OWNER/evidence4"
mkdir -m 0700 -p "$OWNER/evidence4/$GEN"
mkdir -m 0700 -p "$OWNER/evidence4/$GEN/payloads"
mkdir -m 0700 -p "$OWNER/evidence4/$GEN/checkpoints"

echo "=== regenerate the run5 stage scripts ==="
python3 - "$OWNER" <<'PY'
import json, os, sys

owner = sys.argv[1]
chain = json.load(open(os.path.join(owner, "chain_run5.json")))
common = dict(
    owner=owner,
    commit=chain["commit"],
    auth=chain["authorization_sha256"],
    full=chain["full_runtime_sha256"],
    bridge=chain["bridge_runtime_sha256"],
    root=chain["evidence_root"],
    gen=chain["evidence_generation"],
)

stage_a = """#!/bin/bash
# Real Stage A, chain run5. $1=seed_position $2=run_id $3=seed $4=gpu
set -u
ART=/data/repos/fbsource/buck-out/v2/art/fbcode
RT=$ART/c59829718afa7235/buiksat_trm
OWNER={owner}
export CUDA_VISIBLE_DEVICES=$4
export RUN_UPITRM_FULL_EXPERIMENTS=1
$ART/52588bd379d6be4f/buiksat_trm/__phase4_runtime_launcher__/phase4_runtime_launcher.par \\
  --purpose policy-improvement-exp1b-training \\
  --runtime-archive $RT/__policy_improvement_full__/policy_improvement_full.par \\
  --expected-runtime-sha256 {full} \\
  --source-project-root /home/buiksat/trm_bellman \\
  --expected-source-git-commit {commit} \\
  --runtime-authorization $OWNER/exp1b_runtime_authorization_20260913_run5.json \\
  --expected-runtime-authorization-sha256 {auth} \\
  -- \\
  --base-policy-artifact $OWNER/base-policy-artifact/base_policy.pt \\
  --base-policy-amendment $OWNER/base_policy_amendment_exp0_20260913_run5.json \\
  --execution-admission $OWNER/exp1b_execution_admission_20260913_run5.json \\
  --evidence-root {root} --evidence-generation {gen} \\
  --row-id "$2" --seed "$3"
rc=$?
echo "SEED $1 EXIT=$rc"
exit $rc
""".format(**common)

stage_b = """#!/bin/bash
# Real Stage B, chain run5. $1=seed_position
set -u
ART=/data/repos/fbsource/buck-out/v2/art/fbcode
RT=$ART/c59829718afa7235/buiksat_trm
OWNER={owner}
export RUN_UPITRM_FULL_EXPERIMENTS=1
$ART/52588bd379d6be4f/buiksat_trm/__phase4_runtime_launcher__/phase4_runtime_launcher.par \\
  --purpose policy-improvement-exp1b-bridge \\
  --runtime-archive $RT/__policy_improvement_theory_bridge__/policy_improvement_theory_bridge.par \\
  --expected-runtime-sha256 {bridge} \\
  --source-project-root /home/buiksat/trm_bellman_evaluator \\
  --expected-source-git-commit {commit} \\
  --producer-source-project-root /home/buiksat/trm_bellman \\
  --expected-producer-git-commit {commit} \\
  --runtime-authorization $OWNER/exp1b_runtime_authorization_20260913_run5.json \\
  --expected-runtime-authorization-sha256 {auth} \\
  -- \\
  --execution-admission $OWNER/exp1b_execution_admission_20260913_run5.json \\
  --evidence-root {root} --evidence-generation {gen} \\
  --seed-position "$1"
rc=$?
echo "STAGE B POSITION $1 EXIT=$rc"
exit $rc
""".format(**common)

lane = """#!/bin/bash
# Real Stage A lane driver, chain run5: $1=gpu, rest = seed positions.
gpu=$1; shift
REG=/home/buiksat/trm_bellman/configs/policy_improvement_exp1b/registry.json
for pos in "$@"; do
  read -r rid seed < <(python3 -c "
import json
rows={{r['seed_position']:r for r in json.load(open('$REG'))['rows']}}
r=rows[int('$pos')]
print(r['run_id'], r['seed'])
")
  [ -n "$rid" ] && [ -n "$seed" ] || {{ echo "registry lookup failed for position $pos"; exit 1; }}
  echo "=== lane gpu$gpu seed_position $pos ($rid) start $(date '+%H:%M:%S') ==="
  /home/buiksat/real-stage-a-run5.sh "$pos" "$rid" "$seed" "$gpu" \\
    > /tmp/run5-seed$pos.log 2>&1
  rc=$?
  echo "=== lane gpu$gpu seed_position $pos done rc=$rc $(date '+%H:%M:%S') ==="
  [ $rc -eq 0 ] || {{ echo "LANE $gpu ABORT at position $pos"; exit $rc; }}
done
echo "LANE $gpu COMPLETE"
""".format()

# Stage B has no lane: it is strictly sequential in registered seed order and the
# ordering is part of what the audit re-derives. This only sequences and aborts.
lane_b = """#!/bin/bash
# Real Stage B driver, chain run5: args = seed positions, in registered order.
for pos in "$@"; do
  echo "=== stage B position $pos start $(date '+%H:%M:%S') ==="
  /home/buiksat/real-stage-b-run5.sh "$pos" > /tmp/run5-stageb-$pos.log 2>&1
  rc=$?
  echo "=== stage B position $pos done rc=$rc $(date '+%H:%M:%S') ==="
  [ $rc -eq 0 ] || {{ echo "STAGE B ABORT at position $pos"; exit $rc; }}
done
echo "STAGE B COMPLETE"
""".format()

for name, body in (
    ("/home/buiksat/real-stage-a-run5.sh", stage_a),
    ("/home/buiksat/real-stage-b-run5.sh", stage_b),
    ("/home/buiksat/real-lane-run5.sh", lane),
    ("/home/buiksat/real-laneb-run5.sh", lane_b),
):
    with open(name, "w") as out:
        out.write(body)
    os.chmod(name, 0o755)
    print("wrote", name)
PY

echo "=== run5 chain minted ==="
