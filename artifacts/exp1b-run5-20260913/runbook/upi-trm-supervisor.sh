#!/bin/bash
# UPI-TRM experiment supervisor.
#
# A state machine, not a babysitter. Every tick it looks at what is on disk,
# works out the next thing the pipeline owes, and does it. Nothing here depends
# on an agent staying awake: if the box reboots, the next cron tick picks up
# exactly where the evidence tree left off.
#
# Which chain it drives is whatever /home/buiksat/.upi-trm-chain.conf points at,
# so promoting from the scratch rehearsal to the real run is a one-line switch.
#
# Safe to run from cron as often as you like. Ticks are serialised by flock, so
# a tick that lands during a 7-hour Stage A just exits.
set -u

# cron gets a minimal PATH. herdr lives in ~/.local/bin and buck2 in /usr/local/bin;
# without this the agent-liveness block fails silently, which it did on the first
# version of this script.
export PATH=/home/buiksat/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
# herdr resolves its socket from $HOME/.config/herdr/herdr.sock.
export HOME=${HOME:-/home/buiksat}

CONF=${1:-/home/buiksat/.upi-trm-chain.conf}
[ -r "$CONF" ] || { echo "no chain config at $CONF" >&2; exit 2; }
# shellcheck disable=SC1090
. "$CONF"

LOG=/home/buiksat/upi-trm-supervisor.log
LOCK=/home/buiksat/.upi-trm-supervisor.lock

exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0

log() { echo "$(date '+%F %T') [$CHAIN_NAME] $*" >> "$LOG"; }

seals()    { ls -d "$GEN"/checkpoints/seed-* 2>/dev/null | wc -l; }
payloads() { ls "$GEN"/payloads/seed-*.json  2>/dev/null | wc -l; }
busy()     { pgrep -f "$BUSY_PATTERN" >/dev/null 2>&1; }

# Positions whose Stage A seal is missing, keeping each lane on its own GPU.
missing_a() { for p in "$@"; do [ -d "$GEN/checkpoints/seed-$p" ] || printf '%s ' "$p"; done; }

write_status() {
  {
    echo "# $CHAIN_NAME status  (updated $(date '+%F %T'))"
    echo
    echo "supervisor state: $1"
    echo
    echo "Stage A seals: $(seals)/8    Stage B payloads: $(payloads)/8"
    echo "provenance.json:   $([ -f "$GEN/provenance.json" ]   && echo present || echo absent)"
    echo "exp1b_result.json: $([ -f "$GEN/exp1b_result.json" ] && echo present || echo absent)"
    echo
    echo '## Stage A lanes'
    echo '```'
    tail -n 4 "$LANE0_LOG" 2>/dev/null
    tail -n 4 "$LANE1_LOG" 2>/dev/null
    echo '```'
    echo
    echo '## GPUs'
    echo '```'
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
    echo '```'
    if [ -f "$AUDIT_LOG" ]; then
      echo
      echo '## Audit'
      echo '```'
      grep -E 'exp1b-audit|AUDIT EXIT' "$AUDIT_LOG" 2>/dev/null | tail -n 12
      echo '```'
    fi
    echo
    echo '## Agents'
    if command -v herdr >/dev/null 2>&1; then
      herdr agent list 2>/dev/null | python3 -c "
import json,sys
raw = sys.stdin.read().strip()
if not raw:
    print('- herdr returned nothing (server down?)')
else:
    try:
        for a in json.loads(raw)['result']['agents']:
            flag = '  <-- IDLE' if a['agent_status'] == 'idle' else ''
            print(f\"- {a['agent']} ({a['pane_id']}): {a['agent_status']}  cwd={a['cwd']}{flag}\")
    except Exception as exc:
        print('- agent list unparseable:', exc)
"
    else
      echo "- herdr not on PATH"
    fi
  } > "$STATUS".tmp && mv "$STATUS".tmp "$STATUS"
}

# ---------------------------------------------------------------- terminal state
if [ -f "$GEN/.audit-passed" ]; then
  write_status "DONE - $CHAIN_NAME audited clean"
  exit 0
fi

if busy; then
  write_status "running (Stage A seals $(seals)/8, Stage B payloads $(payloads)/8)"
  exit 0
fi

# ---------------------------------------------------------------- stage A
if [ ! -f "$GEN/provenance.json" ]; then
  m0=$(missing_a $LANE0_POS)
  m1=$(missing_a $LANE1_POS)
  if [ -n "$m0$m1" ]; then
    log "Stage A incomplete and nothing running; relaunching lane0=[$m0] lane1=[$m1]"
    [ -n "$m0" ] && setsid nohup "$LANE" 0 $m0 >> "$LANE0_LOG" 2>&1 < /dev/null &
    [ -n "$m1" ] && setsid nohup "$LANE" 1 $m1 >> "$LANE1_LOG" 2>&1 < /dev/null &
    write_status "relaunched Stage A (lane0=[$m0] lane1=[$m1])"
    exit 0
  fi
  # All eight seals on disk but no provenance.json: the gate did not close.
  # Not something to retry blindly, so stop and make it visible.
  log "ALERT: 8 seals present but provenance.json absent - the provenance gate did not close"
  write_status "BLOCKED - 8 seals but no provenance.json; needs a human"
  exit 1
fi

# ---------------------------------------------------------------- stage B
# Strictly sequential in registered order; position 7 publishes the result.
if [ "$(payloads)" -lt 8 ]; then
  for p in 0 1 2 3 4 5 6 7; do
    [ -f "$GEN/payloads/seed-$p.json" ] && continue
    log "Stage B position $p starting"
    write_status "Stage B position $p running"
    "$STAGEB" "$p" > "${STAGEB_LOG_PREFIX}${p}.log" 2>&1
    rc=$?
    log "Stage B position $p rc=$rc"
    if [ $rc -ne 0 ]; then
      log "ALERT: Stage B aborted at position $p; not advancing"
      write_status "BLOCKED - Stage B failed at position $p (see ${STAGEB_LOG_PREFIX}${p}.log)"
      exit $rc
    fi
  done
fi

# ---------------------------------------------------------------- audit
if [ -f "$GEN/exp1b_result.json" ]; then
  log "running the audit"
  /home/buiksat/audit-exp1b.sh "$AUDIT_CELL" "$AUDIT_ROOT" "$GEN" "$ADMISSION" > "$AUDIT_LOG" 2>&1
  if grep -q '^AUDIT EXIT=0$' "$AUDIT_LOG"; then
    : > "$GEN/.audit-passed"
    log "AUDIT PASSED - $CHAIN_NAME is green"
    write_status "DONE - $CHAIN_NAME audited clean"
  else
    log "ALERT: audit did not pass; see $AUDIT_LOG"
    write_status "BLOCKED - audit failed (see $AUDIT_LOG)"
    exit 1
  fi
else
  log "ALERT: Stage B finished but no exp1b_result.json was published"
  write_status "BLOCKED - Stage B complete but no published result"
  exit 1
fi
