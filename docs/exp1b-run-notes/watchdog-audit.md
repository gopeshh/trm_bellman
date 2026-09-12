# UPI-TRM watchdog audit

Date: 2026-09-12 09:25 PDT

## Verdict

**FAIL. The 16-hour failure can recur.** The loop catches the exact idealized case only while its process, Herdr server, pane identity, JSON schema, host, and `/tmp` state all survive. The Claude cron cannot act if its bound session is dead or busy. There is no OS-level supervisor or external dead-man check.

## Findings

### High: the two monitoring layers share the failure modes that matter

- `/tmp/upi-trm-watchdog.sh:13-65` is an unsupervised `nohup` loop. Reboot, OOM kill, SIGKILL, or process-tree/cgroup teardown ends it with no alert and no restart.
- Job `53b17d67` is a Claude session scheduler, not an OS cron. The creation record says it is durable, session-bound, runs only while the REPL is idle, and auto-expires after seven days. `crontab -l` reports no crontab; no UPI-TRM systemd timer exists.
- If the coordinator session dies, layer 2 cannot act. Layer 1 may still write a file, but nobody is guaranteed to read it. If the host or Herdr server is unavailable, neither layer can inspect or recover agents.

The layers are different code paths, but they are not independent end to end. Both depend on this host, Herdr, `/tmp`, and the Claude session for remediation.

### High: the watchdog monitors every non-coordinator pane, not the pipeline owner

`/tmp/upi-trm-watchdog.sh:34-47` treats every status other than exact `working` as stalled, excluding only `w1:p8`.

Current proof: `w1:p5` is an unrelated idle Claude pane. It has been accumulating idle time since the watchdog started at 09:17 and will create a false alert around 09:37 even though implementer2 is working and seed 0 is training. This will train the cron to act on noise and can re-prompt or disturb unrelated agents.

The inverse also fails: an agent can remain `working` while blocked in a hung tool or dead progress loop forever. The daemon checks no process, GPU, log, or evidence heartbeat before declaring it healthy.

Fix direction: monitor explicit pipeline roles/pane IDs and an expected phase, not all panes. Couple agent state to a phase-specific heartbeat: live child PID, GPU/CPU activity, seed log mtime, and checkpoint/evidence progression.

### High: parser failure can silently erase a real alert

At `/tmp/upi-trm-watchdog.sh:23-29`, the JSON parser's exit status is ignored. Valid JSON with a changed shape, a missing key, or a non-list `agents` field yields an empty `line`. The script then finds no stalls and deletes the alert at `:60-62`.

`herdr agent list` also has no timeout at `:17`. A hung CLI blocks the watchdog indefinitely with no new log line or alert. Empty output is handled only by a log message at `:18-20`; it does not create a watchdog-health alert.

Fix direction: require successful command and schema validation before touching the existing alert; use a bounded timeout; emit a distinct monitor-failure alert; keep a last-success timestamp checked by an independent supervisor.

### High: an agent disappearing from the list clears its stall

`idle_since` entries are never reconciled against a seen-pane set. If implementer2 becomes idle and later disappears from `herdr agent list`, the stale array entry remains in memory but is never iterated. `stalled` becomes empty and `:61` removes the alert.

The same happens if a pane is replaced with a new pane ID: its 20-minute clock resets. A disappearing agent should be a failure until explicitly retired or replaced by a named successor.

### Medium: both progress checks point at an abandoned evidence root

The daemon uses `/home/buiksat/upi-trm-owner-20260911/evidence` at `/tmp/upi-trm-watchdog.sh:55`. The Claude cron prompt uses the same path.

The active runtime command uses:

```text
/home/buiksat/upi-trm-owner-20260911/evidence2
generation exp1b-20260912
```

The requested old root contains one stale `.lock`; the active root also currently contains one new `.lock`. Future checkpoints will appear only in `evidence2`, so both watchdog layers will report a frozen count even when the run advances.

### Medium: watchdog state is process-local and `/tmp`-fragile

- `/tmp/upi-trm-watchdog.sh:9` declares `STATE`, but never reads or writes it.
- A process restart forgets every idle start and grants a fresh 20 minutes.
- The alert is overwritten, not append-only or atomic, and is removed on any apparently healthy snapshot.
- There is no singleton lock. Starting a second daemon lets two independent timers race to overwrite or delete one alert.
- A reboot kills the process and normally clears `/tmp`. Manual or tmpfiles cleanup can remove the script, alert, and history; a running Bash may continue from its open script/process state, but it cannot be restarted from the deleted file.

Persist state and alerts under `$XDG_STATE_HOME` or another owner-controlled home path, use a singleton lock, atomic writes, and retain incident history.

## Question a: yesterday's failure simulation

Under ideal conditions, yes:

- First poll after implementer2 becomes idle: `idle_since[pane]` is set.
- At t+20m: because polling is every 120 seconds and minutes are floored, the alert appears between t+20m and just under t+22m.
- At t+2h: the alert is rewritten every two minutes with the larger idle duration.
- At t+16h: it still exists and is rewritten, provided the same pane remains listed idle and every parse succeeds.

It is not durably persistent. One malformed-but-nonempty response deletes it; disappearance of the pane deletes it; daemon death stops updates; `/tmp` cleanup can remove it. The Claude cron acts only at the next `:07` or `:37` tick and only if its session is alive and idle.

## Question b: failure behavior

| Event | Actual behavior |
| --- | --- |
| Herdr server transiently returns no stdout | Daemon survives, logs `ERROR`, sleeps. Existing alert remains; no new alert can be created during outage. |
| `herdr agent list` hangs | Daemon hangs silently because there is no timeout. |
| Valid JSON with unexpected shape or missing fields | Python exits nonzero; status is ignored; empty parse is treated as healthy and existing alert is deleted. |
| Machine reboot | Daemon dies, `/tmp` may clear, no restart unit exists. Silent. |
| OOM/SIGKILL | Daemon dies; log simply stops. Silent. |
| `/tmp` cleared while process remains alive | Existing script/log/alert vanish. The loop may continue and recreate log/alert on a later poll, but restartability and history are lost. Reboot plus clear is total loss. |
| Parent shell exits normally | `nohup` should survive SIGHUP. Current watchdog is still a child of a detached wrapper shell; a session/cgroup teardown or SIGTERM/SIGKILL can still kill both. |
| Disk full or alert write failure | Redirection failure is unchecked; loop can continue without a usable alert. |

## Question c: Bash and threshold details

- Associative-array state does survive iterations while PID `852397` lives.
- The `while ... done <<< "$line"` loop is fed by a here-string, not a pipe, so it runs in the current Bash process. Array updates survive.
- `unset "idle_since[$pane]"` works for the current associative-array keys; a direct shell probe confirmed it.
- Pane disappearance is a real bug, described above.
- Twenty minutes is reasonable for an idle orchestrator waiting on a decision. It is not a valid seed timeout. A 1.9-hour seed should be judged by live PID/GPU/log heartbeat and a phase-specific upper bound, not agent `idle` alone. A background seed can be healthy while its agent is idle; a stuck agent can remain `working` indefinitely.

## Question d: cron independence

If the Claude session dies, the durable job does not execute until that session exists again. It does not provide an independent remediation path. The daemon can survive a normal shell exit, but it can only write an alert. It cannot wake or replace the dead coordinator.

The cron also shares the daemon's broad-pane false positives and stale evidence path. Standing commit authority successfully removes yesterday's specific commit wait, but it does not supervise process death, hung tools, agent disappearance, or a dead coordinator.

## Question e: materially better mechanism

Use a systemd user service plus timer, not the Claude cron or a naked nohup loop.

This host is ready for it:

- `systemctl --user is-system-running` returns `running`;
- `loginctl show-user buiksat` reports `Linger=yes`, so the user manager survives logout and starts after reboot;
- no real user crontab or UPI-TRM systemd timer currently exists.

Recommended design, not implemented:

1. Put the monitor and persistent state under the home directory, not `/tmp`.
2. Run a systemd user service with `Restart=always`, bounded command timeouts, a singleton instance, and journal logging.
3. Use a persistent timer or a continuously supervised service. Add `OnFailure` notification that does not depend on the Claude REPL.
4. Make each check stateless or persist per-role phase/heartbeat data, so a monitor restart does not reset a stall clock.
5. Track the active evidence generation from a pinned state file, not a hard-coded path.

A real crontab with `@reboot`, a frequent check, and `flock` would also beat the current Claude cron, but systemd provides restart policy, boot persistence, status, logs, and failure hooks. Neither protects against the whole host being down; a hard guarantee requires an external heartbeat/dead-man monitor on another machine or service.

## Live pipeline verification

At 09:20-09:24 PDT:

- Herdr: `w1:p9` (`implementer2`) reports `working`; `w1:p8` coordinator and `w1:p4` reviewer are working; unrelated `w1:p5` is idle.
- GPU 0: 24-29% utilization, 592 MiB after a short duplicate probe ended. GPU 1: 0%, 6 MiB.
- One real `policy_improvement_full` process remains, seed `2081976412`, row `exp1b-fixed-base-exact-persistent-seed2081976412`, started 09:17. It was using about 90% of one CPU and the GPU.
- `/tmp/seed0.log` shows completed train steps 0-4. A second identical seed-0 process launched under `timeout 300` at 09:19 and ended at 09:24; it caused temporary duplicate work but did not replace the intended run.
- Requested evidence root `/home/buiksat/upi-trm-owner-20260911/evidence`: **1 file**, the abandoned generation's `.lock`.
- Active evidence root `/home/buiksat/upi-trm-owner-20260911/evidence2`: **1 file**, the fresh generation's `.lock`. No checkpoint is expected only minutes into a roughly 1.9-hour seed.
- `/tmp/upi-trm-run-handoff.md`: mtime `2026-09-12 09:16:31.785103995 -0700`, SHA-256 `fe4e5226f90085a71e32a933ba55185c96797943d5f54dbcfbb15a642cab836d`.
- Git HEAD: `5a0ad34e119a37e44cc49edf00a93264cf63d3b1`, committed `2026-09-12T09:11:03-07:00`, subject `Seal with this backend's training module, and test the real body`; branch is 8 commits ahead of origin.

**Conclusion: implementer2 is genuinely progressing, not merely reporting `working`.** The live process, CPU/GPU activity, recent commit/handoff, and training log prove execution. Operational quality is imperfect: the five-minute duplicate seed consumed the same GPU, and only GPU 0 is active. The evidence count cannot show progress until the seed seals.

## Bottom line

The current setup catches yesterday's exact idle-pane scenario only when everything else stays healthy. It has no self-supervision, no host/session-independent remediation, false-positive pane selection, a disappearing-agent blind spot, parser-driven alert deletion, and a stale evidence path. Do not rely on it as a hard guarantee.
