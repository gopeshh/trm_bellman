# Independent Stage B replay

`output/seed-N.json` contains the lossless per-state export for one sealed checkpoint, and each export has a raw-file SHA-256 sidecar. The eight exports contain 128 census states each. They record Bellman inputs and outputs, policies, masks, rewards, terminals, successor and latent identity data, clocks, advantages, secondary diagnostics, and model-state identities before and after replay.

`recomputation.json` is the output of the separate standard-library consumer. It reports bitwise agreement for all eight signed gaps and their mean, the frozen bootstrap interval, and the shared residual-maximizing state. `comparison-rules.md` was frozen before the first checkpoint replay.

`build_identity.json` records the replay producer used for every per-state export. `consumer_build_identity_final.json` records the independently built final consumer. The source used for both new targets and the matching BUCK file are under `source/`.

The export files are large and must remain outside git. To regenerate them:

1. Start from repository revision `2a14eb656bd983c37b60e92852083524348511d8` in a clean checkout.
2. Install the three files under `source/` into the matching project paths.
3. Build `fbcode//buiksat_trm:policy_improvement_exp1b_replay` and `fbcode//buiksat_trm:policy_improvement_exp1b_replay_consumer` with `@fbcode//mode/opt`. Do not rebuild or relabel a production PAR.
4. Verify the new binary identities against a newly recorded build-identity document.
5. Run the producer once for each registered seed position against a read-only copy of the sealed evidence generation, writing only to a new replay namespace.
6. Run the independent consumer over all eight completed exports under the frozen comparison rules.

The exact successful and pre-output failed invocation logs are under `../logs/replay/`.
