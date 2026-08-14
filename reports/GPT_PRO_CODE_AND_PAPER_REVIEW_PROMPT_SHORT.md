# GPT Pro cross-repository review prompt

Read and obey `reports/CHATGPT_PRO_CODE_AND_PAPER_REVIEW_PROMPT.md` in the
`gopeshh/trm_bellman` `full-implementation` branch. Adversarially review the
exact current heads of that branch and `buiksat/UPI_TRM` branch
`iclr-evidence-aligned-revision`, including the tracked
`UPI_TRM_ICLR/main.pdf`.

Work read-only. Do not edit either repository, train or evaluate a model, run
experiments, or load learned payloads. Independently run every prescribed
paper, Buck, launcher, manifest, and static gate. Default-reject candidate
findings and reconcile every prior `UPITRM-*` item.

Focus on the Phase 4 repairs: pre-import sealed runtimes; independent producer
commit, manifest, and training-PAR identity; strict schema-4 checkpoint and
resume-state validation; exact environment replay of retained transitions; one
training runtime across all 12 cells; staged publication; rooted paths; the
complete imported source closure; and deterministic supported Phase 4 data.
Return a full evidence report. Use `INCOMPLETE` if either exact head, the
tracked PDF, or any required gate is unavailable.
