Blunt read: this paper was not rejected because the core idea was dead. It was rejected because reviewers could not parse the paper, and one of them flatly did not trust the baseline story. Everything else sits behind those two failures.

One logistics correction. NeurIPS 2026 abstract submission is due May 4 AOE, but the full paper and supplementary are due May 6 AOE. NeurIPS gives you 9 content pages instead of ICML's 8, with references, appendices, and the mandatory checklist outside the page limit, all in one PDF. Reviewers are not required to read supplementary, so anything critical must move into the main paper.

My synthesis of the reviews is simple. Reviewer cE8u killed the paper on clarity. Reviewer X1M8 was willing to move if the baseline story and presentation were fixed. Reviewer hD9M wanted sharper novelty positioning and stronger baselines. Reviewer gZb4 was broadly positive but wanted more seeds and a cleaner theory-to-practice story. That is a salvageable rejection.

## 0. Status Update (April 10)

- `M1` hard-4x4 no-mask capability package is locked: UPI `0.574 ± 0.122` over `10` seeds vs in-house A2C `0.000 ± 0.000` over `4` seeds. Masked hard-4x4 is retired as a capability anchor and kept only as a "constraints given away" control.
- The deconfounded no-mask hard-4x4 controlled 2x2 is also locked: `nc_r0 = 0.350 ± 0.089`, `nc_r10 = 0.482 ± 0.071`, `c_r0 = 0.374 ± 0.078`, `c_r10 = 0.502 ± 0.105`.
- Interpretation from that 2x2: projection gives a robust `+0.13` main effect; contraction gives only about `+0.02`; interaction is effectively zero; contraction does not reduce variance under projection.
- Consequence: the remaining empirical blocker is `exp1_v4` (`PF2` / `PF3`). That is now the only place contraction can earn a meaningful main-text role.
- Optional only: backfill in-house no-mask A2C seeds `4-9` if matched seed counts or reviewer optics become necessary. This is not on the critical path.

## 1. Prioritized concerns

Assume 1 day = 5–6 focused hours. The days below are not additive because experiment wall-clock can overlap with writing.

| Rank | Concern | Severity / Feasibility | Needs | Effort | Risk if left unaddressed |
|------|---------|------------------------|-------|--------|--------------------------|
| 1 | Front-half rewrite: plain-language problem statement, reward/objective, original TRM two-loop recap, TRM→MDP mapping, early definitions of f_θ, V_ψ, Π_R, and why the three phases exist | Critical / High | Rewriting | 5 days | Fatal |
| 2 | Baseline credibility: reviewers do not believe the 0% PPO/A2C/DQN result. Re-run at least one trusted recurrent PPO/A2C baseline from a known codebase and add one algorithmically closer baseline if feasible (n-step DQN is the obvious reviewer-requested choice) | Critical / Medium | New experiments + rewriting | 6–8 days | Fatal |
| 3 | More seeds on the key claims: hard 4×4 main comparison, isolation study, best-vs-best 9×9 only if cheap | High / High | New experiments | 3–4 days | High |
| 4 | Novelty positioning: separate what is standard (API/CPI shell, K-step returns, conservative mixture) from what is new (internal evaluator truncation decomposition, L_z^n bias, CPI interaction). Move the bridge from Prop. 5.1 to Thm. 5.6 into the main text so reviewers stop missing it | High / High | Rewriting + minor theory addition | 2–3 days | High |
| 5 | Fix the contraction/projection story: main text currently invites the wrong reading. Move the controlled 2×2 projection×contraction result into the main paper and stop treating contraction as a generic performance knob | Medium-high / High | Rewriting + maybe 1 confirmatory run | 2 days | Medium-high |
| 6 | Theory/practice gap: tiny controlled L_V experiment if you can make it work quickly; otherwise sharpen the limitation and stop hinting at quantitative predictiveness | Medium / Medium-low | New experiment or explicit concession | 2–4 days | Medium |
| 7 | Broader task than Sudoku | Medium / Low | New experiments | 5+ days | Low-medium |

Top 5 are must-do. Item 6 is a stretch goal. Item 7 is a trap unless you already have another checker task nearly ready.

Two concrete calls inside that ranking:
- Do not headline the 1–4-empty toy suite. A 52% random baseline makes it a sanity check, not a persuasive result.
- Do not let 9×9 carry the empirical story unless you can add seeds. A 4% mean over 3 seeds is too fragile. If compute is tight, hard 4×4 should be the main anchor and 9×9 should become supporting evidence.

## 2. Four-week revision schedule

Use your coauthor almost entirely on reruns and seed sweeps. You should own the rewrite, positioning, and theorem surfacing.

| Week | Focus | Concrete milestones | Buffer |
|------|-------|---------------------|--------|
| Week 1 | Story surgery | Rewrite Intro + Setup completely. Add one figure: original TRM loops → your MDP abstraction. Replace Table 1 with a clean split between MDP elements, internal evaluator, and learning rule. Delete duplicated/overlapping algorithm exposition. Lock the exact empirical claim you will defend. Launch seed sweeps and trusted baseline reruns by the end of the week. | 0.5–1 day |
| Week 2 | Empirical credibility | Finish hard 4×4 best-vs-best reruns. Add more seeds to the isolation study. Run at least one trusted PPO/A2C baseline and, if implementation cost is sane, one closer baseline such as n-step DQN. Draft the new experiments section as results land. | 1 day |
| Week 3 | Integrate results and tighten the theory story | Move the deconfounded projection×contraction result into the main text. Rewrite related work/positioning. Add one main-text displayed equation or corollary showing how L_z and n enter the CPI penalty through the advantage-error bound. Spend max 2 days on a tiny L_V-controlled experiment; abort if it destabilizes again. Circulate full draft to coauthors. | 1–1.5 days |
| Week 4 | Compression, packaging, submission | Compress to 9 content pages. Finalize appendix, checklist, anonymization, and code/data ZIP. Do one brutal pass on prose for density and redundancy. Freeze title/abstract internally by May 2, submit abstract by May 4, and treat May 5 as your real paper deadline even though NeurIPS allows until May 6 AOE. | 3–4 days |

This plan uses about 95–110 focused hours. That fits your stated availability. It does not fit if you decide to add a second serious benchmark.

A hard gate: by the end of Week 2, you either have a credible stronger-baseline story or you narrow the claim. No middle ground.

## 3. Reviewer complaints that were misguided or based on misunderstanding

| Complaint | My read | Response |
|-----------|---------|----------|
| "This sounds AI-generated" | Misguided as a substantive objection. It is a readability complaint wearing a cheap accusation. | Do not rebut the accusation. Rewrite ruthlessly. |
| "Inner unroll is just longer-horizon lookahead" | Partly fair, mostly incomplete. The API/CPI shell is standard. Your n is architecture-local computation inside one state evaluation, not environment rollout depth. | Clarify and partially concede. Add a one-paragraph contrast with standard lookahead and VIN. |
| "TRM is just an RNN, so PPO/A2C should work" | Partial misunderstanding, but the empirical skepticism is fair. | Clarify + add stronger evidence. You cannot talk your way out of this one. |
| "z should be part of the MDP state / y_0 is actually latent z" | Mostly a confusion between your abstraction and the original TRM mechanics. | Clarify, do not concede. Keep episodic-z as internal evaluator state, not environment state. |
| "You need strict quantitative validation of the bound" | Fair request, but not mandatory if you stop overselling. | Concede unless a tiny controlled study lands quickly. |
| "Sudoku is not that hard" | Fair caveat. | Concede and reframe. Call it a controlled verifiable-reward mechanism domain, not a reasoning benchmark. |
| "Why no VIN comparison?" | Fair related-work criticism. Direct experimental comparison is optional. | Fix in positioning. Cite VIN earlier and state why it is related but not the same object. |

## 4. Structural changes for a NeurIPS resubmission

NeurIPS gives you one extra content page over ICML, and you should spend it on main-text trust, not more appendix spillover. Both venues make the same practical point: critical material belongs in the paper body. ICML says reviewers may ignore appendices/supplementary and that anything critical to evaluation must be in the main body. NeurIPS says reviewers are not required to read supplementary. Use the extra page for one baseline-sanity table and the controlled projection×contraction result.

NeurIPS 2026 also asks you to choose a contribution type, and reviewers are instructed to evaluate in that context. The available types include General, Theory, Use-Inspired, Concept & Feasibility, and Negative Results. My recommendation is:
- **Default:** General
- **Fallback if the empirical package stays thin:** Theory
- **Avoid:** Concept & Feasibility, because NeurIPS explicitly says the bar is high for that type.

Packaging-wise, NeurIPS requires a single PDF containing paper, references, appendices, and the mandatory checklist. Code/data can go in a separate anonymized ZIP. Put your baseline scripts and exact hyperparameters there if you have them.

The paper structure should change to this:
1. Plain-language problem and why checker-feedback TRM training is hard.
2. Original TRM two-loop mechanism.
3. Your plan-space MDP abstraction.
4. UPI-TRM algorithm in one section, not two.
5. Main theoretical questions and the bridge equation/corollary.
6. Experiments centered on hard 4×4 + deconfounded stability story.
7. Related work positioned around lookahead, VIN, recurrent RL, and CPI.

I would also seriously consider softening the title/abstract. Right now the title invites "new RL algorithm" expectations. The real novelty is narrower: policy improvement with compute-truncated internal evaluators in TRMs.

## 5. What to push back on

1. Do not turn this into a benchmark war. One trusted baseline family plus one closer baseline and more seeds is enough. Five more baselines will burn time and still not settle the argument.
2. Do not chase an end-to-end convergence theorem. That is a bad 4-week bet.
3. Do not sell contraction as a performance booster. The locked no-mask 10-seed hard-4x4 2x2 says projection is the dominant stabilizer (`+0.13`), while contraction contributes only a small average lift (`+0.02`) with no meaningful interaction.
4. Do not concede that the contribution is "just standard lookahead." Concede that the API/CPI shell is standard. Defend the internal-evaluator decomposition and its coupling to conservative improvement.
5. Do not spend a week implementing VIN unless code is already sitting there. Cite and contrast it. That likely suffices.

Hard verdict: rewrite the first three pages from scratch, fix baseline trust with one credible rerun package, and move the deconfounded empirical story into the main text. If you do anything else first, you are likely to get rejected again.

---

## 6. Task Decomposition (April 5 -- May 4)

**Calendar:** 29 days. ~5-6 focused hours/day. Abstract deadline May 4 AOE. Paper deadline May 6 AOE. Internal paper freeze: May 3.

**Parallelism key:** Tasks marked `[PARALLEL-OK]` can run concurrently with writing tasks. GPU experiment wall-clock overlaps with writing time. Coauthor handles experiment execution; you own all writing.

---

### PHASE 0: Setup & Experiment Launch (April 5--6)

Execution doc for this phase: `PHASE0_EXPERIMENT_QUEUE.md`

- [ ] **T0.1 -- Lock empirical claims and experiment queue**
  - Description: Decide the exact set of claims the paper will defend. Write a 1-page internal doc listing: (a) primary comparison = hard 4x4 (6-8 empties), (b) supporting = 9x9 if seeds land, (c) stability story = deconfounded 2x2 projection x contraction. List every experiment run needed with seed counts and hyperparameters.
  - Est. hours: 3
  - Deadline: April 5
  - Depends on: nothing
  - Acceptance: Single markdown file listing every run, its config, expected wall-clock, and which claim it supports. Coauthor can start runs from this doc alone.
  - `[PARALLEL-OK with T0.2]`

- [ ] **T0.2 -- Set up trusted external baseline codebase**
  - Description: Install CleanRL or Stable-Baselines3. Verify PPO and A2C run on a trivial env. Adapt the Sudoku env wrapper so the external codebase can call it. Do NOT tune yet.
  - Est. hours: 5
  - Deadline: April 6
  - Depends on: nothing
  - Acceptance: `python run_baseline.py --algo ppo --env sudoku4x4` completes one episode without error. Env wrapper passes the same action space / observation space as your internal code.
  - `[PARALLEL-OK with T0.1]`

- [ ] **T0.3 -- Launch all long-running experiment jobs**
  - Description: Kick off every GPU job from the T0.1 queue: (a) hard 4x4 UPI-TRM seed sweep (10 seeds), (b) hard 4x4 trusted PPO/A2C from external codebase (10 seeds each), (c) isolation study additional seeds (7 more seeds, total 10), (d) 9x9 additional seeds (7 more, total 10) if compute allows. Set up logging so results land in a shared directory.
  - Est. hours: 4
  - Deadline: April 6
  - Depends on: T0.1, T0.2
  - Acceptance: All jobs are queued/running. A monitoring script or dashboard shows progress. Expected completion: April 12-14 for 4x4 runs, April 16-18 for 9x9.
  - `[PARALLEL-OK -- these run in background for 7-12 days]`

---

### PHASE 1: Story Surgery (April 7--13)

- [ ] **T1.1 -- Outline the new paper structure on paper**
  - Description: Write a section-by-section outline of the rewritten paper following the target structure: (1) plain-language problem, (2) original TRM mechanism, (3) plan-space MDP, (4) UPI-TRM algorithm (single section), (5) theory bridge, (6) experiments, (7) related work. For each section: 1-sentence summary, key equations/figures, and approximate page budget (total: 9 content pages).
  - Est. hours: 3
  - Deadline: April 7
  - Depends on: T0.1
  - Acceptance: Outline fits in 9 pages with specific allocations. Every reviewer complaint from Section 3 of NIPS_PLAN.md has a home in the outline.

- [ ] **T1.2 -- Rewrite Section 1 (Introduction) from scratch**
  - Description: New intro in plain language. Lead with the problem (TRM trained from checker feedback), not the formalism. State three things up front: what TRM is, what the plan-space MDP is, and why truncated evaluation makes standard RL analysis insufficient. End with a crisp 3-item contribution list. Include a forward pointer to the key equation showing L_z^n bias entering the CPI bound. Reframe Sudoku as a controlled verifiable-reward domain, not a reasoning benchmark.
  - Est. hours: 8
  - Deadline: April 9
  - Depends on: T1.1
  - Acceptance: A non-expert can read the first page and explain (a) what problem you solve, (b) why it is hard, (c) what the paper contributes. No undefined notation in the first 1.5 pages.

- [ ] **T1.3 -- Create TRM-to-MDP figure**
  - Description: Design and produce one figure showing: left = original TRM two-loop mechanism (inner latent refinement + outer edit loop), right = your plan-space MDP abstraction with the evaluator as an internal component. Annotate the mapping: inner loop = evaluator with depth n, outer loop = MDP transitions, checker = reward. Use TikZ or a clean vector tool.
  - Est. hours: 4
  - Deadline: April 9
  - Depends on: T1.1
  - Acceptance: Figure is self-contained. A reader unfamiliar with TRM can trace the mapping without reading surrounding text.
  - `[PARALLEL-OK with T1.2]`

- [ ] **T1.4 -- Rewrite Section 2 (Background/Setup) from scratch**
  - Description: Combine the original TRM recap (currently missing from short mode) and the plan-space MDP definition into one section. Define f_theta, V_psi, Pi_R early. Explicitly state episodic-z vs persistent-z distinction with a 2-sentence explanation each. Replace Table 1 with a cleaner version that separates MDP elements, internal evaluator components, and learning-rule knobs into three groups. Add the VIN/lookahead contrast paragraph here (reviewer hD9M).
  - Est. hours: 8
  - Deadline: April 11
  - Depends on: T1.2, T1.3
  - Acceptance: All notation used in later sections is defined here. The episodic-z / persistent-z distinction is clear in under 4 sentences. Table 1 replacement has three visually distinct groups.

- [ ] **T1.5 -- Rewrite Section 3 (Algorithm) as a single unified section**
  - Description: Merge the current split between short-mode algorithm sketch and long-mode full algorithm into one presentation. Show Algorithm 1 pseudocode in the main text. Remove redundant exposition. The section should be at most 1 page.
  - Est. hours: 5
  - Deadline: April 12
  - Depends on: T1.4
  - Acceptance: One algorithm box, one page. A reader can implement the training loop from this section plus the appendix. No forward references to undefined quantities.

- [ ] **T1.6 -- Soften title and rewrite abstract**
  - Description: Draft 2-3 candidate titles that frame the contribution as "policy improvement under compute-truncated evaluation" rather than "new RL algorithm." Rewrite abstract to match the new intro. Remove jargon-heavy first sentence. Lead with the practical motivation.
  - Est. hours: 3
  - Deadline: April 13
  - Depends on: T1.2
  - Acceptance: Abstract is under 200 words. Title does not promise more than the paper delivers. At least 2 candidate titles for coauthor vote.
  - `[PARALLEL-OK with T1.4, T1.5]`

**Phase 1 checkpoint (April 13):** Sections 1-3 are rewritten. New figure exists. Experiments from T0.3 are mid-run. Coauthor reviews the rewritten front half and flags gaps.

---

### PHASE 2: Empirical Credibility (April 14--20)

- [ ] **T2.1 -- Collect and validate hard 4x4 seed sweep results**
  - Description: Pull results from the 10-seed hard 4x4 runs (UPI-TRM and trusted baselines). Compute mean, std, 95% CI. Verify the external baseline numbers are comparable to your original 0% claim or document any differences. If external PPO/A2C achieves >0%, this changes the narrative and you must update claims immediately.
  - Est. hours: 4
  - Deadline: April 15
  - Depends on: T0.3 (experiment completion)
  - Acceptance: A table with 10-seed stats for UPI-TRM and at least 2 baselines. Statistical test (bootstrap or Wilson CI) for the UPI-TRM vs best-baseline gap. Written 1-paragraph interpretation.

- [ ] **T2.2 -- Run n-step DQN closer baseline (if feasible)**
  - Description: Implement or adapt an n-step DQN baseline using the external codebase. Run on hard 4x4 with the same TRM backbone. 5-10 seeds. This is the algorithmically closer baseline reviewers asked for.
  - Est. hours: 8
  - Deadline: April 17
  - Depends on: T0.2, T2.1 (to know if it is still needed)
  - Acceptance: n-step DQN results with CI. If infeasible after 6 hours of implementation effort, write a 1-paragraph explanation of why and skip. Do not sink more than 8 hours here.
  - `[PARALLEL-OK with T2.3]`

- [ ] **T2.3 -- Collect isolation study additional seeds**
  - Description: Pull the 10-seed isolation study results. Recompute all metrics (delta_V, delta_pi, argmax agreement) with the expanded seed count. Update figures and tables.
  - Est. hours: 3
  - Deadline: April 16
  - Depends on: T0.3 (experiment completion)
  - Acceptance: Isolation study tables now show 10-seed stats. Error bars shrink visibly. The contraction-vs-no-contraction effect is either confirmed or honestly reported as not significant.
  - `[PARALLEL-OK with T2.1, T2.2]`

- [ ] **T2.4 -- Collect 9x9 additional seeds (if available)**
  - Description: Pull 10-seed 9x9 results if compute allowed. If only 3 seeds, leave 9x9 as supporting evidence and write accordingly.
  - Est. hours: 2
  - Deadline: April 18
  - Depends on: T0.3
  - Acceptance: Either 10-seed 9x9 table, or a decision doc saying "9x9 stays at 3 seeds, demoted to supporting."
  - `[PARALLEL-OK with T2.1, T2.2, T2.3]`

- [ ] **T2.5 -- Draft new Experiments section**
  - Description: Write the experiments section using the new results. Lead with hard 4x4 as the primary comparison. Present the deconfounded 2x2 (contraction x projection) in the main text, not the appendix. Include the trusted-baseline table. 9x9 as supporting if seeds are sufficient, otherwise appendix. Write honest limitation paragraph.
  - Est. hours: 8
  - Deadline: April 20
  - Depends on: T2.1, T2.2, T2.3, T2.4
  - Acceptance: Experiments section is self-contained, 2.5-3 pages. Every number has a seed count and CI. The 1-4 empty toy suite is a sanity check paragraph, not a headlined result.

**Phase 2 hard gate (April 20):** Either the trusted-baseline story confirms UPI-TRM advantage with 10 seeds (proceed as planned), or external baselines achieve >0% success (narrow the claim to stability/efficiency, not capability). Decision must be made and documented before Phase 3 writing begins.

---

### PHASE 3: Theory Integration & Positioning (April 21--27)

- [ ] **T3.1 -- Surface the theory bridge in the main text**
  - Description: Add one displayed equation or short corollary in the main text showing how L_z and n enter the CPI improvement penalty through the advantage-error bound. This is the bridge from Prop 5.1 to Thm 5.6 that reviewers kept missing. Write 1 paragraph of interpretation after the equation.
  - Est. hours: 4
  - Deadline: April 22
  - Depends on: T1.5
  - Acceptance: A reader who skips proofs can still see the complete chain: contraction -> truncation bias -> advantage error -> CPI penalty. No appendix pointer required for the main claim.

- [ ] **T3.2 -- Rewrite novelty positioning**
  - Description: Add a "What is standard vs. what is new" paragraph or subsection. Explicitly state: the API/CPI shell, K-step returns, and conservative mixture are standard. The internal-evaluator truncation decomposition, the L_z^n bias term, and its coupling to CPI mixture updates are new. Reference the specific propositions/theorems.
  - Est. hours: 3
  - Deadline: April 22
  - Depends on: T3.1
  - Acceptance: A reviewer can point to one paragraph and know exactly what you claim is novel. No ambiguity.
  - `[PARALLEL-OK with T3.3]`

- [ ] **T3.3 -- Rewrite Related Work**
  - Description: Restructure around four groups: (a) TRM/HRM and recursive reasoning, (b) lookahead/VIN/unrolled DP -- add the 1-paragraph contrast with VIN, (c) learning to search / plan-editing RL, (d) CPI/trust-region and safe PI. Cite VIN early. State explicitly why inner unroll != environment rollout depth.
  - Est. hours: 5
  - Deadline: April 23
  - Depends on: T3.2
  - Acceptance: VIN is cited and contrasted within the first 3 related-work paragraphs. The distinction between architecture-internal computation and environment lookahead is stated once, clearly.

- [ ] **T3.4 -- L_V controlled experiment (stretch goal)**
  - Description: Attempt a tiny experiment controlling L_V (value-head Lipschitz) to test whether the bound is qualitatively predictive. Budget: max 2 days. If it destabilizes or results are inconclusive, write an explicit concession paragraph instead.
  - Est. hours: 10 (hard cap)
  - Deadline: April 25
  - Depends on: T0.2 (codebase), T2.5 (knowing the experimental story)
  - Acceptance: EITHER a small table showing L_V control correlates with value stability, OR a 1-paragraph concession stating the bound is structural guidance, not quantitatively predictive, and removing any language that implies otherwise.
  - `[PARALLEL-OK with T3.1, T3.2, T3.3]`

- [ ] **T3.5 -- Full draft assembly and coauthor review**
  - Description: Assemble all rewritten sections into a single coherent draft. Check cross-references, notation consistency, and page count. Circulate to all coauthors with a 48-hour review window.
  - Est. hours: 5
  - Deadline: April 27
  - Depends on: T1.2, T1.4, T1.5, T1.6, T2.5, T3.1, T3.2, T3.3, T3.4
  - Acceptance: Complete draft compiles. Page count is 9-11 content pages (will compress in Phase 4). All coauthors have the PDF.

**Phase 3 checkpoint (April 27):** Full draft circulated. Coauthor feedback due April 29.

---

### PHASE 4: Compression, Packaging & Submission (April 28--May 4)

- [ ] **T4.1 -- Incorporate coauthor feedback**
  - Description: Triage coauthor comments. Fix factual errors and unclear passages. Reject scope-expanding suggestions.
  - Est. hours: 4
  - Deadline: April 29
  - Depends on: T3.5 + coauthor feedback
  - Acceptance: All factual corrections addressed. A response note to each coauthor documenting what was accepted/rejected.

- [ ] **T4.2 -- Compress to 9 content pages**
  - Description: Cut the main body to exactly 9 pages. Move detailed proofs, extended tables, and secondary experiments to the appendix. Tighten prose: eliminate redundant explanations, merge short paragraphs, reduce remark blocks. Do NOT cut the deconfounded 2x2 result or the trusted-baseline table -- those stay in the main text.
  - Est. hours: 6
  - Deadline: April 30
  - Depends on: T4.1
  - Acceptance: `\shortonly` mode compiles to exactly 9 content pages. References start on page 10. No critical result is appendix-only.

- [ ] **T4.3 -- Write NeurIPS mandatory checklist**
  - Description: Complete the NeurIPS paper checklist. Answer every question honestly. For "limitations," point to the concession paragraph from T3.4. For "broader impact," keep it brief and honest.
  - Est. hours: 2
  - Deadline: April 30
  - Depends on: T4.2
  - Acceptance: Every checklist item has an answer. No TODOs remain.
  - `[PARALLEL-OK with T4.4]`

- [ ] **T4.4 -- Prepare code/data anonymized ZIP**
  - Description: Bundle baseline scripts, training configs, evaluation scripts, and exact hyperparameters into an anonymized ZIP. Strip author names, institution references, and git history. Include a README with reproduction instructions.
  - Est. hours: 3
  - Deadline: April 30
  - Depends on: T2.5 (final experiment configs)
  - Acceptance: ZIP unzips cleanly. README lists exact commands to reproduce Table 1 and Figure 1. No deanonymizing info.
  - `[PARALLEL-OK with T4.2, T4.3]`

- [ ] **T4.5 -- Final prose pass**
  - Description: One brutal read-through for density and redundancy. Cut filler words, remove hedging that weakens claims without adding honesty, fix any remaining jargon-before-definition violations. Check every figure caption is self-contained.
  - Est. hours: 4
  - Deadline: May 1
  - Depends on: T4.2, T4.3
  - Acceptance: No sentence in the main body can be deleted without losing information. Every figure caption states what the reader should conclude.

- [ ] **T4.6 -- Anonymization audit**
  - Description: Search the entire PDF for author names, institution names, git URLs, acknowledgment references, and file paths. Check figure metadata. Check the bib file for self-citation patterns that reveal identity.
  - Est. hours: 2
  - Deadline: May 2
  - Depends on: T4.5
  - Acceptance: `grep -i` for all author names and institutions returns zero hits in the .tex source and the generated PDF text. No figure contains metadata with author info.

- [ ] **T4.7 -- Freeze title and abstract**
  - Description: Lock the final title and abstract. No further changes. This is what gets submitted on May 4 for the abstract deadline.
  - Est. hours: 1
  - Deadline: May 2
  - Depends on: T4.5, T1.6
  - Acceptance: Title and abstract match the paper content. Both coauthors sign off.

- [ ] **T4.8 -- Submit abstract (May 4 AOE)**
  - Description: Submit the abstract on the NeurIPS 2026 submission portal. Confirm contribution type (General, or Theory as fallback). Record the submission ID.
  - Est. hours: 1
  - Deadline: **May 4 AOE (HARD)**
  - Depends on: T4.7
  - Acceptance: Submission confirmation email received. Submission ID recorded.

- [ ] **T4.9 -- Final compilation and paper upload (May 5)**
  - Description: Final `pdflatex + bibtex + pdflatex + pdflatex`. Verify no warnings, no undefined refs. Upload the single PDF (paper + references + appendix + checklist) and the code ZIP. Treat May 5 as the real paper deadline.
  - Est. hours: 2
  - Deadline: **May 5** (internal), May 6 AOE (NeurIPS hard deadline)
  - Depends on: T4.6, T4.8
  - Acceptance: Upload confirmation. PDF opens and renders correctly. Page count is correct. Checklist is present.

---

### Buffer allocation

| Buffer | Days | Purpose |
|--------|------|---------|
| Phase 1 end | 1 day (April 13) | Absorb rewrite overruns |
| Phase 2 end | 1 day (April 20) | Absorb experiment delays or negative results requiring narrative change |
| Phase 3 end | 1 day (April 27) | Absorb integration surprises |
| Phase 4 end | 2 days (May 3-4 for paper, May 5-6 for upload) | Emergency fixes, submission portal issues |
| **Total buffer** | **5 days** | |

---

### Parallelism summary

All experiment jobs (T0.3 and its children T2.1-T2.4) run on GPU in the background while writing tasks (T1.x, T3.x) execute. The coauthor handles experiment monitoring and result collection. You handle all writing. The only hard serialization points are:

1. T2.5 (writing experiments section) blocks on T2.1-T2.4 (experiment results).
2. T3.5 (full draft assembly) blocks on all Phase 1-3 writing tasks.
3. T4.2 (compression) blocks on T4.1 (feedback incorporation).
4. T4.8 (abstract submit) blocks on T4.7 (title/abstract freeze).

Everything else can overlap.

---

### Decision gates

| Date | Gate | If YES | If NO |
|------|------|--------|-------|
| April 15 | External baseline confirms 0% on hard 4x4? | Proceed with current narrative | Narrow claim to stability/efficiency advantage; reframe experiments around isolation study |
| April 18 | 9x9 has 10 seeds? | Promote to main text | Demote to appendix supporting evidence |
| April 25 | L_V experiment works? | Add small table to experiments | Write concession paragraph; remove quantitative-predictiveness language |
| April 27 | Draft fits in 11 pages pre-compression? | Proceed to compression | Cut one experiment or move theory details to appendix |

---

### Hour budget

| Phase | Hours | Calendar days |
|-------|-------|---------------|
| Phase 0 | 12 | 2 (April 5-6) |
| Phase 1 | 31 | 7 (April 7-13) |
| Phase 2 | 25 | 7 (April 14-20) |
| Phase 3 | 27 | 7 (April 21-27) |
| Phase 4 | 25 | 7 (April 28-May 4) |
| **Total** | **120** | **29 days** |

At 5 hours/day this requires 24 working days out of 29 calendar days. The remaining 5 are buffer.
