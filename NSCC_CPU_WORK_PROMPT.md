# Prompt for the NSCC LLM — CPU-only paper gates, tasks 1–6

Copy the prompt below into the NSCC-side LLM from the repository root.

---

You are working in the `lambda-zero` / `loss_aversion_rl` repository on NSCC.
Read `AGENTS.md`, `PAPER_READINESS.md`, `RESEARCH_ROADMAP.md`,
`CAUSAL_BASELINE_PROTOCOL.md`, and `KNOWN_ISSUES.md` before acting.

## Current verified status

- Qwen-own-delta GRPO replication is confirmed: 2/2 seeds passed the frozen
  selection, OOD-50, and GSM8K gates.
- The frozen unused-configuration suite and the existing framing suite have
  already been opened. They are closed to training, checkpoint selection, and
  method revision.
- The seed-1 SFT/sign-only pilot is exploratory validation evidence only.
- Full matched SFT training is now complete:
  - seeds 1 and 2 reached 30,000 steps;
  - each seed has all 15 checkpoints from 2k through 30k;
  - measured training runtime was 5,422 s (seed 1) and 4,668 s (seed 2),
    a mean of approximately 5,045 s = 1.40 h/seed = 0.168 s/step;
  - total charged cost was approximately 317 SU;
  - the reported remaining allocation is approximately 3,312 SU;
  - provenance manifests are present on NSCC;
  - the earlier seed-1 pilot is preserved separately.
- The full SFT checkpoint grid has not yet been evaluated and the selector has
  not been run. Training completion is not a behavioral result and does not
  establish that SFT beats GRPO.

## Scope and hard constraints

Work only on the six CPU-only tasks below. Do not submit or run GPU jobs. Do
not run model inference. Do not evaluate any SFT, GRPO, sign-only, OOD,
capability, or framing checkpoint. Do not open, inspect model outputs from, or
adapt to any frozen evaluation suite.

Do not weaken or rewrite existing frozen protocols after seeing results.
Append execution records instead. Preserve all user changes and unrelated
untracked files. Do not delete or move checkpoints, predictions, manifests, or
large artifacts. Do not modify `eval/core_exp_refactored.py` casually; if an
inference improvement requires changing the shared estimator, first add
characterization tests and document the proposed change.

Use a new branch named `codex/cpu-paper-gates`. Start with a read-only audit:

1. confirm the current branch, worktree status, and recent commits;
2. locate the two full-SFT run manifests and 30 checkpoint directories;
3. verify seed, step grid, base-model identity, data/config hashes, completion
   status, and pilot/full-run directory separation;
4. record exact paths and hashes without copying large checkpoint files;
5. stop and report rather than guessing if provenance is inconsistent.

Do not use the NSCC login node for heavy CPU computation. Prepare PBS CPU jobs
for simulations or resampling that exceed a short interactive check.

## Task 1 — Freeze a new untouched method-comparison suite

Design and build a new suite that has not been used for training, reward
construction, checkpoint selection, or prior method development. Its purpose
is the final comparison among matched base, magnitude-weighted GRPO, SFT, and
sign-only/scale-matched controls.

Requirements:

- define the source population and sampling frame;
- prevent overlap with training, `test_goods`, OOD-50, and the already-opened
  frozen-unused suite;
- include stable case IDs and paired X/Y perspectives;
- include a frozen semantic-counterbalancing component covering:
  - Yes/No versus explicit keep/trade responses;
  - X/Y versus A/B item labels;
  - reversed item display order;
  - reversed attribute order;
  - a small fixed set of prompt paraphrases;
- predefine strata and sample sizes without inspecting model responses;
- generate a machine-readable manifest containing dataset SHA-256, generation
  seed, generator/code commit, row counts, overlap checks, and freeze time;
- add tests that hard-fail on duplicate IDs, asymmetric X/Y pairs, overlap,
  missing strata, or hash drift;
- do not evaluate any model on the suite.

If a genuinely untouched source population cannot be constructed from current
data, do not relabel an old suite as untouched. Write a blocking design note
with safe alternatives.

## Task 2 — Freeze the method-comparison protocol

Create a prospective protocol before any full-grid SFT result is opened.
Specify:

- eligible models, seeds, and exact checkpoint grids;
- validation-only use of `test_goods`;
- unchanged per-seed checkpoint selector and tie-break rules;
- the rule that all model checkpoints are selected before opening the new
  untouched suite;
- primary outcome: lambda, with eta jointly reported;
- joint distance `d = sqrt(lambda^2 + eta^2)`;
- direct outcomes: consistency, keep-both, trade-both, target agreement, and
  frozen preference preservation;
- pair/good-aware uncertainty and seed-level reporting;
- malformed-output and missing-run handling;
- capability non-inferiority margins;
- confirmatory versus exploratory claim language;
- a prohibition on selecting checkpoints using training loss, GSM8K, IFEval,
  framing, OOD, or the new untouched suite.

Record that the previous OOD-50, frozen-unused, and framing suites are already
opened. The protocol must not claim that a full `test_goods` grid makes the
SFT-versus-GRPO comparison confirmatory.

## Task 3 — Specify and test a scale-matched reward ablation

The existing sign-only `±1` condition changes both case weighting and effective
reward scale relative to `±|delta|`. Without running training:

- characterize the magnitude-reward distribution from the frozen training
  deltas;
- propose a scale-matching rule and justify which moments/effective gradient
  quantity it matches;
- document remaining limitations caused by group-relative normalization and
  zero-advantage groups;
- implement the option without changing the default magnitude-weighted path;
- hard-fail on unsupported algorithm-defining configuration fields;
- add unit tests for reward signs, scale calculation, configuration parsing,
  default-path equivalence, and deterministic manifests;
- update `ABLATION-001` only if its closure condition is genuinely satisfied.

Do not train the new condition and do not tune its scale using evaluation
results.

## Task 4 — Strengthen structural inference

Use existing committed/raw predictions only; do not generate new model
responses. Implement or prepare:

- ID-based X/Y joins with strict integrity assertions;
- pair-clustered bootstrap intervals;
- leave-one-good-out or good-aware robustness;
- seed-aware summaries without pooling away seed variation;
- multi-start and objective diagnostics;
- Jacobian/conditioning reporting;
- estimator-recovery simulations with known lambda and eta;
- coverage, bias, failure-rate, and weak-identification summaries.

Keep the frozen headline estimator intact unless a separately tested,
documented robustness estimator is added. Small checks may run interactively;
large bootstrap/recovery jobs must be submitted as CPU-only PBS jobs.

## Task 5 — Audit whether lambda tells the full story

Build a deterministic CPU-only aggregation from existing predictions/results
that reports, for every available model and seed:

- lambda and eta with uncertainty;
- `d = sqrt(lambda^2 + eta^2)`;
- consistency;
- keep-both and trade-both rates;
- hard-choice outcomes;
- target agreement;
- pseudo-utility alignment W;
- direct agreement with frozen ownership-free preferences where available;
- sample sizes, parse failures, estimator condition number, and provenance.

Generate machine-readable tables first and Markdown/LaTeX tables from those
files. Do not hand-enter paper numbers. Add checks preventing a small lambda
from being described as success when eta, inconsistency, parse failures, or
choice collapse contradict that interpretation.

## Task 6 — Analyze why GRPO may be inefficient

Using existing logs, manifests, rewards, and training metadata only, quantify:

- fraction of groups with zero differentiated task-reward advantage;
- valid-answer and parse-failure rates;
- KL trajectory;
- reward mean, variance, and filtering over steps;
- update signal by `|delta|` bin;
- prompt exposure, optimizer updates, generated completions, token estimates,
  runtime, GPU-hours, and SU cost;
- stability across GRPO seeds;
- comparable SFT exposure/runtime information without treating unequal
  completion counts as matched;
- whether magnitude weighting concentrates updates on high-`|delta|` cases.

Separate measured quantities from estimates. Do not call zero-advantage-group
filtering “DAPO filtering” unless the implementation actually uses DAPO. Do
not conclude that GRPO is useless: state the conditions under which SFT could
dominate this deterministic one-token task and identify the untouched tests
needed to distinguish efficient learning from shortcut/template learning.

## Required documentation updates

Keep these files mutually consistent:

- `RESEARCH_ROADMAP.md`
- `PAPER_READINESS.md`
- `CAUSAL_BASELINE_PROTOCOL.md` using append-only execution entries
- `PROJECT_OVERVIEW.md`
- `KNOWN_ISSUES.md`
- `HISTORY.md`
- `AGENTS.md`
- `CLAUDE.md`

Record full SFT training as complete but full-grid evaluation/selection as
pending. Do not record full-SFT behavioral results until the evaluations
exist.

## Deliverables and acceptance checks

Produce:

1. new-suite data/generator/manifest/tests, or a precise blocking design note;
2. frozen method-comparison protocol;
3. scale-matched reward specification, code, and tests;
4. inference scripts/tests plus CPU PBS submission files where needed;
5. deterministic full-behavior aggregation and generated tables;
6. GRPO signal/efficiency analysis with machine-readable outputs;
7. synchronized documentation;
8. an audit report listing files changed, checks run, unresolved blockers, and
   commands that still require GPU execution.

Before committing:

- run relevant unit tests and integrity checks;
- confirm no GPU job was submitted;
- confirm no frozen suite was evaluated or opened for method development;
- confirm large artifacts and unrelated files are unstaged;
- inspect the staged diff.

Use small, reviewable commits organized by task. Push the branch, but do not
merge it into `main`. Report the branch name and commit hashes so the changes
can be reviewed safely.

---

This CPU phase should finish with the experiment and analysis frozen and
auditable. The later GPU phase should only execute the predeclared grids,
selection, untouched comparison, capability tests, and framing/semantic tests.
