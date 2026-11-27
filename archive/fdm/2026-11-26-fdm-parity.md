# FDM parity pilot implementation plan

> **For agentic workers:** Execute the already-authorized implementation with independent numerical review. Keep changes limited to the FDM module, its tests/experiment, documentation, and local environment setup.

**Goal:** Add a small-batch Craig–Sneyd SABR reference matching the repository's LN/parity price convention.

**Architecture:** `src/fdm.py` solves the beta-one log-coordinate PDE and exposes prices/IV conversion. `src/fdm_experiment.py` compares fixed paper cases and small deterministic samples against LN and existing ANN checkpoints, recording convergence diagnostics and measured runtimes. Existing datasets and models remain unchanged.

**Tech stack:** Python, NumPy, SciPy/LAPACK, PyTorch for existing-model inference, pytest.

**Spec:** User request of 2026-11-26: install dependencies in the confirmed default Python; add repository-style Python modules; use put-call parity for positive correlation; run a small comparison first; exclude SLN.

## Constraints and numerical choices

- F=1, beta=1, paper parameter ranges and the existing ten-strike sampling rule.
- rho<=0: direct OTM expectation. rho>0: evolve put only, define call=P+F-K.
- Original Craig–Sneyd theta=1/2; explicitly documented finite-domain boundary choices.
- Pilot x focus=min(0.5, alpha*sqrt(T)); hold it fixed when refining an individual case.
- Keep numerical failures and distinguish accepted IV from unvalidated estimates.
- Do not launch paper-scale generation, training, or SLN experiments.

## Tasks and checks

- [x] Install manifest dependencies in the user-confirmed Python, verify imports, dependency consistency, native checkpoint inference and one optimizer step.
- [x] Add failing tests for Black refinement, price convention, negative-correlation regression, and input bounds; implement `sabr_cs_price` and `label_row`; run tests.
- [x] Add experiment selection/acceptance tests; implement deterministic sample selection and independent spatial/time/domain comparisons with per-label statuses and measured timing.
- [x] Run the small pilot, record all settings/results and source versions, compare with LN and ANN at identical strikes. Treat quadrature/model differences separately from PDE convergence.
- [x] Independently review the numerical implementation, run the full test suite, document observed limits, and append verified environment changes to CLAUDE.md.

## Completion evidence

- 27 automated tests passed; no existing test suite was present before this work.
- Native PyTorch inference over both 50,000-row transformed N=750 LN test sets and an in-memory training step passed; `pip check` passed.
- Eight pilot cases / 81 labels completed: 55 pass numerical checks, 26 remain rejected. This is an experimental reference, not full-domain certification.
- Independent positive-rho conditional MC and three bounded follow-up X-domain expansions are recorded separately. No failed pilot labels were replaced by unchecked follow-up values.
- Cases 2–7 were evaluated in two additional processes using identical source hashes to shorten elapsed waiting. Per-case wall times may include contention. All inputs, output counts, statuses, source/checkpoint hashes and plot existence were verified after assembly.

## Review focus

- Short maturity and low alpha: mesh resolution must follow the narrow payoff scale.
- rho>0: never use a direct divergent LN call or conflate raw and parity calls.
- Wing prices: IV inversion failure and insufficient refinement remain visible.
- Expanded domains: retain existing interior nodes so domain and grid effects are separable.
- Failures/provenance: a finite number is not automatically an accepted label; reruns do not reuse unversioned numerical caches.
