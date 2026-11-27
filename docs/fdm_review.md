# Retained FDM code and archive inventory

The retained implementation uses direct call/put Craig-Sneyd pricing. The final
linear-axis comparison includes paper LN and paper FDM. One-off investigations
are archived under `archive/fdm/`; independent validation remains in `src/`.

## Final result entry points

- `src/fdm.py`: retained pricing implementation.
- `src/fdm_paper.py`: the two published cases, convergence checks and reporting.
- `docs/fdm_paper.md`: equations, settings, final IV/price error tables and commands.
- `results/fdm_paper_direct/comparison.png`: final four-panel linear-axis figure.
- `results/fdm_paper_direct/error_summary.csv`: eight aggregate error rows.
- `results/fdm_paper_direct/comparison.csv`: all 61 pointwise results.
- `results/fdm_paper_direct/summary.json`: full aggregate metrics and reference provenance.
- `results/fdm_paper_direct/tables.json`, `figure7.json`: saved numerical solves.
- `results/fdm_paper_direct/figure7_reference.json` and `.csv`: digitized source reference.
- `results/fdm_paper_direct/validation.json`: validation and source-hash record.
- `tests/test_fdm.py`, `tests/test_fdm_paper.py`: solver/reporting tests.

## Supporting code and its dependencies

| File | Role | Retained status |
|---|---|---|
| `archive/fdm/fdm_diagnosis.py` | Grid/domain/interpolation experiments and linear/log companion plots | Archived; not a core dependency |
| `archive/fdm/fdm_fixed_grid_check.py` | One-off common-log-forward payoff-alignment hypothesis test | Archived; not a core dependency |
| `src/fdm_independent.py` | Independent forward-coordinate Crank-Nicolson validation | Retained validation code |
| `src/fdm_mc_check.py` | Independent conditional-QMC benchmark | Retained validation code |
| `src/extract_paper_figure7.py` | Recover reference coordinates from the PDF | Retained optional provenance tool; saved JSON suffices for reporting |
| `src/fdm_experiment.py` | Historical pilot plus shared convergence/serialization functions | Required shared helpers; historical workflow remains available |
| `src/fdm_refinement.py` | Historical eight-case refinement plus shared domain expansion | Required shared helper; historical workflow remains available |

In particular, `fdm_paper.py` imports `TOLERANCES_BP`, `assess_convergence` and
`json_value` from `fdm_experiment.py`, and `expand_domain` from
`fdm_refinement.py`. Their historical workflows are optional, but these modules
currently contain required shared functions. A later cleanup could extract the
four shared items before archiving the old workflows. This archive change
preserves those dependencies without refactoring the validated numerical code.

The two archived scripts and two independent-validation scripts are not imported
by the final comparison. Their evidence remains together in
`results/fdm_diagnosis/` so the research conclusions stay traceable.
The optional extraction script needs
`pypdf`, which is installed locally but is not declared in `pyproject.toml`;
only re-extraction requires it (`python -m pip install pypdf`).

## Historical and local artifacts

| Files | Purpose and location |
|---|---|
| `archive/fdm/figure7_initial.json` | Archived superseded initial-grid result; not a final-report input |
| `results/fdm_diagnosis/`, `docs/fdm_diagnosis.md` | Retained evidence explaining the unresolved paper discrepancy |
| `results/fdm_pilot/`, `results/fdm_refined/`, `docs/fdm.md` | Retained historical pilot/refinement evidence; both result directories support runnable historical refinement |
| `results/fdm_domain_check.json`, `results/fdm_positive_reference.json` | Retained historical positive-correlation checks |
| `results/environment_check.json`, `results/paper_comparison.json` | Local installation and initial repository-review records; excluded from the submission |
| `archive/fdm/2026-11-26-fdm-parity.md` | Archived development plan; historical, not current instructions |
| `CLAUDE.md` | Local project memory with machine-specific paths; excluded from the submission |
| `tests/test_fdm_experiment.py`, `tests/test_fdm_refinement.py` | Retained helper and historical-workflow tests |

The final comparison directory, diagnosis directory and two historical pilot
directories together occupy only a few MB. Cleanup is a clarity decision, not
a storage requirement. The existing tracked datasets, models, integration code,
Hagan code and ANN code are outside this cleanup and remain unchanged.

## Repository scope

The FDM addition includes the core implementation, shared dependencies, tests,
final results, independent validation, research documentation and explicit
archive. Machine-specific installation records and local project memory are
not required to run the repository. Existing integration/Hagan/ANN code,
datasets and checkpoints are outside this cleanup.

See [the archive map and commands](../archive/fdm/README.md) for the historical
files' original locations and optional reproduction instructions.
