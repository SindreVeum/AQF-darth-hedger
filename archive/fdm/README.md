# Archived FDM investigations

These files preserve completed or superseded research. The retained solver is
`src/fdm.py`, and the final comparison entry point is `src/fdm_paper.py`.
Independent validation remains active in `src/fdm_independent.py` and
`src/fdm_mc_check.py`. None of the archived modules is imported by those entry
points.

| Archived file | Previous location | Purpose |
|---|---|---|
| `fdm_diagnosis.py` | `src/fdm_diagnosis.py` | Targeted grid/domain/interpolation experiments |
| `fdm_fixed_grid_check.py` | `src/fdm_fixed_grid_check.py` | Common-log-forward payoff-alignment hypothesis test |
| `figure7_initial.json` | `results/fdm_paper_direct/figure7_initial.json` | Superseded, insufficiently refined Figure 7 grid |
| `2026-11-26-fdm-parity.md` | `docs/superpowers/plans/2026-11-26-fdm-parity.md` | Historical implementation plan; its parity convention is not the current direct-comparison specification |

Original results remain in `results/fdm_diagnosis/`, alongside the independent
validation evidence. Historical source hashes describe the code at the time
of measurement; archiving does not rewrite those measurements or hashes.
Only the diagnostic module command and the fixed-grid script's repository-root
lookup changed during the move. The current FDM solver, accepted final figure
and numerical comparison inputs were preserved byte for byte.

Optional reproduction commands, run **from the repository checkout root**:

```sh
python -m archive.fdm.fdm_diagnosis
python -m archive.fdm.fdm_fixed_grid_check --skip-fine
```

These commands rerun the archived experiments and overwrite their respective
diagnostic outputs. They are not needed for the final report or the tests.
The archive is a checkout-level research directory, not part of the installed
`src` package. See [the investigation report](../../docs/fdm_diagnosis.md) for
the conclusions and limitations.
