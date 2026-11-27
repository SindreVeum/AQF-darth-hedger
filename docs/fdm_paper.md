# Direct call/put FDM comparison with McGhee

## Scope and pricing convention

`src/fdm.py` now defaults to `positive_rho_parity=False`: both call and put
payoffs are evolved with the original Craig-Sneyd method for every correlation.
No call is reconstructed as `put + F - K` in this mode. The original boundaries,
spatial stencil, payoff averaging and PDE interpolation are unchanged.

The retained final comparison is `results/fdm_paper_direct/comparison.png`.
All four panels use a linear K/F axis, keeping the original 11 table strikes
and 50 Figure 7 positions. The chart and error summaries are regenerated from
the saved finest-grid results, without rerunning or modifying the solver.
The left panels show our LN, our FDM, paper FDM and paper LN. The right panels
include paper LN minus paper FDM alongside the two original comparisons.
Paper LN uses the printed Table 6 LN column and the digitized Figure 7 LN
curve; it is not replaced by our computed LN values. The corresponding
`paper_ln_iv` and `paper_ln_minus_paper_fdm_iv_bp` columns are in `comparison.csv`.

Both published cases considered here have negative correlation. They already
used direct pricing in the earlier solver, so switching the positive-rho branch
cannot resolve the existing discrepancy with the paper's FDM column.

The model has beta=1, forward F=1 and undiscounted prices. The comparison selects
the OTM put when K<F and the OTM call otherwise, matching Tables 6/7. LN also
uses direct integration at these negative correlations. SLN, ANN training and
full-dataset generation are outside this experiment.

Direct finite-domain call pricing is not automatically the raw positive-rho
SABR payoff expectation. Boundary/asymptotic conditions still matter. The two
negative-rho cases here do not validate the positive-rho pricing convention.
Historical `src.fdm_experiment` and `src.fdm_refinement` explicitly enable the
old parity option so that their saved experiments keep the same meaning.

## Published parameters and reference precision

| Reference | T in years | alpha | nu | rho | Strike positions |
|---|---:|---:|---:|---:|---:|
| Tables 6/7 | 1.6316 | 0.108266 | 0.887903 | -0.464053 | 11 printed strikes |
| Figure 7 | 540/365 | 0.30 | 0.60 | -0.35 | 50 digitized PDF vertices |

Table 6 IVs and Table 7 prices are printed to 0.01 percent; each column has a
half-rounding unit of 0.5 bp. The input parameters and strikes are also rounded.
Table 7 is a separately rounded price reference: do not replace it with prices
computed from the printed IV column.

Figure 7's vector paths share 50 X coordinates. Axis calibration uses 14
logarithmic strike ticks and nine IV ticks. The recovered strike range is
approximately 0.1107358512 to 4.8229660718. No guessed exact strike linspace is
substituted for those recovered positions. The paper does not say whether all
50 vertices were priced independently or obtained by interpolation.

`src/extract_paper_figure7.py` records the source PDF hash, raw coordinates,
axis maps, colours, strike bounds and coordinate-rounding bounds in
`results/fdm_paper_direct/figure7_reference.json`. It checks the 365-day basis
against the independent Hagan curve: maximum difference 0.021902 IV bp,
versus 1.204467 IV bp for a 360-day basis. This choice is not fitted to FDM.
The approximately 0.01594 IV bp bound for the FDM vertical coordinate concerns
PDF coordinate rounding only, not the author's pricing or interpolation error;
strike-coordinate uncertainty is separately recorded. Thus the Figure 7
reference is a digitized plot, not the author's original numerical output.

## Numerical checks and error units

The Tables 6/7 medium grid is nx=400, ny=200, nt=400; its finest estimate uses
nx=800, ny=400, nt=400. Figure 7 uses medium nx=640, ny=320 and fine nx=1280,
ny=640, with nt=400. Spatial counts are intervals. Both use sinh grids with
X half-width 8, Y half-width 6, X focus 0.5 and Y focus 1. Forward fitting is
disabled to retain the original negative-rho stencil.

The first Figure 7 trial used the Tables 6/7 grid sizes. Its maximum spatial
change was 1.216587 IV bp (29 of 50 points failed the 0.5 bp threshold). The
preceding-to-current change ratio was 3.9602-4.0501 across the 50 points,
consistent with second-order spatial convergence. That trial is retained as
`archive/fdm/figure7_initial.json`; only spatial resolution is increased in the final run.

Seven solves separate the checks: coarse/medium/fine spatial grids, half/double
time steps on the medium spatial grid, and independent X/Y domain expansions
that preserve old interior nodes. Acceptance thresholds remain 0.5 IV bp for
space, 0.05 for time, and 0.1 each for X/Y domain. The time/domain checks are on
the medium grid; they are numerical diagnostics, not rigorous exact-model error
bounds. The paper does not disclose enough boundary/grid detail to reproduce
its internal configuration exactly.

For any paired values a and b, the report records `(a-b)*10000`, its mean
absolute value, root mean square and maximum absolute value. For IV, 1 bp is
0.0001 annual decimal volatility; for price at F=1, 1 bp is 0.0001 price units.
All finest-grid estimates remain visible; metrics restricted to accepted FDM
points are separate. Failed accepted labels remain NaN/null.

Figure 7 provides an IV reference only. Its price comparison below uses Black
OTM prices inferred from that digitized IV curve, explicitly separate from
the independently printed Table 7 prices. It is not a published price table.

The digitized paper Figure 7 LN and FDM curves themselves differ by a maximum
32.21163 IV bp (MAE 5.01069 bp). At the first strike, their IVs are respectively
64.3796177% and 64.0575014%. Our LN differs from the digitized paper LN by at most
0.14440 IV bp, while the independent Hagan cross-check differs by at most
0.02190 bp. Thus a roughly 32 bp discrepancy against the Figure 7 FDM curve is
consistent with the original plot, not evidence of a curve-label swap. The
paper's approximately one-price-bp agreement statement concerns the separate
Tables 6/7 example and must not be generalized to this figure.

## Measured finest-grid comparisons

All 61 requested points passed the retained numerical checks: 11/11 in Tables
6/7 and 50/50 in Figure 7. Maximum spatial/time/X-domain/Y-domain changes were
respectively 0.387453/0.005143/0.000219/4.97e-10 IV bp for Tables 6/7 and
0.473972/0.009687/0.000207/2.50e-11 IV bp for Figure 7. Passing these diagnostics
does not certify exact-model accuracy or reproduce the paper's undisclosed grid.
The 38 automated tests passed, including independent direct-call matrix-exponential
checks and a report regression for all-failed data. `validation.json` records
input/configuration checks and price-to-IV round trips. Its wrapper-hash note
records the Figure-7-only grid change after the Tables run and the final plot
formatting after both runs; the solver/integrator hashes and all saved grid
settings still match. Reporting was rerun without repeating any PDE solves.

All IV differences below are in IV bp; these are differences between numerical
methods/references, not errors against an exact SABR oracle.

| Case | Comparison | MAE | RMSE | Maximum absolute difference | Strike at maximum |
|---|---|---:|---:|---:|---:|
| Tables 6/7 | Our LN - our FDM | 3.131810 | 4.073699 | 7.117317 | 1.5088 |
| Tables 6/7 | Our FDM - printed FDM | 5.217585 | 5.705618 | 8.015529 | 1.2941 |
| Figure 7 | Our LN - our FDM | 2.418073 | 2.837396 | 5.033200 | 1.9379358763 |
| Figure 7 | Our FDM - digitized FDM | 5.581524 | 8.000628 | 31.792408 | 0.1107358512 |

Price differences below are in price bp at F=1:

| Case | Comparison | MAE | RMSE | Maximum absolute difference | Strike at maximum |
|---|---|---:|---:|---:|---:|
| Tables 6/7 | Our LN - our FDM | 0.376728 | 0.601697 | 1.549125 | 0.9520 |
| Tables 6/7 | Our FDM - printed FDM | 0.518863 | 0.610972 | 1.151297 | 0.8166 |
| Figure 7 | Our LN - our FDM | 0.291610 | 0.474381 | 1.428444 | 1.0724231938 |
| Figure 7 | Our FDM - price inferred from paper IV | 1.113719 | 2.173691 | 8.074371 | 1.2647478122 |

At Figure 7's first strike, the maximum 31.792408 IV bp discrepancy corresponds
to only 0.089885 price bp because Vega is small. The maximum price discrepancy
occurs at a different strike, as the table shows. These method/reference
differences are not errors against an exact SABR value.

The Tables 6/7 finest prices are bit-for-bit identical to the earlier saved
negative-rho result in `results/fdm_refined/details.json`. Both old and new
implementations already used direct call/put evolution for this case. Removing
the positive-rho conversion therefore does not remove the discrepancy with the
paper. Figure 7 is also negative-rho, so it does not test the choice between
positive-rho raw and parity call prices.

## Follow-up discrepancy investigation

See [fdm_diagnosis.md](fdm_diagnosis.md) for the independent conditional-QMC
and forward-coordinate PDE checks, controlled boundary/grid/interpolation
experiments, and linear/log-axis comparison of the same Figure 7 strike points.
That investigation preserves the production solver and these saved results.

## Reproduction commands

From the repository root:

```sh
python -m src.fdm_paper
python -m src.fdm_paper --report-only
```

The first command recomputes both cases. `--case tables` and `--case figure7`
select individual cases. The second command only reads saved solves and writes
`summary.json`, `comparison.csv` and `comparison.png`. Per-case JSON includes
all grid results, timings, parameters, checks and source hashes.

`error_summary.csv` provides eight compact rows: two cases, two requested
comparisons, and IV/price units. Each row includes MAE, RMSE, maximum absolute
difference, its strike, sample count and reference provenance. In
`comparison.csv`, the `paper_fdm_price` column remains reserved for printed
Table 7 prices; Figure 7 inferred prices have their own `paper_iv_implied_price`
and `fdm_minus_paper_iv_implied_price_bp` columns. All 61 points passed the saved
convergence checks, so all-estimate and accepted-only metrics currently agree.

The final reporting update passed 39 tests, including a regression preventing
inferred Figure 7 prices from being presented as printed paper prices. The
earlier 38-test count describes the original numerical-comparison validation.

See [the retained code and archive inventory](fdm_review.md) for core
dependencies, independent validation and archived historical investigations.

Optional reference extraction requires the already-installed `pypdf` package:

```sh
python -m src.extract_paper_figure7 --pdf "../McGhee 2020 ANN SABR.pdf"
```

The saved reference is sufficient to run the numerical comparison without a PDF
parser. Original LN/Hagan datasets, ANN checkpoints and historical FDM outputs
are not overwritten.
